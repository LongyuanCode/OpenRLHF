import ray
import itertools
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.utils import get_tokenizer
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    default_data_collator
)
from openrlhf.utils.distributed_sampler import DistributedSampler
from torch.utils.data import DataLoader
from openrlhf.models.loss import DPOLoss
import torch

@ray.remote
class RLAIFTrainer:
    def __init__(
        self,
        labeler_pretrain,
        target_pretrain,
        strategy: DeepspeedStrategy,
        labeler_model_group: RayActorGroup,
        policy_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        labeler_vllm_engines=None,
        policy_vllm_engines=None,
        reference_vllm_engines=None,
        **generate_kwargs
    ) -> None:
        self.strategy = strategy
        self.args = strategy.args
        self.labeler_model_group = labeler_model_group
        self.policy_model_group = policy_model_group
        self.reference_model_group = reference_model_group
        
        # vLLM engines for optimized inference
        self.labeler_vllm_engines = labeler_vllm_engines
        self.policy_vllm_engines = policy_vllm_engines
        self.reference_vllm_engines = reference_vllm_engines
        
        self.labeler_tokenizer = get_tokenizer(labeler_pretrain, None, "left", strategy, use_fast=not self.args.disable_fast_tokenzier)
        self.target_tokenizer = get_tokenizer(target_pretrain, None, "left", strategy, use_fast=not self.args.disable_fast_tokenzier)
        self.target_pretrain_name = target_pretrain
        self.labeler_pretrain_name = labeler_pretrain
    
    def _broadcast_to_vllm(self):
        """Broadcast model weights to vLLM engines for policy model"""
        if self.policy_vllm_engines is not None:
            if self.strategy.args.vllm_enable_sleep:
                from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
                batch_vllm_engine_call(self.policy_vllm_engines, "wake_up")

            # Broadcast from policy model group master to vLLM engines
            self.policy_model_group.async_run_method(method_name="broadcast_to_vllm")

            if self.strategy.args.vllm_enable_sleep:
                batch_vllm_engine_call(self.policy_vllm_engines, "sleep")

    def _data_process_fn(self, item):
        processor = AutoProcessor(self.target_pretrain_name, trust_remote_code=True)
        processed = processor(
            images=item['image'],
            text=item['question'],
            return_tensor="pt",
            padding="max_lenght",
            max_length=self.args.max_len,
            truncation=True
        )

        return {
            "input_ids": processed["input_ids"].squeeze(0),  # [seq_len]
            "attention_mask": processed["attention_mask"].squeeze(0),
            "pixel_values": processed["pixel_values"].squeeze(0),  # [3, H, W]
            "question": item["question"],
            "idx": item["idx"]
        }

    def prepare_dataset(self):
        raw_dataset = load_dataset(self.args.dataset_name)
        if "train" in raw_dataset:
            dataset = raw_dataset["train"]
        else:
            dataset = raw_dataset

        # 处理每个样本
        processed_dataset = dataset.map(
            self._data_process_fn,
            batched=False,  # 每次处理一个样本
            remove_columns=dataset.column_names
        )
        self.processed_dataset = processed_dataset
        # 构建idx到样本的映射
        self.idx2data = {str(item["idx"]): item for item in processed_dataset}
        return processed_dataset

    def get_logp_batch(self, model, contexts, targets, images=None, requires_grad=True):
        """
        批量计算logp，支持多个样本同时推理
        Args:
            model: 模型
            contexts: List[str], 上下文列表
            targets: List[str], 目标文本列表
            images: List, 图片列表
            requires_grad: bool, 是否需要梯度
        Returns:
            torch.Tensor: 每个样本的logp
        """
        # 构造batch输入
        prompts = []
        target_lengths = []
        for context, target in zip(contexts, targets):
            prompt = context
            prompts.append(prompt)
            target_lengths.append(len(self.target_tokenizer.encode(target)))
        
        # 批量编码
        input_ids = self.target_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.target_tokenizer.model_max_length
        ).input_ids
        
        target_ids = self.target_tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.target_tokenizer.model_max_length
        ).input_ids
        
        # 拼接context+target
        input_ids = torch.cat([input_ids, target_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        model_inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
        
        # 处理图片
        if images is not None and any(img is not None for img in images):
            # 这里需要根据你的图片处理逻辑调整
            # 假设images是PIL Image列表，需要转换为tensor
            pixel_values = []
            for img in images:
                if img is not None:
                    # 根据你的图片预处理逻辑处理
                    pixel_values.append(img)
                else:
                    # 处理空图片的情况
                    pixel_values.append(torch.zeros(3, 224, 224))  # 示例尺寸
            pixel_values = torch.stack(pixel_values)
            model_inputs["pixel_values"] = pixel_values
        
        # 前向推理
        if requires_grad:
            outputs = model(**model_inputs)
        else:
            with torch.no_grad():
                outputs = model(**model_inputs)
        
        logits = outputs.logits  # (batch, seq_len, vocab_size)
        
        # 计算每个样本的logp
        batch_logps = []
        for i in range(len(contexts)):
            # 取对应样本的target部分logits
            target_start = input_ids.shape[1] - target_ids.shape[1]
            target_logits = logits[i, target_start - 1:, :]
            target_tokens = target_ids[i, :]
            
            # 计算logp
            log_probs = torch.log_softmax(target_logits, dim=-1)
            target_logp = log_probs.gather(1, target_tokens.unsqueeze(-1)).squeeze(-1)
            
            # 只计算非padding token的logp
            mask = target_tokens != self.target_tokenizer.pad_token_id
            sample_logp = (target_logp * mask.float()).sum()
            
            if not requires_grad:
                sample_logp = sample_logp.detach()
            
            batch_logps.append(sample_logp)
        
        return torch.stack(batch_logps)

    def save_logs_and_checkpoints(self):
        pass

    def train(self):
        train_dataset = self.prepare_dataset()
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=default_data_collator,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        # Initialize all model groups with max_steps for training
        max_steps = len(train_dataloader) * self.args.num_epochs if hasattr(self.args, 'num_epochs') else 1000
        
        refs = []
        refs.extend(self.labeler_model_group.async_init_model_from_pretrained())
        refs.extend(self.policy_model_group.async_init_model_from_pretrained(max_steps=max_steps, vllm_engines=self.policy_vllm_engines))
        refs.extend(self.reference_model_group.async_init_model_from_pretrained())
        ray.get(refs)

        # Set policy model to eval mode for generation
        self.policy_model_group.async_run_method(method_name="set_eval")

        for batch_idx, batch in enumerate(train_dataloader):
            # Step 1: Generate candidate responses using policy model group
            # Use policy model group's multiple actors for parallel generation
            # Each actor will generate n_candidates responses for their assigned questions
            
            futures = self.policy_model_group.async_run_method_batch(
                method_name='forward',
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                questions=batch['question'],
                item_indexes=batch['idx'],
                pixel_values=batch.get('pixel_values', None),
                n_candidates=self.args.n_candidates,
                max_new_tokens=self.args.max_new_tokens,
                temperature=self.args.policy_generate_temperature,
                do_sample=self.args.do_sample,
            )
            results = list(itertools.chain.from_iterable(ray.get(futures)))
            
            # Step 2: Labeler processing - strictly reuse LabelerModelActor methods
            # Dvide step: extract facts from question-candidate_response pairs
            simple_declarative_sentences_sub_batches = self.labeler_model_group.async_run_method_batch(
                method_name='divide',
                q_candidate_a=results,
                max_new_tokens=self.args.max_new_tokens,
                temperature=0.0,
                do_sample=False
            )
            simple_declarative_sentences = \
                list(itertools.chain.from_iterable(ray.get(simple_declarative_sentences_sub_batches)))
            
            # Conquer step: convert facts to simple yes/no questions
            respons_to_simple_questions_sub_batches = self.labeler_model_group.async_run_method_batch(
                method_name='conquer',
                q_facts_batch=simple_declarative_sentences,
                max_new_tokens=self.args.max_new_tokens,
                temperature=0.0,
                do_sample=False
            )
            respons_to_simple_questions = \
                list(itertools.chain.from_iterable(ray.get(respons_to_simple_questions_sub_batches)))
            
            # YesNo step: answer simple questions with yes/no
            yesno_results_sub_batches = self.labeler_model_group.async_run_method_batch(
                method_name='YesNo',
                batch=respons_to_simple_questions,
                dataset=self.idx2data,
                max_new_tokens=3,
                temperature=0.0,
                do_sample=False
            )
            yesno_results = list(itertools.chain.from_iterable(ray.get(yesno_results_sub_batches)))

            # Combine step: select preferred/inferior response pairs
            prefer_inferior_response_sub_batches = \
                self.labeler_model_group.async_run_method_batch(
                    method_name='combine',
                    batch=yesno_results
                )
            prefer_inferior_response = \
                list(itertools.chain.from_iterable(ray.get(prefer_inferior_response_sub_batches)))
            
            # Step 3: Set policy model to train mode for DPO training
            self.policy_model_group.async_run_method(method_name="set_train")

            # Step 4: DPO loss calculation and training
            dpo_loss_fn = DPOLoss(beta=1.0)
            
            # Collect all sample data
            contexts = []
            preferred_targets = []
            inferior_targets = []
            images = []
            
            for item in prefer_inferior_response:
                idx = item['idx']
                question = item['question']
                preferred = item['1']
                inferior = item['0']
                image = self.idx2data.get(str(idx), {}).get('pixel_values', None)
                
                contexts.append(question)
                preferred_targets.append(preferred)
                inferior_targets.append(inferior)
                images.append(image)
            
            # Calculate log probabilities using reference model (no gradients needed)
            # 使用Ray分布式并行logp计算
            ref_chosen_futures = self.reference_model_group.async_run_method_batch(
                method_name="batch_logp",
                contexts=contexts,
                targets=preferred_targets,
                images=images
            )
            ref_rejected_futures = self.reference_model_group.async_run_method_batch(
                method_name="batch_logp",
                contexts=contexts,
                targets=inferior_targets,
                images=images
            )
            reference_chosen_logps = torch.tensor(list(itertools.chain.from_iterable(ray.get(ref_chosen_futures))))
            reference_rejected_logps = torch.tensor(list(itertools.chain.from_iterable(ray.get(ref_rejected_futures))))
            
            # Calculate policy model log probabilities (with gradients for training)
            policy_chosen_logps = self.get_logp_batch(
                self.policy_model_group.master.model, 
                contexts, 
                preferred_targets, 
                images, 
                requires_grad=True
            )
            policy_rejected_logps = self.get_logp_batch(
                self.policy_model_group.master.model, 
                contexts, 
                inferior_targets, 
                images, 
                requires_grad=True
            )
            
            # Calculate DPO loss
            loss, _, _ = dpo_loss_fn(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps
            )
            
            # Step 5: Backward pass and optimization using DeepSpeed
            # This will be handled by the policy model group's master actor
            # Convert loss to CPU and detach for serialization
            loss_cpu = loss.detach().cpu()
            self.policy_model_group.async_run_method(method_name="backward_and_optimize", loss=loss_cpu)
            
            # Step 6: Broadcast updated weights to all policy actors (NCCL)
            self.policy_model_group.broadcast_weights()
            self._broadcast_to_vllm()  # 同步最新权重到vllm engine
            
            # Step 7: Logging and checkpointing
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                self.save_logs_and_checkpoints()
            
            
            