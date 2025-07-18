import ray
import itertools
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils import get_tokenizer, safe_ray_get
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    default_data_collator
)
from openrlhf.utils.distributed_sampler import DistributedSampler
from torch.utils.data import DataLoader
from openrlhf.models.loss import DPOLoss
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision import transforms
import base64
import io
from functools import partial

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
        self.labeler_image_processor = ray.get(self.labeler_model_group._actor_handlers[0].get_image_processor())
        self.policy_image_processor = ray.get(self.policy_model_group._actor_handlers[0].get_image_processor())
        self.reference_image_processor = ray.get(self.reference_model_group._actor_handlers[0].get_image_processor())
        
        # vLLM engines for optimized inference
        self.labeler_vllm_engines = labeler_vllm_engines
        self.policy_vllm_engines = policy_vllm_engines
        self.reference_vllm_engines = reference_vllm_engines
        
        self.labeler_tokenizer = get_tokenizer(labeler_pretrain, None, "left", strategy, use_fast=not self.args.disable_fast_tokenizer)
        self.target_tokenizer = get_tokenizer(target_pretrain, None, "left", strategy, use_fast=not self.args.disable_fast_tokenizer)
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

    def _data_process_fn(self, item, policy_image_processor, labeler_image_processor):
        image = item['image']
        # 如果是 torch.Tensor
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        # 如果是 numpy array
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        # 如果是 base64 字符串
        elif isinstance(image, str):
            if image.strip().startswith('data:image'):
                # base64编码的图片
                image_data = image.split(',')[1]
                image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            else:
                # 假定是图片路径
                image = Image.open(image)
        # 如果已经是 PIL Image，直接用
        elif isinstance(image, Image.Image):
            pass
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # policy/reference处理
        processed = policy_image_processor(
            images=item['image'],
            text=item['question'],
            return_tensors="pt",
            padding="max_length",
            max_length=self.args.max_len,
            truncation=True
        )
        # labeler处理
        labeler_processed = labeler_image_processor(
            images=item['image'],
            text=item['question'],
            return_tensors="pt",
            padding="max_length",
            max_length=self.args.max_len,
            truncation=True
        )

        return {
            "input_ids": processed["input_ids"].squeeze(0),  # [seq_len]
            "attention_mask": processed["attention_mask"].squeeze(0),
            "pixel_values": processed["pixel_values"].squeeze(0),  # [3, H, W]
            "labeler_pixel_values": labeler_processed["pixel_values"].squeeze(0),
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
            partial(
                self._data_process_fn,
                policy_image_processor=self.policy_image_processor,
                labeler_image_processor=self.labeler_image_processor
            ),
            batched=False,  # 每次处理一个样本
            remove_columns=dataset.column_names
        )
        self.processed_dataset = processed_dataset
        # 构建idx到样本的映射
        self.idx2data = {str(item["idx"]): item for item in processed_dataset}
        return processed_dataset

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

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            try:
                # Step 1: Generate candidate responses using policy model group
                # Use policy model group's multiple actors for parallel generation
                # Each actor will generate n_candidates responses for their assigned questions

                # Set policy model to eval mode for generation
                self.policy_model_group.async_run_method(method_name="set_eval")

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
                results = list(itertools.chain.from_iterable(safe_ray_get(futures, desc="policy_model_group.forward")))
                
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
                    list(itertools.chain.from_iterable(safe_ray_get(simple_declarative_sentences_sub_batches, desc="labeler_model_group.divide")))
                
                # Conquer step: convert facts to simple yes/no questions
                respons_to_simple_questions_sub_batches = self.labeler_model_group.async_run_method_batch(
                    method_name='conquer',
                    q_facts_batch=simple_declarative_sentences,
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=0.0,
                    do_sample=False
                )
                respons_to_simple_questions = \
                    list(itertools.chain.from_iterable(safe_ray_get(respons_to_simple_questions_sub_batches, desc="labeler_model_group.conquer")))
                
                # YesNo step: answer simple questions with yes/no
                yesno_results_sub_batches = self.labeler_model_group.async_run_method_batch(
                    method_name='YesNo',
                    batch=respons_to_simple_questions,
                    dataset=self.idx2data,
                    max_new_tokens=3,
                    temperature=0.0,
                    do_sample=False
                )
                yesno_results = list(itertools.chain.from_iterable(safe_ray_get(yesno_results_sub_batches, desc="labeler_model_group.yesno")))

                # Combine step: select preferred/inferior response pairs
                prefer_inferior_response_sub_batches = \
                    self.labeler_model_group.async_run_method_batch(
                        method_name='combine',
                        batch=yesno_results,
                        seed=42
                    )
                prefer_inferior_response = \
                    list(itertools.chain.from_iterable(safe_ray_get(prefer_inferior_response_sub_batches, desc="labeler_model_group.combine")))
                
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
                ref_chosen_futures = self.reference_model_group.async_run_method_batch(
                    method_name="batch_logp",
                    contexts=contexts,
                    targets=preferred_targets,
                    images=images,
                    requires_grad=False
                )
                reference_chosen_logps = torch.tensor(list(itertools.chain.from_iterable(safe_ray_get(ref_chosen_futures, desc="reference_model_group.batch_logp chosen"))))

                ref_rejected_futures = self.reference_model_group.async_run_method_batch(
                    method_name="batch_logp",
                    contexts=contexts,
                    targets=inferior_targets,
                    images=images,
                    requires_grad=False
                )
                reference_rejected_logps = torch.tensor(list(itertools.chain.from_iterable(safe_ray_get(ref_rejected_futures, desc="reference_model_group.batch_logp reject"))))
                
                # Calculate policy model log probabilities (with gradients for training)
                policy_chosen_futures = self.policy_model_group.async_run_method_batch(
                    method_name="batch_logp",
                    contexts=contexts,
                    targets=preferred_targets,
                    images=images,
                    requires_grad=True
                )
                policy_chosen_logps = torch.tensor(list(itertools.chain.from_iterable(safe_ray_get(policy_chosen_futures, desc="policy_model_group.batch_logp chosen"))))
                policy_rejected_futures = self.policy_model_group.async_run_method_batch(
                    method_name="batch_logp",
                    contexts=contexts,
                    targets=inferior_targets,
                    images=images,
                    requires_grad=True
                )
                policy_rejected_logps = torch.tensor(list(itertools.chain.from_iterable(safe_ray_get(policy_rejected_futures, desc="policy_model_group.batch_logp reject"))))
                
                assert policy_chosen_logps.shape == policy_rejected_logps.shape == reference_chosen_logps.shape == reference_rejected_logps.shape, \
                f"Shape mismatch: policy_chosen_logps {policy_chosen_logps.shape},\
                    policy_rejected_logps {policy_rejected_logps.shape},\
                    reference_chosen_logps {reference_chosen_logps.shape},\
                    reference_rejected_logps {reference_rejected_logps.shape}"
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
                self.policy_model_group.async_run_method(method_name="set_train")
                loss_cpu = loss.detach().cpu()
                self.policy_model_group.async_run_method(method_name="backward_and_optimize", loss=loss_cpu)
                
                # Step 6: Broadcast updated weights to all policy actors (NCCL)
                self.policy_model_group.broadcast_weights()
                self._broadcast_to_vllm()  # 同步最新权重到vllm engine
                
                # Step 7: Logging and checkpointing
                if batch_idx % 100 == 0:
                    print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                    self.save_logs_and_checkpoints()
            except Exception as e:
                import traceback
                print(f"[Error] Exception occurred at batch {batch_idx}: {e}")
                traceback.print_exc()
