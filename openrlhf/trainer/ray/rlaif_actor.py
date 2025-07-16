import torch
import random
import math
from transformers.trainer import get_scheduler
import ray

from launcher import BaseModelActor
from openrlhf.models import Actor
from openrlhf.utils.deepspeed import DeepspeedStrategy

from typing import Dict

class TargetModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps=None, vllm_engines=None):
        raise NotImplementedError

    def sync_weights(self):
        """
        使用 torch.distributed.broadcast (NCCL backend) 同步权重，兼容 ZeRO-3。
        group 内所有 actor 都需执行本方法。
        master actor 训练后，调用 RayActorGroup.broadcast_weights()，所有 worker actor 自动同步。
        """
        import torch
        from openrlhf.utils.distributed_util import torch_dist_barrier_and_cuda_sync
        try:
            import deepspeed
            zero_stage = getattr(self.strategy.args, 'zero_stage', 0)
        except ImportError:
            zero_stage = 0
        model = self.model
        if zero_stage == 3:
            # ZeRO-3: 参数分片，需gather后broadcast
            for name, param in model.named_parameters():
                with deepspeed.zero.GatheredParameters([param], enabled=True):
                    if param.ds_numel is not None and param.ds_numel > 0:
                        torch.distributed.broadcast(param.data, 0)
        else:
            # ZeRO-0/1/2: 直接broadcast
            for name, param in model.named_parameters():
                torch.distributed.broadcast(param.data, 0)
        torch_dist_barrier_and_cuda_sync()

    def _sync_weights_direct(self):
        self.sync_weights()

    def _sync_weights_zero3(self):
        self.sync_weights()
    
    def _generate_with_vllm_engines(self, questions, item_indexes, pixel_values, n_candidates, **gen_kwargs):
        """使用vLLM engine进行生成，支持多模态（图片+文本）和纯文本"""
        from vllm import SamplingParams
        from torchvision import transforms
        to_pil = transforms.ToPILImage()

        batch_size = len(questions)
        all_candidates = [[] for _ in range(batch_size)]
        has_images = pixel_values is not None and len(pixel_values) > 0

        if has_images:
            # 多模态：图片+问题
            prompts = []
            multi_modal_data_list = []
            for i, question in enumerate(questions):
                prompt = f"USER: <image>\n{question}\nASSISTANT:"
                prompts.append(prompt)
                image_tensor = pixel_values[i]
                image_pil = to_pil(image_tensor)
                multi_modal_data_list.append({"image": image_pil})

            sampling_params = SamplingParams(
                temperature=gen_kwargs.get('temperature', 0.7),
                top_p=gen_kwargs.get('top_p', 0.9),
                top_k=-1,
                max_tokens=gen_kwargs.get('max_new_tokens', 128),
                min_tokens=1,
                skip_special_tokens=False,
            )

            # 对每个问题采样 n_candidates 次，每次不同seed
            for seed in range(n_candidates):
                sampling_params.seed = seed
                refs = []
                engine_batch_size = (len(prompts) + len(self.vllm_engines) - 1) // len(self.vllm_engines)
                for i, engine in enumerate(self.vllm_engines):
                    start_idx = i * engine_batch_size
                    end_idx = min((i + 1) * engine_batch_size, len(prompts))
                    batch_prompts = prompts[start_idx:end_idx]
                    batch_multi_modal_data = multi_modal_data_list[start_idx:end_idx]
                    if batch_prompts:
                        refs.append(engine.generate_multimodal.remote(
                            batch_prompts,
                            sampling_params,
                            batch_multi_modal_data
                        ))
                responses = []
                for ref in refs:
                    outputs = ray.get(ref)
                    for output in outputs:
                        responses.append(output.outputs[0].text)
                for i in range(batch_size):
                    if i < len(responses):
                        all_candidates[i].append(responses[i])
            return all_candidates

        # 纯文本模式（兼容原有逻辑）
        prompts = []
        for question in questions:
            for _ in range(n_candidates):
                prompt = f"Question: {question}\nAnswer:"
                prompts.append(prompt)

        sampling_params = SamplingParams(
            temperature=gen_kwargs.get('temperature', 0.7),
            top_p=gen_kwargs.get('top_p', 0.9),
            top_k=-1,
            max_tokens=gen_kwargs.get('max_new_tokens', 128),
            min_tokens=1,
            skip_special_tokens=False,
        )

        refs = []
        batch_size_per_engine = (len(prompts) + len(self.vllm_engines) - 1) // len(self.vllm_engines)
        for i, engine in enumerate(self.vllm_engines):
            start_idx = i * batch_size_per_engine
            end_idx = min((i + 1) * batch_size_per_engine, len(prompts))
            batch_prompts = prompts[start_idx:end_idx]
            if batch_prompts:
                refs.append(engine.generate.remote(
                    sampling_params=sampling_params,
                    prompts=batch_prompts
                ))
        all_responses = []
        for ref in refs:
            outputs = ray.get(ref)
            for output in outputs:
                all_responses.append(output.outputs[0].text)
        batch_size = len(questions)
        all_candidates = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            start_idx = i * n_candidates
            end_idx = start_idx + n_candidates
            all_candidates[i] = all_responses[start_idx:end_idx]
        return all_candidates
    
    def _generate_with_model(self, input_ids, attention_mask, pixel_values, n_candidates, **gen_kwargs):
        """使用模型直接生成"""
        batch_size = input_ids.shape[0]
        all_candidates = [[] for _ in range(batch_size)]
        
        # 为每个样本生成 n_candidates 个候选回复
        for i in range(batch_size):
            # 获取当前样本的输入
            sample_input_ids = input_ids[i:i+1]  # (1, seq_len)
            sample_attention_mask = attention_mask[i:i+1]  # (1, seq_len)
            sample_pixel_values = pixel_values[i:i+1] if pixel_values is not None else None
            
            # 生成 n_candidates 个候选回复
            for j in range(n_candidates):
                with torch.no_grad():
                    # 使用模型生成回复
                    generated_ids = self.model.generate(
                        input_ids=sample_input_ids,
                        attention_mask=sample_attention_mask,
                        pixel_values=sample_pixel_values,
                        max_new_tokens=gen_kwargs.get('max_new_tokens', 128),
                        temperature=gen_kwargs.get('temperature', 0.7),
                        top_p=gen_kwargs.get('top_p', 0.9),
                        do_sample=gen_kwargs.get('do_sample', True),
                        pad_token_id=self.model.config.pad_token_id,
                        eos_token_id=self.model.config.eos_token_id,
                    )
                    
                    # 解码生成的文本
                    generated_text = self.tokenizer.decode(
                        generated_ids[0][sample_input_ids.shape[1]:], 
                        skip_special_tokens=True
                    )
                    all_candidates[i].append(generated_text)
        
        return all_candidates

class PolicyModelActor(TargetModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps=None, vllm_engines=None):
        self._setup_distributed(strategy)
        args = strategy.args
        
        # Store vLLM engines for generation
        self.vllm_engines = vllm_engines
        
        # Create model for training (with DeepSpeed support)
        model = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.use_flash_attn_policy,
            bf16=strategy.args.target_bf16,
            load_in_4bit=strategy.args.target_load_in_4bit,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            # TODO: packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        
        strategy.print('model:\n', model)

        # Prepare training model with optimizer and scheduler
        if max_steps is not None:
            # Create optimizer
            train_optim = strategy.create_optimizer(
                model, 
                lr=args.actor_learning_rate, 
                betas=strategy.args.adam_betas, 
                weight_decay=args.l2
            )

            # Create scheduler
            train_scheduler = get_scheduler(
                args.lr_scheduler,
                train_optim,
                num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
                num_training_steps=max_steps,
                scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
            )

            # Prepare training model with DeepSpeed
            self.model, self.train_optim, self.train_scheduler = strategy.prepare(
                (model, train_optim, train_scheduler),
                is_rlhf=True,
            )
        else:
            self.model = None
            self.train_optim = None
            self.train_scheduler = None
        
        # Initialize tokenizer
        from openrlhf.utils import get_tokenizer
        self.tokenizer = get_tokenizer(
            pretrain, 
            self.model, 
            "left", 
            strategy, 
            use_fast=not strategy.args.disable_fast_tokenizer
        )

    def set_train(self):
        if getattr(self, "_rank", 0) == 0 and self.model is not None:
            self.model.train()

    def set_eval(self):
        if self.model is not None:
            self.model.eval()

    def backward_and_optimize(self, loss):
        """Perform backward pass and optimization using DeepSpeed"""
        if getattr(self, "_rank", 0) == 0 and self.model is not None:
            # Backward pass
            self.strategy.backward(loss, self.model, self.train_optim)
            
            # Optimizer step
            self.strategy.optimizer_step(self.train_optim, self.model, self.train_scheduler, name="actor")
            
            # Sync weights between training and generation models if needed
            # This is a simplified approach - in practice you might want more sophisticated sync
            if hasattr(self, 'sync_weights'):
                self.sync_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
        questions,
        item_indexes,
        pixel_values=None,
        n_candidates=5,
        **gen_kwargs
    ):
        """
        Args:
            input_ids: (batch, seq_len) 输入 token IDs
            attention_mask: (batch, seq_len) attention mask
            pixel_values: 图像输入（可选）
            n_candidates: int，采样生成的候选数量 n
            questions: List[str]，每个样本的问题文本（可选，若无则用空串）
            item_indexes: List[str]，每个条数据在数据集中的idx。
            gen_kwargs: 传给 model.generate 的参数，如 temperature、top_p、max_new_tokens 等
        Returns:
            List[{"question": str, "candidate_response": List[str]}]，每个元素包含原始问题和其n个生成的候选回答
        """
        # 训练模式下，直接 forward 并计算 loss 或 logits
        if self.model.training:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=input_ids,  # 或按具体训练需求设置 labels
            )
            return outputs.logits

        # 评估模式下，生成 n_candidates 个候选回复
        batch_size = input_ids.shape[0]
        all_candidates = [[] for _ in range(batch_size)]
        
        # 优先使用vLLM engine进行生成
        if self.vllm_engines is not None:
            # 使用vLLM engine进行批量生成
            all_candidates = self._generate_with_vllm_engines(
                questions, item_indexes, pixel_values, n_candidates, **gen_kwargs
            )
        else:
            # 回退到模型直接生成
            all_candidates = self._generate_with_model(
                input_ids, attention_mask, pixel_values, n_candidates, **gen_kwargs
            )
        
        # 构造返回结果
        results = []
        for i in range(batch_size):
            results.append({
                'question': questions[i] if questions else '',
                'candidate_response': all_candidates[i],
                'idx': item_indexes[i] if item_indexes else i
            })
        
        return results

    def broadcast_to_vllm(self):
        """
        同步当前模型权重到所有vllm engine。
        只在rank 0上执行。
        """
        if not hasattr(self, "vllm_engines") or self.vllm_engines is None or torch.distributed.get_rank() != 0:
            return
        model = self.model
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1
            refs = [
                engine.update_weight.remote(
                    name,
                    param.data.cpu().numpy(),
                    dtype=str(param.dtype),
                    shape=param.shape,
                    empty_cache=(count == num_params)
                )
                for engine in self.vllm_engines
            ]
            ray.get(refs)
        torch.cuda.empty_cache()

class ReferenceModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, vllm_engines=None):
        self._setup_distributed(strategy)
        args = strategy.args
        # Only inference model, no optimizer/scheduler
        model = Actor(
            pretrain,
            use_flash_attention_2=args.use_flash_attn_ref,
            bf16=args.ref_bf16,
            temperature=args.temperature,
            use_liger_kernel=args.use_liger_kernel,
        )
        strategy.print('model:\n', model)
        self.model = strategy.prepare(model, is_rlhf=True)
        self.model.eval()
        from openrlhf.utils import get_tokenizer
        self.tokenizer = get_tokenizer(
            pretrain,
            self.model,
            "left",
            strategy,
            use_fast=not args.disable_fast_tokenizer
        )

    def batch_logp(self, contexts, targets, images=None):
        """
        Efficient batch logp calculation for DPO, supports images (optional).
        Args:
            contexts: List[str], context prompts
            targets: List[str], target completions
            images: List[Tensor or None], optional images
        Returns:
            List[float]: logp for each sample
        """
        import torch
        prompts = []
        target_lengths = []
        for context, target in zip(contexts, targets):
            prompts.append(context)
            target_lengths.append(len(self.tokenizer.encode(target)))
        input_ids = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length
        ).input_ids
        target_ids = self.tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length
        ).input_ids
        input_ids = torch.cat([input_ids, target_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        model_inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if images is not None and any(img is not None for img in images):
            pixel_values = []
            for img in images:
                if img is not None:
                    pixel_values.append(img)
                else:
                    pixel_values.append(torch.zeros(3, 224, 224))
            pixel_values = torch.stack(pixel_values)
            model_inputs["pixel_values"] = pixel_values
        device = next(self.model.parameters()).device
        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor):
                model_inputs[k] = v.to(device)
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        logits = outputs.logits
        batch_logps = []
        for i in range(len(contexts)):
            target_start = input_ids.shape[1] - target_ids.shape[1]
            target_logits = logits[i, target_start:, :]
            target_tokens = target_ids[i, :]
            log_probs = torch.log_softmax(target_logits, dim=-1)
            target_logp = log_probs.gather(1, target_tokens.unsqueeze(-1)).squeeze(-1)
            mask = target_tokens != self.tokenizer.pad_token_id
            sample_logp = (target_logp * mask.float()).sum()
            batch_logps.append(sample_logp.detach().cpu())
        return batch_logps

class LabelerModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)   # TODO：只推理的模型不需要用deepspeed封装
        model_labeler = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn_labeler,
            bf16=strategy.args.labeler_bf16,
            load_in_8bit=strategy.args.labeler_load_in_8bit,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            # packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(model_labeler)

        if strategy.args.labeler_model_offload:
            model_labeler._offload = True

        self.model = self.strategy.prepare(model_labeler, is_rlhf=True)
        self.model.eval()

    def divide(
        self,
        q_candidate_a: list[dict],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        do_sample: bool = False
    ) -> list[dict]:
        """
        将 question–candidate_response 对拆分成"事实陈述句"列表。
        
        Args:
            q_candidate_a: List[{"idx": str, "question": str, "candidate_response": List[str]}]，每个元素包含一个问题和其候选回答列表
            max_new_tokens: 最多生成多少新 token
            temperature: 采样温度（0.0→贪心）
            do_sample: 是否启用采样（False→贪心）
        
        Returns:
            List[Dict]，每个元素包含'question'、'candidate_response'和'facts'字段，'facts'为List[List[str]]，与候选回答一一对应。
        """
        tokenizer = self.model.tokenizer
        device = next(self.model.parameters()).device

        prompts = []
        mapping = []  # 记录每个prompt属于哪个问题和候选索引
        for q_idx, qa in enumerate(q_candidate_a):
            q = qa["question"]
            for r_idx, r in enumerate(qa["candidate_response"]):
                prompt = (
                    "You are an expert in extracting facts from the given question-answer pair for an image. Your task is to extract and rewrite the facts mentioned in the question-answer pair into self-contained sentences. Exclude opinions or subjective statements.\n\n You should present your result in the following format:\n### Facts:\n- {Extracted fact 1}\n- {Extracted fact 2}\n- ...\n\n### Question-response pair:\nQuestion: " + q + "\nResponse: " + r
                )
                prompts.append(prompt)
                mapping.append((q_idx, r_idx))

        # 批量tokenize
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length
        ).to(device)

        # 批量生成
        with torch.no_grad():
            gen_ids = self.model.generate(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 解码并解析"### Facts:"下的列表项
        raw_outputs = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        # 组织成原结构：List[Dict]，每个元素包含question、candidate_response、facts
        num_questions = len(q_candidate_a)
        facts_nested: list[list[list[str]]] = [[] for _ in range(num_questions)]
        for (q_idx, r_idx), text in zip(mapping, raw_outputs):
            lines = []
            in_facts = False
            for line in text.splitlines():
                ls = line.strip()
                if ls.startswith("### Facts"):
                    in_facts = True
                    continue
                if in_facts:
                    if ls.startswith("###"):  # 到下个 section
                        break
                    if ls.startswith("-"):
                        fact = ls.lstrip("- ").rstrip()
                        if fact:
                            lines.append(fact)
            # 确保每个问题有对应的候选列表
            while len(facts_nested[q_idx]) <= r_idx:
                facts_nested[q_idx].append([])
            facts_nested[q_idx][r_idx] = lines
        # 构造最终结果
        result = []
        for q_idx, qa in enumerate(q_candidate_a):
            result.append({
                "idx": qa["idx"],
                "question": qa["question"],
                "candidate_response": qa["candidate_response"],
                "facts": facts_nested[q_idx]
            })
        return result

    def conquer(
        self,
        q_facts_batch: list[dict],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        do_sample: bool = False
    ) -> list[dict]:
        """
        输入为 divide 输出的结构（list[dict]，每个dict包含'question'、'candidate_response'、'facts'），
        将每条陈述句（facts）改写为通用疑问句。
        
        Args:
            q_facts_batch: List[Dict]，每个元素包含'idx'、'question'、'candidate_response'、'facts'（List[List[str]]）
            max_new_tokens: 最多生成多少新 token
            temperature: 采样温度（0.0→贪心）
            do_sample: 是否启用采样（False→贪心）
        
        Returns:
            List[Dict]，每个元素包含'idx'、'question'和'candidates'，'candidates'为List，每个元素为{'candidate_response': ..., 'simple_questions': [...]}
        """
        tokenizer = self.model.tokenizer
        device = next(self.model.parameters()).device

        prompts = []
        q_cand_pairs = []  # (question, candidate_response) 对应关系
        for item in q_facts_batch:
            question = item["question"]
            candidate_responses = item["candidate_response"]
            facts_nested = item.get("facts", [])
            for cand_response, facts in zip(candidate_responses, facts_nested):
                if not facts:
                    prompts.append("")  # 保证对齐
                else:
                    content = (
                        "You are an expert at modifying a given declarative sentence into a general question sentence. Your task is to modify the given declarative sentences one by one into a general question form. Do not change tenses or add extra content.\n    If the given declarative sentence contains not, no or negative meaning words, you need to check the modified general interrogative sentence to make sure that the generated general question sentence retains words with not, no or negative meaning words.\n\nYou should present your result in the following format:\n### Modified sentences:\n- {Modified sentence 1}\n- {Modified sentence 2}\n- ...\n\n### Declarative sentences:"
                    )
                    for fact in facts:
                        content += f"\n- {fact}"
                    prompts.append(content)
                q_cand_pairs.append((question, cand_response))

        # 2) 批量 tokenize
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length
        ).to(device)

        # 3) 生成
        with torch.no_grad():
            gen_ids = self.model.generate(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 4) 解码并解析"### Modified sentences:"下的列表项
        raw_outputs = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        # 组装成每个candidate的simple_questions
        cand_simple_questions = []
        for text in raw_outputs:
            lines = []
            in_mod = False
            for line in text.splitlines():
                ls = line.strip()
                if ls.startswith("### Modified sentences"):
                    in_mod = True
                    continue
                if in_mod:
                    if ls.startswith("###"):  # 到下个 section
                        break
                    if ls.startswith("-"):
                        q = ls.lstrip("- ").rstrip()
                        if q:
                            lines.append(q)
            cand_simple_questions.append(lines)
        # 组装最终结构
        result = []
        idx = 0
        for item in q_facts_batch:
            question = item["question"]
            candidate_responses = item["candidate_response"]
            facts_nested = item.get("facts", [])
            candidates = []
            for cand_response, facts in zip(candidate_responses, facts_nested):
                simple_questions = cand_simple_questions[idx]
                candidates.append({
                    "candidate_response": cand_response,
                    "simple_questions": simple_questions
                })
                idx += 1
            result.append({
                "idx": item["idx"],
                "question": question,
                "candidates": candidates
            })
        return result

    def YesNo(
        self,
        batch: list[dict],
        dataset=None,  # idx到样本的字典
        max_new_tokens: int = 8,
        temperature: float = 0.0,
        do_sample: bool = False
    ) -> list[dict]:
        """
        输入：
            batch: List[Dict]，每个元素包含'idx'、'question'和'candidates'，'candidates'为List，每个元素为{'candidate_response': ..., 'simple_questions': [...]}。
            dataset: idx到样本的字典，通过'idx'查找图片。
        输出：
            List[Dict]，每个元素包含'idx'、'question'和'candidates'，'candidates'为List，每个元素为{'candidate_response': ..., 'simple_questions': [...], 'simple_answers': [{'1': Yes/yes logit, '0': No/no logit}]}。
        """
        tokenizer = self.model.tokenizer
        device = next(self.model.parameters()).device
        results = []
        for item in batch:
            idx = item["idx"]
            question = item["question"]
            # 直接用字典查找图片
            image = dataset.get(str(idx), {}).get("image", None) if dataset is not None else None
            candidates = []
            for cand in item["candidates"]:
                candidate_response = cand["candidate_response"]
                simple_questions = cand["simple_questions"]
                simple_answers = []
                for sq in simple_questions:
                    prompt = sq.strip() + " Please answer Yes or No."
                    # 构造输入
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=tokenizer.model_max_length
                    ).to(device)
                    # 如果需要图片，假设模型支持pixel_values参数
                    model_inputs = dict(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                    if image is not None:
                        model_inputs["pixel_values"] = image.unsqueeze(0) if hasattr(image, 'unsqueeze') else image
                    # 推理，获得logits
                    with torch.no_grad():
                        outputs = self.model(**model_inputs)
                        logits = outputs.logits  # (batch, seq_len, vocab_size)
                        # 取最后一个token的logits
                        last_logits = logits[0, -1, :]
                        # 获取"Yes"、"yes"、"No"、"no"的token id
                        yes_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["Yes", "yes"]]
                        no_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["No", "no"]]
                        # 取最大logit
                        yes_logit = max([last_logits[i].item() for i in yes_ids if i != tokenizer.unk_token_id], default=float('-inf'))
                        no_logit = max([last_logits[i].item() for i in no_ids if i != tokenizer.unk_token_id], default=float('-inf'))
                        simple_answers.append({"1": yes_logit, "0": no_logit})
                candidates.append({
                    "candidate_response": candidate_response,
                    "simple_questions": simple_questions,
                    "simple_answers": simple_answers
                })
            results.append({
                "idx": idx,
                "question": question,
                "candidates": candidates
            })
        return results
        
    def combine(
        self,
        batch: list[Dict]
    ) -> list[Dict]:
        """
        输入：
            batch: List[Dict]，每个元素包含'idx'、'question'和'candidates'，'candidates'为List，每个元素为{'candidate_response': ..., 'simple_questions': [...], 'simple_answers': [{'1': Yes/yes logit, '0': No/no logit}]}。

        输出：
            List[Dict]，优劣回答对的列表，每个元素是一个字典{"1": preffered response, "0": inferior response}
        """
        results = []
        for item in batch:
            candidates = item['candidates']
            scores = []
            for cand in candidates:
                simple_answers = cand['simple_answers']
                num_rejection = 0
                for yes_no in simple_answers:
                    if yes_no['0'] > yes_no['1']:
                        num_rejection -= 1
                scores.append(num_rejection)
            # 随机选取两组索引，要求scores[idx1] > scores[idx2]
            n = len(candidates)
            valid_pairs = [(i, j) for i in range(n) for j in range(n) if i != j and scores[i] > scores[j]]
            if valid_pairs:
                idx1, idx2 = random.choice(valid_pairs)
                candidate_response1 = candidates[idx1]['candidate_response']
                candidate_response2 = candidates[idx2]['candidate_response']
                results.append({
                    "idx": item["idx"],
                    "question": item["question"],
                    "1": candidate_response1,
                    "0": candidate_response2
                })
        return results