import math
import os
import random
import socket

import deepspeed
from typing import Dict, List
from abc import ABC
from torch.optim import Optimizer
from transformers.trainer import get_scheduler
from torch.utils.data import DataLoader, Dataset
from transformers import default_data_collator
import torch
import ray

from openrlhf.models import VisionActor
from openrlhf.models.loss import DPOLoss
from openrlhf.trainer.ray.vllm_engine import LLMRayActor
from openrlhf.utils.distributed_util import init_process_group, torch_dist_cuda_sync_and_barrier
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import offload_deepspeed_states, reload_deepspeed_states
from .launcher import BaseModelActor
from .utils import get_physical_gpu_id

class MultiModalLLMRayActor(LLMRayActor):
    def generate_multimodal(self, prompts, sampling_params, multi_modal_data):
        """
        支持多模态（图片+文本）推理的接口。
        prompts: List[str]
        sampling_params: vllm.SamplingParams 或 list
        multi_modal_data: List[dict]，每个dict如 {"image": PIL.Image}
        """
        # 直接调用vllm.LLM的generate
        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            multi_modal_data=multi_modal_data
        )
        return outputs
    
    def get_image_processor(self):
        model_config = self.llm.model_executor.model_info.model_config
        image_processor = ImagePlugin()._get_hf_image_processor(model_config)
        return image_processor

class DataServerActor:
    def __init__(self, ai_feedback: List[Dict]):
        self.ai_feedback = ai_feedback
    
    def get_data(self, idx: str):
        return self.ai_feedback[idx]
    
    def get_batch(self, idxes: List[str]):
        return [self.ai_feedback[idx] for idx in idxes]

class VisionActorTrainer(ABC):
    def __init__(
        self,
        strategy,
        vision_actor: VisionActor,
        actor_optim: Optimizer,
        actor_scheduler,
        idx2data: dict = None,
        tokenizer=None,
        image_processor=None,
        vllm_engines: List = None,
        **kwargs,
    ):
        """Vision Actor Trainer for training ray actor group."""
        self.strategy = strategy
        self.args = strategy.args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.generate_kwargs = kwargs
        self.vision_actor = vision_actor
        self.vision_actor_optim = actor_optim
        self.vision_actor_scheduler = actor_scheduler
        self.vllm_engines = vllm_engines
        self.loss_fn = DPOLoss(beta=1.0)
        self.idx2data = idx2data

        # weight sync
        backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = False
        if backend == "nccl" and self.args.colocate_all_models:
            self.use_cuda_ipc = True
        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.num_policy_vllm_engines,
                self.strategy.args.vllm_tensor_parallel_size_policy,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            group_name = "openrlhf"
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    group_name,
                    backend=backend,
                    use_ray=use_ray,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            if use_ray:
                import ray.util.collective as collective

                collective.init_collective_group(world_size=world_size, rank=0, backend=backend, group_name=group_name)
                self._model_update_group = group_name
            else:
                self._model_update_group = init_process_group(
                    backend=backend,
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                )

            ray.get(refs)
        
        # log loss
        if getattr(self.args, "wandb_enable", False) and self.strategy.is_rank_0():
            try:
                import wandb
                wandb.init(project=self.args.wandb_project,
                           name=self.args.wandb_name,
                           entity=self.args.wandb_entity,
                           id=self.args.wandb_id,
                           group=self.args.wandb_group,
                           tags=self.args.wandb_tags)
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed, logging disabled")
                self.wandb = None
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                self.wandb = None
        else:
            self.wandb = None

        torch_dist_cuda_sync_and_barrier()

    def setup_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool = True, 
                        drop_last: bool = True, num_workers: int = 4, pin_memory: bool = True):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=None,
            collate_fn=default_data_collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        
        return dataloader
    
    def get_global_loss(self, local_loss):
        """
        聚合所有进程的 loss，返回全局平均 loss。
        local_loss: 当前进程的 loss (tensor, 标量)
        """
        import torch.distributed as dist
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size == 1:
            return local_loss
        # 复制一份，避免原地修改
        loss_clone = local_loss.clone().detach()
        dist.all_reduce(loss_clone, op=dist.ReduceOp.SUM)
        loss_clone /= world_size
        return loss_clone

    def train_step(self, batch):
        device = torch.cuda.current_device()
        # 期望batch为dict of list/tensor，需解包
        # 兼容单条数据
        if isinstance(batch, dict):
            batch = [batch]
        
        # 从batch中提取reference model的logps
        reference_chosen_logps = torch.stack([item['logp_1'] for item in batch]).to(device)
        reference_rejected_logps = torch.stack([item['logp_0'] for item in batch]).to(device)

        # 直接用batch作为输入，调用batch_logp
        # batch的每个元素应包含 'idx', 'question', '1', '0', 'pixel_values'
        logp_results = self.vision_actor.batch_logp(batch, requires_grad=True)
        # 分别收集logp_1和logp_0
        policy_chosen_logps = torch.stack([item['logp_1'] for item in logp_results])
        policy_rejected_logps = torch.stack([item['logp_0'] for item in logp_results])
        loss, _, _ = self.loss_fn(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps
                )
        self.strategy.backward(loss, self.vision_actor, self.vision_actor_optim)
        self.strategy.optimizer_step(self.vision_actor_optim, self.vision_actor, self.vision_actor_scheduler)

        if self.strategy.is_rank_0():
            # If all_reduce all losses every batch makes efficiency low,
            # one can all_reduce all losses every epoch.
            # all_reduce itself is a synchronization operation so we 
            # don't have to barrier here.
            global_loss = self.get_global_loss(loss)
            return global_loss.item()
        
        return None

    def train_epoch(self, dataloader, epoch: int, max_epochs: int):
        num_batches = 0
        epoch_losses = []
        for batch_idx, batch in enumerate(dataloader):
            try:
                global_loss = self.train_step(batch)
                num_batches += 1
                if self.strategy.is_rank_0(): 
                    epoch_losses.append(global_loss)
                    global_step = epoch * len(dataloader) + batch_idx
                    if getattr(self.args, "wandb_enable", False) and self.wandb is not None:
                        if not self.wandb.api.api_key:
                            self.wandb.login(key=self.args.wandb_api_key)
                        self.wandb.log({"train/loss": global_loss}, step=global_step)
                
            except Exception as e:
                print(f"Error in training step {batch_idx}: {e}")
                continue
        if self.strategy.is_rank_0():
            return epoch_losses
        else:
            return None

    def train(self, dataset, batch_size: int, num_epochs: int, 
              shuffle: bool = True, drop_last: bool = True, num_workers: int = 4, 
              pin_memory: bool = True):
        # 用索引和idx2data构建子数据集
        dataloader = self.setup_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        training_history = []
        for epoch in range(num_epochs):
            # 训练一个 epoch
            epoch_global_losses = self.train_epoch(dataloader, epoch, num_epochs)
            if self.strategy.is_rank_0():
                training_history.append(epoch_global_losses)
            
            # 同步所有进程
            torch_dist_cuda_sync_and_barrier()
            if self.strategy.is_rank_0() and not self.args.disable_ds_ckpt:
                client_states = {
                    "epoch": epoch,
                    "loss": epoch_global_losses
                }
                self.save_checkpoint(save_path=self.args.ckpt_path, epoch=epoch, client_states=client_states)
        
        if self.strategy.is_rank_0():
            if self.wandb is not None:
                self.wandb.finish()
            return training_history
        else:
            return None

    def save_checkpoint(self, save_path: str, epoch: int, client_states: dict):
        """
        保存检查点
        
        Args:
            save_path: 保存路径
            epoch: 当前 epoch
        """
        if self.strategy.is_rank_0():
            os.makedirs(save_path, exist_ok=True)
            
            # 保存模型权重
            self.strategy.save_ckpt(
                self.vision_actor.model, 
                os.path.join(save_path, f"checkpoint-{epoch}"),
                client_states=client_states
            )
            
            print(f"Checkpoint saved to {save_path}/checkpoint-{epoch}")
        
        # 同步所有进程
        torch_dist_cuda_sync_and_barrier()

    def _broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        model = self.vision_actor.model.module
        count, num_params = 0, len(list(model.named_parameters()))

        def _broadcast_param(param, count, num_params):
            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            # Fire all vllm engines for broadcast
            if torch.distributed.get_rank() == 0:
                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight.remote(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params)
                    for engine in self.vllm_engines
                ]

                if use_ray:
                    import ray.util.collective as collective

                    collective.broadcast(param.data, 0, group_name=self._model_update_group)
                else:
                    torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                ray.get(refs)

        def _handle_cuda_ipc(param, count, num_params):
            from torch.multiprocessing.reductions import reduce_tensor

            weight = param.data.clone()
            ipc_handle = reduce_tensor(weight)

            ipc_handle = {get_physical_gpu_id(): ipc_handle}
            ipc_handle_list = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

            if torch.distributed.get_rank() == 0:
                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)

                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight_cuda_ipc.remote(
                        name,
                        dtype=param.dtype,
                        shape=shape,
                        ipc_handles=ipc_handles,
                        empty_cache=count == num_params,
                    )
                    for engine in self.vllm_engines
                ]
                ray.get(refs)
            torch_dist_cuda_sync_and_barrier()

        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # broadcast
            if not self.use_cuda_ipc:
                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                if self.strategy.args.ds_tensor_parallel_size > 1:
                    with deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True):
                        _broadcast_param(param, count, num_params)
                else:
                    with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                        _broadcast_param(param, count, num_params)
            # CUDA IPC
            else:
                if self.strategy.args.ds_tensor_parallel_size > 1:
                    with deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True):
                        _handle_cuda_ipc(param, count, num_params)
                else:
                    with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                        _handle_cuda_ipc(param, count, num_params)

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch_dist_cuda_sync_and_barrier()
        

class PolicyModelActor(BaseModelActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps=None, vllm_engines=None):
        args = strategy.args 
        self.vllm_engines = vllm_engines
        self.disable_ds_ckpt = args.disable_ds_ckpt

        if getattr(args, "vllm_num_engines", 0) > 0:
            # To prevent hanging during NCCL synchronization of weights between DeepSpeed and vLLM.
            # see https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"
        
        self._setup_distributed(strategy)

        model_actor = VisionActor(
            pretrain,
            use_flash_attention_2=strategy.args.use_flash_attn_policy,
            bf16=strategy.args.policy_train_bf16,
            load_in_4bit=strategy.args.policy_train_load_in_4bit,
            load_in_8bit=strategy.args.policy_train_load_in_8bit,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
            vision_tower_attr=args.vision_tower_name,
            freeze_vision_tower=args.freeze_vision_tower
        )
        strategy.print('model actor:\n', model_actor)
        self.actor_image_processor = model_actor.get_image_processor()

        actor_optim = strategy.create_optimizer(
            model_actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
        )

        # TODO: 搞清楚这里的学习率是怎么控制的
        actor_scheduler = get_scheduler(
            args.lr_scheduler,
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            model_actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (model_actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        if args.load_checkpoint and os.path.exists(args.ckpt_path):
            _, _ = strategy.load_ckpt(self.actor.model, args.ckpt_path)

        # initial offload
        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)
        
        # 创建 VisionActorTrainer 实例用于训练
        self.vision_trainer = VisionActorTrainer(
            strategy=strategy,
            vision_actor=self.actor,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            idx2data=self.idx2data,
            tokenizer=getattr(self, 'tokenizer', None),
            image_processor=self.actor_image_processor,
            vllm_engines=self.vllm_engines,
        )

    def train_with_dataset(self, dataset, batch_size=8, num_epochs=1, **kwargs):
        torch.cuda.empty_cache()
        self.actor.train()
        train_history = self.vision_trainer.train(
            dataset=dataset,
            batch_size=batch_size,
            num_epochs=num_epochs,
            **kwargs
        )
        # empty_cahce() operates at the process level
        # rather than affecting the entire GPU's memory alloction.
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if self.strategy.is_rank_0():
            return train_history
        else:
            return None
    
    def broadcast_to_vllm(self):
        self.vision_trainer._broadcast_to_vllm()
    
    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)

    def get_checkpoint_states(self):
        return self.checkpoint_states

    def save_checkpoint(self, tag, client_states):
        args = self.strategy.args
        self.strategy.save_ckpt(
            self.actor.model,
            os.path.join(args.ckpt_path, "_actor"),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
            client_states,
        )
        
        # wait
        torch_dist_cuda_sync_and_barrier()

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.actor,
            self.actor.module.get_tokenizer(),
            args.save_path,
        )


class LabelerRayActor(MultiModalLLMRayActor):
    def __init__(self, *args, pixel_values_object_ref=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pixel_values_object_ref = pixel_values_object_ref
    
    def set_pixel_values_object_ref(self, pixel_values_object_ref):
        self.pixel_values_object_ref = pixel_values_object_ref

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
        tokenizer = self.llm.model_executor.model_info.get_tokenizer()

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
            max_length=tokenizer.model_max_length if tokenizer.model_max_length is not None else 1024
        )

        # 批量生成
        gen_ids = self.llm.generate(
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
        tokenizer = self.llm.model_executor.model_info.get_tokenizer()

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
            max_length=tokenizer.model_max_length if tokenizer.model_max_length is not None else 1024
        )

        # 3) 生成
        gen_ids = self.llm.generate(
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
        source_idxes: list[str],
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
        from torchvision import transforms
        to_pil = transforms.ToPILImage()

        pixel_values_dict = ray.get(self.pixel_values_object_ref) if self.pixel_values_object_ref is not None else None
        tokenizer = self.llm.model_executor.model_info.get_tokenizer()
        results = []
        for i, item in enumerate(batch):
            question = item["question"]
            image = to_pil(pixel_values_dict[source_idxes[i]]["labeler_pixel_values"])
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
                        max_length=tokenizer.model_max_length if tokenizer.model_max_length is not None else 1024
                    )
                    # 如果需要图片，假设模型支持pixel_values参数
                    model_inputs = dict(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                    if image is not None:
                        model_inputs["pixel_values"] = image.unsqueeze(0) if hasattr(image, 'unsqueeze') else image
                    # 推理，获得logits
                    outputs = self.llm(**model_inputs)
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
                "idx": item["idx"],
                "question": question,
                "candidates": candidates
            })
        return results

    @staticmethod
    def combine(
        batch: list[Dict],
        seed: int = None
    ) -> list[Dict]:
        """
        输入：
            batch: List[Dict]，每个元素包含'idx'、'question'和'candidates'，'candidates'为List，每个元素为{'candidate_response': ..., 'simple_questions': [...], 'simple_answers': [{'1': Yes/yes logit, '0': No/no logit}]}。

        输出：
            List[Dict]，优劣回答对的列表，每个元素是一个字典{"1": preffered response, "0": inferior response}
        """
        if seed is not None:
            random.seed(seed)
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
            # paper ref: To save the training cost, we randomly sample at most 2 pairs for each instruction 
            # and we find such a filtering process only causes minor performence drops.
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


class PolicyRayActor(MultiModalLLMRayActor):
    def __init__(self, *args, pixel_values_object_ref=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pixel_values_object_ref = pixel_values_object_ref

    def set_pixel_values_object_ref(self, pixel_values_object_ref):
        self.pixel_values_object_ref = pixel_values_object_ref

    def generate_n_candidates(self, questions, item_indexes, n_candidates, **gen_kwargs):
        from vllm import SamplingParams     # or AsyncLLMEngine if needed
        from torchvision import transforms
        to_pil = transforms.ToPILImage()

        batch_size = len(questions)
        all_candidates = [[] for _ in range(batch_size)]

        sampling_params = SamplingParams(
            temperature=gen_kwargs.get('temperature', 0.7),
            top_p=gen_kwargs.get('top_p', 0.9),
            top_k=-1,
            max_tokens=gen_kwargs.get('max_new_tokens', 128),
            min_tokens=1,
            skip_special_tokens=False,
        )

        # 通过ray object store获取图片特征
        pixel_values_dict = ray.get(self.pixel_values_object_ref) if self.pixel_values_object_ref is not None else None
        has_images = pixel_values_dict is not None

        if has_images:
            # Prepare multimodal prompts
            prompts_input = []
            for i, question in enumerate(questions):
                prompt = f"USER: <image>\n{question}\nASSISTANT:"
                img = to_pil(pixel_values_dict[item_indexes[i]]['pixel_values'])
                for seed in range(n_candidates):
                    sd = sampling_params.clone()
                    sd.seed = seed
                    prompts_input.append({
                        "prompt": prompt,
                        "multi_modal_data": {"image": img},
                        "sampling_params": sd
                    })

            # Distribute across engines
            per_engine = (len(prompts_input) + len(self.vllm_engines) - 1) // len(self.vllm_engines)
            refs = []
            for i, engine in enumerate(self.vllm_engines):
                chunk = prompts_input[i * per_engine:(i + 1) * per_engine]
                if chunk:
                    refs.append(engine.generate.remote(chunk))

            # Collect and assign responses
            outputs = []
            for ref in refs:
                out = ray.get(ref)
                for o in out:
                    outputs.append(o.outputs[0].text)

            # Map back to batches
            idx = 0
            for i in range(batch_size):
                for _ in range(n_candidates):
                    if idx < len(outputs):
                        all_candidates[i].append(outputs[idx])
                    idx += 1
            return all_candidates

        # Text‑only fallback
        prompts = []
        for question in questions:
            for _ in range(n_candidates):
                prompts.append({
                    "prompt": f"Question: {question}\nAnswer:",
                    "sampling_params": sampling_params.clone()
                })

        per_engine = (len(prompts) + len(self.vllm_engines) - 1) // len(self.vllm_engines)
        refs = []
        for i, engine in enumerate(self.vllm_engines):
            chunk = prompts[i * per_engine:(i + 1) * per_engine]
            if chunk:
                refs.append(engine.generate.remote(chunk))

        outputs = []
        for ref in refs:
            out = ray.get(ref)
            for o in out:
                outputs.append(o.outputs[0].text)

        # Remap
        idx = 0
        for i in range(batch_size):
            for _ in range(n_candidates):
                all_candidates[i].append(outputs[idx] if idx < len(outputs) else "")
        return all_candidates


class ReferenceRayActor(MultiModalLLMRayActor):
    def batch_logp(self, prefered_inferior_response_list, requires_grad=True):
        """
        Batch logp calculation for DPO, supports images (optional),
        and preserves mapping for each sample. Efficient batch version.
        Args:
            prefered_inferior_response_list: List[dict], each dict contains keys:
                'idx', 'question', '1', '0', 'pixel_values'
            requires_grad: bool, whether to compute gradients
        Returns:
            List[dict]: each dict contains original keys plus 'logp_1', 'logp_0'
        """
        tokenizer = self.llm.model_executor.model_info.get_tokenizer()
        model = self.llm.model_executor.model
        # Prepare batch inputs
        questions = []
        responses = []
        pixel_values_list = []
        for item in prefered_inferior_response_list:
            questions.append(item["question"])
            responses.append(item["1"])
            pixel_values_list.append(item.get("pixel_values", None))
            questions.append(item["question"])
            responses.append(item["0"])
            pixel_values_list.append(item.get("pixel_values", None))
        # Tokenize all pairs
        prompt_enc = tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length if tokenizer.model_max_length is not None else 1024
        )
        target_enc = tokenizer(
            responses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length if tokenizer.model_max_length is not None else 1024
        )
        input_ids = torch.cat([prompt_enc.input_ids, target_enc.input_ids], dim=1)
        attention_mask = torch.cat([prompt_enc.attention_mask, target_enc.attention_mask], dim=1)
        model_inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
        # Handle images if present
        if any(img is not None for img in pixel_values_list):
            pixel_values = []
            for img in pixel_values_list:
                if img is not None:
                    pixel_values.append(img)
                else:
                    pixel_values.append(torch.zeros(3, 224, 224))
            pixel_values = torch.stack(pixel_values)
            model_inputs["pixel_values"] = pixel_values
        device = next(model.parameters()).device
        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor):
                model_inputs[k] = v.to(device)
        # Forward pass
        if requires_grad:
            outputs = model(**model_inputs)
        else:
            with torch.no_grad():
                outputs = model(**model_inputs)
        logits = outputs.logits
        batch_size = len(prefered_inferior_response_list)
        logps = []
        for i in range(batch_size * 2):
            prompt_len = prompt_enc.input_ids[i].shape[0]
            target_len = target_enc.input_ids[i].shape[0]
            target_logits = logits[i, prompt_len - 1 : prompt_len - 1 + target_len - 1, :]
            target_tokens = target_enc.input_ids[i, 1:]
            log_probs = torch.log_softmax(target_logits, dim=-1)
            target_logp = log_probs.gather(1, target_tokens.unsqueeze(-1)).squeeze(-1)
            mask = target_tokens != tokenizer.pad_token_id
            if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
                mask &= (target_tokens != tokenizer.bos_token_id)
            sample_logp = (target_logp * mask.float()).sum()
            if not requires_grad:
                sample_logp = sample_logp.detach().cpu()
            logps.append(sample_logp)
        # Split back to original mapping
        results = []
        for i, item in enumerate(prefered_inferior_response_list):
            result = dict(item)
            result["logp_1"] = logps[2 * i]
            result["logp_0"] = logps[2 * i + 1]
            results.append(result)

        return results