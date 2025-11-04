import math
import os
import random
import socket
import io

import deepspeed
import numpy as np
from typing import Dict, List
from abc import ABC
from torch.optim import Optimizer
from transformers.trainer import get_scheduler
from torch.utils.data import DataLoader, Dataset
from transformers import default_data_collator, AutoProcessor
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
from PIL import Image

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
)
import json
import sys


class ListDataset(Dataset):
    """自定义Dataset类，用于处理列表数据"""
    def __init__(self, data_list: List[dict]):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

class MultiModalLLMRayActor(LLMRayActor):
    def generate_multimodal(self, prompts, sampling_params, multi_modal_data):
        """
        支持多模态（图片+文本）推理的接口。
        prompts: List[str]
        sampling_params: vllm.SamplingParams 或 list
        multi_modal_data: List[dict]，每个dict如 {"image": Tensor}
        """
        # 直接调用vllm.LLM的generate
        inputs = []
        for prompt, mm in zip(prompts, multi_modal_data):
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": mm["image"].unsqueeze(0).contiguous(memory_format=torch.channels_last)},    # Image has to be shape like [b, c, h, w], even when it't one image.
            })
        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        return outputs
    
    def _apply_chat_template(self, tokenizer, user_text: str) -> str:
        """
        使用模型自带的 chat template 格式化 prompt；若不可用则回退到 "USER:"/"ASSISTANT:" 模式。
        """
        try:
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": user_text}]
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            pass
        # 回退：简单 USER/ASSISTANT 结构
        return f"USER: {user_text}\nASSISTANT:"

    def _format_chat_prompts(self, tokenizer, prompts: list[str]) -> list[str]:
        """
        将一组用户文本包装为模型期望的 Chat 模板；如果模板不可用，则原样返回。
        """
        formatted = []
        use_template = hasattr(tokenizer, "apply_chat_template")
        for p in prompts:
            if use_template:
                try:
                    messages = [{"role": "user", "content": p}]
                    chat = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    formatted.append(chat)
                    continue
                except Exception:
                    pass
            formatted.append(f"USER: {p}\nASSISTANT:")
        return formatted

    def _safe_max_length(self, tokenizer, fallback: int = 4096) -> int:
        """
        从 vLLM 引擎或 tokenizer 推断安全的 max_length，上限做裁剪，避免 fast tokenizer 的 size_t 溢出。
        """
        # 1) 优先从 vLLM 引擎配置读取
        max_len = None
        try:
            cfg = self.llm.llm_engine.get_model_config()
            max_len = getattr(cfg, "max_model_len", None)
        except Exception:
            pass

        # 2) 回退到 tokenizer.model_max_length
        if max_len is None:
            max_len = getattr(tokenizer, "model_max_length", None)

        # 3) 合理边界检查与回退
        try:
            if max_len is None:
                return fallback
            max_len = int(max_len)
        except Exception:
            return fallback

        # 过滤异常的大值或非正数
        if max_len <= 0 or max_len > 1000000:
            return fallback
        return max_len

    def get_tokenizer(self):
        return self.llm.get_tokenizer()

    def get_image_processor(self):
        try:
            # 尝试通过 vLLM 的 LLMEngine 获取模型配置
            model_config = self.llm.llm_engine.get_model_config()
            model_name = model_config.model
        except AttributeError:
            # 如果上述方法失败，尝试其他方式
            try:
                # 通过 engine_args 获取模型名称
                model_name = getattr(self.llm, 'model', None)
                if model_name is None:
                    # 最后的备选方案：从 kwargs 中获取
                    model_name = self.kwargs.get('model', None)
            except:
                raise AttributeError("No model config information.")
        
        if model_name is None:
            raise ValueError("No model name.")
        
        try:
            processor = AutoProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True,
                use_fast=True
            )
            return processor
        except Exception as e:
            # 如果加载失败，返回 None
            print(f"警告：无法加载处理器 - {e}")
            return None


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

    def setup_dataloader(self, dataset, batch_size: int, shuffle: bool = True, 
                        drop_last: bool = True, num_workers: int = 4, pin_memory: bool = True):
        import torch.distributed as dist
        
        # 如果dataset是列表，转换为ListDataset
        if isinstance(dataset, list):
            dataset = ListDataset(dataset)
        
        # 检查是否在分布式环境中
        if dist.is_initialized():
            # 使用DistributedSampler确保每个进程拿到不同的数据
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=shuffle
            )
        else:
            sampler = None

        def data_collate_fn(batch):
            """
            自定义的 collate_fn，用于处理包含字典值的数据结构
            """
            if not batch:
                return batch
            
            # 如果batch中只有一个样本，直接返回
            if len(batch) == 1:
                return batch[0]
            
            # 对于多个样本，直接返回列表
            return batch
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=data_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        
        return dataloader
    
    def get_global_loss(self, local_loss):
        """
        聚合所有进程的 loss，返回全局平均 loss。
        local_loss: 当前进程的 loss (tensor, 标量)
        """
        import torch.distributed as dist
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        print(f"chuanwei2 world_size: {world_size}")
        if world_size == 1:
            print(f"chuanwei2 return local loss.")
            return local_loss
        # 复制一份，避免原地修改
        loss_clone = local_loss.clone().detach()
        dist.all_reduce(loss_clone, op=dist.ReduceOp.SUM)
        loss_clone /= world_size
        print(f"chuanwei2 return return all losses.")
        return loss_clone

    def train_step(self, batch):
        device = torch.cuda.current_device()
        # 处理batch数据格式
        # DataLoader返回的batch可能是单个样本或样本列表
        if isinstance(batch, dict):
            # 单个样本，转换为列表
            batch = [batch]
        elif isinstance(batch, (list, tuple)) and len(batch) > 0 and isinstance(batch[0], dict):
            # 已经是字典列表，直接使用
            pass
        else:
            # 其他格式，尝试转换
            batch = list(batch) if hasattr(batch, '__iter__') else [batch]
        
        # 从batch中提取reference model的logps
        reference_chosen_logps = torch.stack([item['logp_1'] for item in batch]).to(device)
        reference_rejected_logps = torch.stack([item['logp_0'] for item in batch]).to(device)

        logp_results = self.vision_actor.batch_logp(batch, requires_grad=True)
        # 分别收集logp_1和logp_0
        policy_chosen_logps = torch.stack([item['logp_1'] for item in logp_results])
        policy_rejected_logps = torch.stack([item['logp_0'] for item in logp_results])
        print(f"chuanwei policy_chosen_logps: {policy_chosen_logps}")
        print(f"chuanwei policy_rejected_logps: {policy_rejected_logps}")
        print(f"chuanwei reference_chosen_logps: {reference_chosen_logps}")
        print(f"chuanwei reference_rejected_logps: {reference_rejected_logps}")
        loss, _, _ = self.loss_fn(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps
                )
        print(f"chuanwei loss: {loss}")
        self.strategy.backward(loss, self.vision_actor, self.vision_actor_optim)
        self.strategy.optimizer_step(self.vision_actor_optim, self.vision_actor, self.vision_actor_scheduler)

        print(f"chaunwei2 368 self.strategy.is_rank_0(): {self.strategy.is_rank_0()}")
        if self.strategy.is_rank_0():
            print(f"chuanwei2 370 self.strategy.is_rank_0()")
            # If all_reduce all losses every batch makes efficiency low,
            # one can all_reduce all losses every epoch.
            # all_reduce itself is a synchronization operation so we 
            # don't have to barrier here.
            global_loss = self.get_global_loss(loss)
            return global_loss.item()
        
        print(f"chuanwei2 378return None")
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
        print(f"chuanwei2 399 self.strategy.is_rank_0(): {self.strategy.is_rank_0()}")
        if self.strategy.is_rank_0():
            print(f"chuanwei2 401 return epoch_losses: {epoch_losses}")
            return epoch_losses
        else:
            print(f"chuanwei2 404 return epoch_losses: None")
            return None

    def train(self, dataset, batch_size: int, num_epochs: int, 
              shuffle: bool = True, drop_last: bool = True, num_workers: int = 4, 
              pin_memory: bool = True):
        """分布式训练方法，使用DistributedSampler确保每个进程拿到不同的数据"""
        # 设置模型为训练模式
        self.vision_actor.train()
        
        # 用DistributedSampler构建dataloader
        # 每个逻辑actor（rank）都要有自己的一份Dataset对象，这是PyTorch分布式的常规做法。
        dataloader = self.setup_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        # 打印调试信息
        if self.strategy.is_rank_0():
            print(f"Training with {len(dataset)} samples, batch_size={batch_size}, num_epochs={num_epochs}")
            if hasattr(dataloader.sampler, 'num_replicas'):
                print(f"DistributedSampler: num_replicas={dataloader.sampler.num_replicas}, rank={dataloader.sampler.rank}")
        
        training_history = []
        for epoch in range(num_epochs):
            # 设置epoch，确保每个epoch的数据分布不同
            if hasattr(dataloader.sampler, 'set_epoch'):
                dataloader.sampler.set_epoch(epoch)
            
            # 训练一个 epoch
            epoch_global_losses = self.train_epoch(dataloader, epoch, num_epochs)
            print(f"chuanwei2 439 epoch_global_losses: {epoch_global_losses}")
            print(f"chuanwei2 440 self.strategy.is_rank_0(): {self.strategy.is_rank_0()}")
            if self.strategy.is_rank_0():
                print(f"chuanwei2 rank0 epoch_global_losses: {epoch_global_losses}")
                training_history.append(epoch_global_losses)
            
            # 同步所有进程
            torch_dist_cuda_sync_and_barrier()
            if self.strategy.is_rank_0() and not self.args.disable_ds_ckpt:
                client_states = {
                    "epoch": epoch,
                    "loss": epoch_global_losses
                }
                self.save_checkpoint(save_path=self.args.ckpt_path, epoch=epoch, client_states=client_states)
        
        print(f"chuanwei2 454 self.strategy.is_rank_0(): {self.strategy.is_rank_0()}")
        if self.strategy.is_rank_0():
            if self.wandb is not None:
                self.wandb.finish()
            print(f"chuanwei2 458 return training_history: {training_history}")
            return training_history
        else:
            print(f"chuanwei2 461 return None")
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
    
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps=None, vllm_engines=None, pixel_values_object_ref=None):
        args = strategy.args 
        self.vllm_engines = vllm_engines
        self.disable_ds_ckpt = args.disable_ds_ckpt
        self.idx2data = pixel_values_object_ref

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
            temperature=strategy.args.temperature,
            vision_tower_attr=args.vision_tower_name,
            freeze_vision_tower=args.freeze_vision_tower,
            idx2pixel_values=self.idx2data
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

        # TODO: 使能梯度检查点功能。https://deepwiki.com/search/openrlhfmodelsactorpy202gradie_92743e54-e9f2-4ef1-ba4e-6c64efcd0eef
        # if args.gradient_checkpointing:
        #     model_actor.gradient_checkpointing_enable(
        #         gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        #     )

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

    def train_with_dataset(self, dataset, batch_size=8, num_epochs=1, shuffle=True, 
                          drop_last=True, num_workers=4, pin_memory=True, **kwargs):
        torch.cuda.empty_cache()
        self.actor.train()
        train_history = self.vision_trainer.train(
            dataset=dataset,
            batch_size=batch_size,
            num_epochs=num_epochs,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
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
        tokenizer = self.get_tokenizer()

        prompts = []
        mapping = []  # 记录每个prompt属于哪个问题和候选索引
        for q_idx, qa in enumerate(q_candidate_a):
            q = qa["question"]
            for r_idx, r in enumerate(qa["candidate_response"]):
                prompt = (
                    "USER: You are an expert in extracting facts from the given question–answer pair for an image."
                    " Extract and rewrite the facts into self-contained sentences. Exclude opinions.\n\n"
                    "You should present your result in the following format:\n"
                    "### Facts:\n- {Extracted fact 1}\n- {Extracted fact 2}\n- ...\n\n"
                    "### Question-response pair:\n"
                    f"Question: {q}\n"
                    f"Response: {r}\n"
                    "ASSISTANT:\n"
                    "### Facts:\n"
                )
                prompts.append(prompt)
                mapping.append((q_idx, r_idx))

        # vLLM 生成（直接使用字符串 prompts）
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            min_tokens=5,
            temperature=temperature,
            top_p=0.95,
            top_k=-1,
            skip_special_tokens=True,
            truncate_prompt_tokens=self._safe_max_length(tokenizer),
            include_stop_str_in_output=False,
            ignore_eos=False,
        )
        # 使用对话模板提升生成概率
        chat_prompts = self._format_chat_prompts(tokenizer, prompts)
        outputs = self.llm.generate(prompts=chat_prompts, sampling_params=sampling_params)

        # 解析生成文本
        raw_outputs = [o.outputs[0].text for o in outputs]
        num_questions = len(q_candidate_a)
        facts_nested: list[list[list[str]]] = [[] for _ in range(num_questions)]
        for (q_idx, r_idx), text in zip(mapping, raw_outputs):
            lines = []
            for line in text.splitlines():
                ls = line.strip()
                if ls.startswith("-"):
                    fact = ls.lstrip("- ").rstrip()
                    if fact:
                        lines.append(fact)
                elif ls.startswith("*"):
                    fact = ls.lstrip("* ").rstrip()
                    if fact:
                        lines.append(fact)
            # 确保每个问题有对应的候选列表
            while len(facts_nested[q_idx]) <= r_idx:
                facts_nested[q_idx].append([])
            facts_nested[q_idx][r_idx] = lines
        # 构造最终结果
        result = []
        for q_idx, qa in enumerate(q_candidate_a):
            result_item = {
                "idx": qa["idx"],
                "question": qa["question"],
                "candidate_response": qa["candidate_response"],
                "facts": facts_nested[q_idx]
            }
            # 如果输入包含图片数据，则传递到输出
            if "image_bytes" in qa:
                result_item["image_bytes"] = qa["image_bytes"]
            result.append(result_item)
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
        tokenizer = self.get_tokenizer()

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
                        "USER: You convert each given declarative English sentence into a grammatically correct Yes/No (polar) question. \
                        Do not change tense, modality, meaning, or add content. Preserve any negation words (not, no, never, etc.). \
                        Do NOT use wh-words: what, which, who, whom, whose, where, when, why, how. \
                        Form the question by subject–auxiliary inversion. \
                        - If the main verb is be, invert be. \
                        - If there is a modal (can/could/will/would/should/must/may/might), invert the modal. \
                        - Otherwise insert Do/Does/Did according to tense and number. \
                        If uncertain, use: \"Is it true that {original sentence}?\" \
                        Output format only: \
                        ASSISTANT:\n\
                        ### Polar questions:"
                    )
                    for fact in facts:
                        content += f"\n- {fact}"
                    prompts.append(content)
                q_cand_pairs.append((question, cand_response))

        # 直接用 vLLM 基于字符串 prompts 生成
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            min_tokens=1,
            temperature=temperature,
            top_p=0.9,
            top_k=-1,
            skip_special_tokens=True,
            truncate_prompt_tokens=self._safe_max_length(tokenizer),
            include_stop_str_in_output=False,
            ignore_eos=False,
        )
        chat_prompts = self._format_chat_prompts(tokenizer, prompts)
        outputs = self.llm.generate(prompts=chat_prompts, sampling_params=sampling_params)
        raw_outputs = [o.outputs[0].text for o in outputs]
        # 组装成每个candidate的simple_questions
        cand_simple_questions = []
        for text in raw_outputs:
            lines = []
            q = None
            for line in text.splitlines():
                ls = line.strip()
                if ls.startswith("###"):
                    break
                if ls.startswith("-"):
                    q = ls.lstrip("- ").rstrip()
                if ls.startswith("*"):
                    q = ls.lstrip("* ").rstrip()
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
            result_item = {
                "idx": item["idx"],
                "question": question,
                "candidates": candidates
            }
            # 如果输入包含图片数据，则传递到输出
            if "image_bytes" in item:
                result_item["image_bytes"] = item["image_bytes"]
            result.append(result_item)
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
                   如果不使用对象存储，还包含'image_bytes'字段。
            source_idxes: 源数据索引列表
        输出：
            List[Dict]，每个元素包含'idx'、'question'和'candidates'，'candidates'为List，每个元素为{'candidate_response': ..., 'simple_questions': [...], 'simple_answers': [{'1': Yes/yes scores, '0': No/no scores}]}。
        """
        # 根据是否使用对象存储来获取图片数据
        if self.pixel_values_object_ref is not None:
            # 使用对象存储
            pixel_values_dict = self.pixel_values_object_ref
        else:
            # 不使用对象存储，从 batch 数据中获取图片数据
            pixel_values_dict = {}
            for item in batch:
                if "image_bytes" in item:
                    pixel_values_dict[item["idx"]] = item["image_bytes"]
            if not pixel_values_dict:
                raise ValueError("No image data found in batch and pixel_values_object_ref is None")

        tokenizer = self.get_tokenizer()
        image_processor = self.get_image_processor()

        # 先构建结果骨架，便于回填
        results: list[dict] = []
        for item in batch:
            result_item = {
                "idx": item["idx"],
                "question": item["question"],
                "candidates": [
                    {
                        "candidate_response": cand["candidate_response"],
                        "simple_questions": list(cand.get("simple_questions", [])),
                        "simple_answers": {"1": 0.0, "0": 0.0}
                    }
                    for cand in item["candidates"]
                ],
            }
            # 如果输入包含图片数据，则传递到输出
            if "image_bytes" in item:
                result_item["image_bytes"] = item["image_bytes"]
            results.append(result_item)

        # 将整个 batch 的所有 simple_questions 扁平化，一次性推理
        from vllm import SamplingParams
        flat_prompts: list[str] = []
        flat_sampling_params: list[SamplingParams] = []
        flat_multi_modal_data: list[dict] = []
        back_mapping: list[tuple[int, int]] = []  # (item_idx, cand_idx)

        for i, item in enumerate(batch):
            # 取该样本对应的图像字节并在消费端处理
            image_bytes = pixel_values_dict[source_idxes[i]]
            if image_bytes is None:
                raise ValueError(f"No image bytes for idx {source_idxes[i]}")
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            if image_processor is None:
                raise ValueError("Image processor is None; cannot process image bytes.")
            processed = image_processor(images=[img], text=[item["question"]], return_tensors="pt", padding=True)
            if "pixel_values" in processed:
                pixel_values_tensor = processed["pixel_values"][0]
            elif "images" in processed:
                pixel_values_tensor = processed["images"][0]
            else:
                raise KeyError(f"Processor output missing pixel_values/images. Keys: {list(processed.keys())}")
            multi_modal_data = {"image": pixel_values_tensor}

            for j, cand in enumerate(item["candidates"]):
                simple_questions = cand.get("simple_questions", [])
                for _sq in simple_questions:
                    # 仅保留用户内容，由模板函数包裹；保留 <image> 标记
                    user_text = "<image>\n" + _sq.strip() + " Please answer Yes or No."
                    flat_prompts.append(user_text)
                    flat_multi_modal_data.append(multi_modal_data)
                    flat_sampling_params.append(
                        SamplingParams(
                            max_tokens=1,
                            temperature=temperature,
                            top_p=0.5,
                            top_k=-1,
                            skip_special_tokens=True,
                            truncate_prompt_tokens=self._safe_max_length(tokenizer),
                            include_stop_str_in_output=False,
                            ignore_eos=False,
                        )
                    )
                    back_mapping.append((i, j))

        if flat_prompts:
            chat_prompts = self._format_chat_prompts(tokenizer, flat_prompts)
            outputs_raw = self.generate_multimodal(chat_prompts, flat_sampling_params, flat_multi_modal_data)
            generated_texts = [o.outputs[0].text for o in outputs_raw]

            for (i_idx, j_idx), text in zip(back_mapping, generated_texts):
                t = (text or "").strip()
                t_low = t.lower()
                yes_score = 1.0 if t_low.startswith("yes") else 0.0
                no_score = 1.0 if t_low.startswith("no") else 0.0
                results[i_idx]["candidates"][j_idx]["simple_answers"]["1"] += float(yes_score)
                results[i_idx]["candidates"][j_idx]["simple_answers"]["0"] += float(no_score)

        return results

    @staticmethod
    def combine(
        batch: list[Dict],
        seed: int = None,
        pair_num: int = 2
    ) -> list[Dict]:
        """
        输入：
            batch: List[Dict]，每个元素包含'idx'、'question'和'candidates'，'candidates'为List，每个元素为{'candidate_response': ..., 'simple_questions': [...], 'simple_answers': [{'1': Yes/yes scores, '0': No/no scores}]}。

        输出：
            List[Dict]，每个元素形如：
            {
                "idx": idx,
                "question": question,
                "prefered_inferior_pairs": [{"1": prefered, "0": inferior}, ...]  # 最多 pair_num 对
            }
        """
        if seed is not None:
            random.seed(seed)
        results = []
        for item in batch:
            candidates = item['candidates']
            scores = []
            for cand in candidates:
                simple_answers = cand['simple_answers']
                scores.append(simple_answers["1"] - simple_answers["0"])
            # 随机选取不重复的索引对，要求scores[idx1] > scores[idx2]
            # 注：最多抽取 pair_num 对
            n = len(candidates)
            valid_pairs = [(i, j) for i in range(n) for j in range(n) if i != j and scores[i] > scores[j]]
            if valid_pairs:
                k = min(pair_num, len(valid_pairs))
                selected = random.sample(valid_pairs, k)
                pairs = []
                for idx1, idx2 in selected:
                    prefered = candidates[idx1]['candidate_response']
                    inferior = candidates[idx2]['candidate_response']
                    pairs.append({"1": prefered, "0": inferior})
                result_item = {
                    "idx": item["idx"],
                    "question": item["question"],
                    "prefered_inferior_pairs": pairs
                }
                # 如果输入包含图片数据，则传递到输出
                if "image_bytes" in item:
                    result_item["image_bytes"] = item["image_bytes"]
                results.append(result_item)
        return results


class PolicyRayActor(MultiModalLLMRayActor):
    def __init__(self, *args, pixel_values_object_ref=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pixel_values_object_ref = pixel_values_object_ref

    def set_pixel_values_object_ref(self, pixel_values_object_ref):
        # 添加调试信息
        self.pixel_values_object_ref = pixel_values_object_ref

    def generate_n_candidates(self, questions=None, item_indexes=None, questions_with_images=None, n_candidates=2, **gen_kwargs):
        from vllm import SamplingParams     # or AsyncLLMEngine if needed
        from torchvision import transforms

        # 处理两种不同的输入格式
        if questions_with_images is not None:
            # 不使用对象存储，数据中已包含图片
            batch_size = len(questions_with_images)
            questions = [item['question'] for item in questions_with_images]
            item_indexes = [item['idx'] for item in questions_with_images]
            image_bytes_dict = {item['idx']: item['image_bytes'] for item in questions_with_images}
            has_images = True
        elif questions is not None and item_indexes is not None:
            # 使用对象存储
            batch_size = len(questions)
            image_bytes_dict = self.pixel_values_object_ref if self.pixel_values_object_ref is not None else None
            has_images = image_bytes_dict is not None
        else:
            raise ValueError("Either questions_with_images or (questions + item_indexes) must be provided")

        sampling_params = SamplingParams(
            temperature=gen_kwargs.get('temperature', 0.7),
            top_p=gen_kwargs.get('top_p', 0.9),
            top_k=-1,
            max_tokens=gen_kwargs.get('max_new_tokens', 128),
            min_tokens=1,
            skip_special_tokens=False,
        )

        if has_images:
            # 直接构建多模态输入列表
            prompts = []
            sampling_params_list = []
            multi_modal_data_list = []
            
            img_processor = self.get_image_processor()
            if img_processor is None:
                raise ValueError("Image processor is None; cannot process image bytes.")

            for i, question in enumerate(questions):
                prompt = f"USER: <image>\n{question}\nASSISTANT:"
                image_bytes = image_bytes_dict[item_indexes[i]]
                if image_bytes is None:
                    raise ValueError(f"No image bytes for idx {item_indexes[i]}")
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                processed = img_processor(images=[img], text=[question], return_tensors="pt", padding=True)
                if "pixel_values" in processed:
                    pixel_values_tensor = processed["pixel_values"][0]
                elif "images" in processed:
                    pixel_values_tensor = processed["images"][0]
                else:
                    raise KeyError(f"Processor output missing pixel_values/images. Keys: {list(processed.keys())}")
                multi_modal_data = {"image": pixel_values_tensor}
                
                for seed in range(n_candidates):
                    sd = sampling_params.clone()
                    sd.seed = seed
                    
                    prompts.append(prompt)
                    sampling_params_list.append(sd)
                    multi_modal_data_list.append(multi_modal_data)
            
            # 直接使用 generate_multimodal 生成
            outputs_raw = self.generate_multimodal(
                prompts=prompts,
                sampling_params=sampling_params_list,
                multi_modal_data=multi_modal_data_list
            )
            
            # 提取文本输出
            outputs = [o.outputs[0].text for o in outputs_raw]

            # Map back to batches
            q_candidate_a = [{} for _ in range(batch_size)]
            idx = 0
            for i in range(batch_size):
                candidates_one_q = []
                for _ in range(n_candidates):
                    if idx < len(outputs):
                        candidates_one_q.append(outputs[idx])
                    idx += 1
                q_candidate_a[i].update({"idx": item_indexes[i]})
                q_candidate_a[i].update({"question": questions[i]})
                q_candidate_a[i].update({"candidate_response": candidates_one_q})

                # 如果不使用对象存储，将图片数据包含在返回结果中
                if questions_with_images is not None:
                    q_candidate_a[i].update({"image_bytes": image_bytes_dict[item_indexes[i]]})

            return q_candidate_a
        else:
            logging.warning("chuanwei No images found, returning None")
            return None

    
class ReferenceModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, pixel_values_object_ref=None):
        self._setup_distributed(strategy)
        self.idx2data = pixel_values_object_ref
        model_actor = VisionActor(
            pretrain,
            use_flash_attention_2=strategy.args.use_flash_attn_policy,
            bf16=strategy.args.policy_train_bf16,
            load_in_4bit=strategy.args.policy_train_load_in_4bit,
            load_in_8bit=strategy.args.policy_train_load_in_8bit,
            temperature=strategy.args.temperature,
            vision_tower_attr=strategy.args.vision_tower_name,
            freeze_vision_tower=strategy.args.freeze_vision_tower,
            idx2pixel_values=self.idx2data
        )
        strategy.print(f"Reference model:\n{model_actor}")

        # Replace self.model_actor.model with deepspeed engine and return model_actor.
        self.model_actor_ds_engine = self.strategy.prepare(model_actor, is_rlhf=False)
        # 避免在 tp>1 时保留一份未封装模型导致重复显存占用：将底层模型指向 DeepSpeed 引擎
        self.model_actor_ds_engine.eval()

    def batch_logp(self, prefered_inferior_response_list, requires_grad=False):
        # 现在 VisionActor 会自己从 idx2pixel_values 获取 pixel_values，直接传递原始数据
        return self.model_actor_ds_engine.batch_logp(prefered_inferior_response_list, requires_grad)