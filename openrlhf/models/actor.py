from typing import Optional
import io

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoConfig, AutoTokenizer, AutoImageProcessor, AutoProcessor, BitsAndBytesConfig
from PIL import Image
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor
from .utils import compute_entropy, log_probs_from_logits


class Actor(nn.Module):
    """
    Base class for Actor models in reinforcement learning.

    This class serves as a foundation for implementing various actor models, which are responsible for selecting actions based on the policy learned from the environment.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        attn_implementation (str, optional): Attention mechanism implementation to use. Defaults to "flash_attention_2".
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        target_modules (list, optional): List of target modules for applying LoRA. Defaults to None.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.
        temperature (float, optional): Temperature for action selection. Defaults to 1.0.
        use_liger_kernel (bool, optional): Whether to use Liger Kernel for the model. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        attn_implementation="flash_attention_2",
        bf16=True,
        load_in_4bit=False,
        load_in_8bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        temperature=1.0,
        use_liger_kernel=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature

        if isinstance(pretrain_or_model, str):
            # Support multiple attention mechanism implementations
            attn_impl = attn_implementation

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            elif load_in_8bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,              # 控制 outlier threshold，可调降低精度损失
                    llm_int8_enable_fp32_cpu_offload=True  # 若显存不足，可将一些权重卸载到 CPU
                )
            else:
                nf4_config = None

            if use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM

                model_class = AutoLigerKernelForCausalLM
            else:
                model_class = AutoModelForCausalLM

            self.model = model_class.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

                # set_z3_leaf_modules is required for MoE models
                for m in self.model.modules():
                    # https://github.com/microsoft/DeepSpeed/pull/4966
                    if "SparseMoeBlock" in m.__class__.__name__:
                        deepspeed.utils.set_z3_leaf_modules(self.model, [m.__class__])
                        print(f"Setting zero3 leaf for model on class with name: {m.__class__.__name__}")
                        break

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples
        else:
            self.model = pretrain_or_model

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        allgather_logits=False,
        return_logprobs=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
        return_entropy=False,
    ) -> torch.Tensor:
        """Returns action log probs"""
        batch, seqlen = sequences.size()
        foward_attention_mask = attention_mask
        if self.packing_samples:
            sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                sequences, attention_mask, ring_attn_group
            )
            foward_attention_mask = None
        else:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(sequences, attention_mask=foward_attention_mask, position_ids=position_ids)
        # https://github.com/OpenRLHF/OpenRLHF/pull/634
        output["logits"] = output["logits"].to(torch.float32)

        if return_entropy:
            assert return_output
            entropy = compute_entropy(output["logits"])
            if self.packing_samples:
                entropy = gather_and_pad_tensor(entropy, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
            setattr(output, "entropy", entropy[:, :-1])

        return_action_log_probs = action_mask is not None
        if not return_action_log_probs and not return_logprobs:
            assert return_output
            if allgather_logits and self.packing_samples:
                output["logits"] = gather_and_pad_tensor(
                    output["logits"], ring_attn_group, ring_attn_pad_len, indices, batch, seqlen
                )
            return output

        log_probs = log_probs_from_logits(output["logits"], rolled_sequences, temperature=self.temperature)

        if self.packing_samples:
            log_probs = gather_and_pad_tensor(log_probs, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)

        log_probs = log_probs[:, :-1]
        if not return_action_log_probs and return_logprobs:
            return (log_probs, output) if return_output else log_probs

        action_log_probs = log_probs[:, -action_mask.shape[1] :] * action_mask.float()

        return (action_log_probs, output) if return_output else action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

class VisionActor(nn.Module):
    def __init__(
        self,
        model_name_or_path,
        trust_remote_code=True,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        load_in_8bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        device_map=None,
        freeze_vision_tower=True,
        vision_tower_attr="vision_tower",
        idx2pixel_values=None,
        **kwargs,
    ):
        super().__init__()
        # 1. 自动加载tokenizer和image_processor
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        # 安全设置 tokenizer 的 max_length，避免 OverflowError
        if not hasattr(self.tokenizer, 'model_max_length') or self.tokenizer.model_max_length is None:
            self.tokenizer.model_max_length = 4096
        elif self.tokenizer.model_max_length > 1000000:
            self.tokenizer.model_max_length = 4096
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                use_fast=True
            )
        except Exception:
            self.image_processor = None
        # 同时尝试加载统一 Processor（可以同时产出 input_ids 与 pixel_values）
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                use_fast=True
            )
        except Exception:
            self.processor = None

        # 2. 加载多模态/纯文本模型
        attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"
        model_kwargs = dict(
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16 if bf16 else "auto",
            device_map=device_map,
        )
        model_kwargs.update(kwargs)
        # 优先尝试以 CausalLM 加载；若遇到多模态配置（如 LlavaConfig）导致不支持，则回退到 Vision2Seq
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        except ValueError as e:
            cfg = None
            try:
                cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
            except Exception:
                pass
            if cfg is not None:
                self.model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, **model_kwargs)
            else:
                raise

        # 冻结视觉模块
        if freeze_vision_tower:
            vision_tower = getattr(self.model, vision_tower_attr, None)
            if vision_tower is not None:
                for param in vision_tower.parameters():
                    param.requires_grad = False

        # 3. LoRA
        if lora_rank > 0:
            self.model.enable_input_require_grads()
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            # LoRA权重精度适配
            for name, module in self.model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if "lm_head" in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)

        # 4. 其它设置
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        
        # 5. 设置 pixel_values 映射
        self.idx2pixel_values = idx2pixel_values
    
    def get_processor(self):
        # 优先返回统一 Processor，便于外部同时处理文本与图像
        return self.processor

    def get_image_processor(self):
        return self.image_processor

    def get_tokenizer(self):
        return self.tokenizer

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        images=None,
        labels=None,
        **kwargs,
    ):
        # 兼容不同模型的输入
        model_inputs = {}
        if input_ids is not None:
            model_inputs["input_ids"] = input_ids
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        if images is not None:
            model_inputs["images"] = images
        if labels is not None:
            model_inputs["labels"] = labels
        model_inputs.update(kwargs)
        
        # 避免递归：直接调用底层模型而不是 self.model
        # 如果 self.model 是 DeepSpeed 引擎，直接调用
        # 如果 self.model 是原始模型，也直接调用
        if hasattr(self.model, 'module'):
            # DeepSpeed 引擎的情况
            return self.model.module(**model_inputs)
        else:
            # 原始模型的情况
            return self.model(**model_inputs)
    
    def batch_logp(self, prefered_inferior_response_list, requires_grad=True):
        """
        仅使用原始图片字节进行多模态 logp 计算。
        输入中每个样本包含 'idx', 'question', 'prefered_inferior_pairs'（两条回答）。
        计算两条回答各自的 logp，并返回到原结构中（logp_1, logp_0）。
        """
        # 优先使用统一 Processor；若不存在再回退到 image_processor
        processor = getattr(self, "processor", None)
        if processor is None:
            raise ValueError("No processor available; cannot process image bytes.")

        tokenizer = self.tokenizer
        if not hasattr(tokenizer, 'image_token') or not hasattr(tokenizer, 'image_token_id'):
            raise ValueError("Tokenizer missing image_token/image_token_id; cannot build multimodal prompts.")

        safe_max_length = getattr(tokenizer, 'model_max_length', None)
        if safe_max_length is None or safe_max_length <= 0 or safe_max_length > 1000000:
            safe_max_length = 4096

        prompts = []
        targets = []
        pil_images = []

        for item in prefered_inferior_response_list:
            idx = item['idx']
            question = item['question']
            pair = item['prefered_inferior_pairs'][0]

            data_entry = self.idx2pixel_values[idx]
            if not isinstance(data_entry, (bytes, bytearray)):
                raise TypeError(f"idx2pixel_values[{idx!r}] must be bytes; got {type(data_entry)}")
            img = Image.open(io.BytesIO(data_entry)).convert("RGB")
            # 注意：部分多模态处理器（如 LlavaProcessor）要求同时提供 text
            prompt_text = f"USER: <image>\n{question}\nASSISTANT: "
            prompts.append(prompt_text)
            targets.append(pair['1'])
            pil_images.append(img)

            prompts.append(prompt_text)
            targets.append(pair['0'])
            pil_images.append(img)

        # 使用 processor 统一生成 prompt 与 full 的编码，确保 image tokens 与视觉特征对齐
        prompt_outputs = processor(
            images=pil_images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=safe_max_length
        )
        full_texts = [p + t for p, t in zip(prompts, targets)]
        full_outputs = processor(
            images=pil_images,
            text=full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=safe_max_length
        )

        prompt_attention_mask = prompt_outputs["attention_mask"]
        full_input_ids = full_outputs["input_ids"]
        full_attention_mask = full_outputs["attention_mask"]
        if "pixel_values" in full_outputs:
            pixel_values_batch = full_outputs["pixel_values"]
        elif "images" in full_outputs:
            pixel_values_batch = full_outputs["images"]
        else:
            raise KeyError(f"Processor output missing pixel_values/images. Keys: {list(full_outputs.keys())}")

        device = next(self.model.parameters()).device
        model_inputs = dict(
            input_ids=full_input_ids.to(device),
            attention_mask=full_attention_mask.to(device),
            pixel_values=pixel_values_batch.to(device)
        )

        if requires_grad:
            outputs = self.model.module(**model_inputs) if hasattr(self.model, 'module') else self.model(**model_inputs)
        else:
            with torch.no_grad():
                outputs = self.model.module(**model_inputs) if hasattr(self.model, 'module') else self.model(**model_inputs)

        logits = outputs.logits
        batch_size = len(prefered_inferior_response_list)

        logps = []
        for i in range(batch_size * 2):
            pl = int(prompt_attention_mask[i].sum().item())
            fl = int(full_attention_mask[i].sum().item())
            ans_len = max(fl - pl, 0)
            if ans_len == 0:
                logps.append(torch.tensor(0.0, device=logits.device))
                continue

            # 目标 token 为 full_input_ids 的回答段
            target_tokens = full_input_ids[i, pl: pl + ans_len].to(device)
            # 对应的 logits 区间从 prompt 的最后一个位置开始预测
            target_logits = logits[i, pl - 1: pl - 1 + ans_len, :]

            log_probs = torch.log_softmax(target_logits, dim=-1)
            token_logp = log_probs.gather(1, target_tokens.unsqueeze(-1)).squeeze(-1)
            sample_logp = token_logp.sum()
            if not requires_grad:
                sample_logp = sample_logp.detach().cpu()
            logps.append(sample_logp)

        results = []
        for i, item in enumerate(prefered_inferior_response_list):
            result = dict(item)
            result['logp_1'] = logps[2 * i]
            result['logp_0'] = logps[2 * i + 1]
            results.append(result)

        return results