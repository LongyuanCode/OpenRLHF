import io
import os

import base64
from modelscope.msdatasets import MsDataset
import numpy as np
import itertools
from PIL import Image
import ray
import torch
from torchvision import transforms
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    default_data_collator
)

from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils import safe_ray_get
from openrlhf.trainer.ray.rlaif_vision_actor import LabelerRayActor, PolicyModelActor, ReferenceModelActor
from ray.util.placement_group import placement_group
from torch.utils.data import DataLoader
from openrlhf.trainer.ray import batch_vllm_engine_call

@ray.remote
class RLAIFTrainer:
    def __init__(
        self,
        strategy: DeepspeedStrategy,
        policy_model_group: RayActorGroup,
        labeler_vllm_engines=None,
        policy_vllm_engines=None,
        reference_vllm_engines=None,
        pg_labeler_vllm=None,
        pg_policy_vllm=None,
        **generate_kwargs
    ) -> None:
        self.strategy = strategy
        self.args = strategy.args
        self.policy_model_group = policy_model_group
        self.labeler_image_processor = ray.get(labeler_vllm_engines[0].get_image_processor.remote()) if labeler_vllm_engines is not None else None
        self.policy_image_processor = ray.get(policy_vllm_engines[0].get_image_processor.remote()) if policy_vllm_engines is not None else None
        self.reference_image_processor = ray.get(reference_vllm_engines[0].get_image_processor.remote()) if reference_vllm_engines is not None else None
        
        # vLLM engines for optimized inference
        self.labeler_vllm_engines = labeler_vllm_engines
        self.policy_vllm_engines = policy_vllm_engines
        self.reference_vllm_engines = reference_vllm_engines
        # PG 句柄（从 driver 传入），用于在本 Actor 内释放
        self.pg_labeler_vllm = pg_labeler_vllm
        self.pg_policy_vllm = pg_policy_vllm

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
            images=image,
            text=item['question'],
            return_tensors="pt",
            padding="max_length",
            max_length=self.args.max_len,
            truncation=True
        )
        # labeler处理
        labeler_processed = labeler_image_processor(
            images=image,
            text=item['question'],
            return_tensors="pt",
            padding="max_length",
            max_length=self.args.max_len,
            truncation=True
        )

        idx = item["idx"]  # 直接用原始字符串
        
        # 修复：检查处理结果中的键，适配不同的处理器输出格式
        def safe_get_tensor(processed_data, key, default_key=None):
            """安全获取张量，处理不同处理器的输出格式"""
            if key in processed_data:
                return processed_data[key].squeeze(0)
            elif default_key and default_key in processed_data:
                return processed_data[default_key].squeeze(0)
            else:
                available_keys = list(processed_data.keys())
                raise KeyError(f"在处理结果中找不到键 '{key}'。可用的键: {available_keys}")
        
        # 返回嵌套字典格式，图片内容单独存储
        return {
            idx: {
                "input_ids": safe_get_tensor(processed, "input_ids"),
                "attention_mask": safe_get_tensor(processed, "attention_mask"),
                "question": item["question"]
            },
            f"{idx}_pixel_values": {
                "pixel_values": safe_get_tensor(processed, "pixel_values"),
                "labeler_pixel_values": safe_get_tensor(labeler_processed, "pixel_values")
            }
        }
    
    def prepare_dataset(self, is_streaming_for_debug=False):
        from modelscope.utils.constant import DownloadMode
        cache_dir = os.getenv('MS_CACHE_HOME')
        if is_streaming_for_debug:
            streaming_dataset = MsDataset.load(
                self.args.dataset_name,
                split="train",
                streaming=True,
                trust_remote_code=False,
                cache_dir=cache_dir,
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
            )
            
            # 只取第一条数据用于调试
            item = next(iter(streaming_dataset))
            
            # 处理这一条数据
            result = self._data_process_fn(
                item,
                policy_image_processor=self.policy_image_processor,
                labeler_image_processor=self.labeler_image_processor
            )
            
            # 拆分主数据和 pixel_values
            processed_dict = {}
            pixel_values_dict = {}
            for k, v in result.items():
                if k.endswith("_pixel_values"):
                    # 将 PyTorch 张量转换为 numpy 数组以便 Ray 序列化
                    converted_v = {}
                    for tensor_key, tensor_value in v.items():
                        if isinstance(tensor_value, torch.Tensor):
                            converted_v[tensor_key] = tensor_value.cpu().numpy()
                        else:
                            converted_v[tensor_key] = tensor_value
                    pixel_values_dict[k.replace("_pixel_values", "")] = converted_v
                else:
                    processed_dict[k] = v
            
            self.processed_dataset = processed_dict
            
            try:
                self.pixel_values_object_ref = ray.put(pixel_values_dict)
            except Exception as e:
                return
            
            # 保存所有idx的列表，便于切片
            self.idx_list = list(self.processed_dataset.keys())
            
        else:
            raw_dataset = MsDataset.load(
                self.args.dataset_name,
                split="train",
                cache_dir=cache_dir,
                trust_remote_code=False,
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
            )
            if "train" in raw_dataset:
                dataset = raw_dataset["train"]
            else:
                dataset = raw_dataset

            # 处理每个样本，收集主数据和 pixel_values
            processed_dict = {}
            pixel_values_dict = {}
            for item in tqdm(dataset, desc="Processing dataset"):
                result = self._data_process_fn(
                    item,
                    policy_image_processor=self.policy_image_processor,
                    labeler_image_processor=self.labeler_image_processor
                )
                # 拆分主数据和 pixel_values
                for k, v in result.items():
                    if k.endswith("_pixel_values"):
                        # 将 PyTorch 张量转换为 numpy 数组以便 Ray 序列化
                        converted_v = {}
                        for tensor_key, tensor_value in v.items():
                            if isinstance(tensor_value, torch.Tensor):
                                converted_v[tensor_key] = tensor_value.cpu().numpy()
                            else:
                                converted_v[tensor_key] = tensor_value
                        pixel_values_dict[k.replace("_pixel_values", "")] = converted_v
                    else:
                        processed_dict[k] = v
            self.processed_dataset = processed_dict
            
            # 创建 ObjectRef
            try:
                self.pixel_values_object_ref = ray.put(pixel_values_dict)
            except Exception as e:
                return

            # 保存所有idx的列表，便于切片
            self.idx_list = list(self.processed_dataset.keys())

    def train(self):
        IS_DUBUG = True
        if not hasattr(self, "processed_dataset"):
            self.prepare_dataset(is_streaming_for_debug=IS_DUBUG)

        # 验证 ObjectRef 是否存在
        if not hasattr(self, "pixel_values_object_ref"):
            return

        train_dataloader = DataLoader(
            self.idx_list,
            batch_size=self.args.labeler_batch_size,
            sampler=None, # 不再使用DistributedSampler
            collate_fn=lambda x: x,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        prefered_inferior_response_list = []
        batch_vllm_engine_call(self.policy_vllm_engines, "wake_up")
        batch_vllm_engine_call(self.labeler_vllm_engines, "wake_up")
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            try:
                # batch 是 idx 列表
                item_indexes = batch

                # Step 1: Generate candidate responses using policy model group
                # Use policy model group's multiple actors for parallel generation
                # Each actor will generate n_candidates responses for their assigned questions
                refs = []
                batch_size_each_policy_vllm_engines = (len(batch) + len(self.policy_vllm_engines) - 1) // len(self.policy_vllm_engines)
                for i, engine in enumerate(self.policy_vllm_engines):
                    ref_set = engine.set_pixel_values_object_ref.remote(self.pixel_values_object_ref)
                    ray.get(ref_set)
                    shard_idx = item_indexes[i * batch_size_each_policy_vllm_engines : (i + 1) * batch_size_each_policy_vllm_engines]
                    if shard_idx == None or len(shard_idx) == 0:
                        continue
                    shard_questions = [self.processed_dataset[idx]['question'] for idx in shard_idx]
                    refs.append(engine.generate_n_candidates.remote(
                        questions=shard_questions,
                        item_indexes=shard_idx,
                        n_candidates=self.args.n_candidates,
                        max_new_tokens=self.args.max_new_tokens,
                        temperature=self.args.policy_generate_temperature,
                    ))
                shard_candidates = ray.get(refs)
                all_candidates = [cand for shard in shard_candidates for cand in shard]
                print(f"chuanwei all_candidates = {all_candidates}")

                # Step2 extract facts from question-candidate_response pairs
                refs = []
                batch_size_each_labeler_vllm_engines = (len(batch) + len(self.labeler_vllm_engines) - 1) // len(self.labeler_vllm_engines)
                for i, engine in enumerate(self.labeler_vllm_engines):
                    ref_set = engine.set_pixel_values_object_ref.remote(self.pixel_values_object_ref)
                    ray.get(ref_set)
                    shard = all_candidates[i * batch_size_each_labeler_vllm_engines : (i + 1) * batch_size_each_labeler_vllm_engines]
                    if shard == None or len(shard) == 0:
                        continue
                    refs.append(engine.divide.remote(
                        q_candidate_a=shard,
                        max_new_tokens=self.args.max_new_tokens,
                        temperature=0.0,
                        do_sample=False
                    ))
                shard_simple_declarative_sentences_list = ray.get(refs)
                simple_declarative_sentences = [sentence for shard in shard_simple_declarative_sentences_list for sentence in shard]
                print(f"chuanwei simple_declarative_sentences: {simple_declarative_sentences}")

                # Conquer step: convert facts to simple yes/no questions
                refs = []
                for i, engine in enumerate(self.labeler_vllm_engines):
                    shard = simple_declarative_sentences[i * batch_size_each_labeler_vllm_engines : (i + 1) * batch_size_each_labeler_vllm_engines]
                    if shard == None or len(shard) == 0:
                        continue
                    refs.append(engine.conquer.remote(
                        q_facts_batch=shard,
                        max_new_tokens=self.args.max_new_tokens,
                        temperature=0.0,
                        do_sample=False
                    ))
                shard_simple_questions_result_list = ray.get(refs)
                simple_questions_result_list = [result for shard in shard_simple_questions_result_list for result in shard]
                print(f"chuanwei simple_questions_result_list: {simple_questions_result_list}")

                # YesNO step: Answer simple questions with yes or no.
                refs = []
                for i, engine in enumerate(self.labeler_vllm_engines):
                    shard = simple_questions_result_list[i * batch_size_each_labeler_vllm_engines : (i + 1) * batch_size_each_labeler_vllm_engines]
                    shard_item_idxes = [item["idx"] for item in shard]
                    if shard == None or len(shard) == 0:
                        continue
                    refs.append(engine.YesNo.remote(
                        batch=shard,
                        source_idxes=shard_item_idxes,
                        max_new_tokens=3,
                        temperature=0.0,
                        do_sample=False
                    ))
                yesno_results_list = ray.get(refs)
                yesno_results = [ans for shard in yesno_results_list for ans in shard]
                print(f"chuanwei yesno_results: {yesno_results}")

                # Combine step: Select prefered and inferior candidate responses.
                prefer_inferior_response = LabelerRayActor.combine(yesno_results, pair_num=self.args.prefered_inferior_pair_num, seed=42)
                print(f"chuanwei prefer_inferior_response: {prefer_inferior_response}")
                # 累加用于后续训练/评估
                if prefer_inferior_response:
                    prefered_inferior_response_list.extend(prefer_inferior_response)
            except Exception as e:
                import traceback
                print(f"[Error] Exception occurred at batch {batch_idx}: {e}")
                traceback.print_exc()
        
        print(f"chuanwei prefered_inferior_response_list = {prefered_inferior_response_list}")
        # we don't need laber vllm engines any more.
        # engine.sleep.remote(level=1/2) can also be used but CPU memory will be occupied.
        # engine.sleep.remote(level=1/2) is used when you needwake up engines later.
        # for engine in self.labeler_vllm_engines:
        #     engine.sleep.remote(level=2)
        try:
            # 1) 调用 Actor 内部的 shutdown（包含 sleep/reset_prefix_cache/empty_cache）
            refs = []
            for engine in (self.labeler_vllm_engines or []):
                refs.append(engine.shutdown.remote(level=2))
            for engine in (self.policy_vllm_engines or []):
                refs.append(engine.shutdown.remote(level=2))
            if refs:
                ray.get(refs)
        except Exception as e:
            print(f"[warn] shutdown engines failed: {e}", flush=True)

        # 2) 先获取共享 PG 句柄（在 kill 前获取，避免 Actor 已退出导致取句柄失败）
        pg_handles = []
        try:
            if self.labeler_vllm_engines:
                try:
                    pg_handles.append(ray.get(self.labeler_vllm_engines[0].get_shared_pg.remote()))
                except Exception as _e:
                    print(f"[warn] get labeler shared PG failed: {_e}", flush=True)
            if self.policy_vllm_engines:
                try:
                    pg_handles.append(ray.get(self.policy_vllm_engines[0].get_shared_pg.remote()))
                except Exception as _e:
                    print(f"[warn] get policy shared PG failed: {_e}", flush=True)
        except Exception as e:
            print(f"[warn] get shared PG failed: {e}", flush=True)

        # 3) 禁止重启地 kill，以释放 GPU 令牌
        try:
            for engine in (self.labeler_vllm_engines or []):
                try:
                    ray.kill(engine, no_restart=True)
                except Exception as _e:
                    print(f"[warn] kill labeler engine failed: {_e}", flush=True)
            for engine in (self.policy_vllm_engines or []):
                try:
                    ray.kill(engine, no_restart=True)
                except Exception as _e:
                    print(f"[warn] kill policy engine failed: {_e}", flush=True)
        except Exception as e:
            print(f"[warn] kill engines failed: {e}", flush=True)

        # 4) 释放创建这些引擎时使用的共享 Placement Group
        try:
            from ray.util.placement_group import remove_placement_group
            removed = 0
            for pg in {pg for pg in pg_handles if pg is not None}:
                try:
                    remove_placement_group(pg)
                    removed += 1
                except Exception as _e:
                    print(f"[warn] remove PG failed: {_e}", flush=True)
            print(f"[info] removed {removed} placement groups; resources now: {ray.available_resources()}", flush=True)
        except Exception as e:
            print(f"[warn] remove placement groups failed: {e}", flush=True)

        # 5) 置空本地引用
        self.labeler_vllm_engines = None
        self.policy_vllm_engines = None
            
        # if self.strategy.args.deepspeed_enable_sleep:
        #     ray.get(self.policy_model_group.async_run_method(method_name="reload_states"))

        # print GPU stuff
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1024**2  # MB
            reserved = torch.cuda.memory_reserved(i) / 1024**2  # MB
            allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
            free = reserved - allocated  # MB

            print(f"GPU {i}: 总显存: {total:.0f} MB, 已分配: {allocated:.0f} MB, 已保留: {reserved:.0f} MB, 剩余: {free:.0f} MB")

        # reference model group
        bundles_ref = [{'CPU': 1, 'GPU':1}]
        pg_ref = placement_group(bundles_ref, strategy='PACK')
        ray.get(pg_ref.ready())
        self.ref_model_group = RayActorGroup(
            # num_nodes=self.args.num_nodes,
            num_nodes=1,
            # num_gpus_per_node=self.args.num_gpus_per_node,
            num_gpus_per_node=1,
            ray_actor_type=ReferenceModelActor,
            pg=pg_ref,
            num_gpus_per_actor=self.args.num_gpus_per_ref_actor,
            duplicate_actors=self.args.ring_attn_size * self.args.ds_tensor_parallel_size_ref,
            resources=None,
        )

        if getattr(self.args, "pretrain_policy", None) is not None:
            refs = self.ref_model_group.async_init_model_from_pretrained(
                strategy=self.strategy,
                pretrain=self.args.pretrain_policy,
                pixel_values_object_ref=self.pixel_values_object_ref
            )
        ray.get(refs)
        print(f'chuanwei Ref ray actor group created.')

        # 使用新的批量计算方法，按逻辑 actor 分片进行 batch 计算
        logp_results = self.ref_model_group.run_method_per_batch_per_logical_actor(
            method_name='batch_logp',
            data_list=prefered_inferior_response_list,
            requires_grad=False
        )
        print(f"chuanwei logp_results: {logp_results}")

        # policy model group
        # 为Zero Stage 2分布式训练，每个actor使用1个GPU
        bundles_policy = [{"CPU": 1, "GPU": 1} for _ in range(2)]  # 2个GPU，每个1个bundle
        pg_policy = placement_group(bundles_policy, strategy="PACK")
        ray.get(pg_policy.ready())

        self.policy_model_group = RayActorGroup(
            # num_nodes=self.args.num_nodes,
            num_nodes=1,
            # num_gpus_per_node=self.args.num_gpus_per_node,
            num_gpus_per_node=2,
            ray_actor_type=PolicyModelActor,
            pg=pg_policy,
            num_gpus_per_actor=1,  # 强制设置为1，确保每个actor只使用1个GPU
            duplicate_actors=self.args.ring_attn_size * self.args.ds_tensor_parallel_size_policy,
            resources=None,
        )
        if getattr(self.args, "pretrain_policy", None) is not None:
            refs = self.policy_model_group.async_init_model_from_pretrained(
                strategy=self.strategy,
                pretrain=self.args.pretrain_policy,
                max_steps=len(self.idx_list)//self.args.train_batch_size,
                pixel_values_object_ref=self.pixel_values_object_ref
            )
        ray.get(refs)
        print("chuanwei policy ray actor group created")

        global_loss_log = []
        # 使用新的分布式训练方法，每个逻辑actor通过DistributedSampler拿到不同的数据
        train_history_refs = self.policy_model_group.async_train_with_distributed_sampling(
            dataset=prefered_inferior_response_list,
            batch_size=self.args.micro_train_batch_size,
            num_epochs=self.args.num_epochs,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True
        )
        train_history = ray.get(train_history_refs)
        print(f"chuanwei train_history: {train_history}")
        global_loss_log.extend([h for h in train_history if h is not None])

        # save_model
        # ray.get(self.policy_model_group.async_save_model())

        if self.strategy.args.deepspeed_enable_sleep:
            ray.get(self.policy_model_group.async_run_method(method_name="offload_states"))

        if self.policy_vllm_engines is not None:
            self._broadcast_to_policy_vllm_engines()
    
    def _broadcast_to_policy_vllm_engines(self):
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.policy_vllm_engines, "wake_up")

        ray.get(self.policy_model_group.async_run_method(method_name="broadcast_to_vllm"))

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.policy_vllm_engines, "sleep")