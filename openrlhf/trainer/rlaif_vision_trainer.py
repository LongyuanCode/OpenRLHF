import io
import os
import json
from datetime import datetime

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

        # 将图片统一序列化为字节，留待下游按需预处理
        if not isinstance(image, Image.Image):
            raise ValueError(f"Image convert failed, got type: {type(image)}")
        # 统一为 RGB 并使用 PNG 无损编码
        image = image.convert("RGB")
        _buf = io.BytesIO()
        image.save(_buf, format="PNG")
        image_bytes = _buf.getvalue()

        idx = item["idx"]  # 直接用原始字符串
        
        # 返回主数据与图片字节分开，避免预计算
        return {
            idx: {
                "question": item["question"]
            },
            f"{idx}_image_bytes": image_bytes
        }
    
    def prepare_dataset(self, is_streaming_for_debug=False, is_put_images_object_store=False, max_data_num=None):
        from modelscope.utils.constant import DownloadMode
        cache_dir = os.getenv('MODELSCOPE_CACHE', '/root/gpufree-data/modelscope_cache')
        if is_streaming_for_debug:
            streaming_dataset = MsDataset.load(
                self.args.dataset_name,
                split="train",
                streaming=False,
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

            # 拆分主数据和 image_bytes
            processed_dict = {}
            image_bytes_dict = {}
            for k, v in result.items():
                if k.endswith("_image_bytes"):
                    image_bytes_dict[k.replace("_image_bytes", "")] = v
                else:
                    processed_dict[k] = v

            # 根据 is_put_images_object_store 参数决定图片数据的存储方式
            if is_put_images_object_store:
                # 将图片数据直接放到 processed_dataset 中
                for idx, image_bytes in image_bytes_dict.items():
                    processed_dict[idx]["image_bytes"] = image_bytes
                self.processed_dataset = processed_dict
                self.pixel_values_object_ref = None  # 不使用对象存储
            else:
                # 原有逻辑：将图片数据放到 Ray 对象存储中
                self.processed_dataset = processed_dict
                try:
                    # 为保持下游接口不变，仍沿用 pixel_values_object_ref 命名，但内容为 idx->image_bytes
                    self.pixel_values_object_ref = ray.put(image_bytes_dict)
                except Exception as e:
                    return

            # 保存所有idx的列表，便于切片
            self.idx_list = list(self.processed_dataset.keys())
            
        else:
            # 如果给定了 max_data_num，优先尝试使用分片语法，避免在 MsDataset.load 阶段就扫描完整数据集
            split_to_use = "train"
            if max_data_num is not None and isinstance(max_data_num, int) and max_data_num > 0:
                split_to_use = f"train[:{max_data_num}]"

            print(f"chuanwei Loading dataset from ModelScope... (split={split_to_use})", flush=True)
            try:
                raw_dataset = MsDataset.load(
                    self.args.dataset_name,
                    split=split_to_use,
                    cache_dir=cache_dir,
                    trust_remote_code=False,
                    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                    streaming=False
                )
            except Exception as e:
                # 兼容不支持切片分片的场景，回退到原有 split="train" + 事后 islice
                print(f"chuanwei MsDataset split slicing not supported or failed ({e}), fallback to split='train'", flush=True)
                raw_dataset = MsDataset.load(
                    self.args.dataset_name,
                    split="train",
                    cache_dir=cache_dir,
                    trust_remote_code=False,
                    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                    streaming=False
                )
            if "train" in raw_dataset:
                dataset = raw_dataset["train"]
            else:
                dataset = raw_dataset
            print(f"chuanwei Dataset loaded successfully", flush=True)

            # 若 split 已经使用了切片表达式，则不再二次切片；否则使用 islice 做兜底限制
            if max_data_num is not None and isinstance(max_data_num, int) and max_data_num > 0 and split_to_use == "train":
                print(f"chuanwei Limiting dataset to first {max_data_num} samples via islice", flush=True)
                dataset = itertools.islice(dataset, max_data_num)

            # 处理每个样本，收集主数据和 image_bytes
            processed_dict = {}
            image_bytes_dict = {}

            # 设置进度条描述
            if max_data_num is not None:
                desc = f"Processing dataset (limited to {max_data_num} samples)"
                total = max_data_num
            else:
                desc = "Processing dataset"
                total = None

            print(f"chuanwei Starting to process dataset items...", flush=True)
            # 直接迭代数据集（已经在上面限制了数量）
            processed_count = 0
            for item in tqdm(dataset, desc=desc, total=total):
                result = self._data_process_fn(
                    item,
                    policy_image_processor=self.policy_image_processor,
                    labeler_image_processor=self.labeler_image_processor
                )
                # 拆分主数据和 image_bytes
                for k, v in result.items():
                    if k.endswith("_image_bytes"):
                        image_bytes_dict[k.replace("_image_bytes", "")] = v
                    else:
                        processed_dict[k] = v

            # 根据 is_put_images_object_store 参数决定图片数据的存储方式
            if is_put_images_object_store:
                # 将图片数据直接放到 processed_dataset 中
                for idx, image_bytes in image_bytes_dict.items():
                    processed_dict[idx]["image_bytes"] = image_bytes
                self.processed_dataset = processed_dict
                self.pixel_values_object_ref = None  # 不使用对象存储
            else:
                # 原有逻辑：将图片数据放到 Ray 对象存储中
                self.processed_dataset = processed_dict
                # 创建 ObjectRef
                try:
                    # 为保持下游接口不变，仍沿用 pixel_values_object_ref 命名，但内容为 idx->image_bytes
                    self.pixel_values_object_ref = ray.put(image_bytes_dict)
                except Exception as e:
                    return

            # 保存所有idx的列表，便于切片
            self.idx_list = list(self.processed_dataset.keys())

    def train(self):
        # 检查是否已经有预计算好的数据
        precomputed_dir = "/root/gpufree-data/rlaif/RLAIF-V-Dataset/run-20250820_183133"
        has_precomputed_data = False
        
        if os.path.exists(precomputed_dir):
            # 检查是否有预计算好的数据文件
            batch_files = [f for f in os.listdir(precomputed_dir) if f.startswith('prefer_inferior_batch-') and f.endswith('.jsonl')]
            
            if batch_files:
                print(f"[info] Found pre-computed data in {precomputed_dir}, skipping data preprocessing...", flush=True)
                has_precomputed_data = True
                # 使用预计算数据的输出目录
                _out_dir = precomputed_dir
            else:
                print(f"[info] No batch files found in {precomputed_dir}, will perform data preprocessing...", flush=True)
        else:
            print(f"[info] Pre-computed data directory {precomputed_dir} does not exist, will perform data preprocessing...", flush=True)
        
        if not has_precomputed_data:
            IS_DUBUG = False
            if not hasattr(self, "processed_dataset"):
                # 从 args 中获取参数
                is_put_images_object_store = getattr(self.args, "is_put_images_object_store", False)
                max_data_num = getattr(self.args, "max_data_num", None)  # 获取最大样本数限制
                print(f"chuanwei Start prepare_dataset.", flush=True)
                self.prepare_dataset(
                    is_streaming_for_debug=IS_DUBUG,
                    is_put_images_object_store=is_put_images_object_store,
                    max_data_num=max_data_num
                )

            # 验证数据是否准备完成（无论是否使用对象存储）
            if not hasattr(self, "processed_dataset") or not hasattr(self, "pixel_values_object_ref"):
                return

            # 执行数据预计算逻辑
            train_dataloader = DataLoader(
                self.idx_list,
                batch_size=self.args.labeler_batch_size,
                sampler=None, # 不再使用DistributedSampler
                collate_fn=lambda x: x,
                num_workers=self.args.num_workers,
                pin_memory=True
            )

            # 输出目录：数据盘 /root/gpufree-data/rlaif/<dataset_name>/run-<ts>
            _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            _dname = os.path.basename(getattr(self.args, 'dataset_name', 'dataset')).replace('/', '_')
            _out_dir = os.path.join('/root/gpufree-data/rlaif', _dname, f'run-{_ts}')
            os.makedirs(_out_dir, exist_ok=True)

            prefered_inferior_response_list = []
            batch_vllm_engine_call(self.policy_vllm_engines, "wake_up")
            batch_vllm_engine_call(self.labeler_vllm_engines, "wake_up")
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Assess candidate responses")):
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
                        # 根据是否使用对象存储来准备数据
                        if self.pixel_values_object_ref is None:
                            # 不使用对象存储，将图片数据包含在问题数据中
                            shard_questions_with_images = []
                            for idx in shard_idx:
                                question_data = {
                                    'idx': idx,
                                    'question': self.processed_dataset[idx]['question'],
                                    'image_bytes': self.processed_dataset[idx]['image_bytes']
                                }
                                shard_questions_with_images.append(question_data)

                            refs.append(engine.generate_n_candidates.remote(
                                questions_with_images=shard_questions_with_images,
                                n_candidates=self.args.n_candidates,
                                max_new_tokens=self.args.max_new_tokens,
                                temperature=self.args.policy_generate_temperature,
                            ))
                        else:
                            # 使用对象存储，传递原有格式
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
                    # print(f"chuanwei all_candidates = {all_candidates}")

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
                    # print(f"chuanwei simple_declarative_sentences: {simple_declarative_sentences}")

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
                    # print(f"chuanwei simple_questions_result_list: {simple_questions_result_list}")

                    # YesNO step: Answer simple questions with yes or no.
                    refs = []
                    for i, engine in enumerate(self.labeler_vllm_engines):
                        shard = simple_questions_result_list[i * batch_size_each_labeler_vllm_engines : (i + 1) * batch_size_each_labeler_vllm_engines]
                        shard_item_idxes = [item["idx"] for item in shard]
                        if shard == None or len(shard) == 0:
                            continue
                        # YesNo 方法现在从 batch 数据中自动获取图片数据
                        refs.append(engine.YesNo.remote(
                            batch=shard,
                            source_idxes=shard_item_idxes,
                            max_new_tokens=3,
                            temperature=0.0,
                            do_sample=False
                        ))
                    yesno_results_list = ray.get(refs)
                    yesno_results = [ans for shard in yesno_results_list for ans in shard]
                    # print(f"chuanwei yesno_results: {yesno_results}")

                    # Combine step: Select prefered and inferior candidate responses.
                    prefer_inferior_response = LabelerRayActor.combine(yesno_results, pair_num=self.args.prefered_inferior_pair_num, seed=42)

                    # 每个 batch 立即落盘保存 prefer_inferior_response
                    try:
                        shard_path = os.path.join(_out_dir, f'prefer_inferior_batch-{batch_idx:05d}.jsonl')
                        # 若使用对象存储，则提前取回整张 idx->image_bytes 字典，避免循环内频繁 ray.get
                        image_bytes_dict_cache = None
                        if self.pixel_values_object_ref is not None:
                            try:
                                image_bytes_dict_cache = ray.get(self.pixel_values_object_ref)
                            except Exception:
                                image_bytes_dict_cache = None

                        with open(shard_path, 'w', encoding='utf-8') as f:
                            for it in (prefer_inferior_response or []):
                                # 处理图片数据：将字节数据转换为 base64 编码
                                item_to_save = it.copy()
                                if "image_bytes" in item_to_save:
                                    item_to_save["image_base64"] = base64.b64encode(item_to_save["image_bytes"]).decode('utf-8')
                                    del item_to_save["image_bytes"]  # 删除原始字节数据
                                elif image_bytes_dict_cache is not None:
                                    # 从对象存储中按 idx 取回图片字节
                                    _idx = item_to_save.get("idx")
                                    if _idx in image_bytes_dict_cache:
                                        try:
                                            item_to_save["image_base64"] = base64.b64encode(image_bytes_dict_cache[_idx]).decode('utf-8')
                                        except Exception:
                                            pass
                                f.write(json.dumps(item_to_save, ensure_ascii=False) + '\n')
                        print(f"[info] Saved prefer_inferior_response for batch {batch_idx} to {shard_path}")
                    except Exception as _e:
                        print(f"[warn] Failed to save prefer_inferior_response for batch {batch_idx}: {_e}")

                    # 累加用于后续训练/评估（如仍需要一并计算 logp）
                    # if prefer_inferior_response:
                    #     prefered_inferior_response_list.extend(prefer_inferior_response)
                except Exception as e:
                    import traceback
                    print(f"[Error] Exception occurred at batch {batch_idx}: {e}")
                    traceback.print_exc()
            
            print(f"[info] Data preprocessing completed, results saved to {_out_dir}", flush=True)
        else:
            print(f"[info] Using pre-computed data from {_out_dir}", flush=True)
        
        # print(f"chuanwei prefered_inferior_response_list = {prefered_inferior_response_list}")
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

        return []
        # reference model group
        bundles_ref = [{'CPU': 1, 'GPU':1} for _ in range(2)]
        pg_ref = placement_group(bundles_ref, strategy='PACK')
        ray.get(pg_ref.ready())
        self.ref_model_group = RayActorGroup(
            # num_nodes=self.args.num_nodes,
            num_nodes=1,
            # num_gpus_per_node=self.args.num_gpus_per_node,
            num_gpus_per_node=2,
            ray_actor_type=ReferenceModelActor,
            pg=pg_ref,
            num_gpus_per_actor=self.args.num_gpus_per_ref_actor,
            duplicate_actors=self.args.ring_attn_size * self.args.ds_tensor_parallel_size_ref,
            resources=None,
        )

        if getattr(self.args, "pretrain_policy", None) is not None:
            # 根据是否使用对象存储来传递不同的参数
            if self.pixel_values_object_ref is None:
                # 不使用对象存储，传递 processed_dataset 中的图片数据
                image_bytes_dict = {idx: self.processed_dataset[idx]["image_bytes"] for idx in self.processed_dataset.keys()}
                refs = self.ref_model_group.async_init_model_from_pretrained(
                    strategy=self.strategy,
                    pretrain=self.args.pretrain_policy,
                    pixel_values_object_ref=image_bytes_dict
                )
            else:
                # 使用对象存储
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

        # 将 logp_results 落盘到数据盘 /root/gpufree-data/rlaif/<dataset>/run-<ts>/logp 目录
        try:
            ts2 = datetime.now().strftime('%Y%m%d_%H%M%S')
            dname2 = os.path.basename(getattr(self.args, 'dataset_name', 'dataset')).replace('/', '_')
            out_dir2 = os.path.join('/root/gpufree-data/rlaif', dname2, f'run-{ts2}', 'logp')
            os.makedirs(out_dir2, exist_ok=True)

            def _sanitize(item):
                new_item = dict(item)
                v1 = new_item.get('logp_1')
                v0 = new_item.get('logp_0')
                if isinstance(v1, torch.Tensor):
                    new_item['logp_1'] = v1.item() if v1.numel() == 1 else v1.tolist()
                if isinstance(v0, torch.Tensor):
                    new_item['logp_0'] = v0.item() if v0.numel() == 1 else v0.tolist()
                return new_item

            out_path2 = os.path.join(out_dir2, 'logp_results.jsonl')
            with open(out_path2, 'w', encoding='utf-8') as f:
                for it in logp_results:
                    f.write(json.dumps(_sanitize(it), ensure_ascii=False) + '\n')
            print(f"[info] Saved logp_results to {out_path2}")
        except Exception as e:
            print(f"[warn] Failed to save logp_results: {e}")

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
            # 根据是否使用对象存储来传递不同的参数
            if self.pixel_values_object_ref is None:
                # 不使用对象存储，传递 processed_dataset 中的图片数据
                image_bytes_dict = {idx: self.processed_dataset[idx]["image_bytes"] for idx in self.processed_dataset.keys()}
                refs = self.policy_model_group.async_init_model_from_pretrained(
                    strategy=self.strategy,
                    pretrain=self.args.pretrain_policy,
                    max_steps=len(self.idx_list)//self.args.train_batch_size,
                    pixel_values_object_ref=image_bytes_dict
                )
            else:
                # 使用对象存储
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
            dataset=logp_results,
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