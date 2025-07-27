import itertools
import os
import pickle
import ray
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils import safe_ray_get
from openrlhf.trainer.ray.rlaif_vision_actor import LabelerRayActor
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    default_data_collator
)
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
        strategy: DeepspeedStrategy,
        policy_model_group: RayActorGroup,
        labeler_vllm_engines=None,
        policy_vllm_engines=None,
        reference_vllm_engines=None,
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
        # 返回嵌套字典格式，图片内容单独存储
        return {
            idx: {
                "input_ids": processed["input_ids"].squeeze(0),
                "attention_mask": processed["attention_mask"].squeeze(0),
                "question": item["question"]
            },
            f"{idx}_pixel_values": {
                "pixel_values": processed["pixel_values"].squeeze(0),
                "labeler_pixel_values": labeler_processed["pixel_values"].squeeze(0)
            }
        }
    
    def prepare_dataset(self, is_streaming_for_debug=False):
        if is_streaming_for_debug:
            streaming_dataset = load_dataset(self.args.dataset_name, split="train", streaming=True)
            
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
                    pixel_values_dict[k.replace("_pixel_values", "")] = v
                else:
                    processed_dict[k] = v
            
            self.processed_dataset = processed_dict
            self.pixel_values_object_ref = ray.put(pixel_values_dict)
            
            # 保存所有idx的列表，便于切片
            self.idx_list = list(self.processed_dataset.keys())
            
        else:
            raw_dataset = load_dataset(self.args.dataset_name, split="train")
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
                        pixel_values_dict[k.replace("_pixel_values", "")] = v
                    else:
                        processed_dict[k] = v
            self.processed_dataset = processed_dict
            self.pixel_values_object_ref = ray.put(pixel_values_dict)

            # 保存所有idx的列表，便于切片
            self.idx_list = list(self.processed_dataset.keys())

    def train(self):
        IS_DUBUG = True
        if not hasattr(self, "processed_dataset"):
            self.prepare_dataset(is_streaming_for_debug=IS_DUBUG)

        train_dataloader = DataLoader(
            self.idx_list,
            batch_size=self.args.labeler_batch_size,
            sampler=None, # 不再使用DistributedSampler
            collate_fn=lambda x: x,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        prefered_inferior_response_list = []
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
                    shard_idx = item_indexes[i * batch_size_each_policy_vllm_engines : (i + 1) * batch_size_each_policy_vllm_engines]
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
                print(all_candidates)

                # Step2 extract facts from question-candidate_response pairs
                refs = []
                batch_size_each_labeler_vllm_engines = (len(batch) + len(self.labeler_vllm_engines) - 1) // len(self.labeler_vllm_engines)
                for i, engine in enumerate(self.labeler_vllm_engines):
                    shard = all_candidates[i * batch_size_each_labeler_vllm_engines : (i + 1) * batch_size_each_labeler_vllm_engines]
                    refs.append(engine.divide.remote(
                        q_candidate_a=shard,
                        max_new_tokens=self.args.max_new_tokens,
                        temperature=0.0,
                        do_sample=False
                    ))
                shard_simple_declarative_sentences_list = ray.get(refs)
                simple_declarative_sentences = [sentence for shard in shard_simple_declarative_sentences_list for sentence in shard]
                print(simple_declarative_sentences)

                # Conquer step: convert facts to simple yes/no questions
                refs = []
                for i, engine in enumerate(self.labeler_vllm_engines):
                    shard = simple_declarative_sentences[i * batch_size_each_labeler_vllm_engines : (i + 1) * batch_size_each_labeler_vllm_engines]
                    refs.append(engine.conquer.remote(
                        batch=shard,
                        max_new_tokens=self.args.max_new_tokens,
                        temperature=0.0,
                        do_sample=False
                    ))
                shard_simple_questions_result_list = ray.get(refs)
                simple_questions_result_list = [result for shard in shard_simple_questions_result_list for result in shard]
                print(simple_questions_result_list)
                # YesNO step: Answer simple questions with yes or no.
                refs = []
                for i, engine in enumerate(self.labeler_vllm_engines):
                    shard = simple_questions_result_list[i * batch_size_each_labeler_vllm_engines : (i + 1) * batch_size_each_labeler_vllm_engines]
                    shard_item_idxes = [item["idx"] for item in shard]
                    refs.append(engine.YesNo.remote(
                        batch=shard,
                        source_idxes=shard_item_idxes,
                        max_new_tokens=3,
                        temperature=0.0,
                        do_sample=False
                    ))
                yesno_results_list = ray.get(refs)
                yesno_results = [ans for shard in yesno_results_list for ans in shard]
                print(yesno_results)

                # Combine step: Select prefered and inferior candidate responses.
                prefer_inferior_response = LabelerRayActor.combine(yesno_results, 42)

                # Calculate log probabilities using reference model (no gradients needed)
                refs = []
                batch_size_each_reference_vllm_engines = (len(prefer_inferior_response) + len(self.reference_vllm_engines) - 1) // len(self.reference_vllm_engines)
                for i, engine in enumerate(self.reference_vllm_engines):
                    shard = prefer_inferior_response[i * batch_size_each_reference_vllm_engines : (i + 1) * batch_size_each_reference_vllm_engines]
                    refs.append(engine.batch_logp.remote(
                        prefered_inferior_response_list=shard,
                        requires_grad=False
                    ))
                reference_chosen_rejected_logps = list(itertools.chain.from_iterable(safe_ray_get(refs, desc="reference_model_group.batch_logp chosen and rejected")))

                prefered_inferior_response_list.extend(reference_chosen_rejected_logps)
                print(prefered_inferior_response_list)
            except Exception as e:
                import traceback
                print(f"[Error] Exception occurred at batch {batch_idx}: {e}")
                traceback.print_exc()
        
        # we don't need laber vllm engines any more.
        # engine.sleep.remote(level=1/2) can also be used but CPU memory will be occupied.
        # engine.sleep.remote(level=1/2) is used when you needwake up engines later.
        for engine in self.labeler_vllm_engines:
            engine.sleep.remote(level=2)
            
        if self.strategy.args.deepspeed_enable_sleep:
            ray.get(self.policy_model_group.async_run_method(method_name="reload_states"))

        logical_actor_num = self.args.num_nodes * self.args.num_gpus_per_node_policy_group // self.args.num_gpus_per_policy_actor
        max_steps = ((len(prefered_inferior_response_list) // logical_actor_num) // self.args.micro_train_batch_size) * self.args.num_epochs
        if getattr(self.args, "pretrain_policy", None) is not None:
            refs = self.policy_model_group.async_init_model_from_pretrained(self.strategy, self.args.pretrain_policy, max_steps)
        ray.get(refs)

        global_loss_log = []
        # for i in range(0, len(prefered_inferior_response_list), self.args.train_large_batch_size):
        #     batch_data = prefered_inferior_response_list[i:i+self.args.train_large_batch_size]
        train_history_refs = self.policy_model_group.async_run_method_batch(
            method_name="train_with_dataset",
            dataset=prefered_inferior_response_list,
            batch_size=self.args.micro_train_batch_size,
            num_epochs=self.args.num_epochs)
        train_history = ray.get(train_history_refs)
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