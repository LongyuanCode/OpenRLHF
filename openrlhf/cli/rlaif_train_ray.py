import argparse

import ray
from ray.util.placement_group import placement_group, PlacementGroupSchedulingStrategy
from openrlhf.trainer.ray import create_vllm_engines
from ..utils import get_bundle_indices

from openrlhf.trainer.ray.launcher import RayActorGroup

from openrlhf.utils import get_strategy

from openrlhf.trainer.ray.rlaif_actor import PolicyModelActor, ReferenceModelActor, LabelerModelActor

def train(args):
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "DEBUG"}})
    
    strategy = get_strategy(args)   # TODO(chuanwei.tang) 这里的deepspeed的策略参数需要定制
    strategy.print(args)

    # TODO：将对RayActorGroup的修改剥离开来，不然会影响其他功能

    # policy model group
    bundles_policy = [{"CPU": 1, "GPU_MEM": 20},     # rank0 for training and generation
                      {"CPU": 1, "GPU_MEM": 12}]     # rank1 only for generation
    pg_policy = placement_group(bundles_policy, strategy="STRICT_SPREAD")
    ray.get(pg_policy.ready())

    policy_group = RayActorGroup(
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.num_gpus_per_node_policy_group,
        ray_actor_type=PolicyModelActor,
        pg=pg_policy,
        num_gpus_per_actor=args.num_gpus_per_policy_actor,                # 不用整卡 GPU
        resources=None,
    )

    # reference model group
    bundles_ref = [{"CPU": 1, "GPU_MEN": 12},
                   {"CPU": 1, "GPU_MEN": 12}]
    pg_ref = placement_group(bundles_ref, strategy="STRICT_SPREAD")
    reference_group = RayActorGroup(
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.num_gpus_per_node_ref_group,
        ray_actor_type=ReferenceModelActor,
        pg=pg_ref,
        num_gpus_per_actor=args.num_gpus_per_ref_actor,
        resources=None
    )

    # 打标模型分组
    bundles_labeler = [{"CPU": 1, "GPU_MEM": 36} for _ in range(3)]
    pg_labeler = placement_group(bundles_labeler, strategy="STRICT_SPREAD")
    ray.get(pg_labeler.ready())

    labeler_group = RayActorGroup(
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.num_gpus_per_node_labeler_group,
        ray_actor_type=LabelerModelActor,
        pg=pg_labeler,
        num_gpus_per_actor=args.num_gpus_per_labeler_actor,
        resources={"GPU_MEM": 36},
    )

    if (args.num_labeler_vllm_engines is not None and args.num_labeler_vllm_engines > 0) and\
       (args.num_policy_vllm_engines is not None and args.num_policy_vllm_engines > 0) and\
       (args.num_ref_vllm_engines is not None and args.num_ref_vllm_engines > 0):
        max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        from openrlhf.trainer.ray.vllm_engine import LLMRayActor
        # ref: https://docs.vllm.ai/en/v0.7.2/api/offline_inference/llm.html
        labeler_vllm_engines = create_vllm_engines(
            args.num_labeler_vllm_engines,
            args.vllm_tensor_parallel_size_labeler,
            args.pretrain_labeler,
            args.seed_labeler,
            args.full_determinism_labeler,
            args.enable_prefix_caching_labeler,
            args.enforce_eager_labeler_labeler,
            max_len,
            args.gpu_memory_utilization_labeler,   # 估算为33GB，冗余配给3GB = 36/48
            args.vllm_enable_sleep_labeler,
            LLMRayActor,
            shared_pg=labeler_group,
        )

        policy_vllm_engines = create_vllm_engines(
            args.num_policy_vllm_engines,
            args.vllm_tensor_parallel_size_policy,
            args.pretrain_policy,
            args.seed_policy,
            args.full_determinism_policy,
            args.enable_prefix_caching_policy,
            args.enforce_eager_labeler_policy,
            max_len,
            args.gpu_memory_utilization_policy,   # 2 * (估算推理12GB、训练20GB)/48
            args.vllm_enable_sleep_policy,
            LLMRayActor,
            shared_pg=policy_group,
        )
    else:
        labeler_vllm_engines = None
        policy_vllm_engines = None
        reference_vllm_engines = None

    # Create RLAIF trainer
    from openrlhf.trainer.rlaif_trainer import RLAIFTrainer
    
    trainer = RLAIFTrainer.remote(
        labeler_pretrain=args.pretrain_labeler,
        target_pretrain=args.pretrain_policy,
        strategy=strategy,
        labeler_model_group=labeler_group,
        policy_model_group=policy_group,
        reference_model_group=reference_group,
        labeler_vllm_engines=labeler_vllm_engines,
        policy_vllm_engines=policy_vllm_engines,
        reference_vllm_engines=None,
    )
    
    # Start training
    ray.get(trainer.train.remote())
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Ray and vLLM
    parser.add_argument("--num_labeler_vllm_engines", type=int, default=None, help="Number of vllm engines for labeler model, setted 0 to disable vLLM.")
    parser.add_argument("--num_policy_vllm_engines", type=int, default=None, help="Number of vllm engines for policy model, setted 0 to disable vLLM.")
    parser.add_argument(
        "--vllm_tensor_parallel_size_labeler",
        type=int,
        default=1,
        help="Tensor parallel size of vLLM engine of labeler model for multi-GPU inference."
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size_policy",
        type=int,
        default=1,
        help="Tensor parallel size of vLLM engine of policy model for multi-GPU inference."
    )
    parser.add_argument("--enable_prefix_caching_labeler", action="store_true", default=False)
    parser.add_argument("--enforce_eager_labeler", action="store_true", default=False, help="Always use eager-mode PyTorch. If False, will use eager mode and CUDA graph in hybrid for maximal performance and flexibility.")
    parser.add_argument("--gpu_memory_utilization_labeler", type=float, default=35/48, help="GPU utilization of labeler model.")
    parser.add_argument("--vllm_enable_sleep_labeler", action="store_true", default=False, help="Enable sleep mode of vLLM engine of labeler model.")
    parser.add_argument("--enable_prefix_caching_policy", action="store_true", default=False)
    parser.add_argument("--enforce_eager_policy", action="store_true", default=False, help="Always use eager-mode PyTorch. If False, will use eager mode and CUDA graph in hybrid for maximal performance and flexibility.")
    parser.add_argument("--gpu_memory_utilization_policy", type=float, default=20/48, help="GPU utilization of policy model.")
    parser.add_argument("--vllm_enable_sleep_policy", action="store_true", default=False, help="Enable sleep mode of vLLM engine of policy model.")
    # Checkpoints
    parser.add_argument(
        "--use_ds_universal_ckpt", action="store_true", help="Use deepspeed universal checkpoint", default=False
    )
    # DeepSpeed
    parser.add_argument("--policy_master_actor_local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepcompile", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    # Vision RLAIF
    parser.add_argument("--labeler_load_in_8bit", action="stroe_true", default=True, help="Enable 8 bits quantinization for labeler model.")
    parser.add_argument("--labeler_load_in_4bit", action="stroe_true", default=True, help="Enable 4 bits quantinization for labeler model.")
    parser.add_argument("--labeler_bf16", action="stroe_true", default=True, help="Enable bfloat16 for labeler model.")
    parser.add_argument("--policy_load_in_4bit", action="stroe_true", default=False, help="Enable 4 bits quantinization for policy/reference model.")
    parser.add_argument("--policy_load_in_8bit", action="stroe_true", default=False, help="Enable 8 bits quantinization for policy/reference model.")
    parser.add_argument("--policy_bf16", action="stroe_true", default=True, help="Enable bfloat16 for policy/reference model.")
    parser.add_argument("--ref_load_in_4bit", action="stroe_true", default=False, help="Enable 4 bits quantinization for policy/reference model.")
    parser.add_argument("--ref_load_in_8bit", action="stroe_true", default=False, help="Enable 8 bits quantinization for policy/reference model.")
    parser.add_argument("--ref_bf16", action="stroe_true", default=True, help="Enable bfloat16 for policy/reference model.")
    parser.add_argument("--use_flash_attn_labeler", action="store_true", default=False, help="Enable FlashAttention2 of labeler model.")
    parser.add_argument("--use_flash_attn_policy", action="store_true", default=False, help="Enable FlashAttention2 of policy model.")
    parser.add_argument("--use_flash_attn_ref", action="store_true", default=False, help="Enable FlashAttention2 of reference model.")
    parser.add_argument("--pretrain_labeler", type=str, default=None, help="Huggingface model name or path of labeler model.")
    parser.add_argument("--seed_labeler", type=int, default=42)
    parser.add_argument(
        "--full_determinism_labeler",
        action="store_true",
        default=False,
        help="Enable reproducible behavior of labeler model.",
    )
    parser.add_argument("--pretrain_policy", type=str, default=None, help="Huggingface model name or path of policy model.")
    parser.add_argument("--seed_policy", type=int, default=42)
    parser.add_argument(
        "--full_determinism_policy",
        action="store_true",
        default=False,
        help="Enable reproducible behavior of policy model.",
    )
    parser.add_argument("--pretrain_ref", type=str, default=None, help="Huggingface model name or path of reference model.")
    parser.add_argument("--seed_ref", type=int, default=42)
    parser.add_argument(
        "--full_determinism_ref",
        action="store_true",
        default=False,
        help="Enable reproducible behavior of reference model.",
    )
    parser.add_argument("--n_candidates", type=int, default=5, help="Number of candidates for each question.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens for each candidate.")
    parser.add_argument("--policy_generate_temperature", type=float, default=0.7, help="Temperature for each candidate.")
    parser.add_argument("--do_sample", action="store_true", default=True, help="Do sample for each candidate.")
    parser.add_argument("--max_len", type=int, default=None, help="Max length of the input sequence.")
    parser.add_argument("--prompt_max_len", type=int, default=None, help="Max length of the prompt.")
    parser.add_argument("--generate_max_len", type=int, default=None, help="Max length of the generated sequence.")
    parser.add_argument("--train_batch_size", type=int, default=None, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs for training.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers for training data loader.")
    parser.add_argument("--num_gpus_per_node_policy_group", type=int, default=None, help="Number of GPUs per node for policy group.")
    parser.add_argument("--num_gpus_per_node_labeler_group", type=int, default=None, help="Number of GPUs per node for labeler group.")
    parser.add_argument("--num_gpus_per_node_ref_group", type=int, default=None, help="Number of GPUs per node for ref group.")
    parser.add_argument("--num_nodes", type=int, default=None, help="Number of nodes for training.")
    parser.add_argument("--num_gpus_per_policy_actor", type=int, default=None, help="Number of GPUs each policy actor.")
    parser.add_argument("--num_gpus_per_labeler_actor", type=int, default=None, help="Number of GPUs each labeler actor.")
    parser.add_argument("--num_gpus_per_ref_actor", type=int, default=None, help="Number of GPUs each ref actor.")
    parser.add_argument("--num_labeler_vllm_engines", type=int, default=None, help="Number of vllm engines for labeler model, setted 0 to disable vLLM.")
    parser.add_argument("--num_policy_vllm_engines", type=int, default=None, help="Number of vllm engines for policy model, setted 0 to disable vLLM.")
    parser.add_argument("--num_ref_vllm_engines", type=int, default=None, help="Number of vllm engines for reference model, setted 0 to disable vLLM.")
    parser.add_argument("--vllm_tensor_parallel_size_labeler", type=int, default=1, help="Tensor parallel size of vLLM engine of labeler model for multi-GPU inference.")
    parser.add_argument("--vllm_tensor_parallel_size_policy", type=int, default=1, help="Tensor parallel size of vLLM engine of policy model for multi-GPU inference.")
    parser.add_argument("--vllm_tensor_parallel_size_ref", type=int, default=1, help="Tensor parallel size of vLLM engine of reference model for multi-GPU inference.")
    parser.add_argument("--enable_prefix_caching_labeler", action="store_true", default=False, help="Enable prefix caching for labeler model.")

    args = parser.parse_args()
    train(args)