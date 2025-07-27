import argparse

import ray
from ray.util.placement_group import placement_group
from openrlhf.trainer.ray import create_vllm_engines

from openrlhf.trainer.ray.launcher import RayActorGroup

from openrlhf.utils import get_strategy

from openrlhf.trainer.ray.rlaif_vision_actor import PolicyModelActor

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
)


def train(args):
    logging.info("chuanwei Here we go.")
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "DEBUG"}})
        logging.info("chuanwei Ray initialized")
    
    strategy = get_strategy(args)
    strategy.print("chuanwei args: ", args)

    if (args.num_labeler_vllm_engines is not None and args.num_labeler_vllm_engines > 0) and\
       (args.num_policy_vllm_engines is not None and args.num_policy_vllm_engines > 0):
        max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        from openrlhf.trainer.ray.vllm_engine import LLMRayActor
        # ref: https://docs.vllm.ai/en/v0.7.2/api/offline_inference/llm.html
        bundle_labeler = [{"CPU": 1, "GPU": 1},
                           {"CPU": 1, "GPU": 1}]
        pg_labeler_vllm = placement_group(bundle_labeler, strategy="PACK")
        labeler_vllm_engines = create_vllm_engines(
            num_engines=args.num_labeler_vllm_engines,
            tensor_parallel_size=args.vllm_tensor_parallel_size_labeler,
            pretrain=args.pretrain_labeler,
            seed=args.seed_labeler,
            full_determinism=args.full_determinism_labeler,
            enable_prefix_caching=args.enable_prefix_caching_labeler,
            enforce_eager=args.enforce_eager_labeler,
            max_model_len=max_len,
            gpu_memory_utilization=args.gpu_memory_utilization_labeler,
            vllm_enable_sleep=args.vllm_enable_sleep_labeler,
            llm_actor_cls=LLMRayActor,
            shared_pg=pg_labeler_vllm,
            quantization=args.labeler_quantization,
        )
        logging.info("chuanwei Ray labeler vllm engines created")

        bundles_policy = [{"CPU": 1, "GPU": 1},
                          {"CPU": 1, "GPU": 1}]
        pg_policy_vllm = placement_group(bundles_policy, strategy="PACK")
        policy_vllm_engines = create_vllm_engines(
            num_engines=args.num_policy_vllm_engines,
            tensor_parallel_size=args.vllm_tensor_parallel_size_policy,
            pretrain=args.pretrain_policy,
            seed=args.seed_policy,
            full_determinism=args.full_determinism_policy,
            enable_prefix_caching=args.enable_prefix_caching_policy,
            enforce_eager=args.enforce_eager_policy,
            max_model_len=max_len,
            gpu_memory_utilization=args.gpu_memory_utilization_policy,
            vllm_enable_sleep=args.vllm_enable_sleep_policy,
            llm_actor_cls=LLMRayActor,
            shared_pg=pg_policy_vllm,
        )
        logging.info("chuanwei Ray policy vllm engines created")

        bundles_ref  = [{"CPU": 1, "GPU": 1},
                        {"CPU": 1, "GPU": 1}]
        pg_ref_vllm = placement_group(bundles_ref, strategy="PACK")
        ref_vllm_engines = create_vllm_engines(
            num_engines=args.num_ref_vllm_engines,
            tensor_parallel_size=args.vllm_tensor_parallel_size_ref,
            pretrain=args.pretrain_ref,
            seed=args.seed_ref,
            full_determinism=args.full_determinism_ref,
            enable_prefix_caching=args.enable_prefix_caching_ref,
            enforce_eager=args.enforce_eager_ref,
            max_model_len=max_len,
            gpu_memory_utilization=args.gpu_memory_utilization_ref,
            vllm_enable_sleep=args.vllm_enable_sleep_ref,
            llm_actor_cls=LLMRayActor,
            shared_pg=pg_ref_vllm
        )
        logging.info("chuanwei Ray ref vllm engines created")
    else:
        labeler_vllm_engines = None
        policy_vllm_engines = None
        ref_vllm_engines = None

    # policy model group
    bundles_policy = [{"CPU": 1, "GPU": 2}]
    pg_policy = placement_group(bundles_policy, strategy="PACK")
    ray.get(pg_policy.ready())

    policy_group = RayActorGroup(
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.num_gpus_per_node_policy_group,
        ray_actor_type=PolicyModelActor,
        pg=pg_policy,
        num_gpus_per_actor=args.num_gpus_per_policy_actor,
        duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
        resources=None,
    )
    logging.info("chuanwei Ray actor group created")

    # Create RLAIF trainer
    from openrlhf.trainer.rlaif_vision_trainer import RLAIFTrainer
    
    trainer = RLAIFTrainer.remote(
        strategy=strategy,
        policy_model_group=policy_group,
        labeler_vllm_engines=labeler_vllm_engines,
        policy_vllm_engines=policy_vllm_engines,
        reference_vllm_engines=ref_vllm_engines,
    )
    
    # Start training
    ray.get(trainer.train.remote())
    logging.info("chuanwei Ray trainer created")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Labeler model vLLM parameters
    parser.add_argument("--num_labeler_vllm_engines", type=int, default=2, help="Number of vllm engines for labeler model, setted 0 to disable vLLM.")
    parser.add_argument("--enforce_eager_labeler", action="store_true", default=False, help="Disable CUDA graph in vllms.")
    parser.add_argument("--gpu_memory_utilization_labeler", type=float, default=0.95, help="GPU utilization of labeler model.")
    parser.add_argument("--vllm_enable_sleep_labeler", default=True, help="Enable sleep mode of vLLM engine of labeler model.")
    # Policy model vLLM parameters
    parser.add_argument("--num_policy_vllm_engines", type=int, default=2, help="Number of vllm engines for policy model, setted 0 to disable vLLM.")
    parser.add_argument("--enable_prefix_caching_policy", default=False)
    parser.add_argument("--enforce_eager_policy", action="store_true", default=False, help="Disable CUDA graph in vllms.")
    parser.add_argument("--gpu_memory_utilization_policy", type=float, default=0.95, help="GPU utilization of policy model.")
    parser.add_argument("--vllm_enable_sleep_policy", default=True, help="Enable sleep mode of vLLM engine of policy model.")
    
    # Reference model vLLM parameters
    parser.add_argument("--enable_prefix_caching_ref", default=False, help="Enable prefix caching for reference model.")
    parser.add_argument("--enforce_eager_ref", action="store_true", default=False, help="Disable CUDA graph in vllms for reference model.")
    parser.add_argument("--gpu_memory_utilization_ref", type=float, default=0.95, help="GPU utilization of reference model.")
    parser.add_argument("--vllm_enable_sleep_ref", default=True, help="Enable sleep mode of vLLM engine of reference model.")
    
    # Dataset parameters
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name or path for training.")
    
    # Checkpoints
    parser.add_argument(
        "--use_ds_universal_ckpt", action="store_true", help="Use deepspeed universal checkpoint", default=False
    )
    # DeepSpeed
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", default=False)
    parser.add_argument("--deepcompile", default=False)
    parser.add_argument("--bf16", default=False, help="Enable bfloat16 for training policy model.")
    # Vision RLAIF
    parser.add_argument("--labeler_load_in_8bit", default=False, help="Enable 8 bits quantinization for labeler model.")
    parser.add_argument("--labeler_load_in_4bit", default=False, help="Enable 4 bits quantinization for labeler model.")
    parser.add_argument("--labeler_bf16", default=False, help="Enable bfloat16 for labeler model.")
    parser.add_argument("--policy_load_in_4bit", default=False, help="Enable 4 bits quantinization for policy/reference model.")
    parser.add_argument("--policy_load_in_8bit", default=False, help="Enable 8 bits quantinization for policy/reference model.")
    parser.add_argument("--policy_bf16", default=False, help="Enable bfloat16 for policy/reference model.")
    parser.add_argument("--ref_load_in_4bit", default=False, help="Enable 4 bits quantinization for policy/reference model.")
    parser.add_argument("--ref_load_in_8bit", default=False, help="Enable 8 bits quantinization for policy/reference model.")
    parser.add_argument("--ref_bf16", default=False, help="Enable bfloat16 for policy/reference model.")
    parser.add_argument("--use_flash_attn_policy", default=False, help="Enable FlashAttention2 of policy model.")
    parser.add_argument("--pretrain_labeler", type=str, default="liuhaotian/llava-v1.5-13b", help="Huggingface model name or path of labeler model.")
    parser.add_argument("--seed_labeler", type=int, default=42)
    parser.add_argument(
        "--full_determinism_labeler",
        default=False,
        help="Enable reproducible behavior of labeler model.",
    )
    parser.add_argument("--pretrain_policy", type=str, default="liuhaotian/llava-v1.5-7b", help="Huggingface model name or path of policy model.")
    parser.add_argument("--seed_policy", type=int, default=42)
    parser.add_argument(
        "--full_determinism_policy",
        default=False,
        help="Enable reproducible behavior of policy model.",
    )
    parser.add_argument("--pretrain_ref", type=str, default="liuhaotian/llava-v1.5-7b", help="Huggingface model name or path of reference model.")
    parser.add_argument("--seed_ref", type=int, default=42)
    parser.add_argument(
        "--full_determinism_ref",
        default=False,
        help="Enable reproducible behavior of reference model.",
    )
    parser.add_argument("--n_candidates", type=int, default=5, help="Number of candidates for each question.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens for each candidate.")
    parser.add_argument("--policy_generate_temperature", type=float, default=0.7, help="Temperature for each candidate.")
    parser.add_argument("--do_sample", default=False, help="Do sample for each candidate.")
    parser.add_argument("--max_len", type=int, default=None, help="Deprecated by OpenRLHF.")
    parser.add_argument("--prompt_max_len", type=int, default=600, help="Max length of the prompt.")
    parser.add_argument("--generate_max_len", type=int, default=256, help="Max length of the generated sequence.")
    parser.add_argument("--labeler_batch_size", type=int, default=500, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for training data loader.")
    parser.add_argument("--num_gpus_per_node_policy_group", type=int, default=4, help="Number of GPUs per node for policy group.")
    parser.add_argument("--num_gpus_per_policy_actor", type=int, default=2, help="Number of GPUs each policy actor.")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes for training.")
    parser.add_argument("--num_ref_vllm_engines", type=int, default=2, help="Number of vllm engines for reference model, setted 0 to disable vLLM.")
    parser.add_argument("--vllm_tensor_parallel_size_labeler", type=int, default=1, help="Tensor parallel size of vLLM engine of labeler model for multi-GPU inference.")
    parser.add_argument("--vllm_tensor_parallel_size_policy", type=int, default=1, help="Tensor parallel size of vLLM engine of policy model for multi-GPU inference.")
    parser.add_argument("--vllm_tensor_parallel_size_ref", type=int, default=1, help="Tensor parallel size of vLLM engine of reference model for multi-GPU inference.")
    parser.add_argument("--enable_prefix_caching_labeler", default=False, help="Enable prefix caching for labeler model.")
    parser.add_argument("--freeze_vision_tower", default=False, help="Whether to freeze vision tower.")
    parser.add_argument("--vision_tower_name", type=str, default="vision_tower", help="Name of vision tower of model to be post-trained.")

    # Training parameters
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6, help="Learning rate for actor model")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="Learning rate scheduler type")
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.1, help="Learning rate warmup ratio")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=[0.9, 0.999], help="Adam optimizer betas")
    parser.add_argument("--l2", type=float, default=0.0, help="Weight decay")
    
    # Model parameters
    parser.add_argument("--policy_train_bf16", default=False, help="Enable bfloat16 for target model")
    parser.add_argument("--policy_train_load_in_4bit", default=False, help="Enable 4-bit quantization for target model")
    parser.add_argument("--policy_train_load_in_8bit", default=False, help="Enable 8-bit quantization for target model")

    # Checkpoint and save parameters
    # parser.add_argument("--save_hf_ckpt", default=False, help="Save HuggingFace checkpoint")
    parser.add_argument("--disable_ds_ckpt", default=False, help="Disable DeepSpeed checkpoint")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints", help="Checkpoint save path")
    parser.add_argument("--load_checkpoint", default=False, help="Load checkpoint")
    parser.add_argument("--max_ckpt_num", type=int, default=3, help="Maximum number of checkpoints to keep")
    parser.add_argument("--max_ckpt_mem", type=float, default=20.0, help="Maximum checkpoint memory in GB. The old checkpoint will be deleted if the total size exceeds this value.")
    parser.add_argument("--save_path", type=str, default="./trained_policy_model", help="Save path")
    
    # DeepSpeed parameters
    parser.add_argument("--deepspeed_enable_sleep", default=False, help="Enable DeepSpeed sleep mode")
    parser.add_argument("--ds_tensor_parallel_size", type=int, default=2, help="DeepSpeed tensor parallel size")
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention size")
    # DeepSpeed demands that train_batch_size = micro_train_batch_size * (world_size / ds_tensor_parallel_size / ring_attention_size) * gradient_accumulation_steps
    # ds_config["train_batch_size"] =
    #     (micro_batch × grad_acc_steps × dp) × sp × tp
    #   = micro_batch × grad_acc_steps × dp × sp × tp
    #   = micro_batch × grad_acc_steps × world_size
    # train_batch_size = 100 × gradient_accumulation_steps × dp  =  100 × 1 × 2  = 200
    # dp_config["train_batch_size"] = micro_train_batch_size_per_gpu (100) × grad_acc (1) × world_size (4)
    parser.add_argument("--train_batch_size", type=int, default=200, help="Batch size of data parallel group.")
    parser.add_argument("--micro_train_batch_size", type=int, default=100, help="DeepSpeed train micro batch size per GPU.")
    
    # vLLM sync parameters
    parser.add_argument("--vllm_sync_backend", type=str, default="nccl", help="vLLM sync backend")
    parser.add_argument("--vllm_sync_with_ray", default=False, help="Use Ray for vLLM sync")
    parser.add_argument("--colocate_all_models", default=False, help="Colocate all models")

    # wandb
    parser.add_argument("--wandb_enable", default=False, help="Enable wandb")
    parser.add_argument("--wandb_project", type=str, default="rlaif-vision", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default="rlaif-vision", help="Wandb run name")
    parser.add_argument("--wandb_entity", type=str, default="openrlhf", help="Wandb entity name")
    parser.add_argument("--wandb_id", type=str, default=None, help="Wandb run id")
    parser.add_argument("--wandb_group", type=str, default=None, help="Wandb group name")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Wandb tags")
    parser.add_argument("--wandb_api_key", type=str, default=None, help="Wandb api key")

    # 8bit quantization
    # --pretrain_labeler "TheBloke/llava-v1.5-13B-GPTQ" \
    # --labeler_quantization "gptq" \
    parser.add_argument("--labeler_quantization", type=str, default=None, help="Quantization type for labeler model")

    args = parser.parse_args()
    train(args)