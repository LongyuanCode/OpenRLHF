import ray
from ray.util.placement_group import placement_group, PlacementGroupSchedulingStrategy
from openrlhf.trainer.ray import create_vllm_engines
from ..utils import get_bundle_indices

from openrlhf.trainer.ray.launcher import RayActorGroup

from openrlhf.utils import get_strategy

from openrlhf.trainer.ray.rlaif_actor import TargetModelActor, LabelerModelActor

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
        num_nodes=1,
        num_gpus_per_node=2,
        ray_actor_type=TargetModelActor,
        pg=pg_policy,
        num_gpus_per_actor=1,                # 不用整卡 GPU
        resources=None,
    )

    # reference model group
    bundles_ref = [{"CPU": 1, "GPU_MEN": 12},
                   {"CPU": 1, "GPU_MEN": 12}]
    pg_ref = placement_group(bundles_ref, strategy="STRICT_SPREAD")
    reference_group = RayActorGroup(
        num_nodes=1,
        num_gpus_per_node=2,
        ray_actor_type=TargetModelActor,
        pg=pg_ref,
        num_gpus_per_actor=1,
        resources=None
    )

    # 打标模型分组
    bundles_labeler = [{"CPU": 1, "GPU_MEM": 36} for _ in range(3)]
    pg_labeler = placement_group(bundles_labeler, strategy="STRICT_SPREAD")
    ray.get(pg_labeler.ready())

    labeler_group = RayActorGroup(
        num_nodes=1,
        num_gpus_per_node=3,
        ray_actor_type=LabelerModelActor,
        pg=pg_labeler,
        num_gpus_per_actor=1,
        resources={"GPU_MEM": 36},
    )

    if args.vllm_num_engines is not None and args.vllm_num_engines > 0:
        max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        from openrlhf.trainer.ray.vllm_engine import LLMRayActor
        # ref: https://docs.vllm.ai/en/v0.7.2/api/offline_inference/llm.html
        labeler_vllm_engines = create_vllm_engines(
            3,  # num of labeler vllm engines, TODO: use args to specify.
            1, # tensor parallel size of labeler, TODO: use args to specify.
            args.pretrain_labeler,  # labeler name or path.
            args.seed_labeler,
            args.full_determinism_labeler,
            args.enable_prefix_caching_labeler,
            args.enforce_eager_labeler_labeler,
            max_len,
            args.gpu_memory_utilization_labeler,   # 估算为33GB，冗余配给3GB = 36/48
            args.vllm_enable_sleep_labeler,
            LLMRayActor,
            args.agent_func_path_labeler,
            shared_pg=labeler_group,
        )

        policy_vllm_engines = create_vllm_engines(
            2,
            1,
            args.pretrain_policy,
            args.seed_policy,
            args.full_determinism_policy,
            args.enable_prefix_caching_policy,
            args.enforce_eager_labeler_policy,
            max_len,
            args.gpu_memory_utilization_policy,   # 2 * (估算推理12GB、训练20GB)/48
            args.vllm_enable_sleep_policy,
            LLMRayActor,
            args.agent_func_path_policy,
            shared_pg=policy_group,
        )

        reference_vllm_engines = create_vllm_engines(
            2,
            1,
            args.pretrain_ref,
            args.seed_ref,
            args.full_determinism_ref,
            args.enable_prefix_caching_ref,
            args.enforce_eager_ref,
            max_len,
            args.gpu_memory_utilization_ref,
            args.vllm_enable_sleep_ref,
            LLMRayActor,
            args.agent_func_path_ref,
            shared_pg=reference_group
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
        reference_vllm_engines=reference_vllm_engines,
    )
    
    # Start training
    ray.get(trainer.train.remote())
    
