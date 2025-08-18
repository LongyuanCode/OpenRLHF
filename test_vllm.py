import os

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode

# 1. 添加conv2d debug hook
_orig_conv2d = F.conv2d
def debug_conv2d(*args, **kwargs):
    inp, w = args[0], args[1]
    print(">>> [DEBUG conv2d] input:", inp.shape, inp.dtype, inp.device,
          "contig(NCHW):", inp.is_contiguous(),
          "contig(NHWC):", inp.is_contiguous(memory_format=torch.channels_last),
          "stride:", inp.stride())
    print(">>> [DEBUG conv2d] weight:", w.shape, w.dtype, w.device,
          "contig:", w.is_contiguous(),
          "stride:", w.stride())
    return _orig_conv2d(*args, **kwargs)
F.conv2d = debug_conv2d

# 2. 加载模型和处理器
model_name = "/root/gpufree-data/modelscope_cache/models/llava-hf/llava-1.5-7b-hf"
engine_kwargs = {
    "model": model_name,
    "enforce_eager": True,
    "tensor_parallel_size": 1,
    "seed": 42,
    "max_model_len": 1024,
    "enable_prefix_caching": False,
    "dtype": torch.float16,
    "trust_remote_code": True,
    "gpu_memory_utilization": 0.95,
    "mm_processor_kwargs": {"use_fast": True},
    "limit_mm_per_prompt": {"image": 1}
}
llm = LLM(**engine_kwargs)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# 3. 加载一条数据
# dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train", streaming=True)
# item = next(iter(dataset))
streaming_dataset = MsDataset.load(
                dataset_name="OpenBMB/RLAIF-V-Dataset",
                split="train",
                streaming=True,
                cache_dir=os.getenv('MS_CACHE_HOME'),
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
            )
item = next(iter(streaming_dataset))
print("原始数据：", item)

question = item["question"]

# 4. 获取三种图片格式
# 4.1 PIL.Image
if isinstance(item["image"], str):
    pil_image = Image.open(item["image"])
elif isinstance(item["image"], Image.Image):
    pil_image = item["image"]
else:
    # 其他格式先转PIL
    pil_image = Image.fromarray(item["image"])

# 4.2 numpy.ndarray
np_image = np.array(pil_image)

# 4.3 torch.Tensor
torch_image = torch.from_numpy(np_image).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

image_variants = [
    ("PIL.Image", pil_image),
    ("numpy.ndarray", np_image),
    ("torch.Tensor", torch_image)
]

# 5. 依次推理
for fmt, img in image_variants:
    print(f"\n====== 当前图片格式: {fmt} ======")
    # 用processor处理图片
    inputs = processor(images=img, text=question, return_tensors="pt")
    pixel_values = inputs["pixel_values"][0]  # [c, h, w]
    print("图片处理后shape:", pixel_values.shape, "dtype:", pixel_values.dtype)

    # 构造vllm输入
    multi_modal_data = {"image": pixel_values.unsqueeze(0).contiguous(memory_format=torch.channels_last)}
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    vllm_input = {
        "prompt": prompt,
        "multi_modal_data": multi_modal_data
    }

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=-1,
        max_tokens=128,
        min_tokens=1,
        skip_special_tokens=False,
    )

    print("vllm输入prompt:", vllm_input["prompt"])
    print("vllm输入图片shape:", vllm_input["multi_modal_data"]["image"].shape)
    print("vllm输入图片dtype:", vllm_input["multi_modal_data"]["image"].dtype)
    print("vllm输入图片device:", vllm_input["multi_modal_data"]["image"].device)

    # 推理
    outputs = llm.generate(vllm_input, sampling_params=sampling_params)
    print("推理输出：", outputs)