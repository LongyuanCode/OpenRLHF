import base64
import io
import json
import sys
from typing import Any, Iterable, Optional

try:
    from PIL import Image
except Exception:  # Pillow 未安装时的兜底
    Image = None  # type: ignore


DATA_FILE = \
    "/root/gpufree-data/rlaif/RLAIF-V-Dataset/run-20250820_183133/prefer_inferior_batch-00000.jsonl"


def load_first_jsonl_record(file_path: str) -> Optional[dict]:
    """读取 JSONL 文件的第一条非空记录并返回为字典。"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            return json.loads(stripped)
    return None


def iter_image_base64_strings(obj: Any) -> Iterable[str]:
    """递归查找对象中名含 'image_base64' 的字段，返回其 base64 字符串。"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            lower_key = str(key).lower()
            if "image_base64" in lower_key and isinstance(value, str):
                yield value
            elif isinstance(value, (dict, list)):
                yield from iter_image_base64_strings(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_image_base64_strings(item)


def normalize_base64(data: str) -> str:
    """去除 data URL 前缀与空白字符。"""
    text = data.strip()
    if "," in text and "base64" in text[:100].lower():
        # 处理 data:image/png;base64,xxxx
        text = text.split(",", 1)[1]
    # 移除可能的换行与空白
    return "".join(text.split())


def base64_to_image_bytes(b64_text: str) -> bytes:
    normalized = normalize_base64(b64_text)
    return base64.b64decode(normalized)


def print_image_info_from_record(record: dict) -> None:
    # 优先使用显式字段
    candidates = []
    if isinstance(record.get("image_base64"), str):
        candidates.append(record["image_base64"])  # type: ignore[arg-type]

    # 递归兜底搜寻
    if not candidates:
        candidates = list(iter_image_base64_strings(record))

    if not candidates:
        print("未在首条记录中找到 image_base64 字段。可检查键名或数据结构。")
        return

    first_b64 = candidates[0]
    try:
        image_bytes = base64_to_image_bytes(first_b64)
    except Exception as exc:
        print(f"base64 解码失败: {exc}")
        return

    if Image is None:
        print("未安装 Pillow，无法解析图片。请先安装: pip install pillow")
        print(f"图片字节长度: {len(image_bytes)} bytes")
        return

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            print("首条数据的图片信息：")
            print(f"- 尺寸: {width}x{height}")
            print(f"- 模式: {img.mode}")
            print(f"- 格式: {img.format}")
    except Exception as exc:
        print(f"图片解码或读取失败: {exc}")
        print(f"图片字节长度: {len(image_bytes)} bytes")


def main() -> None:
    file_path = DATA_FILE
    if len(sys.argv) > 1 and sys.argv[1]:
        file_path = sys.argv[1]

    record = load_first_jsonl_record(file_path)
    if record is None:
        print("未从文件中读取到任何记录。")
        return

    print_image_info_from_record(record)


if __name__ == "__main__":
    main()


