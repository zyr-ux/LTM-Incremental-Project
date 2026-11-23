from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw

def interpolate_frames(
    img_start: Image.Image,
    img_end: Image.Image,
    num_steps: int = 15
) -> List[Image.Image]:
    arr_start = np.array(img_start.convert("RGB"), dtype=np.float32)
    arr_end = np.array(img_end.convert("RGB"), dtype=np.float32)
    frames = []
    for step in range(num_steps):
        alpha = step / float(num_steps - 1)
        blended_arr = (1.0 - alpha) * arr_start + alpha * arr_end
        frames.append(Image.fromarray(blended_arr.astype(np.uint8)))
    return frames

def build_promo_video(
    keyframes: List[Image.Image],
    output_path: str,
    fps: int = 10,
    transition_frames: int = 12
) -> str:
    if not keyframes:
        raise ValueError("At least one keyframe is required to generate a video.")
    
    first_frame = keyframes[0].convert("RGB")
    width, height = first_frame.size

    all_frames: List[Image.Image] = []
    for i in range(len(keyframes) - 1):
        k_curr = keyframes[i].convert("RGB").resize((width, height))
        k_next = keyframes[i + 1].convert("RGB").resize((width, height))
        for _ in range(5):
            all_frames.append(k_curr)
        interpolated = interpolate_frames(k_curr, k_next, num_steps=transition_frames)
        all_frames.extend(interpolated)
    
    last_frame = keyframes[-1].convert("RGB").resize((width, height))
    for _ in range(6):
        all_frames.append(last_frame)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    video_writer_fourcc = getattr(cv2, "VideoWriter_fourcc")
    fourcc = video_writer_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    for frame in all_frames:
        bgr_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)

    video_writer.release()
    return str(out_file)

def generate_product_promo_clip(
    base_image: Image.Image,
    variation_images: List[Image.Image],
    output_path: str,
    fps: int = 10
) -> Dict[str, Any]:
    keyframes = [base_image] + list(variation_images)
    video_file = build_promo_video(keyframes=keyframes, output_path=output_path, fps=fps)
    return {
        "output_path": video_file,
        "total_keyframes": len(keyframes),
        "fps": fps,
        "status": "Generated"
    }
