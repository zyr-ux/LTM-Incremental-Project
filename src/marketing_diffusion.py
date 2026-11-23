import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

try:
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, text, watermark, logo, bad proportions, "
    "low resolution, oversaturated, noisy, disfigured, cartoon, amateur"
)

BRAND_STYLE_PRESETS = {
    "studio_minimalist": "clean white studio background, high-end commercial retail photography, soft studio lighting, ultra-detailed 8k, sharp focus, elegant composition",
    "lifestyle_editorial": "vibrant lifestyle setting, natural daylight, architectural background, modern retail magazine editorial, photorealistic, premium aesthetic",
    "urban_streetwear": "moody urban street setting, cinematic natural shadows, 35mm lens photography, hyperrealistic texture, authentic retail marketing look",
    "flatlay_ecommerce": "top-down flatlay composition, aesthetic beige surface, minimal decor accessories, neat catalog styling, crisp shadows"
}

class MarketingImageGenerator:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        use_mock: bool = False
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mock = use_mock or (self.device == "cpu")
        self.pipeline = None

    def build_prompt(
        self,
        product_name: str,
        category: str,
        style_preset: str = "studio_minimalist",
        setting_context: Optional[str] = None
    ) -> str:
        preset_desc = BRAND_STYLE_PRESETS.get(style_preset, BRAND_STYLE_PRESETS["studio_minimalist"])
        setting_text = f", situated in a {setting_context}" if bool(setting_context) else ""
        return f"RetailMind promotional commercial photography of {product_name}, {category}{setting_text}, {preset_desc}"

    def _load_pipeline(self):
        if self.pipeline is not None or self.use_mock or not DIFFUSERS_AVAILABLE:
            return
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            safety_checker=None
        )
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.to(self.device)

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        seed: Optional[int] = 42,
        width: int = 512,
        height: int = 512
    ) -> Image.Image:
        if self.use_mock or not DIFFUSERS_AVAILABLE or self.device == "cpu":
            return self._generate_synthetic_preview(prompt, seed, width, height)

        self._load_pipeline()
        if self.pipeline is None:
            return self._generate_synthetic_preview(prompt, seed, width, height)
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            width=width,
            height=height
        )
        result_img = result.images[0]
        assert isinstance(result_img, Image.Image)
        return result_img

    def _generate_synthetic_preview(
        self,
        prompt: str,
        seed: Optional[int],
        width: int,
        height: int
    ) -> Image.Image:
        effective_seed = seed if seed is not None else 42
        rng = np.random.RandomState(effective_seed % (2**32 - 1))
        
        base_color = rng.randint(40, 90, size=3)
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        y_coords, x_coords = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x_coords - width / 2) ** 2 + (y_coords - height / 2) ** 2)
        max_dist = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
        gradient = (1.0 - (dist_from_center / max_dist) * 0.45)[:, :, None]
        
        for c in range(3):
            img_array[:, :, c] = np.clip(base_color[c] * gradient[:, :, 0], 20, 240).astype(np.uint8)
            
        noise = rng.randint(0, 15, size=(height, width, 3), dtype=np.uint8)
        img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)
        
        draw.rectangle([15, 15, width - 15, height - 15], outline=(200, 200, 200), width=2)
        draw.text((25, 25), "RetailMind AI — Marketing Visual", fill=(240, 240, 240))
        draw.text((25, 50), f"Seed: {effective_seed} | Synthetic Preview (CPU)", fill=(180, 180, 180))
        
        words = prompt.split()
        shortened_prompt = " ".join(words[:14]) + ("..." if len(words) > 14 else "")
        draw.text((25, height - 40), shortened_prompt, fill=(220, 220, 220))
        return img
