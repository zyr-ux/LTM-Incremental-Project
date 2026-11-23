import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import torch

try:
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, watermark, text, low resolution, cartoon, amateur"
)

VARIATION_STYLES = {
    "seasonal_autumn": "autumn theme, warm golden lighting, fallen leaves accents, cozy retail campaign background",
    "seasonal_summer": "bright sunny summer ambiance, vibrant warm daylight, outdoor resort retail aesthetic",
    "minimalist_studio": "monochrome studio podium, crisp softbox highlights, high-end commercial luxury catalogue",
    "lifestyle_urban": "modern city loft interior, contemporary architectural background, stylish marketing atmosphere"
}

class ProductVariationGenerator:
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

    def _load_pipeline(self):
        if self.pipeline is not None or self.use_mock or not DIFFUSERS_AVAILABLE:
            return
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            safety_checker=None
        )
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.to(self.device)

    def generate_variation(
        self,
        init_image: Image.Image,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        strength: float = 0.55,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        seed: Optional[int] = 42
    ) -> Image.Image:
        init_resized = init_image.convert("RGB").resize((512, 512))
        if self.use_mock or not DIFFUSERS_AVAILABLE or self.device == "cpu":
            return self._generate_synthetic_variation(init_resized, prompt, strength, seed)

        self._load_pipeline()
        if self.pipeline is None:
            return self._generate_synthetic_variation(init_resized, prompt, strength, seed)

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        result = self.pipeline(
            prompt=prompt,
            image=init_resized,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        )
        out_img = result.images[0]
        assert isinstance(out_img, Image.Image)
        return out_img

    def sweep_variation_strengths(
        self,
        init_image: Image.Image,
        prompt: str,
        strengths: Optional[List[float]] = None,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        eval_strengths = strengths or [0.3, 0.55, 0.8]
        results = []
        for s in eval_strengths:
            var_img = self.generate_variation(
                init_image=init_image,
                prompt=prompt,
                strength=s,
                seed=seed
            )
            results.append({
                "strength": s,
                "image": var_img,
                "fidelity": "High Fidelity" if s < 0.45 else ("Balanced" if s <= 0.65 else "Creative Reimagination")
            })
        return results

    def _generate_synthetic_variation(
        self,
        init_image: Image.Image,
        prompt: str,
        strength: float,
        seed: Optional[int]
    ) -> Image.Image:
        effective_seed = seed if seed is not None else 42
        rng = np.random.RandomState(effective_seed % (2**32 - 1))
        
        base_img = init_image.copy()
        
        if "autumn" in prompt.lower():
            tint = np.array([30, 10, -20])
        elif "summer" in prompt.lower():
            tint = np.array([20, 20, -10])
        elif "minimalist" in prompt.lower():
            tint = np.array([-15, -15, -15])
        else:
            tint = rng.randint(-15, 25, size=3)

        img_np = np.array(base_img, dtype=np.int16)
        clamped_strength = float(np.clip(strength, 0.1, 1.0))
        
        modified_np = img_np + (tint * clamped_strength).astype(np.int16)
        noise = rng.randint(-20, 20, size=img_np.shape).astype(np.int16)
        blended = ((1.0 - clamped_strength * 0.4) * modified_np + (clamped_strength * 0.4) * (img_np + noise))
        blended_np = np.clip(blended, 0, 255).astype(np.uint8)
        
        result = Image.fromarray(blended_np)
        if clamped_strength > 0.6:
            result = result.filter(ImageFilter.SMOOTH_MORE)
        
        draw = ImageDraw.Draw(result)
        draw.rectangle([10, 10, result.width - 10, result.height - 10], outline=(220, 220, 220), width=2)
        draw.text((20, 20), f"Variation (Strength: {strength:.2f})", fill=(245, 245, 245))
        draw.text((20, result.height - 35), f"Seed: {effective_seed} | On-Brand Creative Preview", fill=(200, 200, 200))
        return result
