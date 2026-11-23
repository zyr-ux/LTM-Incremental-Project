import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from src.marketing_diffusion import (
    BRAND_STYLE_PRESETS,
    DEFAULT_NEGATIVE_PROMPT,
    MarketingImageGenerator,
)
from src.product_variations import (
    VARIATION_STYLES,
    ProductVariationGenerator,
)
from src.promo_video_generator import (
    build_promo_video,
    generate_product_promo_clip,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPT_LIBRARY_PATH = ROOT_DIR / "docs" / "marketing_prompt_library.json"

DEFAULT_CAMPAIGN_PROMPTS = [
    {
        "campaign_id": "summer_apparel_01",
        "product_name": "Turtle Check Navy Blue Linen Shirt",
        "category": "Apparel - Topwear",
        "style_preset": "lifestyle_editorial",
        "setting_context": "sunlit minimalist seaside resort terrace",
        "seed": 101,
        "guidance_scale": 7.5
    },
    {
        "campaign_id": "luxury_accessories_02",
        "product_name": "Titan Chronograph Silver Edition Watch",
        "category": "Accessories - Watches",
        "style_preset": "studio_minimalist",
        "setting_context": "polished dark slate pedestal with subtle rim lighting",
        "seed": 202,
        "guidance_scale": 8.0
    },
    {
        "campaign_id": "street_denim_03",
        "product_name": "Peter England Slim Fit Indigo Denim Jeans",
        "category": "Apparel - Bottomwear",
        "style_preset": "urban_streetwear",
        "setting_context": "downtown urban loft with concrete textures and soft natural sidelight",
        "seed": 303,
        "guidance_scale": 7.5
    },
    {
        "campaign_id": "activewear_launch_04",
        "product_name": "Manchester United Performance Training Track Pants",
        "category": "Apparel - Sportswear",
        "style_preset": "flatlay_ecommerce",
        "setting_context": "contemporary matte gym studio flat surface",
        "seed": 404,
        "guidance_scale": 7.0
    }
]

def load_prompt_library() -> List[Dict]:
    if PROMPT_LIBRARY_PATH.exists():
        with open(PROMPT_LIBRARY_PATH, "r") as f:
            return json.load(f)
    return DEFAULT_CAMPAIGN_PROMPTS

def save_prompt_library(prompts: List[Dict]):
    with open(PROMPT_LIBRARY_PATH, "w") as f:
        json.dump(prompts, f, indent=2)

class MarketingCreativeService:
    def __init__(
        self,
        generator: Optional[MarketingImageGenerator] = None,
        variation_generator: Optional[ProductVariationGenerator] = None
    ):
        self.generator = generator or MarketingImageGenerator()
        self.variation_generator = variation_generator or ProductVariationGenerator()
        if not PROMPT_LIBRARY_PATH.exists():
            save_prompt_library(DEFAULT_CAMPAIGN_PROMPTS)

    def create_product_variations(
        self,
        init_image: Image.Image,
        prompt: str,
        strengths: Optional[List[float]] = None,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        return self.variation_generator.sweep_variation_strengths(
            init_image=init_image,
            prompt=prompt,
            strengths=strengths,
            seed=seed
        )

    def create_promo_clip(
        self,
        base_image: Image.Image,
        variation_images: List[Image.Image],
        output_path: str = "docs/marketing_assets/promo_clip.mp4",
        fps: int = 10
    ) -> Dict[str, Any]:
        return generate_product_promo_clip(
            base_image=base_image,
            variation_images=variation_images,
            output_path=output_path,
            fps=fps
        )

    def create_product_marketing_asset(
        self,
        product_name: str,
        category: str,
        style_preset: str = "studio_minimalist",
        setting_context: Optional[str] = None,
        seed: int = 42,
        guidance_scale: float = 7.5,
        save_path: Optional[str] = None
    ) -> Dict:
        prompt = self.generator.build_prompt(
            product_name=product_name,
            category=category,
            style_preset=style_preset,
            setting_context=setting_context
        )
        image = self.generator.generate_image(
            prompt=prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            guidance_scale=guidance_scale,
            seed=seed
        )
        if bool(save_path):
            out_file = Path(save_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            image.save(out_file)

        return {
            "product_name": product_name,
            "category": category,
            "style_preset": style_preset,
            "prompt": prompt,
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "image_size": [image.width, image.height],
            "saved_path": str(save_path) if bool(save_path) else None
        }

    def generate_campaign_assets(
        self,
        output_dir: str = "docs/marketing_assets",
        max_items: int = 4
    ) -> List[Dict]:
        prompts = load_prompt_library()[:max_items]
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        results = []
        for item in prompts:
            save_file = out_path / f"{item['campaign_id']}.png"
            res = self.create_product_marketing_asset(
                product_name=item["product_name"],
                category=item["category"],
                style_preset=item.get("style_preset", "studio_minimalist"),
                setting_context=item.get("setting_context"),
                seed=item.get("seed", 42),
                guidance_scale=item.get("guidance_scale", 7.5),
                save_path=str(save_file)
            )
            results.append(res)
        return results

_default_service: Optional[MarketingCreativeService] = None

def get_marketing_service() -> MarketingCreativeService:
    global _default_service
    if _default_service is None:
        _default_service = MarketingCreativeService()
    return _default_service

def generate_marketing_image(
    product_name: str,
    category: str,
    style_preset: str = "studio_minimalist",
    setting_context: Optional[str] = None,
    seed: int = 42,
    save_path: Optional[str] = None
) -> Dict:
    service = get_marketing_service()
    return service.create_product_marketing_asset(
        product_name=product_name,
        category=category,
        style_preset=style_preset,
        setting_context=setting_context,
        seed=seed,
        save_path=save_path
    )
