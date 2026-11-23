from pathlib import Path
from src.marketing_diffusion import (
    BRAND_STYLE_PRESETS,
    DEFAULT_NEGATIVE_PROMPT,
    MarketingImageGenerator,
)
from src.marketing_service import (
    MarketingCreativeService,
    generate_marketing_image,
    load_prompt_library,
)

def test_prompt_template_builder():
    generator = MarketingImageGenerator(use_mock=True)
    prompt = generator.build_prompt(
        product_name="Classic White Sneakers",
        category="Footwear - Casual Shoes",
        style_preset="studio_minimalist",
        setting_context="wooden display podium"
    )
    assert "Classic White Sneakers" in prompt
    assert "Footwear - Casual Shoes" in prompt
    assert "wooden display podium" in prompt
    assert "clean white studio background" in prompt

def test_negative_prompt_and_presets():
    assert "blurry" in DEFAULT_NEGATIVE_PROMPT
    assert "distorted" in DEFAULT_NEGATIVE_PROMPT
    assert "studio_minimalist" in BRAND_STYLE_PRESETS
    assert "lifestyle_editorial" in BRAND_STYLE_PRESETS
    assert "urban_streetwear" in BRAND_STYLE_PRESETS
    assert "flatlay_ecommerce" in BRAND_STYLE_PRESETS

def test_seed_reproducibility():
    generator = MarketingImageGenerator(use_mock=True)
    img1 = generator.generate_image("RetailMind test product", seed=42)
    img2 = generator.generate_image("RetailMind test product", seed=42)
    img3 = generator.generate_image("RetailMind test product", seed=99)
    data1 = list(img1.getdata())  # type: ignore
    data2 = list(img2.getdata())  # type: ignore
    data3 = list(img3.getdata())  # type: ignore
    assert data1 == data2
    assert data1 != data3

def test_marketing_service_asset_generation():
    service = MarketingCreativeService()
    prompts = load_prompt_library()
    assert len(prompts) >= 4
    
    test_out = Path("docs/marketing_assets/test_asset.png")
    result = service.create_product_marketing_asset(
        product_name="Navy Blue Shirt",
        category="Apparel",
        style_preset="lifestyle_editorial",
        seed=123,
        save_path=str(test_out)
    )
    assert result["product_name"] == "Navy Blue Shirt"
    assert result["seed"] == 123
    assert test_out.exists()
    assert test_out.stat().st_size > 0
    if test_out.exists():
        test_out.unlink()
