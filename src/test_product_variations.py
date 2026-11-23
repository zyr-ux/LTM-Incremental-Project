from pathlib import Path
import numpy as np
from PIL import Image
from src.product_variations import (
    VARIATION_STYLES,
    ProductVariationGenerator,
)
from src.promo_video_generator import (
    build_promo_video,
    generate_product_promo_clip,
    interpolate_frames,
)
from src.marketing_service import get_marketing_service

def test_product_variation_generator_synthetic():
    generator = ProductVariationGenerator(use_mock=True)
    init_img = Image.new("RGB", (60, 60), color=(100, 150, 200))
    
    var_img = generator.generate_variation(
        init_image=init_img,
        prompt="autumn theme for retail shirt",
        strength=0.5,
        seed=42
    )
    assert var_img.size == (512, 512)
    assert isinstance(var_img, Image.Image)

def test_sweep_variation_strengths():
    generator = ProductVariationGenerator(use_mock=True)
    init_img = Image.new("RGB", (60, 60), color=(120, 120, 120))
    
    results = generator.sweep_variation_strengths(
        init_image=init_img,
        prompt="summer resort style",
        strengths=[0.3, 0.6, 0.85],
        seed=101
    )
    assert len(results) == 3
    assert results[0]["fidelity"] == "High Fidelity"
    assert results[1]["fidelity"] == "Balanced"
    assert results[2]["fidelity"] == "Creative Reimagination"

def test_frame_interpolation():
    img_a = Image.new("RGB", (100, 100), color=(0, 0, 0))
    img_b = Image.new("RGB", (100, 100), color=(200, 200, 200))
    
    frames = interpolate_frames(img_a, img_b, num_steps=5)
    assert len(frames) == 5
    mid_arr = np.array(frames[2])
    assert 90 <= mid_arr[50, 50, 0] <= 110

def test_promo_video_generation():
    test_out = Path("docs/marketing_assets/test_promo.mp4")
    f1 = Image.new("RGB", (64, 64), color=(255, 0, 0))
    f2 = Image.new("RGB", (64, 64), color=(0, 255, 0))
    
    res = generate_product_promo_clip(
        base_image=f1,
        variation_images=[f2],
        output_path=str(test_out),
        fps=10
    )
    assert res["status"] == "Generated"
    assert test_out.exists()
    assert test_out.stat().st_size > 0
    if test_out.exists():
        test_out.unlink()

def test_marketing_service_variations_and_video():
    service = get_marketing_service()
    init_img = Image.new("RGB", (64, 64), color=(50, 100, 150))
    
    variations = service.create_product_variations(
        init_image=init_img,
        prompt="lifestyle urban collection",
        strengths=[0.4, 0.7]
    )
    assert len(variations) == 2
    
    video_out = Path("docs/marketing_assets/test_service_clip.mp4")
    clip_res = service.create_promo_clip(
        base_image=init_img,
        variation_images=[variations[0]["image"], variations[1]["image"]],
        output_path=str(video_out),
        fps=10
    )
    assert clip_res["status"] == "Generated"
    assert video_out.exists()
    if video_out.exists():
        video_out.unlink()
