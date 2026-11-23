# Day 15 DevLog: Diffusion — Marketing & Product Imagery

**Date**: 2026-11-19  
**Platform**: RetailMind AI  
**Increment**: 15 / 35  
**Deliverable**: Retail Creative Asset Generator & Stable Diffusion Marketing Module  
**Focus**: Latent Diffusion Models (Hugging Face `diffusers`), Brand Consistency via Style Presets, Negative Prompting, Guidance Scale Control, Reproducible Seeding, Prompt Library Operationalization  
**Stack**: `diffusers`, `torch`, `Pillow`, `numpy`, `json`  

---

## 1. Objectives & Retail Context

High-conversion retail campaigns and e-commerce product pages require rich visual assets:
1. **Design Velocity & Scalability**: Creating unique marketing visuals for thousands of catalog SKUs across social ads, seasonal campaigns, and web banners by hand is slow and cost-prohibitive.
2. **Brand & Visual Consistency**: Uncontrolled generative models produce hallucinations, deformed artifacts, or inconsistent aesthetic tones. Controlled diffusion ensures on-brand composition, specific catalog attributes, and consistent lighting.

### Day 15 Key Deliverables
- **Stable Diffusion Pipeline Architecture**: Integrate `diffusers` `StableDiffusionPipeline` pointing to `runwayml/stable-diffusion-v1-5` with DPMSolverMultistepScheduler for fast convergence.
- **Hardware-Aware Design (Non-Nvidia / CPU Resilient)**: Standard laptops without discrete Nvidia CUDA GPUs run generation via deterministic, lightweight CPU preview synthesis, with automatic GPU acceleration when CUDA is available.
- **Structured Prompt Templating**: Combine product metadata (`productDisplayName`, `masterCategory`, `subCategory`, `baseColour`) with curated retail marketing presets (`studio_minimalist`, `lifestyle_editorial`, `urban_streetwear`, `flatlay_ecommerce`).
- **Quality & Artifact Suppression**: Utilize negative prompts targeting blurry, low-resolution, warped, or cartoonish imagery, with tunable classifier-free guidance scales ($7.0 - 8.5$).
- **Seed-Controlled Prompt Library**: Curate and persist reproducible campaign recipes (`data/processed/marketing_prompt_library.json`) and store reproducible campaign image previews in `data/processed/marketing_assets/`.

---

## 2. Architecture & Design

```text
 Product Catalog Metadata (Category, Style, Setting)
                     │
                     ▼
       [Brand Prompt Templating Engine]
        - Studio Minimalist / Lifestyle Editorial / Streetwear
        - Negative Prompt (Distortion / Blurry Filters)
                     │
                     ▼
       [Marketing Image Generator Service]
       ┌─────────────┴─────────────┐
       ▼                           ▼
[CUDA GPU Device]          [CPU Hardware Fallback]
 (Diffusers SD 1.5)         (Synthetic Preview Rendering)
       │                           │
       └─────────────┬─────────────┘
                     ▼
       [Reproducible Asset Delivery]
        - Fixed Generator Seed (e.g. 101, 202)
        - data/processed/marketing_prompt_library.json
        - data/processed/marketing_assets/*.png
```

### Components
1. **Diffusion Pipeline & Generator (`src/marketing_diffusion.py`)**:
   - `MarketingImageGenerator`: Encapsulates prompt compilation, guidance scale tuning, negative prompt injection, seed control, and device detection.
2. **Creative Service & Prompt Library (`src/marketing_service.py`)**:
   - Manages prompt library loading/saving (`data/processed/marketing_prompt_library.json`).
   - Generates reproducible campaign batches (`generate_campaign_assets`).
   - Exposes clean module facade `generate_marketing_image`.
3. **Prompt Library (`data/processed/marketing_prompt_library.json`)**:
   - Standardized recipes for seasonal apparel, luxury timepieces, denim streetwear, and activewear launches.

---

## 3. Verification & Testing

Unit tests implemented in `src/test_marketing_service.py` verify:
1. `test_prompt_template_builder`: Prompt expansion containing product name, category, background setting, and style aesthetic keywords.
2. `test_negative_prompt_and_presets`: Integrity of negative suppression tokens and all 4 preset templates.
3. `test_seed_reproducibility`: Deterministic image synthesis matching identical seeds while diverging on alternate seeds.
4. `test_marketing_service_asset_generation`: End-to-end asset generation and file creation.

All tests passed successfully on CPU.
