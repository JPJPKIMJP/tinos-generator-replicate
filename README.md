# Tinos Image Generator (Replicate)

This repository contains the Replicate model for generating images using trained LoRA models.

## Overview

This service handles image generation for the Tinos AI photo platform. It:
- Loads user-specific LoRA models
- Generates professional photos in various styles
- Supports multiple output formats and quality settings

## Model

This model is deployed on Replicate and can be called via:
```
replicate.run("jpjpkimjp/tinos-generator-replicate:latest", input={...})
```

## Deployment

### Automatic Deployment
The model is automatically deployed when pushing to the main branch.

### Manual Deployment
```bash
# Install Replicate CLI
pip install replicate

# Push model to Replicate
cog push r8.im/jpjpkimjp/tinos-generator-replicate
```

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| prompt | string | "professional headshot..." | Text prompt for generation |
| negative_prompt | string | "low quality..." | Features to avoid |
| lora_url | string | required | URL to LoRA weights |
| lora_scale | float | 0.8 | LoRA strength (0.0-2.0) |
| num_outputs | int | 4 | Number of images (1-8) |
| width | int | 1024 | Image width |
| height | int | 1024 | Image height |
| num_inference_steps | int | 30 | Denoising steps |
| guidance_scale | float | 7.5 | Prompt adherence |
| scheduler | string | "DPM++ 2M" | Sampling algorithm |
| seed | int | -1 | Random seed |
| output_format | string | "webp" | Output format |
| output_quality | int | 95 | JPEG/WebP quality |

## Supported Styles

The model is optimized for professional photo styles:
- Professional headshots
- Studio portraits
- LinkedIn profile photos
- Passport/ID photos
- Casual outdoor portraits

## Example Usage

```python
import replicate

output = replicate.run(
    "jpjpkimjp/tinos-generator-replicate:latest",
    input={
        "prompt": "professional headshot photo of tinos_user_123",
        "lora_url": "https://storage.example.com/user123_lora.safetensors",
        "num_outputs": 4,
        "style": "professional"
    }
)
```

## Local Testing

```bash
# Install Cog
pip install cog

# Run predictions locally
cog predict -i prompt="test prompt" -i lora_url="https://..."
```

## Performance

- Average generation time: 15-30 seconds per image
- Supports batch generation up to 8 images
- Optimized for SDXL with xformers