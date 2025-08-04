import os
import time
from typing import List, Optional
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DDIMScheduler
)
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from safetensors.torch import load_file
import requests
import tempfile

class Predictor(BasePredictor):
    """Replicate predictor for generating images with LoRA models"""
    
    def setup(self):
        """Load the base model into memory"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load base SDXL model
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(self.device)
        
        # Enable memory efficient attention
        self.pipe.enable_xformers_memory_efficient_attention()
        
        # Compile for faster inference (requires PyTorch 2.0+)
        # self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        
        # Store available schedulers
        self.schedulers = {
            "DPM++ 2M": DPMSolverMultistepScheduler,
            "Euler A": EulerAncestralDiscreteScheduler,
            "Euler": EulerDiscreteScheduler,
            "DDIM": DDIMScheduler
        }
    
    def download_lora(self, lora_url: str) -> str:
        """Download LoRA weights from URL"""
        response = requests.get(lora_url, stream=True)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            return tmp_file.name
    
    def load_lora(self, lora_path: str, scale: float = 1.0):
        """Load LoRA weights into the pipeline"""
        # Load LoRA weights
        state_dict = load_file(lora_path)
        
        # Clear any existing LoRA
        self.pipe.unload_lora_weights()
        
        # Load new LoRA weights
        self.pipe.load_lora_weights(state_dict)
        self.pipe.fuse_lora(lora_scale=scale)
    
    def predict(
        self,
        prompt: str = Input(
            description="Text prompt for image generation",
            default="professional headshot photo of a person"
        ),
        negative_prompt: str = Input(
            description="Negative prompt to avoid certain features",
            default="low quality, blurry, distorted, ugly, bad anatomy, wrong proportions, extra limbs, cloned face, mutated hands, poor quality, low resolution, bad hands, bad face"
        ),
        lora_url: str = Input(
            description="URL to the LoRA model weights (.safetensors file)"
        ),
        lora_scale: float = Input(
            description="LoRA strength (0.0 to 2.0)",
            default=0.8,
            ge=0.0,
            le=2.0
        ),
        num_outputs: int = Input(
            description="Number of images to generate",
            default=4,
            ge=1,
            le=8
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
            choices=[512, 768, 1024, 1280, 1536]
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
            choices=[512, 768, 1024, 1280, 1536]
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            default=30,
            ge=1,
            le=100
        ),
        guidance_scale: float = Input(
            description="Guidance scale for text prompt adherence",
            default=7.5,
            ge=1.0,
            le=20.0
        ),
        scheduler: str = Input(
            description="Scheduler algorithm",
            default="DPM++ 2M",
            choices=["DPM++ 2M", "Euler A", "Euler", "DDIM"]
        ),
        seed: int = Input(
            description="Random seed (-1 for random)",
            default=-1,
            ge=-1,
            le=2147483647
        ),
        output_format: str = Input(
            description="Output image format",
            default="webp",
            choices=["webp", "png", "jpeg"]
        ),
        output_quality: int = Input(
            description="Output image quality (for JPEG/WebP)",
            default=95,
            ge=1,
            le=100
        )
    ) -> List[Path]:
        """Generate images using the loaded model and LoRA"""
        
        # Download and load LoRA if provided
        if lora_url:
            print(f"Downloading LoRA from {lora_url}")
            lora_path = self.download_lora(lora_url)
            print(f"Loading LoRA weights with scale {lora_scale}")
            self.load_lora(lora_path, lora_scale)
            
            # Clean up temporary file
            try:
                os.unlink(lora_path)
            except:
                pass
        
        # Set scheduler
        scheduler_class = self.schedulers[scheduler]
        self.pipe.scheduler = scheduler_class.from_config(self.pipe.scheduler.config)
        
        # Set seed
        if seed == -1:
            seed = int(time.time() * 1000) % 2147483647
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate images
        print(f"Generating {num_outputs} images...")
        output_paths = []
        
        for i in range(num_outputs):
            # Generate with different seeds for variety
            if i > 0:
                generator = torch.Generator(device=self.device).manual_seed(seed + i)
            
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            
            # Save image
            output_path = f"/tmp/output_{i}.{output_format}"
            save_kwargs = {}
            
            if output_format in ["webp", "jpeg"]:
                save_kwargs["quality"] = output_quality
                if output_format == "webp":
                    save_kwargs["lossless"] = False
            
            image.save(output_path, **save_kwargs)
            output_paths.append(Path(output_path))
            
            print(f"Generated image {i+1}/{num_outputs}")
        
        return output_paths