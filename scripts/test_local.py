#!/usr/bin/env python3
"""
Local testing script for the Replicate model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import Predictor
import tempfile

def test_generation():
    """Test the model locally"""
    print("Initializing model...")
    predictor = Predictor()
    predictor.setup()
    
    # Test parameters
    test_inputs = {
        "prompt": "professional headshot photo of a person, studio lighting",
        "negative_prompt": "low quality, blurry",
        "lora_url": "",  # Add a real LoRA URL for testing
        "lora_scale": 0.8,
        "num_outputs": 2,
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "scheduler": "DPM++ 2M",
        "seed": 42,
        "output_format": "png",
        "output_quality": 95
    }
    
    print("Generating images...")
    try:
        # Convert inputs to match Cog's Input class
        from types import SimpleNamespace
        inputs = SimpleNamespace(**test_inputs)
        
        # Run prediction
        outputs = predictor.predict(
            prompt=inputs.prompt,
            negative_prompt=inputs.negative_prompt,
            lora_url=inputs.lora_url,
            lora_scale=inputs.lora_scale,
            num_outputs=inputs.num_outputs,
            width=inputs.width,
            height=inputs.height,
            num_inference_steps=inputs.num_inference_steps,
            guidance_scale=inputs.guidance_scale,
            scheduler=inputs.scheduler,
            seed=inputs.seed,
            output_format=inputs.output_format,
            output_quality=inputs.output_quality
        )
        
        print(f"Generated {len(outputs)} images:")
        for i, output in enumerate(outputs):
            print(f"  - Image {i+1}: {output}")
            
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    test_generation()