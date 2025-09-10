# Extending the Pipeline with Diffusion-Based Inpainting

This document explains how to extend the current implementation with more advanced diffusion-based inpainting for occlusion filling.

## Current Implementation

The current implementation uses traditional inpainting methods (Navier-Stokes or Telea) which are fast but not always producing the best results for complex occlusions.

## Adding Diffusion-Based Inpainting

### 1. Create a new inpainting module

Create a new file `inpainting/diffusion.py`:

```python
"""
Diffusion-based inpainting for high-quality occlusion filling.
"""

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np

class DiffusionInpainter:
    def __init__(self, model_id="runwayml/stable-diffusion-inpainting"):
        """
        Initialize diffusion inpainter.
        
        Args:
            model_id (str): HuggingFace model ID for Stable Diffusion inpainting
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        
    def inpaint(self, image, mask, prompt="", strength=0.99):
        """
        Inpaint occluded regions using diffusion model.
        
        Args:
            image (np.ndarray): Input image
            mask (np.ndarray): Binary mask where 255 indicates regions to inpaint
            prompt (str): Text prompt for inpainting guidance
            strength (float): Strength of diffusion process
            
        Returns:
            np.ndarray: Inpainted image
        """
        # Convert numpy to PIL
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)
        
        # Run diffusion inpainting
        result = self.pipe(
            prompt=prompt,
            image=image_pil,
            mask_image=mask_pil,
            strength=strength,
            num_inference_steps=50
        ).images[0]
        
        # Convert back to numpy
        result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        return result_np
```

### 2. Update the pipeline to support the new inpainter

Modify `pipeline.py` to support both basic and diffusion inpainting:

```python
# In pipeline.py __init__ method
def __init__(self, depth_profile="balanced", baseline=0.05, focal_length=1000, inpainting_method="basic"):
    # ... existing code ...
    self.inpainting_method = inpainting_method
    if inpainting_method == "diffusion":
        from inpainting.diffusion import DiffusionInpainter
        self.inpainter = DiffusionInpainter()
    else:
        self.inpainter = BasicInpainter(method="telea")
```

### 3. Requirements

Add the following to `requirements.txt`:

```
diffusers>=0.10.0
transformers>=4.21.0
accelerate>=0.12.0
xformers>=0.0.16  # For attention optimization
```

### 4. Memory Considerations

Diffusion models require significant GPU memory (16GB+ recommended). To optimize:

1. Use model offloading:
```python
self.pipe.enable_model_cpu_offload()
```

2. Use attention slicing:
```python
self.pipe.enable_attention_slicing()
```

3. Use xFormers for optimized attention:
```python
self.pipe.enable_xformers_memory_efficient_attention()
```

### 5. Performance Optimization

For real-time applications, consider:

1. Using smaller diffusion models (e.g., SDXL Turbo)
2. Implementing progressive inpainting
3. Using tiled diffusion for large images
4. Caching intermediate results

## Integration Example

Here's how to integrate diffusion inpainting in your workflow:

```python
# For high-quality results with more computational resources
pipeline = Pipeline(
    depth_profile="precision",
    inpainting_method="diffusion"
)

# Process as usual
left_path, right_path = pipeline.process_image("input.jpg")
```

## Future Enhancements

1. Implement custom diffusion models trained specifically for occlusion filling
2. Add controlnet integration for better structural preservation
3. Implement multi-scale diffusion for different occlusion sizes
4. Add temporal consistency for video applications