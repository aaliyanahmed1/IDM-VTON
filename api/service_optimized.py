"""
Optimized Try-On Service - Memory efficient and fast processing
"""
import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Optional, Tuple
import concurrent.futures
from functools import lru_cache
import gc

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer,
)
from diffusers import DDPMScheduler, AutoencoderKL
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from gradio_demo.utils_mask import get_mask_location
from gradio_demo import apply_net
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

logger = logging.getLogger(__name__)


class OptimizedTryOnService:
    """Memory-efficient and fast Try-On service with optimizations"""
    
    def __init__(self):
        """Initialize models with optimizations"""
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Enable optimizations
        if self.device.startswith('cuda'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.base_path = 'yisol/IDM-VTON'
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        # Cache for processed images
        self._image_cache = {}
        self._max_cache_size = 10
        
        # Initialize models
        self._load_models_optimized()
        
    def _load_models_optimized(self):
        """Load models with memory optimizations"""
        logger.info("Loading models with optimizations...")
        
        dtype = torch.float16 if self.device.startswith('cuda') else torch.float32
        
        # Load UNet with optimizations
        self.unet = UNet2DConditionModel.from_pretrained(
            self.base_path,
            subfolder="unet",
            torch_dtype=dtype,
        )
        self.unet.requires_grad_(False)
        self.unet.eval()
        if self.device.startswith('cuda') and hasattr(torch, 'compile'):
            try:
                self.unet = torch.compile(self.unet, mode="reduce-overhead")  # PyTorch 2.0+ optimization
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed, continuing without: {e}")
        
        # Load tokenizers (lightweight, no GPU needed)
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            self.base_path,
            subfolder="tokenizer",
            use_fast=False,
        )
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            self.base_path,
            subfolder="tokenizer_2",
            use_fast=False,
        )
        
        # Load schedulers
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.base_path,
            subfolder="scheduler"
        )
        
        # Load text encoders with optimizations
        self.text_encoder_one = CLIPTextModel.from_pretrained(
            self.base_path,
            subfolder="text_encoder",
            torch_dtype=dtype,
        )
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_one.eval()
        
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            self.base_path,
            subfolder="text_encoder_2",
            torch_dtype=dtype,
        )
        self.text_encoder_two.requires_grad_(False)
        self.text_encoder_two.eval()
        
        # Load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.base_path,
            subfolder="image_encoder",
            torch_dtype=dtype,
        )
        self.image_encoder.requires_grad_(False)
        self.image_encoder.eval()
        
        # Load VAE with optimizations
        self.vae = AutoencoderKL.from_pretrained(
            self.base_path,
            subfolder="vae",
            torch_dtype=dtype,
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        # Enable VAE slicing for memory efficiency
        if hasattr(self.vae, 'enable_slicing'):
            self.vae.enable_slicing()
        if hasattr(self.vae, 'enable_tiling'):
            self.vae.enable_tiling()
        
        # Load garment encoder UNet
        self.unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            self.base_path,
            subfolder="unet_encoder",
            torch_dtype=dtype,
        )
        self.unet_encoder.requires_grad_(False)
        self.unet_encoder.eval()
        
        # Initialize pipeline
        self.pipe = TryonPipeline.from_pretrained(
            self.base_path,
            unet=self.unet,
            vae=self.vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            scheduler=self.noise_scheduler,
            image_encoder=self.image_encoder,
            torch_dtype=dtype,
        )
        self.pipe.unet_encoder = self.unet_encoder
        self.pipe.to(self.device)
        
        # Enable attention optimizations
        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing(1)  # Reduce memory usage
        if hasattr(self.pipe, 'enable_vae_slicing'):
            self.pipe.enable_vae_slicing()
        
        # Load preprocessing models (lazy loading)
        self.parsing_model = None
        self.openpose_model = None
        self._preprocessing_loaded = False
        
        # Transforms (cached)
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # Clear cache
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        logger.info("All models loaded with optimizations")
    
    def _load_preprocessing_models(self):
        """Lazy load preprocessing models only when needed"""
        if not self._preprocessing_loaded:
            logger.info("Loading preprocessing models...")
            self.parsing_model = Parsing(0)
            self.openpose_model = OpenPose(0)
            self._preprocessing_loaded = True
    
    def _optimize_image(self, img: Image.Image, max_size: Tuple[int, int] = (768, 1024)) -> Image.Image:
        """Optimize image size before processing"""
        width, height = img.size
        target_width, target_height = max_size
        
        # Calculate optimal resize
        if width > target_width or height > target_height:
            ratio = min(target_width / width, target_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        return img
    
    def _pil_to_binary_mask(self, pil_image, threshold=0):
        """Convert PIL image to binary mask (optimized)"""
        import numpy as np
        np_image = np.array(pil_image.convert("L"))
        mask = (np_image > threshold).astype(np.uint8) * 255
        return Image.fromarray(mask)
    
    async def process_tryon(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        garment_description: str = "clothing",
        denoise_steps: int = 30,
        seed: Optional[int] = None,
        auto_mask: bool = True,
        crop_image: bool = False
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Optimized processing with memory management
        """
        # Run in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._process_tryon_optimized,
            person_image,
            garment_image,
            garment_description,
            denoise_steps,
            seed,
            auto_mask,
            crop_image
        )
        return result
    
    def _process_tryon_optimized(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        garment_description: str,
        denoise_steps: int,
        seed: Optional[int],
        auto_mask: bool,
        crop_image: bool
    ) -> Tuple[Image.Image, Image.Image]:
        """Optimized synchronous processing"""
        try:
            # Load preprocessing models if needed
            self._load_preprocessing_models()
            
            # Optimize images before processing
            person_image = self._optimize_image(person_image)
            garment_image = self._optimize_image(garment_image)
            
            # Move models to device only when needed
            if not self._preprocessing_loaded:
                self.openpose_model.preprocessor.body_estimation.model.to(self.device)
            
            # Ensure pipeline is on device
            self.pipe.to(self.device)
            if hasattr(self.pipe, 'unet_encoder'):
                self.pipe.unet_encoder.to(self.device)
            
            # Prepare images
            garment_img = garment_image.convert("RGB").resize((768, 1024), Image.LANCZOS)
            human_img_orig = person_image.convert("RGB")
            
            # Crop if requested
            if crop_image:
                width, height = human_img_orig.size
                target_width = int(min(width, height * (3 / 4)))
                target_height = int(min(height, width * (4 / 3)))
                left = (width - target_width) / 2
                top = (height - target_height) / 2
                right = (width + target_width) / 2
                bottom = (height + target_height) / 2
                cropped_img = human_img_orig.crop((left, top, right, bottom))
                crop_size = cropped_img.size
                human_img = cropped_img.resize((768, 1024), Image.LANCZOS)
            else:
                human_img = human_img_orig.resize((768, 1024), Image.LANCZOS)
                crop_size = None
                left = top = None
            
            # Generate mask (optimized)
            if auto_mask:
                # Use smaller size for preprocessing
                preprocess_size = (384, 512)
                keypoints = self.openpose_model(human_img.resize(preprocess_size, Image.LANCZOS))
                model_parse, _ = self.parsing_model(human_img.resize(preprocess_size, Image.LANCZOS))
                mask, _ = get_mask_location('hd', "upper_body", model_parse, keypoints)
                mask = mask.resize((768, 1024), Image.LANCZOS)
            else:
                mask = self._pil_to_binary_mask(human_img)
            
            # Generate mask_gray
            mask_gray = (1 - transforms.ToTensor()(mask)) * self.tensor_transform(human_img)
            mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)
            
            # Get DensePose (optimized)
            human_img_arg = _apply_exif_orientation(human_img.resize((384, 512), Image.LANCZOS))
            human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
            
            args = apply_net.create_argument_parser().parse_args((
                'show',
                './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
                './ckpt/densepose/model_final_162be9.pkl',
                'dp_segm',
                '-v',
                '--opts',
                'MODEL.DEVICE',
                'cuda' if self.device.startswith('cuda') else 'cpu'
            ))
            pose_img = args.func(args, human_img_arg)
            pose_img = pose_img[:, :, ::-1]
            pose_img = Image.fromarray(pose_img).resize((768, 1024), Image.LANCZOS)
            
            # Clear cache before generation
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
            
            # Generate image with optimizations
            with torch.no_grad():
                with torch.cuda.amp.autocast() if self.device.startswith('cuda') else torch.no_grad():
                    dtype = torch.float16 if self.device.startswith('cuda') else torch.float32
                    
                    # Encode prompts (cached if same description)
                    prompt = "model is wearing " + garment_description
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    
                    with torch.inference_mode():
                        (
                            prompt_embeds,
                            negative_prompt_embeds,
                            pooled_prompt_embeds,
                            negative_pooled_prompt_embeds,
                        ) = self.pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=True,
                            negative_prompt=negative_prompt,
                        )
                        
                        prompt_c = "a photo of " + garment_description
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = self.pipe.encode_prompt(
                            prompt_c,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )
                    
                    # Prepare tensors
                    pose_img_tensor = self.tensor_transform(pose_img).unsqueeze(0).to(self.device, dtype=dtype)
                    garm_tensor = self.tensor_transform(garment_img).unsqueeze(0).to(self.device, dtype=dtype)
                    
                    generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
                    
                    # Generate with reduced steps if possible
                    effective_steps = min(denoise_steps, 30)  # Cap at 30 for speed
                    
                    # Generate
                    images = self.pipe(
                        prompt_embeds=prompt_embeds.to(self.device, dtype=dtype),
                        negative_prompt_embeds=negative_prompt_embeds.to(self.device, dtype=dtype),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(self.device, dtype=dtype),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(self.device, dtype=dtype),
                        num_inference_steps=effective_steps,
                        generator=generator,
                        strength=1.0,
                        pose_img=pose_img_tensor,
                        text_embeds_cloth=prompt_embeds_c.to(self.device, dtype=dtype),
                        cloth=garm_tensor,
                        mask_image=mask,
                        image=human_img,
                        height=1024,
                        width=768,
                        ip_adapter_image=garment_img.resize((768, 1024), Image.LANCZOS),
                        guidance_scale=2.0,
                    )[0]
            
            # Post-process result
            if crop_image and crop_size is not None:
                out_img = images[0].resize(crop_size, Image.LANCZOS)
                human_img_orig.paste(out_img, (int(left), int(top)))
                result = human_img_orig, mask_gray
            else:
                result = images[0], mask_gray
            
            # Cleanup
            del images, pose_img_tensor, garm_tensor, prompt_embeds, prompt_embeds_c
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
            gc.collect()
            
            return result
                
        except Exception as e:
            logger.error(f"Error in try-on processing: {str(e)}", exc_info=True)
            # Cleanup on error
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
            gc.collect()
            raise
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        logger.info("Service cleanup complete")

