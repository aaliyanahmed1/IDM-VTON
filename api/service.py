"""
Try-On Service - Wraps IDM-VTON inference logic
"""
import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Optional, Tuple
import concurrent.futures

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

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


class TryOnService:
    """Production-ready Try-On service"""
    
    def __init__(self):
        """Initialize models and pipeline"""
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        self.base_path = 'yisol/IDM-VTON'
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        # Initialize models
        self._load_models()
        
    def _load_models(self):
        """Load all required models"""
        logger.info("Loading models...")
        
        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.base_path,
            subfolder="unet",
            torch_dtype=torch.float16 if self.device.startswith('cuda') else torch.float32,
        )
        self.unet.requires_grad_(False)
        
        # Load tokenizers
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
        
        # Load text encoders
        self.text_encoder_one = CLIPTextModel.from_pretrained(
            self.base_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16 if self.device.startswith('cuda') else torch.float32,
        )
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            self.base_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16 if self.device.startswith('cuda') else torch.float32,
        )
        
        # Load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.base_path,
            subfolder="image_encoder",
            torch_dtype=torch.float16 if self.device.startswith('cuda') else torch.float32,
        )
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.base_path,
            subfolder="vae",
            torch_dtype=torch.float16 if self.device.startswith('cuda') else torch.float32,
        )
        
        # Load garment encoder UNet
        self.unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            self.base_path,
            subfolder="unet_encoder",
            torch_dtype=torch.float16 if self.device.startswith('cuda') else torch.float32,
        )
        self.unet_encoder.requires_grad_(False)
        
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
            torch_dtype=torch.float16 if self.device.startswith('cuda') else torch.float32,
        )
        self.pipe.unet_encoder = self.unet_encoder
        self.pipe.to(self.device)
        
        # Load preprocessing models
        self.parsing_model = Parsing(0)
        self.openpose_model = OpenPose(0)
        
        # Transforms
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        logger.info("All models loaded successfully")
    
    def _pil_to_binary_mask(self, pil_image, threshold=0):
        """Convert PIL image to binary mask"""
        import numpy as np
        np_image = np.array(pil_image)
        grayscale_image = Image.fromarray(np_image).convert("L")
        binary_mask = np.array(grayscale_image) > threshold
        mask = np.zeros(binary_mask.shape, dtype=np.uint8)
        for i in range(binary_mask.shape[0]):
            for j in range(binary_mask.shape[1]):
                if binary_mask[i, j] == True:
                    mask[i, j] = 1
        mask = (mask * 255).astype(np.uint8)
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
        Process virtual try-on
        
        Args:
            person_image: PIL Image of person
            garment_image: PIL Image of garment
            garment_description: Text description of garment
            denoise_steps: Number of diffusion steps
            seed: Random seed
            auto_mask: Use automatic masking
            crop_image: Crop to optimal aspect ratio
            
        Returns:
            Tuple of (result_image, mask_image)
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._process_tryon_sync,
            person_image,
            garment_image,
            garment_description,
            denoise_steps,
            seed,
            auto_mask,
            crop_image
        )
        return result
    
    def _process_tryon_sync(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        garment_description: str,
        denoise_steps: int,
        seed: Optional[int],
        auto_mask: bool,
        crop_image: bool
    ) -> Tuple[Image.Image, Image.Image]:
        """Synchronous processing (runs in thread pool)"""
        try:
            # Move models to device
            self.openpose_model.preprocessor.body_estimation.model.to(self.device)
            self.pipe.to(self.device)
            self.pipe.unet_encoder.to(self.device)
            
            # Prepare images
            garment_img = garment_image.convert("RGB").resize((768, 1024))
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
                human_img = cropped_img.resize((768, 1024))
            else:
                human_img = human_img_orig.resize((768, 1024))
                crop_size = None
                left = top = None
            
            # Generate mask
            if auto_mask:
                keypoints = self.openpose_model(human_img.resize((384, 512)))
                model_parse, _ = self.parsing_model(human_img.resize((384, 512)))
                mask, _ = get_mask_location('hd', "upper_body", model_parse, keypoints)
                mask = mask.resize((768, 1024))
            else:
                # Use simple binary mask (you can enhance this)
                mask = self._pil_to_binary_mask(human_img)
            
            # Generate mask_gray
            mask_gray = (1 - transforms.ToTensor()(mask)) * self.tensor_transform(human_img)
            mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)
            
            # Get DensePose
            human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
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
            pose_img = Image.fromarray(pose_img).resize((768, 1024))
            
            # Generate image
            with torch.no_grad():
                with torch.cuda.amp.autocast() if self.device.startswith('cuda') else torch.no_grad():
                    # Encode prompts
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
                    pose_img_tensor = self.tensor_transform(pose_img).unsqueeze(0).to(
                        self.device, torch.float16 if self.device.startswith('cuda') else torch.float32
                    )
                    garm_tensor = self.tensor_transform(garment_img).unsqueeze(0).to(
                        self.device, torch.float16 if self.device.startswith('cuda') else torch.float32
                    )
                    
                    generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
                    
                    # Generate
                    images = self.pipe(
                        prompt_embeds=prompt_embeds.to(
                            self.device, torch.float16 if self.device.startswith('cuda') else torch.float32
                        ),
                        negative_prompt_embeds=negative_prompt_embeds.to(
                            self.device, torch.float16 if self.device.startswith('cuda') else torch.float32
                        ),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(
                            self.device, torch.float16 if self.device.startswith('cuda') else torch.float32
                        ),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(
                            self.device, torch.float16 if self.device.startswith('cuda') else torch.float32
                        ),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength=1.0,
                        pose_img=pose_img_tensor,
                        text_embeds_cloth=prompt_embeds_c.to(
                            self.device, torch.float16 if self.device.startswith('cuda') else torch.float32
                        ),
                        cloth=garm_tensor,
                        mask_image=mask,
                        image=human_img,
                        height=1024,
                        width=768,
                        ip_adapter_image=garment_img.resize((768, 1024)),
                        guidance_scale=2.0,
                    )[0]
            
            # Post-process result
            if crop_image and crop_size is not None:
                out_img = images[0].resize(crop_size)
                human_img_orig.paste(out_img, (int(left), int(top)))
                return human_img_orig, mask_gray
            else:
                return images[0], mask_gray
                
        except Exception as e:
            logger.error(f"Error in try-on processing: {str(e)}", exc_info=True)
            raise
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        logger.info("Service cleanup complete")

