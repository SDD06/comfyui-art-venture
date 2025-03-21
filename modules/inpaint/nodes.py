import cv2
import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from typing import Dict

from .sam.nodes import SAMLoader, GetSAMEmbedding, SAMEmbeddingToImage
from .lama import LaMaInpaint

from ..masking import get_crop_region, expand_crop_region
from ..image_utils import ResizeMode, resize_image, flatten_image
from ..utils import numpy2pil, tensor2pil, pil2tensor


class PrepareImageAndMaskForInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_blur": ("INT", {"default": 4, "min": 0, "max": 64}),
                "inpaint_masked": ("BOOLEAN", {"default": False}),
                "mask_padding": ("INT", {"default": 32, "min": 0, "max": 256}),
                "tight_focus": ("BOOLEAN", {"default": False}),
                "focus_padding": ("INT", {"default": 16, "min": 0, "max": 128}),
                "target_size": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "width": ("INT", {"default": 0, "min": 0, "max": 2048}),
                "height": ("INT", {"default": 0, "min": 0, "max": 2048}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "CROP_REGION", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("inpaint_image", "inpaint_mask", "overlay_image", "crop_region", "left", "top", "right", "bottom")
    CATEGORY = "Art Venture/Inpainting"
    FUNCTION = "prepare"

    def prepare(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        mask_blur: int,
        inpaint_masked: bool,
        mask_padding: int,
        tight_focus: bool,
        focus_padding: int,
        target_size: int,
        width: int,
        height: int,
    ):
        if image.shape[0] != mask.shape[0]:
            raise ValueError("image and mask must have same batch size")

        if image.shape[1] != mask.shape[1] or image.shape[2] != mask.shape[2]:
            raise ValueError("image and mask must have same dimensions")

        if width == 0 and height == 0:
            height, width = image.shape[1:3]
            
        sourceheight, sourcewidth = image.shape[1:3]

        masks = []
        images = []
        overlay_masks = []
        overlay_images = []
        crop_regions = []
        lefts = []
        tops = []
        rights = []
        bottoms = []

        for img, msk in zip(image, mask):
            np_mask: np.ndarray = msk.cpu().numpy()

            if mask_blur > 0:
                kernel_size = 2 * int(2.5 * mask_blur + 0.5) + 1
                np_mask = cv2.GaussianBlur(np_mask, (kernel_size, kernel_size), mask_blur)

            pil_mask = numpy2pil(np_mask, "L")
            crop_region = None

            if inpaint_masked:
                # Get initial crop region
                crop_region = get_crop_region(np_mask, mask_padding)
                
                # Extract crop dimensions
                left, top, right, bottom = crop_region
                
                if tight_focus:
                    # Find the tightest bounding box around the mask
                    # Get mask pixels
                    mask_indices = np.where(np_mask > 0.05)
                    
                    if len(mask_indices[0]) > 0:
                        # Use actual mask pixels to determine boundaries
                        min_y, max_y = np.min(mask_indices[0]), np.max(mask_indices[0])
                        min_x, max_x = np.min(mask_indices[1]), np.max(mask_indices[1])
                        
                        # Add minimal padding
                        left = max(0, min_x - focus_padding)
                        top = max(0, min_y - focus_padding)
                        right = min(sourcewidth, max_x + focus_padding)
                        bottom = min(sourceheight, max_y + focus_padding)
                        
                        # Update crop region with tight focus
                        crop_region = (left, top, right, bottom)
                
                # Get output dimensions for resizing
                crop_width = right - left
                crop_height = bottom - top
                
                # Ensure valid crop dimensions
                if crop_width <= 0 or crop_height <= 0:
                    # Fallback to standard crop
                    crop_region = get_crop_region(np_mask, mask_padding)
                    left, top, right, bottom = crop_region
                    crop_width = right - left
                    crop_height = bottom - top
                
                # Calculate target dimensions to maintain aspect ratio
                aspect_ratio = crop_width / crop_height
                
                if aspect_ratio >= 1.0:
                    # Landscape or square
                    target_w = target_size
                    target_h = int(target_size / aspect_ratio)
                else:
                    # Portrait
                    target_h = target_size
                    target_w = int(target_size * aspect_ratio)
                
                # Store individual coordinates
                lefts.append(torch.tensor(left, dtype=torch.int64))
                tops.append(torch.tensor(top, dtype=torch.int64))
                rights.append(torch.tensor(right, dtype=torch.int64))
                bottoms.append(torch.tensor(bottom, dtype=torch.int64))
                
                # Crop mask
                overlay_mask = pil_mask
                pil_mask = resize_image(pil_mask.crop(crop_region), target_w, target_h, ResizeMode.RESIZE_TO_FIT)
                pil_mask = pil_mask.convert("L")
            else:
                np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                overlay_mask = numpy2pil(np_mask, "L")
                crop_region = (0, 0, 0, 0)
                lefts.append(torch.tensor(0, dtype=torch.int64))
                tops.append(torch.tensor(0, dtype=torch.int64))
                rights.append(torch.tensor(0, dtype=torch.int64))
                bottoms.append(torch.tensor(0, dtype=torch.int64))

            pil_img = tensor2pil(img)
            pil_img = flatten_image(pil_img)

            image_masked = Image.new("RGBa", (pil_img.width, pil_img.height))
            image_masked.paste(pil_img.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(overlay_mask))
            overlay_images.append(pil2tensor(image_masked.convert("RGBA")))
            overlay_masks.append(pil2tensor(overlay_mask))

            if crop_region is not None and crop_region != (0, 0, 0, 0):
                # When tight focus is enabled, use the natural aspect ratio of the mask
                if tight_focus:
                    # Use the dimensions we calculated earlier
                    resize_width, resize_height = target_w, target_h
                else:
                    # Use specified width/height
                    resize_width = width if width > 0 else right - left
                    resize_height = height if height > 0 else bottom - top
                
                pil_img = resize_image(pil_img.crop(crop_region), resize_width, resize_height, ResizeMode.RESIZE_TO_FIT)
            else:
                crop_region = (0, 0, 0, 0)

            images.append(pil2tensor(pil_img))
            masks.append(pil2tensor(pil_mask))
            crop_regions.append(torch.tensor(crop_region, dtype=torch.int64))

        return (
            torch.cat(images, dim=0),
            torch.cat(masks, dim=0),
            torch.cat(overlay_images, dim=0),
            torch.stack(crop_regions),
            torch.stack(lefts),
            torch.stack(tops),
            torch.stack(rights),
            torch.stack(bottoms),
        )


class OverlayInpaintedLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original": ("LATENT",),
                "inpainted": ("LATENT",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    CATEGORY = "Art Venture/Inpainting"
    FUNCTION = "overlay"

    def overlay(self, original: Dict, inpainted: Dict, mask: torch.Tensor):
        s_original: torch.Tensor = original["samples"]
        s_inpainted: torch.Tensor = inpainted["samples"]

        if s_original.shape[0] != s_inpainted.shape[0]:
            raise ValueError("original and inpainted must have same batch size")

        if s_original.shape[0] != mask.shape[0]:
            raise ValueError("original and mask must have same batch size")

        overlays = []

        for org, inp, msk in zip(s_original, s_inpainted, mask):
            latmask = tensor2pil(msk.unsqueeze(0), "L").convert("RGB").resize((org.shape[2], org.shape[1]))
            latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.around(latmask)
            latmask = np.tile(latmask[None], (4, 1, 1))

            msk = torch.asarray(1.0 - latmask)
            nmask = torch.asarray(latmask)

            overlayed = inp * nmask + org * msk
            overlays.append(overlayed)

        samples = torch.stack(overlays)
        return ({"samples": samples},)


class OverlayInpaintedImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inpainted": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "crop_region": ("CROP_REGION",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Inpainting"
    FUNCTION = "overlay"

    def overlay(self, inpainted: torch.Tensor, overlay_image: torch.Tensor, crop_region: torch.Tensor):
        if inpainted.shape[0] != overlay_image.shape[0]:
            raise ValueError("inpainted and overlay_image must have same batch size")
        if inpainted.shape[0] != crop_region.shape[0]:
            raise ValueError("inpainted and crop_region must have same batch size")

        images = []
        for image, overlay, region in zip(inpainted, overlay_image, crop_region):
            image = tensor2pil(image.unsqueeze(0))
            overlay = tensor2pil(overlay.unsqueeze(0), mode="RGBA")

            x1, y1, x2, y2 = region.tolist()
            if (x1, y1, x2, y2) == (0, 0, 0, 0):
                pass
            else:
                base_image = Image.new("RGBA", (overlay.width, overlay.height))
                image = resize_image(image, x2 - x1, y2 - y1, ResizeMode.RESIZE_TO_FILL)
                base_image.paste(image, (x1, y1))
                image = base_image

            image = image.convert("RGBA")
            image.alpha_composite(overlay)
            image = image.convert("RGB")

            images.append(pil2tensor(image))

        return (torch.cat(images, dim=0),)


class CreateExpandedCanvasForOutpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "focused_image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "top": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "right": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "focus_mode": (["only_focused", "mask_outside_focus", "mask_both"], {"default": "mask_outside_focus"}),
                "feather_edges": ("INT", {"default": 10, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_REGION")
    RETURN_NAMES = ("expanded_image", "outpaint_mask", "new_crop_region")
    CATEGORY = "Art Venture/Inpainting"
    FUNCTION = "create_outpaint_canvas"

    def create_outpaint_canvas(self, original_image, focused_image, left, top, right, bottom, focus_mode, feather_edges):
        images = []
        masks = []
        crop_regions = []
        
        for orig_img, focus_img in zip(original_image, focused_image):
            # Convert tensors to PIL images
            orig_pil = tensor2pil(orig_img.unsqueeze(0))
            focus_pil = tensor2pil(focus_img.unsqueeze(0))
            
            # Get image dimensions
            orig_width, orig_height = orig_pil.size
            focus_width, focus_height = focus_pil.size
            
            # Validate crop coordinates
            left = max(0, min(left, orig_width-1))
            top = max(0, min(top, orig_height-1))
            right = max(left+1, min(right, orig_width))
            bottom = max(top+1, min(bottom, orig_height))
            
            # Resize focused image to match crop dimensions
            crop_width = right - left
            crop_height = bottom - top
            focus_pil = resize_image(focus_pil, crop_width, crop_height, ResizeMode.RESIZE_TO_FILL)
            
            # Convert to RGBA for compositing
            orig_rgba = orig_pil.convert("RGBA")
            focus_rgba = focus_pil.convert("RGBA")
            
            # Create the output mask based on focus_mode
            if focus_mode == "only_focused":
                # Only include the focused region in the result
                # Create RGBA image with focused region
                result = Image.new("RGBA", (orig_width, orig_height), (0, 0, 0, 0))
                result.paste(focus_rgba, (left, top))
                
                # Create mask where focused area is black (0) and rest is white (255)
                mask = Image.new("L", (orig_width, orig_height), 255)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle((left, top, right, bottom), fill=0)
            
            elif focus_mode == "mask_outside_focus":
                # Include both original and focused region, but mask outside focus
                # Start with original image
                result = orig_rgba.copy()
                
                # Paste focused region on top
                result.paste(focus_rgba, (left, top))
                
                # Create mask where focused area is black (0) and rest is white (255)
                mask = Image.new("L", (orig_width, orig_height), 255)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle((left, top, right, bottom), fill=0)
            
            else:  # "mask_both"
                # Include original but prepare masks for both inside and outside focus
                # Start with original image
                result = orig_rgba.copy()
                
                # Paste focused region (for reference, will be masked anyway)
                result.paste(focus_rgba, (left, top))
                
                # Create mask where both focused and outside areas are marked for outpainting
                # This is a special mask where:
                # - Focused region is gray (128) - will be outpainted differently
                # - Rest is white (255) - will be outpainted normally
                mask = Image.new("L", (orig_width, orig_height), 255)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle((left, top, right, bottom), fill=128)
            
            # Apply feathering to the mask edges if requested
            if feather_edges > 0 and focus_mode != "mask_both":
                # Create a feathered mask by blurring
                feathered_mask = mask.copy()
                feathered_mask = feathered_mask.filter(ImageFilter.GaussianBlur(radius=feather_edges))
                
                # For "only_focused" we need to invert the feathering effect
                if focus_mode == "only_focused":
                    # Create an inverted mask for the alpha channel
                    alpha_mask = Image.new("L", (orig_width, orig_height), 0)
                    alpha_draw = ImageDraw.Draw(alpha_mask)
                    alpha_draw.rectangle((left, top, right, bottom), fill=255)
                    alpha_mask = alpha_mask.filter(ImageFilter.GaussianBlur(radius=feather_edges))
                    
                    # Apply the alpha mask to result
                    result.putalpha(alpha_mask)
                else:
                    # Normal feathering for "mask_outside_focus"
                    mask = feathered_mask
            
            # Convert back to RGB
            if result.mode == "RGBA":
                # Create white background
                bg = Image.new("RGB", result.size, (255, 255, 255))
                # Composite with alpha
                bg.paste(result, mask=result.split()[3])
                result = bg
            
            # Append results
            images.append(pil2tensor(result))
            masks.append(pil2tensor(mask))
            crop_regions.append(torch.tensor((left, top, right, bottom), dtype=torch.int64))
        
        return (
            torch.cat(images, dim=0),
            torch.cat(masks, dim=0),
            torch.stack(crop_regions),
        )


NODE_CLASS_MAPPINGS = {
    "AV_SAMLoader": SAMLoader,
    "GetSAMEmbedding": GetSAMEmbedding,
    "SAMEmbeddingToImage": SAMEmbeddingToImage,
    "LaMaInpaint": LaMaInpaint,
    "PrepareImageAndMaskForInpaint": PrepareImageAndMaskForInpaint,
    "OverlayInpaintedLatent": OverlayInpaintedLatent,
    "OverlayInpaintedImage": OverlayInpaintedImage,
    "CreateExpandedCanvasForOutpaint": CreateExpandedCanvasForOutpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AV_SAMLoader": "SAM Loader",
    "GetSAMEmbedding": "Get SAM Embedding",
    "SAMEmbeddingToImage": "SAM Embedding to Image",
    "LaMaInpaint": "LaMa Remove Object",
    "PrepareImageAndMaskForInpaint": "Prepare Image & Mask for Inpaint",
    "OverlayInpaintedLatent": "Overlay Inpainted Latent",
    "OverlayInpaintedImage": "Overlay Inpainted Image",
    "CreateExpandedCanvasForOutpaint": "Create Expanded Canvas for Outpaint",
}
