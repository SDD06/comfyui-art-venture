import cv2
import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw
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
        # resize_mode: str,
        mask_blur: int,
        inpaint_masked: bool,
        mask_padding: int,
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
                crop_region = get_crop_region(np_mask, mask_padding)
                crop_region = expand_crop_region(crop_region, width, height, sourcewidth, sourceheight)
                # Store individual coordinates
                left, top, right, bottom = crop_region
                lefts.append(torch.tensor(left, dtype=torch.int64))
                tops.append(torch.tensor(top, dtype=torch.int64))
                rights.append(torch.tensor(right, dtype=torch.int64))
                bottoms.append(torch.tensor(bottom, dtype=torch.int64))
                # crop mask
                overlay_mask = pil_mask
                pil_mask = resize_image(pil_mask.crop(crop_region), width, height, ResizeMode.RESIZE_TO_FIT)
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

            if crop_region is not None:
                pil_img = resize_image(pil_img.crop(crop_region), width, height, ResizeMode.RESIZE_TO_FIT)
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
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "top": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "right": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "expansion_direction": (["left", "right", "top", "bottom", "all", "custom"], {"default": "all"}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "expansion_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("expanded_image", "outpaint_mask", "new_left", "new_top", "new_right", "new_bottom")
    CATEGORY = "Art Venture/Inpainting"
    FUNCTION = "expand_canvas"

    def expand_canvas(self, image, left, top, right, bottom, 
                     expansion_direction, custom_width, custom_height, expansion_factor):
        images = []
        masks = []
        new_lefts = []
        new_tops = []
        new_rights = []
        new_bottoms = []
        
        for img in image:
            pil_img = tensor2pil(img.unsqueeze(0))
            orig_width, orig_height = pil_img.size
            
            # Original crop width and height
            crop_width = right - left
            crop_height = bottom - top
            
            # Calculate expansion sizes based on direction
            if expansion_direction == "custom":
                new_width = custom_width if custom_width > 0 else orig_width
                new_height = custom_height if custom_height > 0 else orig_height
            else:
                # Default expansion based on factor
                if expansion_direction in ["left", "right", "all"]:
                    width_expansion = int(crop_width * (expansion_factor - 1))
                    new_width = orig_width + width_expansion
                else:
                    new_width = orig_width
                    
                if expansion_direction in ["top", "bottom", "all"]:
                    height_expansion = int(crop_height * (expansion_factor - 1))
                    new_height = orig_height + height_expansion
                else:
                    new_height = orig_height
            
            # Create new image with transparency
            new_img = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
            
            # Calculate paste position
            if expansion_direction == "left":
                paste_x = new_width - orig_width
                paste_y = 0
            elif expansion_direction == "top":
                paste_x = 0
                paste_y = new_height - orig_height
            elif expansion_direction == "right":
                paste_x = 0
                paste_y = 0
            elif expansion_direction == "bottom":
                paste_x = 0
                paste_y = 0
            elif expansion_direction == "all":
                paste_x = width_expansion // 2
                paste_y = height_expansion // 2
            else:  # custom
                paste_x = (new_width - orig_width) // 2
                paste_y = (new_height - orig_height) // 2
            
            # Paste original image
            new_img.paste(pil_img.convert("RGBA"), (paste_x, paste_y))
            
            # Create mask for outpainting (255 for areas to outpaint, 0 for original image)
            mask = Image.new("L", (new_width, new_height), 255)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rectangle((paste_x, paste_y, paste_x + orig_width, paste_y + orig_height), fill=0)
            
            # Calculate new coordinates for the region to outpaint
            if expansion_direction == "left":
                new_left = 0
                new_top = top
                new_right = paste_x
                new_bottom = bottom
            elif expansion_direction == "top":
                new_left = left
                new_top = 0
                new_right = right
                new_bottom = paste_y
            elif expansion_direction == "right":
                new_left = paste_x + orig_width
                new_top = top
                new_right = new_width
                new_bottom = bottom
            elif expansion_direction == "bottom":
                new_left = left
                new_top = paste_y + orig_height
                new_right = right
                new_bottom = new_height
            elif expansion_direction == "all":
                # Adjust original crop region to new position
                new_left = left + paste_x
                new_top = top + paste_y
                new_right = right + paste_x
                new_bottom = bottom + paste_y
            else:  # custom
                new_left = left + paste_x
                new_top = top + paste_y
                new_right = right + paste_x
                new_bottom = bottom + paste_y
            
            # Convert back to tensor - MODIFY THIS PART
            # Before:
            # new_img_tensor = pil2tensor(new_img)
            
            # After - Convert RGBA to RGB before creating tensor:
            if new_img.mode == "RGBA":
                # Create white background
                bg = Image.new("RGB", new_img.size, (255, 255, 255))
                # Composite the image with alpha on the background
                bg.paste(new_img, mask=new_img.split()[3])  # Use alpha channel as mask
                new_img = bg
            
            new_img_tensor = pil2tensor(new_img)
            mask_tensor = pil2tensor(mask)
            
            images.append(new_img_tensor)
            masks.append(mask_tensor)
            new_lefts.append(torch.tensor(new_left, dtype=torch.int64))
            new_tops.append(torch.tensor(new_top, dtype=torch.int64))
            new_rights.append(torch.tensor(new_right, dtype=torch.int64))
            new_bottoms.append(torch.tensor(new_bottom, dtype=torch.int64))
        
        return (
            torch.cat(images, dim=0),
            torch.cat(masks, dim=0),
            torch.stack(new_lefts),
            torch.stack(new_tops),
            torch.stack(new_rights),
            torch.stack(new_bottoms),
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
