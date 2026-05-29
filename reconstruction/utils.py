import torch
import os 
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from util import box_ops
import torch.nn.functional as F
from torch import nn
import torch
import cv2
import wandb
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from PIL import ImageDraw, ImageFont


def checkpoint_weights(checkpoint):
    """Return state dict from a checkpoint (dict with 'weights' or raw state dict)."""
    if isinstance(checkpoint, dict) and "weights" in checkpoint:
        return checkpoint["weights"]
    return checkpoint


def sprite_size_from_weights(weights):
    """Infer (H, W) sprite size from generator.proto in a state dict."""
    proto = weights.get("generator.proto")
    if proto is None:
        raise ValueError("Checkpoint has no generator.proto")
    _, _, height, width = proto.shape
    return int(height), int(width)


def sprite_size_from_checkpoint_path(path, map_location="cpu"):
    """Load checkpoint from path and return (H, W) sprite size."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    return sprite_size_from_weights(checkpoint_weights(checkpoint))


def add_character_labels(grid_img, charset):
    """
    Add character labels to the grid image.
    Args:
        grid_img: PIL Image of the grid
        charset: List of characters (excluding empty character)
    """
    # Convert to PIL Image if needed
    if not isinstance(grid_img, Image.Image):
        grid_img = Image.fromarray(grid_img)
    
    # Create a copy to draw on
    labeled_img = grid_img.copy()
    draw = ImageDraw.Draw(labeled_img)
    
    # Try to use a default font, fallback to default if not available
    try:
        font = ImageFont.truetype("/home/vlachoum/learnable-DTLR/Junicode.ttf", 12)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
    
    # Calculate grid dimensions
    img_width, img_height = grid_img.size
    num_chars = len(charset)
    
    # Estimate grid layout (assuming square-ish grid)
    grid_size = int(np.ceil(np.sqrt(num_chars)))
    
    # Calculate sprite size
    sprite_width = img_width // grid_size
    sprite_height = img_height // grid_size
    
    # Add labels for each character
    for i, char in enumerate(charset):
        if i >= num_chars:
            break
            
        # Calculate position in grid
        row = i // grid_size
        col = i % grid_size
        
        # Calculate text position (above the sprite, centered horizontally)
        x = col * sprite_width + sprite_width // 2
        y = row * sprite_height + 5  # Position above the sprite
        
        # Get text size for centering
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Adjust position to center text horizontally
        x -= text_width // 2
        
        # Draw text with black outline for visibility
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), char, font=font, fill=(0, 0, 0))
        
        # Draw main text
        draw.text((x, y), char, font=font, fill=(255, 255, 255))
    
    return labeled_img

def transform_sprite(sprite_tensor, i, sprite_bbox_stats=None):
    """
    Transform a sprite tensor using bbox statistics.
    Args:
        sprite_tensor: Tensor of sprite (C, H, W) in range [0, 1]
        i: Index in the loop (0-indexed)
        sprite_bbox_stats: Dictionary mapping sprite indices to bbox statistics
    Returns:
        PIL Image of the transformed sprite
    """
    # Invert the colors: 1 - sprite
    sprite_inverted = sprite_tensor.clamp(0, 1)
    
    # Get sprite dimensions
    sprite_h, sprite_w = sprite_inverted.shape[1], sprite_inverted.shape[2]
    
    # Always use sprite dimensions
    output_h = sprite_h
    output_w = sprite_w 
    
    # Get bbox statistics for this sprite if available
    # Exact same logic as debug_sprite_grid.py: if i>0:
    if i > 0 and sprite_bbox_stats is not None and i in sprite_bbox_stats:
        stats = sprite_bbox_stats[i]
        mean_width = stats.get('mean_width', 0.0)
        mean_height = stats.get('mean_height', 0.0)
    else:
        mean_width = 0
        mean_height = 0
    
    # Apply aspect-ratio deformation only when bbox stats are available.
    if mean_width <= 0 or mean_height <= 0:
        sprite_out = 1.0 - sprite_tensor.clamp(0, 1)
        return ToPILImage()(sprite_out.cpu())

    max_size = max(mean_width, mean_height)
    scale_x = max_size / mean_width
    scale_y = max_size / mean_height
    tx, ty = 0, 0

    device = sprite_inverted.device
    theta = torch.tensor([[scale_x, 0.0,tx],
                         [0.0, scale_y, ty]], dtype=torch.float32, device=device)
    
    # Prepare sprite tensor: (C, H, W) -> (1, C, H, W) for grid_sample
    sprite_tensor_batch = sprite_inverted.unsqueeze(0)  # (1, C, H, W)
    
    # Create affine grid with specified size (or default sprite_h, sprite_w+2)
    grid = F.affine_grid(theta.unsqueeze(0), 
                        size=(1, sprite_tensor_batch.shape[1], output_h, output_w),
                        align_corners=True)
    
    # Apply transformation
    sprite_transformed = F.grid_sample(sprite_tensor_batch, grid, 
                                      mode='bilinear', 
                                      padding_mode='zeros',
                                      align_corners=True)
    
    # Remove batch dimension, clamp values, and convert to PIL
    sprite_transformed = 1-sprite_transformed.squeeze(0)  # (C, H, W)
    sprite_transformed = torch.clamp(sprite_transformed, 0.0, 1.0)  # Ensure values are in [0, 1]
    sprite_img = ToPILImage()(sprite_transformed.cpu())
    #save sprite_img
    return sprite_img

def add_padding_to_sprite(sprite_img, padding):
    """
    Add padding to a PIL sprite image.
    Args:
        sprite_img: PIL Image of the sprite
        padding: List of [left, right, top, bottom] padding values
    Returns:
        PIL Image with padding added
    """
    if sum(padding) == 0:
        return sprite_img
    
    left, right, top, bottom = padding
    width, height = sprite_img.size
    
    # Create new image with padding (white background)
    new_width = width + left + right
    new_height = height + top + bottom
    padded_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    
    # Paste original sprite in the center (with padding offsets)
    padded_img.paste(sprite_img, (left, top))
    
    return padded_img

def add_padding_to_sprites(sprites_pil, padding):
    """
    Add padding to a list of PIL sprite images.
    Args:
        sprites_pil: List of PIL Images
        padding: List of [left, right, top, bottom] padding values
    Returns:
        List of PIL Images with padding added
    """
    if sum(padding) == 0:
        return sprites_pil
    
    return [add_padding_to_sprite(sprite, padding) for sprite in sprites_pil]

def create_manual_grid(proto_pixel, charset, num_sprites_per_letter=1, sprite_bbox_stats=None, padding=None):
    """
    Create a manual grid with character names in first row and sprites in second row.
    For multiple sprites per character, create separate grids side by side.
    Args:
        proto_pixel: Tensor of sprites (N, C, H, W)
        charset: List of characters (excluding empty character)
        num_sprites_per_letter: Number of sprites per character
        sprite_bbox_stats: Dictionary mapping sprite indices (1-indexed) to bbox statistics
                          with keys: mean_width, mean_height, var_width, var_height, count
        padding: Optional list of [left, right, top, bottom] padding values
    """
    # Convert sprites to PIL images with size increase and color inversion
    # Apply aspect ratio transformation based on bbox statistics using affine transform
    sprites = []
    
    # Get actual sprite dimensions from the tensor (H, W)
    sprite_h, sprite_w = proto_pixel.shape[2], proto_pixel.shape[3]
    
    for i in range(proto_pixel.shape[0]):
        sprite = proto_pixel[i]
        
        # Transform sprite using the utility function (exact same logic as debug_sprite_grid.py)
        sprite_img = transform_sprite(sprite, i, sprite_bbox_stats)
        
        sprites.append(sprite_img)
    
    # Add padding to sprites if specified
    if padding is not None and len(padding) == 4 and sum(padding) > 0:
        sprites = add_padding_to_sprites(sprites, padding)
    
    # Calculate grid dimensions
    num_chars = len(charset)
    chars_per_row = 8  # Fixed number of characters per row
    num_rows = (num_chars + chars_per_row - 1) // chars_per_row  # Ceiling division
    
    # Use fixed SQUARE cell size for all sprites in the grid (to keep grid uniform)
    # Find maximum dimensions to ensure all sprites fit
    max_sprite_width = max(sprite.size[0] for sprite in sprites) if sprites else sprite_w
    max_sprite_height = max(sprite.size[1] for sprite in sprites) if sprites else sprite_h
    
    # Use fixed SQUARE dimensions (same width and height) to ensure cells are square
    # Take the maximum of width and height to ensure all sprites fit
    cell_size = max(max_sprite_width, max_sprite_height, sprite_w, sprite_h)
    sprite_width = cell_size  # Square cell
    sprite_height = cell_size  # Square cell
    padding = 5
    
    # Calculate total grid dimensions for a single sprite grid
    single_grid_width = chars_per_row * sprite_width + (chars_per_row - 1) * padding
    single_grid_height = num_rows * (sprite_height + 60)  # Each row: character + sprite + small spacing
    
    # For multiple sprites, create separate grids side by side
    if num_sprites_per_letter > 1:
        # Calculate total width for all grids side by side
        total_width = single_grid_width * num_sprites_per_letter + (num_sprites_per_letter - 1) * 20  # 20px spacing between grids
        total_height = single_grid_height
    else:
        total_width = single_grid_width
        total_height = single_grid_height
    
    # Create blank image
    grid_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(grid_img)
    
    # Use the specified font with higher resolution
    try:
        font = ImageFont.truetype("/home/vlachoum/learnable-DTLR/Junicode.ttf", 36)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
            except:
                font = ImageFont.load_default()
    
    # Create grids for each sprite index
    for sprite_idx in range(num_sprites_per_letter):
        # Calculate offset for this grid
        grid_offset_x = sprite_idx * (single_grid_width + 20)
        
        # Draw character names in first row for this grid
        for i, char in enumerate(charset):
            if i >= num_chars:
                break          
            row = i // chars_per_row
            col = i % chars_per_row
            
            # Calculate position for character name
            x = grid_offset_x + col * (sprite_width + padding)
            y = row * (sprite_height + 60) + 5  # Position at top of each row group
            
            # Center the character name
            bbox = draw.textbbox((0, 0), char, font=font)
            text_width = bbox[2] - bbox[0]
            x += (sprite_width - text_width) // 2
            
            # Draw character name with white outline for contrast
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), char, font=font, fill=(255, 255, 255))
            
            # Draw main text in black
            draw.text((x, y), char, font=font, fill=(0, 0, 0))  # Black text
        
        # Draw sprites for this grid (only sprites corresponding to this sprite index)
        for i in range(num_chars):
            # Calculate the actual sprite index in the proto_pixel tensor
            # Note: sprites are already filtered (sprite 0 removed), so indices are shifted by 1
            # For num_sprites_per_letter = 2: 
            # Grid 0: sprites 0,1,2,3,4,5,6,7... (correspond to original sprites 1,2,3,4,5,6,7,8...)
            # Grid 1: sprites n,n+1,n+2,n+3,n+4,n+5,n+6,n+7... (correspond to original sprites n+1,n+2,n+3,n+4,n+5,n+6,n+7,n+8...)
            actual_sprite_idx = sprite_idx * num_chars + i
            if actual_sprite_idx >= len(sprites):
                break
                
            row = i // chars_per_row
            col = i % chars_per_row
            
            # Calculate position for sprite (centered in the cell)
            cell_x = grid_offset_x + col * (sprite_width + padding)
            cell_y = row * (sprite_height + 60) + 40  # Below character name in same row group
            
            # Get actual sprite dimensions
            actual_sprite = sprites[actual_sprite_idx]
            actual_w, actual_h = actual_sprite.size
            
            # Center the sprite in the cell
            x = cell_x + (sprite_width - actual_w) // 2
            y = cell_y + (sprite_height - actual_h) // 2
            
            # Paste sprite centered in the cell
            grid_img.paste(actual_sprite, (x, y))
    
    return grid_img, sprites

def create_individual_sprite_grids(proto_pixel, charset, num_sprites_per_letter=1, sprite_bbox_stats=None):
    """
    Create separate grid images for each sprite index.
    Args:
        proto_pixel: Tensor of sprites (N, C, H, W)
        charset: List of characters (excluding empty character)
        num_sprites_per_letter: Number of sprites per character
        sprite_bbox_stats: Dictionary mapping sprite indices (1-indexed) to bbox statistics
    Returns:
        List of PIL Images, one for each sprite grid
    """
    grids = []
    for sprite_idx in range(num_sprites_per_letter):
        # Extract sprites for this sprite index
        # Note: sprites are already filtered (sprite 0 removed), so indices are shifted by 1
        # For num_sprites_per_letter = 2: 
        # Grid 0: sprites 0,1,2,3,4,5,6,7... (correspond to original sprites 1,2,3,4,5,6,7,8...)
        # Grid 1: sprites n,n+1,n+2,n+3,n+4,n+5,n+6,n+7... (correspond to original sprites n+1,n+2,n+3,n+4,n+5,n+6,n+7,n+8...)
        start_idx = sprite_idx * len(charset)
        end_idx = start_idx + len(charset)
        sprite_indices = list(range(start_idx, min(end_idx, proto_pixel.shape[0])))
        
        if not sprite_indices:
            continue
            
        # Extract the sprites for this grid
        sprites_for_grid = proto_pixel[sprite_indices]
        
        # Create subset of bbox stats for this grid
        # The sprites in sprites_for_grid have indices start_idx in the filtered array
        # Their real sprite indices are start_idx + 1 (because sprite 0 is empty)
        grid_bbox_stats = None
        if sprite_bbox_stats:
            grid_bbox_stats = {}
            for local_idx, global_idx in enumerate(sprite_indices):
                real_sprite_idx = global_idx + 1  # +1 because sprite 0 is empty
                if real_sprite_idx in sprite_bbox_stats:
                    # Map to local index (0, 1, 2, ...) for this grid
                    grid_bbox_stats[local_idx + 1] = sprite_bbox_stats[real_sprite_idx]
        
        # Create grid for this sprite index
        grid, _ = create_manual_grid(sprites_for_grid, charset, num_sprites_per_letter=1, sprite_bbox_stats=grid_bbox_stats)
        grids.append(grid)
    
    return grids

def renorm(img: torch.FloatTensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        -> torch.FloatTensor:
    # img: tensor(3,H,W) or tensor(B,3,H,W)
    # return: same as img
    assert img.dim() == 3 or img.dim() == 4, "img.dim() should be 3 or 4 but %d" % img.dim() 
    if img.dim() == 3:
        assert img.size(0) == 3, 'img.size(0) shoule be 3 but "%d". (%s)' % (img.size(0), str(img.size()))
        img_perm = img.permute(1,2,0)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(2,0,1)
    else: # img.dim() == 4
        assert img.size(1) == 3, 'img.size(1) shoule be 3 but "%d". (%s)' % (img.size(1), str(img.size()))
        img_perm = img.permute(0,2,3,1)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(0,3,1,2)

class SoftClamp(nn.Module):
    def __init__(self, alpha=0.01, inplace=False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x):
        x0 = torch.min(x, torch.zeros(x.shape, device=x.device))
        x1 = torch.max(x - 1, torch.zeros(x.shape, device=x.device))
        if self.inplace:
            return x.clamp_(0, 1).add_(x0, alpha=self.alpha).add_(x1, alpha=self.alpha)
        else:
            return torch.clamp(x, 0, 1) + self.alpha * x0 + self.alpha * x1

def CTC_loss(new_pred_logits,targets,num_fines_classes = None, device = None, num_sprites_per_letter = 1):
    if num_fines_classes is None:
        # Handle multiple sprites per character
        if num_sprites_per_letter > 1:
            # Reshape to group sprites by character
            batch_size, num_queries, num_classes = new_pred_logits.shape
            num_chars = num_classes // num_sprites_per_letter
            
            # Reshape to (batch_size, num_queries, num_chars, num_sprites_per_letter)
            pred_logits_reshaped = new_pred_logits.view(batch_size, num_queries, num_chars, num_sprites_per_letter)
            
            # Take max probability for each character across sprites
            pred_logits_summed = pred_logits_reshaped.max(dim=3)[0]
            
            # Add blank class (index 0)
            pred_final = torch.zeros(batch_size, num_queries, num_chars + 1, device=device)
            pred_final[:, :, 0] = 1e-5  # blank probability
            pred_final[:, :, 1:] = pred_logits_summed
            
            new_pred_logits = pred_final
        else:
            new_pred_logits = new_pred_logits
            
        blank_tensor = torch.zeros_like(new_pred_logits) + 1e-5
        blank_tensor[:, :, 0] = 1

        # ## add blank tokens
        pred_logits_padded = torch.zeros(
            (
                new_pred_logits.shape[0],
                new_pred_logits.shape[1] * 2,
                new_pred_logits.shape[2],
            )
        ).to(device)  # + 1e-5
        pred_logits_padded[:, ::2, :] = new_pred_logits
        pred_logits_padded[:, 1::2, :] = blank_tensor

        length_pred = torch.full(
            size=(pred_logits_padded.shape[0],),
            fill_value=pred_logits_padded.shape[1],
            dtype=torch.int64,
        )

        with torch.no_grad():
            length_input = torch.zeros_like(length_pred)
            max_length = 0
            for i, target in enumerate(targets):
                len_target = len(target["labels"])
                length_input[i] = len_target
                if len_target > max_length:
                    max_length = len_target
            targets_tensor = torch.zeros(new_pred_logits.shape[0], max_length)
            for i, target in enumerate(targets):
                targets_tensor[i, : len(target["labels"])] = target["labels"] + 1
        ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
        loss = ctc_loss(
            torch.log(pred_logits_padded.permute(1, 0, 2)),
            targets_tensor,
            length_pred,
            length_input,
        )  
        # check if loss is nan
        if torch.isnan(loss):
            print("CTC loss is nan")
            print("pred_logits_padded min/max:", pred_logits_padded.min(), pred_logits_padded.max())
            print("pred_logits_padded has nan:", torch.isnan(pred_logits_padded).any())
            print("log input has zero or neg:", (pred_logits_padded <= 0).any())
            print(loss)
    else:   
        if num_sprites_per_letter > 1:
            
            b,q,c = new_pred_logits.shape
            num_char = (c-1)//(num_sprites_per_letter*num_fines_classes)

           
            sprites_proba =  new_pred_logits[:,:,1:].view(b,q,num_fines_classes,num_sprites_per_letter,num_char)
            #shape is b,q,num_fines_classes,num_sprites_per_letter,num_char (1 character has num_sprites_per_letter sprites, each query has num_fines_classes elements)
            #eg if num_fines_classes = 2 (one for character, one for accent), num_sprites_per_letter = 2, then one query has 2*2 = 4 elements: 2 sprites for first character, 2 sprites for second character
            new_new_pred_logits =  torch.zeros(b,q,num_char*num_fines_classes+1,device=device)
            #first element is blank
            new_new_pred_logits[...,0] = new_pred_logits[...,0]
            pred_logits = sprites_proba.max(dim=-2)[0] # take the max probability for each sprite
            new_new_pred_logits[:,:,1:] = pred_logits.view(b,q,num_char*num_fines_classes)
            new_pred_logits = new_new_pred_logits
            # new_new_pred_logits =  torch.zeros(b,q,num_char+1,device=device)
            # pred_logits =  new_pred_logits[:,:,1:].view(b,q,num_char,num_sprites_per_letter)
            # pred_logits = pred_logits.max(dim=-1)[0]
            # new_new_pred_logits[:,:,0] = new_pred_logits[:,:,0]
            # new_new_pred_logits[:,:,1:] = pred_logits
            # new_pred_logits = new_new_pred_logits
        else:
            new_pred_logits = new_pred_logits

        num_char = new_pred_logits.shape[-1]-1
        blank_tensor = torch.zeros([new_pred_logits.shape[0], new_pred_logits.shape[1]//num_fines_classes, new_pred_logits.shape[2]]) + 1e-5
        blank_tensor[:, :, 0] = 1

        # ## add blank tokens
        pred_logits_padded = torch.zeros(
            (
                new_pred_logits.shape[0],
                new_pred_logits.shape[1] + blank_tensor.shape[1] * num_fines_classes,
                new_pred_logits.shape[2],
            )
        ).to(device)
        pred_logits_padded[:,1::num_fines_classes*2] = blank_tensor
        pred_logits_padded[:,3::num_fines_classes*2] = blank_tensor
        for i in range(num_fines_classes):
            pred_logits_padded[:,i*2::num_fines_classes*2,:] = new_pred_logits[:,i::num_fines_classes,:]
        length_pred = torch.full(
            size=(pred_logits_padded.shape[0],),
            fill_value=pred_logits_padded.shape[1],
            dtype=torch.int64,
        )
        with torch.no_grad():
            length_input = torch.zeros_like(length_pred)
            max_length = 0
            list_len_target = []
            for i, target in enumerate(targets):
                num_fcs = len(target["labels"])
                len_target = 0
                for j in range(1,num_fcs):
                    mask = target["labels"][j] != num_char
                    len_target += len(target["labels"][j][mask])

                length_input[i] = len_target
                if len_target > max_length:
                    max_length = len_target
            targets_tensor = torch.zeros(pred_logits_padded.shape[0], max_length)
            for i, target in enumerate(targets):

                target_char = target["labels"][1]
                other_target_char = []
                for j in range(2,num_fcs):
                    other_target_char.append(target["labels"][j])
                true_idx = 0
                for idx in range(len(targets_tensor[i])):
                    if idx < len(target_char):
                        targets_tensor[i][true_idx] = target_char[idx] + 1
                        true_idx += 1
                        for j in range(len(other_target_char)):
                            if other_target_char[j][idx] != num_char:
                                targets_tensor[i][true_idx] = other_target_char[j][idx] + 1
                                true_idx+=1
        ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
        loss = ctc_loss(
            torch.log(pred_logits_padded.permute(1, 0, 2)),
            targets_tensor,
            length_pred,
            length_input,
        )  
        if torch.isnan(loss):
            print("loss is nan")
            print(loss)
            print(pred_logits_padded)
            print(targets_tensor)
            print(length_pred)
    return loss
def process_bboxes(bboxes, w, h,  upscale_factor = 2):
    num_dim = bboxes.shape[-1] // 2

    if num_dim ==2:
        scale_tensor = torch.stack([w, h, w, h], dim=1) 
    else:
        scale_tensor = torch.tensor([w, h, w, h, 2*h, 2*w])
        bboxes = bboxes -  torch.tensor([0,0,0,0,0.5,0.5]).to(bboxes.device)
    scale_tensor = scale_tensor.to(bboxes.device)
    
    # Scale bboxes to image coordinates
    bboxes = bboxes * scale_tensor.unsqueeze(1)
    
    # Clamp bboxes to reasonable bounds - use clone to avoid in-place modification
    # For cx, cy: allow some margin outside image (up to 2x image size)
    # For w, h: minimum 2 pixels, maximum 10x image size (to prevent extreme values)
    if num_dim == 2:
        # Create new tensor to avoid in-place modification
        bboxes_clamped = bboxes.clone()
        bboxes_clamped[:, :, 2] = torch.clamp(bboxes[:, :, 2], min=2.0, max=w.max())  # w
        bboxes_clamped[:, :, 3] = torch.clamp(bboxes[:, :, 3], min=2.0, max=h.max())  # h
        bboxes = bboxes_clamped
    else:
        # Format: [cx, cy, w, h, theta1, theta2] - clamp first 4 dims
        # Create new tensor to avoid in-place modification
        bboxes_clamped = bboxes.clone()
        bboxes_clamped[:, :, 0] = torch.clamp(bboxes[:, :, 0], min=-w.max()*0.5, max=w.max()*2.5)  # cx
        bboxes_clamped[:, :, 1] = torch.clamp(bboxes[:, :, 1], min=-h.max()*0.5, max=h.max()*2.5)  # cy
        bboxes_clamped[:, :, 2] = torch.clamp(bboxes[:, :, 2], min=2.0, max=w.max()*10.0)  # w
        bboxes_clamped[:, :, 3] = torch.clamp(bboxes[:, :, 3], min=2.0, max=h.max()*10.0)  # h
        # theta1, theta2 can stay as is (they're rotation parameters)
        bboxes = bboxes_clamped

    return bboxes

def upscale_bbox(bboxes, upscale_factor = 2):
    width = bboxes[:,:,2] - bboxes[:,:,0]
    height = bboxes[:,:,3] - bboxes[:,:,1]
    center_x = bboxes[:,:,0] + width / 2
    center_y = bboxes[:,:,1] + height / 2

    new_width = width * upscale_factor
    new_height = height * upscale_factor
    new_bb_left = center_x - new_width / 2
    new_bb_top = center_y - new_height / 2
    new_bb_right = center_x + new_width / 2
    new_bb_bottom = center_y + new_height / 2

    upscaled_bboxes =  torch.stack([new_bb_left,new_bb_top,new_bb_right,new_bb_bottom],dim = -1)
    return upscaled_bboxes


def get_bboxes_scores(image, model, indice_space = None, process_bbox = True, num_fine_classes = None):
    device = image.device
    # output = model.to(device)(image[None].to(device))

    output,features_decoder,features_resnet = model(image, return_feature = True)
    

    
    h,w = output['true_dims'][:,0],output['true_dims'][:,1]



    if num_fine_classes is None:
        bboxes = output['pred_boxes']
        scores = output['pred_logits'].sigmoid()
    else: #bboxes is a tensor of shape num_fines_classes+1 x batch_size x num_queries x 4
        idx_box = torch.randint(0,output['pred_boxes'].shape[0],(1,))

        bboxes = output['pred_boxes'][idx_box]
        scores = output['pred_logits'][1].sigmoid()
        
        

    __, topk_indexes = torch.topk(scores.max(-1).values, 100)
    topk_indexes = topk_indexes
    bboxes = output['pred_boxes']
    scores = output['pred_logits'].sigmoid() #shape b,q,c, c = charset * num_sprites_per_letter


    if len(bboxes.shape) == 3:
        bboxes = torch.gather(bboxes, 1, topk_indexes.unsqueeze(-1).expand(-1, -1, bboxes.shape[-1]))
        scores = torch.gather(scores, 1, topk_indexes.unsqueeze(-1).expand(-1, -1, scores.shape[-1]))
    else:
        # For num_fine_classes case where bboxes is num_ffs x batch x num_query x 4
        bboxes = torch.gather(bboxes, 2, topk_indexes.unsqueeze(0).unsqueeze(-1).expand(bboxes.shape[0], -1, -1, bboxes.shape[-1]))
        scores = torch.gather(scores, 2, topk_indexes.unsqueeze(0).unsqueeze(-1).expand(scores.shape[0], -1, -1, scores.shape[-1]))
        num_fines_classes,batch_size,num_queries,charset_size = scores.shape
        padded_scores = torch.ones(num_fines_classes,batch_size,num_queries,charset_size*2,device=device)*1e-5
        #we need to padd here for the last fc we pad with zeros at the beginning charset_size, ie  a block of size charset_size at the beginning of the tensor with zeros
        # for the other fc we pad with zeros at end of the tensor charset, ie a block of size charset_size at the end of the tensor with zeros
        padded_scores[-1,:,:,charset_size:] = scores[-1]
        for i in range(num_fines_classes-1):
            padded_scores[i,:,:,:charset_size] = scores[i]
        # def hook_block_padding(grad):
        #     mask = torch.zeros_like(grad)
        #     mask[-1, :, :, charset_size:] = 1
        #     mask[:-1, :, :, :charset_size] = 1
            
        #     # CORRECTION CRITIQUE : Remplace les Inf/NaN par 0 avant la multiplication
        #     # Cela évite le problème "0 * Inf = NaN" causé par le log(1e-9)
        #     grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            
        #     return grad * mask
        # padded_scores.register_hook(hook_block_padding)
        scores = padded_scores

    features_decoder = torch.gather(features_decoder, 1, topk_indexes.unsqueeze(-1).expand(-1, -1, features_decoder.shape[-1]))
    if len(bboxes.shape) == 3:
        __, idx = torch.sort(bboxes[:, :, 0])
    else:
        __, idx = torch.sort(bboxes[1,:, :, 0])

    if len(scores.shape) == 3:
        scores = torch.gather(
                scores,
                1,
                idx.unsqueeze(-1).expand(-1, -1, scores.shape[-1]),
            )
        bboxes = torch.gather(
                bboxes,
                1,
                idx.unsqueeze(-1).expand(-1, -1, bboxes.shape[-1]),
            )
    else:
        scores = torch.gather(
                scores,
                2,
                idx.unsqueeze(-1).expand(scores.shape[0],scores.shape[1], -1, scores.shape[-1]),
            )
        bboxes = torch.gather(
                bboxes,
                2,
                idx.unsqueeze(-1).expand(bboxes.shape[0], bboxes.shape[1], -1, bboxes.shape[-1]),
            )
    features_decoder = torch.gather(
            features_decoder,
            1,
            idx.unsqueeze(-1).expand(-1, -1, features_decoder.shape[-1]),
        )
    if len(scores.shape) ==3:
        new_scores = torch.zeros(
            (
                scores.shape[0],
            scores.shape[1],
            scores.shape[2] + 1,
        )
        ).to(device)
        

        new_scores = new_scores
        new_scores[:, :, 1:] = scores
        ## computes the proba of the blank token
        eps = 0.003  # / pred_logits.shape[-1] # 0.01/num_classes
        mask = scores.sum(-1) < 1 - eps
        new_scores[:, :, 0][mask] = 1 - scores[mask].sum(-1)

        mask = ~mask

        new_scores[:, :, 0][mask] = eps
        new_scores[:, :, 1:][mask] = (
            (1 - eps)
            * scores[mask]
            / scores[mask].sum(-1).unsqueeze(-1)
        )
        scores = new_scores
    
    else:

        new_scores = torch.zeros(
            (
                scores.shape[0],
            scores.shape[1],
            scores.shape[2],
            scores.shape[3] + 1,
        )
        ).to(device)
        eps = 0.003  # / pred_logits.shape[-1] # 0.01/num_classes
        
        # Handle each fc independently
        for fc in range(scores.shape[0]):
            mask = scores[fc].sum(-1) < 1 - eps
            new_scores[fc, :, :, 1:] = scores[fc]
            new_scores[fc, :, :, 0][mask] = 1 - scores[fc][mask].sum(-1)

            mask = ~mask
            new_scores[fc, :, :, 0][mask] = eps
            new_scores[fc, :, :, 1:][mask] = (
                (1 - eps) 
                * scores[fc][mask]
                / scores[fc][mask].sum(-1).unsqueeze(-1)
            )   

        last_fcs = new_scores[1:] # shape [2, batch, num_queries, num_classes]
        last_fcs = last_fcs.transpose(0,1)
        combined_queries = torch.zeros( 
            last_fcs.shape[0],
            last_fcs.shape[1] * last_fcs.shape[2], 
            last_fcs.shape[3],
            device=last_fcs.device
        )
        combines_bboxes = torch.zeros(
            last_fcs.shape[0],
            last_fcs.shape[1] * last_fcs.shape[2], 
            bboxes.shape[-1],
            device=last_fcs.device
        )
        last_fcs_bboxes = bboxes[1:]
        last_fcs_bboxes = last_fcs_bboxes.transpose(0,1)
        for i in range(last_fcs.shape[1]):
            combined_queries[:,i::last_fcs.shape[1],:] = last_fcs[:,i,:,:]
            combines_bboxes[:,i::last_fcs.shape[1],:] = last_fcs_bboxes[:,i,:,:]
        scores = combined_queries
        bboxes = combines_bboxes


    bboxes = process_bboxes(bboxes,w,h)
    bboxes = bboxes.to(device)
    

    return bboxes,scores, features_decoder,features_resnet.decompose()[0], features_resnet.decompose()[1]

def composite(reconstruction_transformed, background_transformed, upscaled_bboxes, cc):
    perm = torch.randperm(len(reconstruction_transformed))
    
    reconstruction_transformed = [reconstruction_transformed[i]for i in perm] 
    #bboxes = bboxes[perm]
    perm = perm

    upscaled_bboxes = upscaled_bboxes[perm]
    cc = cc
    cc = cc[perm]

    comp = background_transformed

    H,W = comp.shape[-2:]
    upscaled_bboxes = upscaled_bboxes
    W_upscaled = upscaled_bboxes[:,2] - upscaled_bboxes[:,0]
    H_upscaled = upscaled_bboxes[:,3] - upscaled_bboxes[:,1]
    X_A = torch.maximum(torch.tensor(0.),upscaled_bboxes[:,0])
    Y_A = torch.maximum(torch.tensor(0.),upscaled_bboxes[:,1])

    X_AA = torch.maximum(torch.tensor(0.), -upscaled_bboxes[:,0])
    Y_AA = torch.maximum(torch.tensor(0.), -upscaled_bboxes[:,1])

    X_B = torch.minimum(torch.tensor(W), X_A + W_upscaled-X_AA)
    Y_B = torch.minimum(torch.tensor(H), Y_A  + H_upscaled-Y_AA)

    X_BB = X_AA + X_B -X_A
    Y_BB = Y_AA + Y_B -Y_A

    X_BB = torch.floor(X_BB).int()
    Y_BB = torch.floor(Y_BB).int()

    X_AA = torch.ceil(X_AA).int()
    Y_AA = torch.ceil(Y_AA).int()

    X_A, X_B =  torch.ceil(X_A).int(), torch.floor(X_B).int()
    Y_A, Y_B = torch.ceil(Y_A).int(), torch.floor(Y_B).int()
    

    for reco_,x_a,x_b,y_a,y_b,x_aa,y_aa,x_bb,y_bb,ccc in zip(reconstruction_transformed,X_A,X_B,Y_A,Y_B,X_AA,Y_AA,X_BB,Y_BB,cc):
        mask = torch.zeros(comp.shape[-2:], device=comp.device)
        colors =  torch.zeros_like(comp, device = comp.device)

        colors[:,y_a:y_b, x_a:x_b] = (reco_[y_aa:y_bb, x_aa:x_bb].unsqueeze(-1) * ccc.unsqueeze(0).unsqueeze(0)).permute(2, 0, 1)
        mask[y_a:y_b, x_a:x_b] = reco_[y_aa:y_bb, x_aa:x_bb]

        comp = (1-mask.unsqueeze(0)) * comp + colors


    return comp


def affine_transform_reconstruction(reconstruction,boxes,generator):
    
    device = "cuda"

    w_bbox = boxes[:,:,2] 
    h_bbox = boxes[:,:,3]
    # Clamp w and h to reasonable bounds (min 2, max very large but finite)
    w_bbox = torch.clamp(w_bbox, min=2.0, max=1e6)
    h_bbox = torch.clamp(h_bbox, min=2.0, max=1e6)
    c_x = boxes[:,:,0]
    c_y = boxes[:,:,1]
    x1 = c_x - w_bbox/2
    y1 = c_y - h_bbox/2
    x2 = c_x + w_bbox/2
    y2 = c_y + h_bbox/2

    padding_left,padding_right,padding_top,padding_bottom = generator.padding
    hs,ws = generator.old_sprite_size
    bottom = padding_bottom / hs *h_bbox
    top = padding_top / hs *h_bbox
    left = padding_left / ws *w_bbox
    right = padding_right / ws *w_bbox

    x1 = x1 - left
    y1 = y1 - top
    x2 = x2 + right
    y2 = y2 + bottom
    Cx = (x1 + x2) / 2
    Cy = (y1 + y2) / 2
    W = x2 - x1
    H = y2 - y1

    
    max_W = torch.max(W)
    max_H = torch.max(H)
    upscaled_bboxes = torch.zeros_like(boxes)
    upscaled_bboxes[:,:,0] = torch.floor(Cx - max_W*0.5)
    upscaled_bboxes[:,:,1] = torch.floor(Cy - max_H*0.5)
    upscaled_bboxes[:,:,2] = torch.ceil(Cx + max_W*0.5)
    upscaled_bboxes[:,:,3] = torch.ceil(Cy + max_H*0.5)

    W_upscaled = upscaled_bboxes[:,:,2] - upscaled_bboxes[:,:,0]
    H_upscaled = upscaled_bboxes[:,:,3] - upscaled_bboxes[:,:,1]

    #compute the integer bboxes that can fit the upscaled bboxes
    W_int = torch.max(W_upscaled)
    H_int = torch.max(H_upscaled)


    #compute the delta to make sure the bboxes have all the same width and height (otherwise they can differs because of the integer rounding)
    delta_int_W = W_int - W_upscaled 
    delta_int_H = H_int - H_upscaled 
    upscaled_bboxes[:,:,2] += delta_int_W
    upscaled_bboxes[:,:,3] += delta_int_H

    #compute the center of the integer bboxes 

    Cx_int = (upscaled_bboxes[:,:,0] + upscaled_bboxes[:,:,2]) / 2
    Cy_int = (upscaled_bboxes[:,:,1] + upscaled_bboxes[:,:,3]) / 2

    S_x =  W_int /W
    S_y =  H_int /H

    
    theta = torch.zeros((boxes.shape[0],boxes.shape[1],2,3)).to(device)

    Delta_Cx = Cx - Cx_int
    Delta_Cy = Cy - Cy_int

    Tx = -2 / W  * Delta_Cx
    Ty = -2 / H  * Delta_Cy


    theta[:,:,0,0] = S_x 
    theta[:,:,1,1] = S_y 
    theta[:,:,0,2] = Tx
    theta[:,:,1,2] = Ty


    reconstruction_transformed  = []
    b,q = theta.shape[0],theta.shape[1]
    theta = theta.view(theta.shape[0] * theta.shape[1],2,3)
    reconstruction = reconstruction.view(reconstruction.shape[0] * reconstruction.shape[1],reconstruction.shape[2],reconstruction.shape[3],reconstruction.shape[4])
    grid = F.affine_grid(theta,size = (theta.shape[0],1,H_int.int(),W_int.int()),align_corners=True)
    reconstruction_transformed = F.grid_sample(reconstruction,grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    reconstruction_transformed = reconstruction_transformed.view(b,q,reconstruction_transformed.shape[2],reconstruction_transformed.shape[3])

    
    # CRITICAL: Cut NaN/Inf gradients to prevent weight corruption
    if reconstruction_transformed.requires_grad:
        def clamp_grad_reconstruction(grad):
            if grad is None:
                return None
            nan_count = torch.isnan(grad).sum().item() if torch.isnan(grad).any() else 0
            inf_count = torch.isinf(grad).sum().item() if torch.isinf(grad).any() else 0
            if nan_count > 0 or inf_count > 0:
                print(f"⚠️ Cutting {nan_count} NaN and {inf_count} Inf gradients in reconstruction_transformed")
            # Replace NaN/Inf with zeros and clamp to reasonable range
            grad = torch.nan_to_num(grad, nan=0.0, posinf=5.0, neginf=-5.0)
            return torch.clamp(grad, min=-5.0, max=5.0)
        reconstruction_transformed.register_hook(clamp_grad_reconstruction)

    return reconstruction_transformed, upscaled_bboxes




def affine_transform_background(background, image_sizes, generator):
    device = generator.proto.device

    w_bbox = image_sizes[:,1]
    h_bbox = image_sizes[:,0]
    ratio_w =  generator.sprite_size[1] /w_bbox 
    ratio_h =  generator.sprite_size[0] /h_bbox
    theta = torch.zeros(image_sizes.shape[0],2,3).to(device)
    theta[:,0,0] = 1
    theta[:,1,1] = 1
    batch_size = image_sizes.shape[0]
    background_transformed  = []

    w_bbox = w_bbox.int()
    h_bbox = h_bbox.int()
    for i in range(batch_size):
            grid = F.affine_grid(theta[i,].unsqueeze(0),size = (1,1,h_bbox[i],w_bbox[i]),align_corners=False)
            reconstruction_background_i = F.grid_sample(background.unsqueeze(0),grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            background_transformed.append(reconstruction_background_i)
    return background_transformed

def save_reconstruction_visualization(image, mask, target, reconstructor, model, charset, args = None, e = None, device=None, num_fine_classes = 2, space_index = None, batch_idx = 0):
    if args is not None:
        if args.space_index is not None:
            space_index = args.space_index
        else:
            space_index = 0
    image = image.unsqueeze(0)
    # mask = mask.unsqueeze(0)
    target = [target]
    # mask = mask.to(device)
    mask = torch.zeros_like(image)
    mask = mask.unsqueeze(0)
    mask  = mask.to(device)

    bboxes, scores, features_decoder, features_resnet, mask_resnet = get_bboxes_scores(image.cuda(), model, num_fine_classes=num_fine_classes)
    
    renorm_images = [renorm(image[i]).cuda() for i in range(image.shape[0])]
    list_image = [image[i].cuda() for i in range(image.shape[0])]
    list_reco_image = reconstructor(renorm_images, scores, bboxes, features_decoder, features_resnet, space_index=space_index, mask_resnet=mask_resnet)

    reco = list_reco_image
    renorm_images = torch.stack(renorm_images)
    reco_image = reco.detach().cpu().numpy()
    
    step_to_use = getattr(args, 'step', e)
    step_folder = f"{e:03d}"
    figure_folder = f"{e:03d}"

    # Create main figure folder
    if figure_folder is not None:
        # Create sprites and reco subfolders within the main output directory
        sprites_folder = os.path.join(args.output_dir, "sprites", figure_folder)
        reconstruction_folder = os.path.join(args.output_dir, "reco", figure_folder)
        
        if not os.path.exists(sprites_folder):
            os.makedirs(sprites_folder)
        if not os.path.exists(reconstruction_folder):
            os.makedirs(reconstruction_folder)
    else:
        sprites_folder = None
        reconstruction_folder = None
    
    # Process reconstruction
    reco = list_reco_image[0] * (1-mask[0].long())
    reco = reco[0]
    reco_image = np.transpose(reco.detach().cpu().numpy(), (1, 2, 0))
    
    # Process color image
    if args is not None:
        if args.mask:
            extra_mask = target[0]["masks"]
            color_image = renorm_images[0] * (1-mask[0].long()) * extra_mask.to(device)
        else:
            color_image = renorm_images[0] * (1-mask[0].long())
    else:
        color_image = renorm_images[0] * (1-mask[0].long())
    color_image = color_image[0]
    color_image = np.transpose(color_image.detach().cpu().numpy(), (1, 2, 0))
    
    # Clip images
    reco_image = np.clip(reco_image, 0, 1)
    color_image = np.clip(color_image, 0, 1)
    
    # Convert to BGR for OpenCV
    color_image_bgr = (color_image * 255).astype(np.uint8)
    color_image_bgr = cv2.cvtColor(color_image_bgr, cv2.COLOR_RGB2BGR)
    
    # Draw bounding boxes
    if args is not None and args.skew:
            bbox = bboxes[0].detach().cpu().numpy()
            cx, cy, w, h, theta1, theta2 = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:,3], bbox[:,4], bbox[:,5]
            
            c = np.array([cx,cy])
            v1 = np.array([w, theta1])
            v2 = np.array([theta2,h])
            
            A = c - (v1 + v2)
            B = A + 2*v1
            C = A + 2*(v1 + v2)
            D = A + 2*v2
            polygon = np.stack([A,B,C,D]).transpose(2,0,1)
            
            for it, pp in enumerate(polygon):
                if scores[0][it].argmax().item() != 0:
                    pp = pp.astype(np.int32)
                    pp = pp[:,np.newaxis,:]
                    cv2.polylines(color_image_bgr, [pp.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=1)
    else:
        bboxes = box_ops.box_cxcywh_to_xyxy(bboxes)
        for it, bb in enumerate(bboxes[0].detach()):
            bb = bb.int().cpu().numpy()
            if scores[0][it].argmax().item() != 0:
                cv2.rectangle(color_image_bgr, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)
                char = charset[(scores[0][it].argmax().item()-1)%len(charset)]
                char = 'char {}'.format(char)
                text_size = cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_y = bb[3] - 5
                cv2.putText(color_image_bgr, char, (bb[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
    # Convert back to RGB and normalize
    color_image_with_bboxes = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
    color_image_with_bboxes = color_image_with_bboxes.astype(np.float32) / 255.0
    
    # Stack images
    stacked_image = np.vstack((color_image_with_bboxes, reco_image))
    
    # Process transcription
    transcription_list = [charset[(i-1)%len(charset)] for i in scores[0].argmax(dim=1).detach().cpu().numpy() if i != 0]
    transcription = "".join(transcription_list)
    transcription_list_str = str(transcription_list)
    
    # Convert to uint8 and add padding
    stacked_image_uint8 = (stacked_image * 255).astype(np.uint8)
    padding_height = 50
    padded_image = np.zeros((stacked_image_uint8.shape[0] + padding_height, stacked_image_uint8.shape[1], 3), dtype=np.uint8)
    padded_image[:stacked_image_uint8.shape[0], :stacked_image_uint8.shape[1]] = stacked_image_uint8
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    font_color = (255, 255, 255)
    
    text_size = cv2.getTextSize(transcription, font, font_scale, font_thickness)[0]
    list_text_size = cv2.getTextSize(transcription_list_str, font, font_scale, font_thickness)[0]
    
    text_x = (padded_image.shape[1] - max(text_size[0], list_text_size[0])) // 2
    text_y = stacked_image_uint8.shape[0] + 20
    list_text_y = stacked_image_uint8.shape[0] + 40
    
    # Add black background for text
    cv2.rectangle(padded_image, 
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + max(text_size[0], list_text_size[0]) + 5, list_text_y + 5),
                (0, 0, 0),
                -1)
    
    # Add text
    cv2.putText(padded_image, transcription, (text_x, text_y), font, font_scale, font_color, font_thickness)
    cv2.putText(padded_image, transcription_list_str, (text_x, list_text_y), font, font_scale, font_color, font_thickness)
    
    # Convert back to float and save
    stacked_image = padded_image.astype(np.float32) / 255.0
    # Save reco image only if figure_folder is set (i.e., every 25 epochs)
    if figure_folder is not None:
        plt.imsave(os.path.join(reconstruction_folder, f"sample_{batch_idx}.jpg"), stacked_image)
    else:
        plt.imshow(stacked_image)
        plt.show()
    
    if args is not None and args.wandb and figure_folder is not None:
        wandb.log({f"reco_image_{e}_{batch_idx}": wandb.Image(os.path.join(reconstruction_folder, f"sample_{batch_idx}.jpg"))})
    return stacked_image