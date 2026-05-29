import numpy as np
import torch
import torch.nn as nn
from .utils import composite, affine_transform_reconstruction
from .generator import Generator, Background_Color
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from .utils import SoftClamp
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class Reconstructor(nn.Module):
    def __init__(self, n_outputs, sprite_size, type='mlp', mask_sprite=False, composite_mode='batch', init_zeros=False):
        super().__init__()
        self.generator = Generator(n_outputs, sprite_size, type, init_zeros=init_zeros).to(device)
        self.composite_mode = composite_mode
        self.proto_pixel = self.generator()
        # color predictor takes as input a tensor of size Bx Qx 256 and outputs a tensor of size BxQ x3
        self.color_predictor = MLP(256, 128, 3, 3)
        self.background_color = Background_Color()
        self.soft_clamp = SoftClamp(alpha=0.01)
        self.sigmoid = nn.Sigmoid()
        self.mask_sprite = mask_sprite
    
    def add_padding(self,padding):
        self.generator.add_padding(padding)


    def predict_color(self, features):
        # Flatten the features from BxQx256 to (B*Q)x256
        B, Q, _ = features.shape
        features = features.view(B * Q, 256)
        
        # Pass through the color predictor
        output = self.color_predictor(features)
        
        # Reshape output back to BxQx3
        return output.view(B, Q, 3)
    

    def forward(self,images,scores,bboxes, features_decoder,features_resnet,space_index = None,  mask_resnet = None, true_mask = None): # batch size 1 for now
        sim = scores.permute(2,0,1)
        
        # Filtre les scores d'attention minuscules pour éviter que les prototypes non utilisés
        # ne reçoivent des micro-gradients (bruit) que l'optimiseur Adam amplifierait.
        sim = torch.where(sim < 1e-3, torch.zeros_like(sim), sim)
        
        proto_pixel = self.generator(space_index = space_index,mask_sprite = self.mask_sprite)
        #shape scores b,q,c . q = 200 = 100 (characters) + 100 (accents)
        # c = charset * num_sprites_per_letter + 1 (blank)
        #proto_pixel shape = charset * num_sprites_per_letter + 1 (blank)
        #
        if scores.shape[-1]-1 == proto_pixel.shape[0]-1: #there is no accent
            reconstruction = torch.einsum('pbq, pchw -> bqchw', sim, proto_pixel)
        else:
            sim_characters = sim[:proto_pixel.shape[0]]
            reconstruction_characters = torch.einsum('pbq, pchw -> bqchw', sim_characters, proto_pixel)
            sim_accents = torch.cat([sim[0].unsqueeze(0), sim[proto_pixel.shape[0]:]], dim=0)
            reconstruction_accents = torch.einsum('pbq, pchw -> bqchw', sim_accents, proto_pixel)
            reconstruction = reconstruction_characters + reconstruction_accents


        
        features_decoder = features_decoder
        colors = self.predict_color(features_decoder)
        colors = self.soft_clamp(colors)
        
        if bboxes.shape[1] > colors.shape[1]:
            # Duplicate colors to match bbox shape
            colors = colors.repeat(1, bboxes.shape[1] // colors.shape[1], 1)
        
        H,W = images[0].shape[-2:]
        true_mask_dims = []
        for i in range(len(features_resnet)):
            if true_mask is not None:
                mask = true_mask[i]
                # Find valid pixels (where mask is False for content, True for padding)
                valid_coords = torch.nonzero(~mask)
                if len(valid_coords) > 0:
                    min_coords = valid_coords.min(dim=0)[0]
                    max_coords = valid_coords.max(dim=0)[0]
                    min_h, min_w = min_coords[0].item(), min_coords[1].item()
                    max_h, max_w = max_coords[0].item(), max_coords[1].item()
                    true_h = max_h - min_h + 1
                    true_w = max_w - min_w + 1
                    true_mask_dims.append((true_h, true_w, min_h, min_w, max_h, max_w))
                else:
                    # If no valid content found, use full image size
                    true_mask_dims.append((H, W, 0, 0, H-1, W-1))
            else:
                true_mask_dims.append((H, W, 0, 0, H-1, W-1))
        colors_background = self.background_color(features_resnet, (H,W), mask_resnet, true_mask_dims)

        old_sprite_size = self.generator.old_sprite_size
        sprite_size = self.generator.sprite_size
        r_h = sprite_size[0]/old_sprite_size[0]
        r_w = sprite_size[1]/old_sprite_size[1]
    
        reconstruction_transformed,upscaled_bboxes = affine_transform_reconstruction(reconstruction, bboxes,self.generator)

        image_shapes = [image.shape[1:] for image in images]
        image_shapes = torch.tensor(image_shapes).to(device)

        if self.composite_mode == 'sequential':
            list_reco_image = composite_batch_sequential(reconstruction_transformed, colors_background, upscaled_bboxes, colors)
        elif self.composite_mode == 'additive':
            list_reco_image = composite_batch_additive(reconstruction_transformed, colors_background, upscaled_bboxes, colors)
        else:
            list_reco_image = composite_batch(reconstruction_transformed, colors_background, upscaled_bboxes, colors)

        return list_reco_image

def composite_batch_additive(reconstruction_transformed, background_transformed, upscaled_bboxes, colors):
    """
    Vectorized compositing for batched input.
    
    Args:
        reconstruction_transformed: (B, Q, H_patch, W_patch) -- alpha masks
        background_transformed: (B, 3, H, W)
        upscaled_bboxes: (B, Q, 4) -- [x0, y0, x1, y1]
        colors: (B, Q, 3)
    
    Returns:
        comp: (B, 3, H, W) -- composited images
    """
    B, Q, H_patch, W_patch = reconstruction_transformed.shape
    _, _, H_full, W_full = background_transformed.shape
    device = background_transformed.device
    dtype = background_transformed.dtype

    # Generate local grids
    yy = torch.arange(H_patch, device=device).view(1, 1, H_patch, 1).expand(B, Q, H_patch, W_patch)
    xx = torch.arange(W_patch, device=device).view(1, 1, 1, W_patch).expand(B, Q, H_patch, W_patch)

    # Absolute coordinates
    x0 = upscaled_bboxes[..., 0].unsqueeze(-1).unsqueeze(-1)
    y0 = upscaled_bboxes[..., 1].unsqueeze(-1).unsqueeze(-1)
    Y = torch.clamp(yy + y0, 0, H_full - 1).long()
    X = torch.clamp(xx + x0, 0, W_full - 1).long()

    # Flatten indices
    flat_Y = Y.reshape(B, Q, -1)
    flat_X = X.reshape(B, Q, -1)
    flat_idx = flat_Y * W_full + flat_X  # shape (B, Q, H_patch*W_patch)
    
    # Clamp invalid indices
    if (flat_idx < 0).any() or (flat_idx >= H_full * W_full).any():
        flat_idx = torch.clamp(flat_idx, 0, H_full * W_full - 1)

    # Flatten colors and masks
    alpha_flat = reconstruction_transformed.reshape(B, Q, -1)  # (B,Q,N)
    color_flat = (colors.unsqueeze(-1).unsqueeze(-1) * reconstruction_transformed.unsqueeze(2))  # (B,Q,3,H,W)
    color_flat = color_flat.permute(0,1,3,4,2).reshape(B, Q, -1, 3)  # (B,Q,N,3)

    # Prepare accumulators
    comp = background_transformed.clone()
    accum_colors = torch.zeros(B, 3, H_full*W_full, device=device, dtype=dtype)
    accum_alpha = torch.zeros(B, 1, H_full*W_full, device=device, dtype=dtype)

    for b in range(B):
        # Flatten Q*H_patch*W_patch per batch
        idx = flat_idx[b].reshape(-1)
        alpha = alpha_flat[b].reshape(-1)
        color = color_flat[b].reshape(-1, 3)

        accum_colors[b].index_add_(1, idx, color.T)
        accum_alpha[b].index_add_(1, idx, alpha.unsqueeze(0))

    # Reshape back
    accum_colors = accum_colors.view(B, 3, H_full, W_full)
    accum_alpha = accum_alpha.view(B, 1, H_full, W_full).clamp(0, 1)

    # Blend with background
    comp = (1 - accum_alpha) * comp + accum_colors
    
    return comp

def composite_batch(reconstruction_transformed, background_transformed, upscaled_bboxes, colors):
    """
    Vectorized compositing for batched input with modulo-based layering.
    Splits queries into 4 groups (modulo 4) sorted by X-coordinate to minimize 
    intra-group overlap. 
    The rendering order of the 4 groups is randomized to prevent the model 
    from learning a strict left-to-right depth bias.
    """
    # On garde ton tri original par coordonnée X
    x_coords = upscaled_bboxes[..., 0] # (B, Q)
    sorted_indices = torch.argsort(x_coords, dim=1) # (B, Q)

    B, Q, H_patch, W_patch = reconstruction_transformed.shape
    
    # Gather sorted data
    batch_indices = torch.arange(B, device=reconstruction_transformed.device).unsqueeze(1).expand(B, Q)
    reconstruction_transformed = reconstruction_transformed[batch_indices, sorted_indices]
    upscaled_bboxes = upscaled_bboxes[batch_indices, sorted_indices]
    colors = colors[batch_indices, sorted_indices]

    _, _, H_full, W_full = background_transformed.shape
    device = background_transformed.device
    dtype = background_transformed.dtype

    # Generate local grids
    yy = torch.arange(H_patch, device=device).view(1, 1, H_patch, 1).expand(B, Q, H_patch, W_patch)
    xx = torch.arange(W_patch, device=device).view(1, 1, 1, W_patch).expand(B, Q, H_patch, W_patch)

    # Absolute coordinates
    x0 = upscaled_bboxes[..., 0].unsqueeze(-1).unsqueeze(-1)
    y0 = upscaled_bboxes[..., 1].unsqueeze(-1).unsqueeze(-1)
    Y = torch.clamp(yy + y0, 0, H_full - 1).long()
    X = torch.clamp(xx + x0, 0, W_full - 1).long()

    # Flatten indices
    flat_Y = Y.reshape(B, Q, -1)
    flat_X = X.reshape(B, Q, -1)
    flat_idx = flat_Y * W_full + flat_X  # shape (B, Q, H_patch*W_patch)
    
    # Clamp invalid indices
    if (flat_idx < 0).any() or (flat_idx >= H_full * W_full).any():
        flat_idx = torch.clamp(flat_idx, 0, H_full * W_full - 1)

    # Flatten colors and masks
    alpha_flat = reconstruction_transformed.reshape(B, Q, -1)  # (B,Q,N)
    color_flat = (colors.unsqueeze(-1).unsqueeze(-1) * reconstruction_transformed.unsqueeze(2))  # (B,Q,3,H,W)
    color_flat = color_flat.permute(0,1,3,4,2).reshape(B, Q, -1, 3)  # (B,Q,N,3)

    # Initialize composite with background
    comp = background_transformed.clone()
    
    num_steps = 4
    
    random_steps = torch.randperm(num_steps).tolist()
    
    for step in random_steps:
    # -----------------------------------------------------------
        # Prepare accumulators for this step
        step_accum_colors = torch.zeros(B, 3, H_full*W_full, device=device, dtype=dtype)
        step_accum_alpha = torch.zeros(B, 1, H_full*W_full, device=device, dtype=dtype)

        for b in range(B):
            # Select queries for this step: index % 4 == step
            idx = flat_idx[b, step::num_steps].reshape(-1)
            alpha = alpha_flat[b, step::num_steps].reshape(-1)
            color = color_flat[b, step::num_steps].reshape(-1, 3)

            # Optimization: skip if empty (e.g. if Q < step)
            if idx.numel() > 0:
                step_accum_colors[b].index_add_(1, idx, color.T)
                step_accum_alpha[b].index_add_(1, idx, alpha.unsqueeze(0))

        # Reshape accumulators
        step_accum_colors = step_accum_colors.view(B, 3, H_full, W_full)
        step_accum_alpha = step_accum_alpha.view(B, 1, H_full, W_full).clamp(0, 1)

        # Blend this layer onto the composite
        comp = step_accum_colors + (1 - step_accum_alpha) * comp
    
    return comp


def composite_batch_sequential(
    reconstruction_transformed, background_transformed, upscaled_bboxes, colors
):
    """
    Batched compositing with RANDOM order (Painter's Algorithm).
    
    Args:
        reconstruction_transformed: (B, Q, H_patch, W_patch) -- alpha masks (0.0 to 1.0)
        background_transformed: (B, 3, H, W) -- Base canvas
        upscaled_bboxes: (B, Q, 4) -- [x0, y0, x1, y1]
        colors: (B, Q, 3) -- RGB colors
        
    Returns:
        comp: (B, 3, H, W) -- Composited image
    """
    B, Q, H_patch, W_patch = reconstruction_transformed.shape
    _, _, H_full, W_full = background_transformed.shape
    device = background_transformed.device

    # 1. Generate local grids
    # Shape: (B, Q, H_patch, W_patch)
    yy = torch.arange(H_patch, device=device).view(1, 1, H_patch, 1).expand(B, Q, H_patch, W_patch)
    xx = torch.arange(W_patch, device=device).view(1, 1, 1, W_patch).expand(B, Q, H_patch, W_patch)

    # 2. Calculate absolute coordinates
    x0 = upscaled_bboxes[..., 0, None, None]
    y0 = upscaled_bboxes[..., 1, None, None]
    
    y_abs = yy + y0
    x_abs = xx + x0

    # 3. Create Valid Mask (Prevent clamping streaks)
    # We only want to paint pixels that actually fall inside the canvas dimensions
    valid_mask = (
        (y_abs >= 0) & (y_abs < H_full) & 
        (x_abs >= 0) & (x_abs < W_full)
    )

    # Clamp for safe indexing (invalid pixels will be filtered out later)
    Y = torch.clamp(y_abs, 0, H_full - 1).long()
    X = torch.clamp(x_abs, 0, W_full - 1).long()

    # 4. Flatten for the loop
    # Shape: (B, Q, N_pixels) where N_pixels = H_patch * W_patch
    flat_idx = (Y * W_full + X).reshape(B, Q, -1)
    flat_alpha = reconstruction_transformed.reshape(B, Q, -1)
    flat_valid = valid_mask.reshape(B, Q, -1)

    # 5. Iterative Compositing with Random Order
    # Clone background so we don't modify the input in-place
    canvas = background_transformed.clone()
    
    # View canvas as (B, 3, H*W) for easy flat indexing
    canvas_flat = canvas.view(B, 3, -1)

    for b in range(B):
        # Generate a RANDOM permutation of indices for this batch item
        # e.g., if Q=3, might be [2, 0, 1]
        order = torch.randperm(Q, device=device)

        for q in order:
            # Extract data for this specific query
            indices = flat_idx[b, q]    # Pixel indices on canvas
            alpha = flat_alpha[b, q]    # Alpha values
            valid = flat_valid[b, q]    # Validity mask
            
            # Get color and reshape to (3, 1) for broadcasting
            rgb = colors[b, q].view(3, 1) 

            # Optimization: Skip if no valid pixels
            if not valid.any():
                continue

            # Select only valid pixels to avoid artifacts at image borders
            indices = indices[valid]
            alpha = alpha[valid].unsqueeze(0)  # Shape (1, N_valid)

            # --- Alpha Blending ---
            # 1. Read current background pixels
            current_bg_pixels = canvas_flat[b, :, indices]
            
            # 2. Blend: New = Alpha * Color + (1 - Alpha) * Old
            # This ensures that if we draw over an existing object, it obscures it correctly
            blended_pixels = (rgb * alpha) + (current_bg_pixels * (1.0 - alpha))
            
            # 3. Write back
            canvas_flat[b, :, indices] = blended_pixels

    # Reshape back to (B, 3, H, W)
    return canvas_flat.view(B, 3, H_full, W_full)