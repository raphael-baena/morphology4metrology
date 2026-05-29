import torch
import torch.nn as nn

import torch.nn.functional as F
from .utils import SoftClamp
import numpy as np
class Generator(nn.Module):
    def __init__(self, n_outputs, sprite_size, type='mlp', init_zeros=False):
        super().__init__()
        self.mode = type
        self.old_sprite_size = sprite_size
        self.sprite_size = sprite_size
        self.n_outputs = n_outputs
        if init_zeros:
            self.proto = nn.Parameter(torch.zeros((n_outputs, 1) + tuple(sprite_size)))  #size (K,1,H,W)
        else:
            self.proto = nn.Parameter(torch.rand((n_outputs, 1) + tuple(sprite_size)))  #size (K,1,H,W)

        self.register_buffer('empty_sprite', torch.zeros((1, 1) + tuple(sprite_size)))

        self.flat_latents = self.proto.squeeze(1).flatten(start_dim=-2)
        self.latent_dim = sprite_size[0]*sprite_size[1]
        self.border_size = 1
        self.mask = self.create_mask(sprite_size, self.border_size)
        self.soft_clamp = SoftClamp(alpha=0.01)
        self.activation = nn.Sigmoid()
        self.padding = [0,0,0,0]

    def add_padding(self,padding):
        #padding is (left, right, top, bottom)
        device  =self.proto.device
        self.old_sprite_size = self.sprite_size
        new_sprite_size = (self.sprite_size[0] + padding[2] + padding[3], self.sprite_size[1] + padding[0] + padding[1])
        # Initialize the new padded prototype with zeros
        new_proto  = torch.zeros((self.n_outputs, 1) + tuple(new_sprite_size), device=device)
        new_proto[:, :, padding[2]:padding[2]+self.sprite_size[0], padding[0]:padding[0]+self.sprite_size[1]] = self.proto
        self.proto = nn.Parameter(new_proto)
        self.sprite_size = new_sprite_size
        self.mask = self.create_mask(new_sprite_size, self.border_size)
        self.padding = padding
        self.register_buffer('empty_sprite', torch.zeros((1, 1) + tuple(new_sprite_size), device=device))


    def create_mask(self, sprite_size, border_size):
        """Create a mask that zeros out the border and keeps the center learnable."""
        H, W = sprite_size
        mask = torch.ones((1, 1, H, W), device = self.proto.device)
        mask[:, :, :border_size, :] = 0  # Top border
        mask[:, :, -border_size:, :] = 0  # Bottom border
        mask[:, :, :, :border_size] = 0  # Left border
        mask[:, :, :, -border_size:] = 0  # Right border
  
        return mask

    def forward(self, space_index=None, mask_sprite=False):
        proto = self.proto

        proto = torch.cat([self.empty_sprite, proto], dim=0)

        if space_index is not None:
            proto[space_index + 1] = self.empty_sprite
        proto = self.soft_clamp(proto)
        if mask_sprite:
            proto = proto * self.mask.to(self.proto.device)

        return proto  # size (K,1,H,W)


class Background_Color(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=256, out_channels=3):
        super(Background_Color, self).__init__()

        # Convolutional layers for spatial feature processing
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU()

        # Final convolution for color prediction
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding=0)

        # No global pool - will use dynamic pooling in forward pass

    def forward(self, activation_map, original_size, mask_resnet, true_mask_dims=None):
        """
        Convolutional background color prediction:
        1. Crop activation map according to mask
        2. Apply convolutions on cropped region
        3. Global average pool and predict background color
        """
        batch_size = activation_map.shape[0]
        color_maps = []

        for i in range(batch_size):
            # Extract single sample and its mask
            single_activation = activation_map[i : i + 1]  # Shape: [1, C, H, W]
            h_f, w_f = single_activation.shape[-2:]
            true_h, true_w, img_min_h, img_min_w, img_max_h, img_max_w = true_mask_dims[i]
            H, W = original_size
            r_h = true_h / H
            r_w = true_w / W
            y_max = int(np.floor(h_f * float(r_h)))
            x_max = int(np.floor(w_f * float(r_w)))

            cropped_activation = single_activation[:, :, 0:y_max, 0:x_max]

            # Ensure cropped_activation has minimum dimensions of 3x3 for kernel size 3
            if cropped_activation.shape[-2] < 3 or cropped_activation.shape[-1] < 3:
                # Interpolate cropped_activation to max(3, original_h), max(3, original_w)
                target_h = max(3, cropped_activation.shape[-2])
                target_w = max(3, cropped_activation.shape[-1])
                cropped_activation = F.interpolate(
                    cropped_activation,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=True,
                )
            # Apply convolutions
            hidden_map = self.relu(self.bn1(self.conv1(cropped_activation)))
            color_map = self.conv2(hidden_map)  # Shape: [1, 3, H_crop, W_crop]
            color_map = torch.sigmoid(color_map)

            kernel_w = min(8, color_map.shape[-1])
            kernel_size = (color_map.shape[-2], kernel_w)

            color_vector = F.max_pool2d(color_map, kernel_size)  # Shape: [1, 3, 1, 1]

            color_map_full = torch.zeros(1, 3, H, W, device=color_vector.device)

            color_map_crop = F.interpolate(
                color_vector, size=(true_h, true_w), mode="bilinear", align_corners=True
            )

            color_map_full[:, :, :true_h, :true_w] = color_map_crop
            color_maps.append(color_map_full)

        # Stack all color maps
        background_maps = torch.cat(color_maps, dim=0)  # Shape: [B, 3, H, W]

        return background_maps
