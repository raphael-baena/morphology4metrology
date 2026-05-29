import os
import torch, json
import torch.nn as nn
from datasets import build_dataset
from util.visualizer import renorm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from reconstruction.reconstructor import Reconstructor
from reconstruction.utils import (
    get_bboxes_scores,
    CTC_loss,
    save_reconstruction_visualization,
    sprite_size_from_checkpoint_path,
)
import matplotlib.pyplot as plt
import util.misc as utils
from torchvision.transforms import ToPILImage
from util.slconfig import load_model
from util.get_param_dicts import get_param_dict
from main_synthetic import build_model_main
from PIL import Image
import argparse
from tqdm import tqdm
import json
from PIL import Image
import glob
from collections import defaultdict
import shutil


parser = argparse.ArgumentParser()
parser.add_argument("--model_config_path", type=str, default="config/Latin_CTC.py", help="path to the model config file")
parser.add_argument("--model_checkpoint_path", type=str, default="/home/vlachoum/learnable-DTLR/checkpoint/checkpoint.pth", help="path to the model checkpoint file")
parser.add_argument("--reconstructor_path", type=str, default=None, help="path to the reconstructor checkpoint file")
parser.add_argument("--reconstructor_unfrozen_path", type=str, default=None, help="path to the unfrozen reconstructor checkpoint file")
parser.add_argument("--dataset_file", type=str, default="dataset", help="dataset module name (see datasets/__init__.py)")
parser.add_argument("--data_folder", type=str, default=None, help="path to the dataset")
parser.add_argument("--document", nargs='+', default=None, help="list of documents to process")
parser.add_argument("--documents", action="store_true", help="Automatically process all documents from annotation file")
parser.add_argument("--annotation_file", type=str, help="Path to annotation JSON file for document discovery")
parser.add_argument("--split", type=str, default="all", choices=["train", "all"], help="Split to use for document discovery")
parser.add_argument("--script", nargs='+', default=None, help="list of scripts to process")
parser.add_argument("--space_index", type=int, default=None, help="space index")
parser.add_argument("--output_dir", type=str, default=None, help="output directory")
parser.add_argument("--step", type=int, default=0, help="step number (0: frozen, 1: unfrozen)")
parser.add_argument("--resume", action="store_true", help="resume")
parser.add_argument("--wandb" , action="store_true", help="use wandb")
parser.add_argument("--tag", type=str, default=None, help="optional run name for wandb and output directory")
parser.add_argument("--no_weight_decay" , action="store_true", help="weight decay")
parser.add_argument("--max_e" , type=int, default=10, help="max epoch")
parser.add_argument("--skew", action="store_true", help="skew")
parser.add_argument("--mask", action="store_true", help="mask")
parser.add_argument("--loss", type=str, default="MSE", choices=["MSE", "L1"], help="reconstruction loss type (MSE or L1)")
parser.add_argument("--num_fine_classes", type = int, default = None, help = "number of fine classes, 2 means one class for character one for accent")
parser.add_argument("--bbox_only", action="store_true", help="bbox only")
parser.add_argument("--init", action="store_true", help="init")
parser.add_argument("--weight_loss_reconstruction", type=float, default=10.0, help="weight for the reconstruction loss")
parser.add_argument("--prototypes_only_path", type=str, default=None, help="path to the prototypes only model (used with step=2)")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
parser.add_argument("--num_sprites_per_letter", type=int, default=1, help="number of sprites per letter (default: 1)")
parser.add_argument(
    "--sprite_size",
    type=int,
    nargs="+",
    default=[32, 32],
    metavar=("H", "W"),
    help="Sprite size in pixels: one value for square (e.g. 32) or H W (default: 32 32)",
)
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for the reconstructor")
parser.add_argument("--learning_rate_background", type=float, default=1e-5, help="learning rate for background predictor in step 2")
parser.add_argument("--right_padding", type=int, default=0, help="right padding")
parser.add_argument("--left_padding", type=int, default=0, help="left padding")
parser.add_argument("--top_padding", type=int, default=0, help="top padding")
parser.add_argument("--bottom_padding", type=int, default=0, help="bottom padding")
parser.add_argument("--mask_sprite", action="store_true", help="mask reconstruction")
parser.add_argument("--old_data_augmentation", action="store_true", help="old data augmentation")
parser.add_argument(
    "--line_resize_h_ref",
    type=int,
    default=90,
    help="Line image resize target height (no old_data_augmentation)",
)
parser.add_argument(
    "--line_resize_max_width",
    type=int,
    default=1400,
    help="Line image max width before width compression",
)
parser.add_argument("--composite_mode", type=str, default="batch", choices=["sequential", "batch", "additive"],
    help="Composite mode: 'sequential' (random order per query), 'batch' (modulo 4 groups), 'additive'")

args = parser.parse_args()
if len(args.sprite_size) == 1:
    args.sprite_size = (args.sprite_size[0], args.sprite_size[0])
elif len(args.sprite_size) == 2:
    args.sprite_size = (args.sprite_size[0], args.sprite_size[1])
else:
    raise ValueError("--sprite_size expects 1 or 2 integers (H, or H W)")

padding = [ args.left_padding, args.right_padding, args.top_padding, args.bottom_padding]
if sum(padding) != 0 and args.step != 2:
    raise ValueError("Padding is only supported for step 2")


def resolve_sprite_size_for_training(args, model_folder):
    """Use checkpoint proto shape when loading weights; else --sprite_size."""
    candidates = []
    if args.step == 2:
        if args.resume:
            candidates.append(os.path.join(model_folder, "reconstructor.pth"))
        if args.prototypes_only_path:
            candidates.append(args.prototypes_only_path)
    if args.step == 1:
        if args.reconstructor_unfrozen_path:
            candidates.append(args.reconstructor_unfrozen_path)
        if args.reconstructor_path:
            candidates.append(args.reconstructor_path)
    if args.resume and args.step in (0, 1):
        candidates.append(os.path.join(model_folder, "reconstructor_unfrozen.pth"))
        candidates.append(os.path.join(model_folder, "reconstructor.pth"))
    if args.step == 0 and args.reconstructor_path:
        candidates.append(args.reconstructor_path)

    for path in candidates:
        if path and os.path.isfile(path):
            try:
                size = sprite_size_from_checkpoint_path(path)
                print(f"Sprite size {size} inferred from checkpoint: {path}")
                return size
            except (ValueError, KeyError, OSError) as exc:
                print(f"Could not read sprite size from {path}: {exc}")
    print(f"Sprite size from --sprite_size: {args.sprite_size}")
    return args.sprite_size

def get_documents(path, split=None, filter=None):
    with open(path, 'r') as f:
        annotation = json.load(f)

    document_keys = ['_'.join(k.split('_')[:2]) for k, v in annotation.items() if split == 'all' or v.get('split') == split]
    split_dict = defaultdict(set)
    for k, v in annotation.items():
        if v['split'] == split or split == 'all':
            doc_key = '_'.join(k.split('_')[:2])
            split_dict[doc_key].add(v['split'])

    for k, v in split_dict.items():
        split_dict[k] = 'all' if len(v) == 2 else list(v)[0]

    documents = sorted(set(document_keys))  # Ensure consistent order
    return list(zip(documents, [split_dict[doc] for doc in documents]))

def save_sprites_and_mapping(reconstructor, charset, results_folder, epoch=None, step=None, max_e=None, num_sprites_per_letter=1, sprite_bbox_stats=None, sprite_selection_stats=None):
    """
    Save individual sprites and create a JSON mapping file.
    Args:
        reconstructor: The reconstructor model
        charset: List of characters
        results_folder: Folder to save results
        epoch: Current epoch number
        step: Training step number
        max_e: Maximum number of epochs
        num_sprites_per_letter: Number of sprites per character
    """
    # Get sprites from reconstructor
    proto_pixel = reconstructor.generator()

    # Skip the first sprite (index 0) which corresponds to empty
    sprites = proto_pixel[1:]  # Remove first sprite

    
    # Create mapping dictionary with bbox statistics
    sprite_mapping = {}
    
    # Get total number of sprites (excluding sprite 0 which is empty)
    total_sprites = len(charset) * num_sprites_per_letter
    
    for sprite_idx in range(1, total_sprites + 1):  # Start from 1 (skip sprite 0)
        # Calculate character index using modulo operation (same as grid creation)
        char_idx = sprite_idx % len(charset)
        char = charset[char_idx]
        
        # Get bbox stats
        if sprite_bbox_stats and sprite_idx in sprite_bbox_stats:
            bbox_stats = sprite_bbox_stats[sprite_idx]
        else:
            bbox_stats = {
                "mean_width": 0.0,
                "mean_height": 0.0,
                "var_width": 0.0,
                "var_height": 0.0,
                "count": 0
            }
        
        # Get sprite selection stats for this character
        if sprite_selection_stats and char_idx in sprite_selection_stats:
            selection_stats = sprite_selection_stats[char_idx]
        else:
            selection_stats = {
                "sprite_0_ratio": 0.0,
                "sprite_1_ratio": 0.0,
                "sprite_0_count": 0,
                "sprite_1_count": 0,
                "total_count": 0
            }
        
        sprite_mapping[sprite_idx] = {
            "character": char,
            "bbox_stats": bbox_stats,
            "sprite_selection_stats": selection_stats
        }
    
    # Save mapping to JSON
    mapping_file = os.path.join(item_output_dir, "transcribe.json")
    with open(mapping_file, 'w') as f:
        json.dump(sprite_mapping, f, indent=2, ensure_ascii=False)
    
    # Determine if we should save sprites (every 5 epochs or at the end)
    should_save = False
    if epoch is None:
        step_folder = "current"
        should_save = True
    else:
        should_save = (epoch % 5 == 0) or (max_e is not None and epoch == max_e - 1)
        step_folder = f"{epoch:03d}"
    if should_save:
        # Create step-specific folder
        step_sprites_folder = os.path.join(results_folder, step_folder)
        if not os.path.exists(step_sprites_folder):
            os.makedirs(step_sprites_folder)
        
        # Save the combined grid for this epoch and get transformed sprites
        from reconstruction.utils import create_manual_grid
        # Create a version of create_manual_grid that works with already filtered sprites
        grid_with_labels, transformed_sprites = create_manual_grid(sprites, charset, num_sprites_per_letter, sprite_bbox_stats)
        grid_with_labels.save(os.path.join(step_sprites_folder, "grid.jpg"))
        
        # Save individual sprites (using transformed version from the grid)
        for i, sprite_img in enumerate(transformed_sprites):
            # Save sprite with index only
            sprite_file = os.path.join(step_sprites_folder, f"{i}.png")
            sprite_img.save(sprite_file)
        
        # Save individual sprite grids if multiple sprites per letter
        if num_sprites_per_letter > 1:
            from reconstruction.utils import create_individual_sprite_grids
            individual_grids = create_individual_sprite_grids(sprites, charset, num_sprites_per_letter, sprite_bbox_stats)
            for i, grid in enumerate(individual_grids):
                grid.save(os.path.join(step_sprites_folder, f"grid_sprite_{i}.jpg"))

def create_grid_gif(results_folder, step=None):
    """
    Create a GIF showing grid evolution across all epochs.
    Args:
        results_folder: Folder containing the results
        step: Training step number to determine correct grid folder
    """
    try:
        # Find all grid images from step folders (000, 025, 050, etc.)
        grid_pattern = os.path.join(results_folder, "*", "grid.jpg")
        grid_files = glob.glob(grid_pattern)
        
        if not grid_files:
            print("No grid files found for GIF creation")
            return
        
        # Sort files by folder name (epoch number)
        grid_files.sort(key=lambda x: os.path.basename(os.path.dirname(x)))
        
        # Load images
        images = []
        for grid_file in grid_files:
            img = Image.open(grid_file)
            images.append(img)
        
        # Create GIF
        gif_path = os.path.join(results_folder, "grid.gif")
        if images:
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=1000,  # 1 second per frame
                loop=0
            )
        else:
            print("No images found for GIF creation")
            
    except Exception as e:
        print(f"⚠️ Error creating grid GIF: {e}")

def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors
# wandb will be initialized inside the loop for each document/script


if not os.path.exists("logs_reconstruction"):
    os.makedirs("logs_reconstruction")

if args.output_dir is None:
    args.output_dir = f"logs_reconstruction/{args.dataset_file}"
if args.tag is not None:
    args.output_dir = f"logs_reconstruction/{args.tag}"
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# --- Baseline copy logic for step 2 (finetuning) ---
if args.step == 2:
    # Get pretrain folder from model_checkpoint_path
    pretrain_folder = os.path.dirname(args.model_checkpoint_path)
    pretrain_sprites_folder = os.path.join(pretrain_folder, "sprites")
    finetune_baseline_folder = os.path.join(args.output_dir, "baseline")

    if os.path.exists(pretrain_sprites_folder):
        if not os.path.exists(finetune_baseline_folder):
            os.makedirs(finetune_baseline_folder)
        # Copy all PNGs from pretrain sprites to baseline
        for file in os.listdir(pretrain_sprites_folder):
            if file.endswith(".png"):
                shutil.copy2(
                    os.path.join(pretrain_sprites_folder, file),
                    os.path.join(finetune_baseline_folder, file)
                )
        # Copy mapping file if it exists (sprite_mapping.json or transcribe.json)
        for mapping_name in ["sprite_mapping.json", "transcribe.json"]:
            mapping_path = os.path.join(pretrain_folder, mapping_name)
            if os.path.exists(mapping_path):
                shutil.copy2(mapping_path, os.path.join(finetune_baseline_folder, mapping_name))
        print(f"Copied baseline sprites and mapping from {pretrain_sprites_folder} to {finetune_baseline_folder}")
    else:
        print(f"Warning: Pretrain sprites folder {pretrain_sprites_folder} does not exist.")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device

# Initialize loss function based on argument
if args.loss == "MSE":
    reconstruction_loss = nn.MSELoss(reduction="none")
else:  # L1
    reconstruction_loss = nn.L1Loss(reduction="none")

# Check that user is not passing both document and script
# Validate that only one mode is selected
if sum([args.documents, args.document is not None, args.script is not None]) > 1:
    raise ValueError("Use only one of --documents, --document, or --script")

# Automatic document discovery
if args.documents:
    if not args.annotation_file:
        raise ValueError("You must provide --annotation_file when using --documents")
    discovered = get_documents(args.annotation_file, args.split)
    items = [doc for doc, _ in discovered]
    item_type = "document"
elif args.document is not None:
    items = args.document
    item_type = "document"
elif args.script is not None:
    items = args.script
    item_type = "script"
else:
    items = [None]
    item_type = "all"

# Show what will be processed first
print(f"\n{'='*60}")
if len(items) > 1:
    print(f"📋 ITEMS TO PROCESS ({len(items)} total):")
    for i, item_name in enumerate(items, 1):
        print(f"   {i}. {item_type}: {item_name}")
elif items[0] is not None:
    print(f"📋 PROCESSING: Single {item_type} '{items[0]}'")
else:
    print(f"📋 PROCESSING: Full dataset '{args.dataset_file}'")
print(f"{'='*60}")
print(f"🎯 TRAINING STEP: {args.step}")
# Print step information
if args.step == 0:
    print("📚 STEP 0: Learning a good initialization for the prototypes and classification.")
    print("   • The bounding boxes are frozen")
    print("   • Only prototypes and classification parameters are optimized")
    print("   • This stage give a good initialization")
elif args.step == 1:
    print("🔄 STEP 1: Optimizing prototypes with unfrozen model.")
    print("   • All parameters are optimized including bounding boxes")
    print("   • This stage should lead to a good reconstruction")
elif args.step == 2:
    print("🎨 STEP 2: Optimizing prototypes only.")
    print("   • At this stage, bounding boxes are no longer frozen")
    print("   • Only prototype parameters are optimized")
    print("   • Use this stage to learn cleaner prototype or finetuning")
else:
    print(f"❓ STEP {args.step}: Unknown step configuration")
    
for item_idx, item in enumerate(items):
    print(f"\n{'='*60}")

    model_path = os.path.join(args.output_dir, *( ["model.pth"] if item is None else [item, "model.pth"]))
    
    # For finetuning (step 2), skip if model.pth exists in the item's output directory
    # Unless we explicitly want to resume training.
    if args.step == 2 and os.path.isfile(model_path) and not args.resume:
        print(f"✅ Skipping already processed document: {item} (model.pth exists)")
        continue

    if item is not None:
        print(f"📄 Processing: {item_type} '{item}'")
    else:
        print(f"📊 Processing: full dataset '{args.data_folder}'")
    print(f"{'='*60}")

    print(f"{'='*60}")
    
    print(f"\n=== Processing {item_type}: {item} ===")
    
    # Initialize wandb for this specific item
    if args.wandb:
        import wandb
        run_name = item if item is not None else args.tag #"all_data"
        wandb.init(
            project='DTLR-reconstruction',
            name=run_name,
            config=vars(args),
            tags=[item_type, run_name] if item is not None else [item_type]
        )
    
    # Create item-specific folder structure
    if item is not None:
        # For specific documents/scripts, create subfolder
        item_output_dir = os.path.join(args.output_dir, item)
    else:
        # For full dataset, save directly in dataset folder
        item_output_dir = args.output_dir
    
    if not os.path.exists(item_output_dir):
        os.makedirs(item_output_dir)
    
    # Create model folder for checkpoints
    model_folder = os.path.join(item_output_dir)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    # Create results folder for sprites and mapping
    results_folder = os.path.join(item_output_dir, "sprites")
    
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Create dataset for this item (document or script)
    if item_type == "document":
        args.document = item
        args.script = None
    elif item_type == "script":
        args.script = item
        args.document = None
    else:
        args.document = None
        args.script = None

    dataset_train = build_dataset(image_set='train', args=args)
    charset = dataset_train.charset

    # Load model for this specific document/charset.
    # Important: for step-2 resume, we must load the local per-subfolder `model.pth`,
    # otherwise we'd keep using the global `--model_checkpoint_path` for every folder.
    model_checkpoint_path_for_item = args.model_checkpoint_path
    if args.step == 2 and args.resume:
        local_model_path = os.path.join(model_folder, "model.pth")
        if os.path.exists(local_model_path):
            model_checkpoint_path_for_item = local_model_path
            print(f"Step 2 resume: loading local model from {local_model_path}")
        else:
            print(f"Step 2 resume: local model not found at {local_model_path}, using {args.model_checkpoint_path}")

    model, args_model = load_model(
        args.model_config_path,
        device,
        model_checkpoint_path_for_item,
        build_model_main,
        charset,
        expand_bbox=args.skew,
        resume=args.resume,
        num_fine_classes=args.num_fine_classes,
        dataset_train=dataset_train,
        init=args.init,
        num_sprites_per_letter=args.num_sprites_per_letter,
    )

    model.eval()

    model = model.to(device)

    # Create reconstructor for this specific document/charset
    sprite_size = resolve_sprite_size_for_training(args, model_folder)
    reconstructor = Reconstructor(
        n_outputs=len(dataset_train.charset) * args.num_sprites_per_letter,
        sprite_size=sprite_size,
        mask_sprite=args.mask_sprite,
        composite_mode=args.composite_mode,
        init_zeros=(args.step == 2)
    ).to(device)
    lr = args.learning_rate

    # Load existing reconstructor if resuming
    start_epoch = 0
    
    # Case 1: Resume training (files should be in model_folder)
    if args.resume and args.step in [0,1]:
        reconstructor_path = os.path.join(model_folder, "reconstructor.pth")
        print('Reconstruction path is:' + reconstructor_path)
        if os.path.exists(reconstructor_path):
            checkpoint = torch.load(reconstructor_path, weights_only=True)

            if isinstance(checkpoint, dict) and 'weights' in checkpoint:
                # New format with weights and epoch
                reconstructor.load_state_dict(checkpoint['weights'])
                start_epoch = checkpoint.get('epoch', 0)
                print(f"Resuming from epoch {start_epoch}")
            else:
                # Old format - just weights
                reconstructor.load_state_dict(checkpoint)
                print("Loaded old format checkpoint")

        if args.step == 1:
            model_path = os.path.join(model_folder, "model.pth")
            print('model_path is:' + model_path)
            if os.path.exists(model_path):
                model, args_model = load_model(args.model_config_path, device, model_path, build_model_main, charset, expand_bbox=args.skew, num_fine_classes=args.num_fine_classes, init=args.init, num_sprites_per_letter=args.num_sprites_per_letter)
            model.eval()
            model = model.to(device)
            reconstructor_unfrozen_path = os.path.join(model_folder, "reconstructor_unfrozen.pth")
            if os.path.exists(reconstructor_unfrozen_path):
                checkpoint = torch.load(reconstructor_unfrozen_path, weights_only=True)
                if isinstance(checkpoint, dict) and 'weights' in checkpoint:
                    # New format with weights and epoch
                    reconstructor.load_state_dict(checkpoint['weights'])
                    start_epoch = checkpoint.get('epoch', 0)
                else:
                    # Old format - just weights
                    reconstructor.load_state_dict(checkpoint)
            print('resume and step 1')
        
    
    # Case 2: Step 1 without resume (load from specific paths if provided)
    elif args.step == 1:
        # Use specific paths if provided, otherwise use model_folder
        reconstructor_path = args.reconstructor_path if args.reconstructor_path else os.path.join(model_folder, "reconstructor.pth")
        if os.path.exists(reconstructor_path):
            checkpoint = torch.load(reconstructor_path, weights_only=True)
            if isinstance(checkpoint, dict) and 'weights' in checkpoint:
                # New format with weights and epoch
                reconstructor.load_state_dict(checkpoint['weights'])
                print(f"Loaded reconstructor from {reconstructor_path}")
            else:
                # Old format - just weights
                reconstructor.load_state_dict(checkpoint)
                print(f"Loaded old format reconstructor from {reconstructor_path}")
        else:
            raise ValueError(f"Reconstructor path {reconstructor_path} does not exist")

        model_path = args.model_checkpoint_path if args.model_checkpoint_path else os.path.join(model_folder, "model.pth")
        if os.path.exists(model_path):
            model, args_model = load_model(args.model_config_path, device, model_path, build_model_main, charset, expand_bbox=args.skew, num_fine_classes=args.num_fine_classes, init=args.init, num_sprites_per_letter=args.num_sprites_per_letter)
            model.eval()
            model = model.to(device)
            print(f"Loaded model from {model_path}")

        # Only load unfrozen reconstructor if resuming or explicitly provided
        loaded_unfrozen = False
        if args.resume:
            local_unfrozen_path = os.path.join(model_folder, "reconstructor_unfrozen.pth")
            if os.path.exists(local_unfrozen_path):
                checkpoint = torch.load(local_unfrozen_path, weights_only=True)
                if isinstance(checkpoint, dict) and 'weights' in checkpoint:
                    reconstructor.load_state_dict(checkpoint['weights'])
                    start_epoch = checkpoint.get('epoch', 0)
                    print(f"Loaded unfrozen reconstructor from {local_unfrozen_path}")
                else:
                    reconstructor.load_state_dict(checkpoint)
                    print(f"Loaded old format unfrozen reconstructor from {local_unfrozen_path}")
                loaded_unfrozen = True
        elif args.reconstructor_unfrozen_path:
            reconstructor_unfrozen_path = args.reconstructor_unfrozen_path
            if os.path.exists(reconstructor_unfrozen_path):
                checkpoint = torch.load(reconstructor_unfrozen_path, weights_only=True)
                if isinstance(checkpoint, dict) and 'weights' in checkpoint:
                    reconstructor.load_state_dict(checkpoint['weights'])
                    start_epoch = checkpoint.get('epoch', 0)
                    print(f"Loaded unfrozen reconstructor from {reconstructor_unfrozen_path}")
                else:
                    reconstructor.load_state_dict(checkpoint)
                    print(f"Loaded old format unfrozen reconstructor from {reconstructor_unfrozen_path}")
                loaded_unfrozen = True
        if not loaded_unfrozen:
            print("No unfrozen reconstructor loaded; using base reconstructor weights.")

        param_dicts = get_param_dict(args_model, model)
        if args.no_weight_decay:
            optimizer_DTLR = torch.optim.AdamW(
                param_dicts, lr=args_model.lr, weight_decay=0
            )
        else:
            optimizer_DTLR = torch.optim.AdamW(
                param_dicts, lr=args_model.lr, weight_decay=args_model.weight_decay
            )
    if args.bbox_only:
        # For bbox_only, use specific path if provided, otherwise use model_folder
        model_bbox_path = args.model_checkpoint_path if args.model_checkpoint_path else os.path.join(model_folder, "model_bbox_only.pth")
        if os.path.exists(model_bbox_path):
            model.load_state_dict(torch.load(model_bbox_path, weights_only=True))
            print(f"Loaded bbox model from {model_bbox_path}")
    else:
        param_dicts = get_param_dict(args_model, model)
        optimizer_DTLR = torch.optim.AdamW(
            param_dicts, lr=args_model.lr, weight_decay=args_model.weight_decay
        )
    model.train() 

    if args.step == 2:
        # In step 2, avoid requiring a local reconstructor checkpoint.
        # Prefer prototypes_only_path if provided; if --resume is set and a local reconstructor exists, use it; otherwise skip.
        proto_ckpt = None
        proto_ckpt_path = None
        local_proto_path = None
        if args.resume:
            local_proto_path = os.path.join(model_folder, "reconstructor.pth")
            if os.path.exists(local_proto_path):
                proto_ckpt_path = local_proto_path
        if proto_ckpt_path is None and args.prototypes_only_path is not None:
            proto_ckpt_path = args.prototypes_only_path
        if proto_ckpt_path is not None and os.path.exists(proto_ckpt_path):
            proto_ckpt = torch.load(proto_ckpt_path, map_location='cpu')
            if isinstance(proto_ckpt, dict) and 'weights' in proto_ckpt:
                reconstructor.load_state_dict(proto_ckpt['weights'])
                # If we're resuming from a previous step-2 reconstructor checkpoint,
                # continue from the next epoch (avoid re-running the last saved epoch).
                if args.resume and proto_ckpt_path == local_proto_path:
                    start_epoch = int(proto_ckpt.get("epoch", start_epoch)) + 1
            else:
                reconstructor.load_state_dict(proto_ckpt)
            if sum(padding) != 0:
                reconstructor.add_padding(padding)
                print(f"Padding added to prototypes: {padding}")
            print(f"Prototypes loaded for step 2 from {proto_ckpt_path}")
        else:
            print("No prototype checkpoint provided or found for step 2; proceeding without loading prototypes.")

    # Step 2: freeze only character color predictor; background uses lower LR (learning_rate_background)
    if args.step == 2:
        for p in reconstructor.color_predictor.parameters():
            p.requires_grad = False
        print("Frozen character color predictor for step 2; background predictor uses LR={}.".format(args.learning_rate_background))

    # Setup optimizers
    def _reconstructor_param_groups_step2(reconstructor, lr_prototypes, lr_background):
        """For step 2: two groups — prototypes (and rest) at lr_prototypes, background_color at lr_background. color_predictor is frozen."""
        frozen_ids = {id(p) for p in reconstructor.color_predictor.parameters()}
        main_params = [p for p in reconstructor.parameters() if id(p) not in frozen_ids and p.requires_grad]
        bg_params = list(reconstructor.background_color.parameters())
        # Params in main_params that are background_color get their own group
        bg_ids = {id(p) for p in bg_params}
        main_params = [p for p in main_params if id(p) not in bg_ids]
        return [
            {"params": main_params, "lr": lr_prototypes},
            {"params": bg_params, "lr": lr_background},
        ]

    if args.bbox_only:
        parameters_to_optimize = (
            list(model.transformer.decoder.bbox_embed.parameters())
            + list(model.transformer.enc_out_bbox_embed.parameters())
            + list(model.bbox_embed.parameters()))
        optimizer_DTLR = torch.optim.AdamW(
            parameters_to_optimize, lr=args_model.lr, weight_decay=args_model.weight_decay
        )
        
        if args.step == 2:
            param_groups = _reconstructor_param_groups_step2(reconstructor, lr, args.learning_rate_background)
            optimizer = torch.optim.AdamW(param_groups, weight_decay=0)
        elif args.step == 1:
            optimizer = torch.optim.AdamW(reconstructor.parameters(), lr=lr, weight_decay=0)
    else:
        # Collect all parameters and remove duplicates; for step 2 use two LRs (prototypes vs background)
        if args.step == 2:
            recon_groups = _reconstructor_param_groups_step2(reconstructor, lr, args.learning_rate_background)
            other_params = list(model.class_embed.parameters()) + list(model.transformer.decoder.class_embed.parameters()) + list(model.transformer.enc_out_class_embed.parameters())
            seen = set()
            unique_other = []
            for param in other_params:
                if param not in seen:
                    seen.add(param)
                    unique_other.append(param)
            param_groups = recon_groups + [{"params": unique_other, "lr": lr}]
            optimizer = torch.optim.AdamW(param_groups, weight_decay=0)
        else:
            all_params = list(reconstructor.parameters()) + list(model.class_embed.parameters()) + list(model.transformer.decoder.class_embed.parameters()) + list(model.transformer.enc_out_class_embed.parameters())
            seen = set()
            unique_params = []
            for param in all_params:
                if param not in seen:
                    seen.add(param)
                    unique_params.append(param)
            optimizer = torch.optim.AdamW(unique_params, lr=lr, weight_decay=0)

    e = start_epoch
    max_e = args.max_e

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )
    train_loader =  DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=4,
    )

    max_iterations = 1000

    # Create progress bar for all epochs
    total_batches = len(train_loader) * (max_e - start_epoch + 1)
    current_batch = 0
    
    # Create a single progress bar for all epochs
    if item is not None:
        progress_desc = f"{item_type}: {item}"
    else:
        progress_desc = f"dataset: {args.dataset_file}"
    
    pbar = tqdm(total=total_batches, desc=progress_desc, leave=True)
    
    # Initialize metrics file - erase if not resuming
    metrics_file = os.path.join(item_output_dir, f"metrics_step_{args.step}.txt")
    if not args.resume and os.path.exists(metrics_file):
        os.remove(metrics_file)
        print(f"🗑️ Erased existing metrics file: {metrics_file}")
    
    for e in range(start_epoch, max_e + 1):
        # Initialize epoch metrics
        epoch_loss = 0.0
        epoch_loss_reco = 0.0
        epoch_loss_ctc = 0.0
        num_batches = 0
        
        # Initialize bbox statistics collection for this epoch
        bbox_stats = {}  # Dictionary to collect bbox statistics per character
        sprite_counts = {}  # Dictionary to collect sprite counts per character: {char_idx: {sprite_idx: count}}
        grad_norm = 0.0
        max_grad_norm = 0.0
        max_grad_norm_after = 0.0
        grad_norm_after = 0.0
        
        for exemple, (samples, target) in enumerate(train_loader):
            # try:

            image,mask  = samples.decompose()
            mask  = mask.to(device)


            bboxes, scores,features_decoder,features_resnet,mask_resnet =  get_bboxes_scores(samples.to(device), model, num_fine_classes = args.num_fine_classes)
            # Filter out boxes with cx=0 (problematic boxes)
            # bboxes format: [batch, num_queries, 4] where 4 is [cx, cy, w, h]
            cx = bboxes[:, :, 0]  # Get cx coordinates
            cy = bboxes[:, :, 1]  # Also check cy
            valid_mask = (torch.abs(cx) > 1e-3) & (torch.abs(cy) > 1e-3)  # Keep boxes where cx and cy are not close to 0
            

            
            invalid_mask = ~valid_mask
            
            # Set scores: blank class (index 0) gets probability 1, others get 0
            if invalid_mask.any():
                scores = scores.clone()  # Avoid in-place modification
                scores[invalid_mask][:,0] = 1.0
                scores[invalid_mask][:,1:] = 1e-6
            bboxes = bboxes.clamp(min=1.)
            renorm_images = [ renorm(image[i]).cuda()  for i in range(image.shape[0])]

        
            list_image = [image[i].cuda() for i in range(image.shape[0])]

            list_reco_image = reconstructor(renorm_images,scores,bboxes,  features_decoder,features_resnet,space_index = args.space_index, mask_resnet = mask_resnet, true_mask = mask) 
            
            # Collect bbox statistics for each sprite (except sprite 0)
            batch_scores = scores.detach().cpu().numpy()  # Shape: (batch_size, num_queries, num_classes)
            batch_bboxes = bboxes.detach().cpu().numpy()  # Shape: (batch_size, num_queries, 4)
            
            # Collect sprite selection statistics
            if args.num_sprites_per_letter > 1:
                # Do the same operation as CTC_loss but collect which sprite was selected
                batch_size, num_queries, num_classes = batch_scores.shape
                # The scores include the blank token (class 0), so we need to exclude it
                scores_no_blank = batch_scores[:, :, 1:]  # Remove blank token
                num_chars = (num_classes - 1) // args.num_sprites_per_letter
                
                # Reshape to group sprites by character (same as CTC function)
                scores_reshaped = scores_no_blank.reshape(batch_size, num_queries, num_chars, args.num_sprites_per_letter)
                
                # Get which sprite was selected for each character (argmax)
                selected_sprites = scores_reshaped.argmax(axis=3)  # Shape: (batch_size, num_queries, num_chars)
                
                # Collect sprite selection counts for each character
                for batch_idx in range(batch_size):
                    for query_idx in range(num_queries):
                        for char_idx in range(num_chars):
                            # Get which sprite was selected for this character
                            selected_sprite_idx = selected_sprites[batch_idx, query_idx, char_idx]
                            
                            # Initialize character dict if not exists
                            if char_idx not in sprite_counts:
                                sprite_counts[char_idx] = {}
                            
                            # Count this sprite selection
                            if selected_sprite_idx not in sprite_counts[char_idx]:
                                sprite_counts[char_idx][selected_sprite_idx] = 0
                            sprite_counts[char_idx][selected_sprite_idx] += 1
            
            for batch_idx in range(batch_scores.shape[0]):
                batch_score = batch_scores[batch_idx]  # Shape: (num_queries, num_classes)
                batch_bbox = batch_bboxes[batch_idx]   # Shape: (num_queries, 4)
                
                # Get predicted sprite for each query
                predicted_sprites = np.argmax(batch_score, axis=1)  # Shape: (num_queries,)
                
                for query_idx, pred_sprite in enumerate(predicted_sprites):
                    if pred_sprite == 0:  # Skip empty/background sprite
                        continue
                    pred_sprite = pred_sprite - 1
                    # Get bbox dimensions for this sprite
                    bbox = batch_bbox[query_idx]  # Shape: (4,) - [x1, y1, x2, y2] or [cx, cy, w, h]
                    
                    # Calculate width and height
                    # Bboxes are in [cx, cy, w, h] format after processing
                    width = bbox[2]  # w
                    height = bbox[3]  # h
                    
                    # Initialize sprite statistics if not exists
                    if pred_sprite not in bbox_stats:
                        bbox_stats[pred_sprite] = {
                            'widths': [],
                            'heights': []
                        }
                    
                    # Add measurements
                    bbox_stats[pred_sprite]['widths'].append(width)
                    bbox_stats[pred_sprite]['heights'].append(height)
            
            loss = 0
            reco = list_reco_image
            renorm_images = torch.stack(renorm_images)

            loss = (reconstruction_loss(renorm_images,reco) * (1-mask.long()).unsqueeze(1))#.mean()
            loss = loss.mean() * args.weight_loss_reconstruction

            loss_ctc = CTC_loss(scores, target,num_fines_classes = args.num_fine_classes, device = device, num_sprites_per_letter = args.num_sprites_per_letter)
            loss_reco = loss.clone()
            loss += loss_ctc
                
            # Update epoch metrics
            epoch_loss += loss.item()
            epoch_loss_reco += loss_reco.item()
            epoch_loss_ctc += loss_ctc.item()
            num_batches += 1
            if args.wandb:
                wandb.log({"loss_ctc":loss_ctc.item()})
                wandb.log({"loss_reconstruction":loss_reco.item()})
                wandb.log({"loss":loss.item()})
                wandb.log({"example":exemple})
                
            # Update progress bar with current losses
            avg_loss = epoch_loss / num_batches
            avg_loss_reco = epoch_loss_reco / num_batches
            avg_loss_ctc = epoch_loss_ctc / num_batches
            
            # Update global progress
            current_batch += 1
            
            # Update tqdm progress bar
            pbar.set_postfix({
                'Epoch': f'{e}/{max_e}',
                'Loss': f'{avg_loss:.4f}',
                'Reco': f'{avg_loss_reco:.4f}',
                'CTC': f'{avg_loss_ctc:.4f}',
                'Grad': f'{grad_norm:.4f}',
                'MaxGrad': f'{max_grad_norm:.4f}',
                'MaxGradAfter': f'{max_grad_norm_after:.6f}'
            })
            pbar.update(1)

            optimizer.zero_grad()
            if args.step == 1 or args.bbox_only:
                optimizer_DTLR.zero_grad()
            loss.backward()
            
            # DEBUG: Check gradients for NaNs/Infs, especially bbox-related parameters
            has_nan_grad = False
            has_inf_grad = False
            bbox_param_names = []
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        has_nan_grad = True
                        if 'bbox' in name.lower():
                            bbox_param_names.append(name)
                            print(f"🚨 NaN gradient in bbox param: {name}")
                            print(f"   grad stats: min={param.grad.min()}, max={param.grad.max()}, nan_count={torch.isnan(param.grad).sum()}")
                    if torch.isinf(param.grad).any():
                        has_inf_grad = True
                        if 'bbox' in name.lower():
                            bbox_param_names.append(name)
                            print(f"🚨 Inf gradient in bbox param: {name}")
                            print(f"   grad stats: min={param.grad.min()}, max={param.grad.max()}, inf_count={torch.isinf(param.grad).sum()}")
            
            # Also check reconstructor gradients
            for name, param in reconstructor.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        has_nan_grad = True
                        print(f"🚨 NaN gradient in reconstructor param: {name}")
                        print(f"   grad stats: min={param.grad.min()}, max={param.grad.max()}, nan_count={torch.isnan(param.grad).sum()}")
                    if torch.isinf(param.grad).any():
                        has_inf_grad = True
                        print(f"🚨 Inf gradient in reconstructor param: {name}")
                        print(f"   grad stats: min={param.grad.min()}, max={param.grad.max()}, inf_count={torch.isinf(param.grad).sum()}")
            
            if has_nan_grad or has_inf_grad:
                optimizer.zero_grad()
                if args.step == 1 or args.bbox_only:
                    optimizer_DTLR.zero_grad()
            else:
                # Only do optimizer step if gradients are clean
                if args_model.clip_max_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args_model.clip_max_norm).item()
                    
                    # Calculate norm AFTER clipping
                    total_norm_after = 0.0
                    count_params = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            count_params += 1
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm_after += param_norm.item() ** 2
                    grad_norm_after = total_norm_after ** 0.5
                    # print(f"DEBUG: Found {count_params} params with grad. Norm before: {grad_norm}, Norm after: {grad_norm_after}")
                else:
                    grad_norm = 0.0
                    grad_norm_after = 0.0
                
                if grad_norm > max_grad_norm:
                    max_grad_norm = grad_norm
                if grad_norm_after > max_grad_norm_after:
                    max_grad_norm_after = grad_norm_after

                optimizer.step()
                if (args.step == 1 or args.bbox_only) and args.step != 2:
                    optimizer_DTLR.step()


        
        if args.step != 1:
            # Save reconstructor with weights and epoch
            torch.save({
                'weights': reconstructor.state_dict(),
                'epoch': e
            }, os.path.join(model_folder, "reconstructor.pth"))
            torch.save(model.state_dict(),os.path.join(model_folder, "model.pth"))
        if args.step != 1 and args.bbox_only:
            torch.save(model.state_dict(),os.path.join(model_folder, "model_bbox_only.pth"))
        if args.step == 1:
            torch.save(model.state_dict(),os.path.join(model_folder, "model.pth"))
            # Save unfrozen reconstructor with weights and epoch
            torch.save({
                'weights': reconstructor.state_dict(),
                'epoch': e
            }, os.path.join(model_folder, "reconstructor_unfrozen.pth"))
            # Compute bbox statistics for this epoch
        sprite_bbox_stats = {}
        for sprite_idx, stats in bbox_stats.items():
            if len(stats['widths']) > 0:
                widths = np.array(stats['widths'])
                heights = np.array(stats['heights'])
                
                sprite_bbox_stats[sprite_idx] = {
                    'mean_width': float(np.mean(widths)),
                    'mean_height': float(np.mean(heights)),
                    'var_width': float(np.var(widths)),
                    'var_height': float(np.var(heights)),
                    'count': len(widths)
                }
            else:
                sprite_bbox_stats[sprite_idx] = {
                    'mean_width': 0.0,
                    'mean_height': 0.0,
                    'var_width': 0.0,
                    'var_height': 0.0,
                    'count': 0
                }
        
        # Compute sprite selection ratios for each character
        sprite_selection_stats = {}
        for char_idx, sprite_count_dict in sprite_counts.items():
            total_count = sum(sprite_count_dict.values())
            if total_count > 0:
                char_stats = {}
                for sprite_idx in range(args.num_sprites_per_letter):
                    count = sprite_count_dict.get(sprite_idx, 0)
                    ratio = count / total_count
                    char_stats[f'sprite_{sprite_idx}_ratio'] = float(ratio)
                    char_stats[f'sprite_{sprite_idx}_count'] = count
                char_stats['total_count'] = total_count
                sprite_selection_stats[char_idx] = char_stats
            else:
                sprite_selection_stats[char_idx] = {
                    'sprite_0_ratio': 0.0,
                    'sprite_1_ratio': 0.0,
                    'sprite_0_count': 0,
                    'sprite_1_count': 0,
                    'total_count': 0
                }
        
        # Save sprites and mapping to results folder at the end of each epoch
        save_sprites_and_mapping(reconstructor, charset, results_folder, e, args.step, args.max_e, args.num_sprites_per_letter, sprite_bbox_stats, sprite_selection_stats)
        
        # Save reconstruction images every 5 epochs or at the end
        should_save_reconstruction = False
        if e % 5 == 0 or e == args.max_e - 1:  # Save every 5 epochs or at the end
            should_save_reconstruction = True
            if e == args.max_e - 1:
                print(f"Saving final reconstruction images for epoch {e}...")
            else:
                print(f"Saving reconstruction images for epoch {e}...")
        
        if should_save_reconstruction:
            for i in range(5):
                image, target = dataset_train[i]
                # Create a temporary args object with the document-specific output directory
                temp_args = type('Args', (), {})()
                for attr in dir(args):
                    if not attr.startswith('_'):
                        setattr(temp_args, attr, getattr(args, attr))
                temp_args.output_dir = item_output_dir
                # Add unfrozen attribute for compatibility with save_reconstruction_visualization
                temp_args.unfrozen = (args.step == 1)
                save_reconstruction_visualization(image, mask, target, reconstructor, model, dataset_train.charset, temp_args, e, device, batch_idx=i)
        
        # Create grid evolution GIF
        create_grid_gif(results_folder, args.step)
        

        
        # Save training metrics to text file
        with open(metrics_file, 'a') as f:
            f.write(f"Epoch: {e}, Loss: {avg_loss:.6f}, Loss_Reco: {avg_loss_reco:.6f}, Loss_CTC: {avg_loss_ctc:.6f}\n")
        
        # Create/update loss graphs
        epochs = []
        losses = []
        losses_reco = []
        losses_ctc = []
        
        # Read all metrics from file
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split(', ')
                        epoch = int(parts[0].split(': ')[1])
                        loss = float(parts[1].split(': ')[1])
                        loss_reco = float(parts[2].split(': ')[1])
                        try:
                            loss_ctc_str = parts[3].split(': ')[1]
                            # Handle potential file corruption where next line merges (e.g. "0.078Epoch")
                            if 'Epoch' in loss_ctc_str:
                                loss_ctc_str = loss_ctc_str.split('Epoch')[0]
                            loss_ctc = float(loss_ctc_str)
                        except (ValueError, IndexError):
                            print(f"Skipping malformed metric line: {line.strip()}")
                            continue
                        
                        epochs.append(epoch)
                        losses.append(loss)
                        losses_reco.append(loss_reco)
                        losses_ctc.append(loss_ctc)
        
        # Create graphs
        if epochs:
            # Total Loss graph
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, losses, 'b-', linewidth=2, label='Total Loss', marker='o', markersize=4)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss - Step {args.step}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(epochs)  # Show all epoch numbers on x-axis
            plt.tight_layout()
            plt.savefig(os.path.join(results_folder, f'loss_step_{args.step}.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Reconstruction Loss graph
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, losses_reco, 'r-', linewidth=2, label='Reconstruction Loss', marker='o', markersize=4)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Reconstruction Loss - Step {args.step}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(epochs)  # Show all epoch numbers on x-axis
            plt.tight_layout()
            plt.savefig(os.path.join(results_folder, f'loss_reco_step_{args.step}.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # CTC Loss graph
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, losses_ctc, 'g-', linewidth=2, label='CTC Loss', marker='o', markersize=4)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'CTC Loss - Step {args.step}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(epochs)  # Show all epoch numbers on x-axis
            plt.tight_layout()
            plt.savefig(os.path.join(results_folder, f'loss_ctc_step_{args.step}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Create comprehensive summary figure
        if epochs:
            # Create a large figure with subplots
            fig = plt.figure(figsize=(20, 12))
            
            # Create grid layout: 2 rows, 3 columns (reconstruction spans top row, 3 losses in bottom row)
            gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
            
            # 1. Grid visualization (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            # Load and display the latest grid image
            if args.step in [0, 1]:
                grid_pattern = os.path.join(item_output_dir, f"figure_{args.step}", "grid", f"proto_pixel_grid_{e}.jpg")
            else:
                grid_pattern = os.path.join(item_output_dir, "figure", "grid", f"proto_pixel_grid_{e}.jpg")
            if os.path.exists(grid_pattern):
                grid_img = plt.imread(grid_pattern)
                ax1.imshow(grid_img)
                ax1.set_title(f'Prototype Grid - Epoch {e}', fontsize=12, fontweight='bold')
                ax1.axis('off')
            else:
                ax1.text(0.5, 0.5, 'Grid not available', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title(f'Prototype Grid - Epoch {e}', fontsize=12, fontweight='bold')
            
            # 2. Reconstruction examples (spans top middle and right)
            ax2 = fig.add_subplot(gs[0, 1:])
            # Load and display reconstruction examples from the last epoch
            if args.step in [0, 1]:
                recon_pattern = os.path.join(item_output_dir, f"figure_{args.step}", "reconstruction", f"reco_image_{e}_*.jpg")
            else:
                recon_pattern = os.path.join(item_output_dir, "figure", "reconstruction", f"reco_image_{e}_*.jpg")
            recon_files = glob.glob(recon_pattern)
            if recon_files:
                # Create a 2x2 subplot for the 4 reconstruction images
                ax2.remove()  # Remove the original subplot
                gs_recon = gs[0, 1:].subgridspec(2, 2, hspace=0.05, wspace=0.05)
                
                for i, recon_file in enumerate(sorted(recon_files)[:4]):  # Take first 4 files
                    row = i // 2
                    col = i % 2
                    ax_recon = fig.add_subplot(gs_recon[row, col])
                    recon_img = plt.imread(recon_file)
                    ax_recon.imshow(recon_img)
                    ax_recon.axis('off')
                    ax_recon.set_title(f'Reconstruction {i+1}', fontsize=10, fontweight='bold')
                
                # Add main title for reconstruction section
                fig.text(0.75, 0.75, f'Reconstruction Examples - Epoch {e}', fontsize=14, fontweight='bold', ha='center')
            else:
                ax2.text(0.5, 0.5, 'Reconstruction not available', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title(f'Reconstruction Examples - Epoch {e}', fontsize=12, fontweight='bold')
            
            # 3. Total Loss graph (bottom left)
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(epochs, losses, 'b-', linewidth=2, label='Total Loss', marker='o', markersize=4)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.set_title(f'Total Loss', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xticks(epochs)
            
            # 4. Reconstruction Loss graph (bottom middle)
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(epochs, losses_reco, 'r-', linewidth=2, label='Reconstruction Loss', marker='o', markersize=4)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title(f'Reconstruction Loss', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_xticks(epochs)
            
            # 5. CTC Loss graph (bottom right)
            ax5 = fig.add_subplot(gs[1, 2])
            ax5.plot(epochs, losses_ctc, 'g-', linewidth=2, label='CTC Loss', marker='o', markersize=4)
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Loss')
            ax5.set_title(f'CTC Loss', fontsize=12, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.set_xticks(epochs)
            
            # Add overall title
            fig.suptitle(f'Training Summary - Epoch {e}', fontsize=16, fontweight='bold', y=0.98)
            
            # Save the comprehensive summary
            plt.savefig(os.path.join(results_folder, f'summary.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
    if args.wandb:
        wandb.log({"epoch":e})
        # Finish the wandb run for this item
        wandb.finish()

    try:
        if os.path.exists("stop"):
            break
    except:
        pass

    # Close the progress bar
    pbar.close()

    print(f"\n=== Finished processing {item_type}: {item} ===")
