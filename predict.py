#!/usr/bin/env python3
"""
Prediction script for trained DTLR models.
Supports three mutually exclusive modes:
1. Multiple documents (--documents): iterate over models in subfolders
2. Single model (--single_model): use model.pth and reconstructor directly in path_models
3. Scripts (--script): iterate over a list of scripts (like reconstruction.py)
"""

import os
import torch
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import editdistance
import unicodedata

# Project imports
from datasets import build_dataset
from util.slconfig import load_model
from util.get_param_dicts import get_param_dict
from main_synthetic import build_model_main
from reconstruction.utils import get_bboxes_scores, sprite_size_from_checkpoint_path
from reconstruction.reconstructor import Reconstructor
import util.misc as utils
from torch.utils.data import DataLoader


def find_reconstructor_file(folder_path):
    """Find reconstructor checkpoint file (several possible names)."""
    possible_names = [
        "reconstructor_unfrozen.pth",
        "reconstructor.pth",
    ]
    
    for name in possible_names:
        path = os.path.join(folder_path, name)
        if os.path.exists(path):
            return path
    
    return None


def get_documents_from_annotation(path, split=None, filter=None):
    """Extract document list from annotation JSON file."""
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

    documents = sorted(set(document_keys))
    return list(zip(documents, [split_dict[doc] for doc in documents]))


def get_documents_from_models_folder(models_path):
    """Extract document list by scanning the models folder."""
    documents = []
    
    if not os.path.exists(models_path):
        raise ValueError(f"Models folder not found: {models_path}")
    
    for item in os.listdir(models_path):
        item_path = os.path.join(models_path, item)
        
        if os.path.isdir(item_path):
            model_path = os.path.join(item_path, "model.pth")
            reconstructor_path = find_reconstructor_file(item_path)
            
            if os.path.exists(model_path) and reconstructor_path:
                documents.append(item)
            else:
                print(f"WARNING: Skipping folder {item} (missing models)")
    
    return sorted(documents)


def compute_cer(predicted_text, ground_truth_text):
    """Compute Character Error Rate (CER)."""
    if len(ground_truth_text) == 0:
        return 1.0 if len(predicted_text) > 0 else 0.0
    
    distance = editdistance.eval(predicted_text, ground_truth_text)
    return distance / len(ground_truth_text)


def get_alignment_labels(ground_truth, prediction):
    """Compute alignment between ground truth and prediction."""
    m, n = len(ground_truth), len(prediction)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ground_truth[i - 1] == prediction[j - 1]:
                cost = 0
            else:
                cost = 1
            
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
            
    i, j = m, n
    labels = [None] * n
    
    while i > 0 or j > 0:
        cost = dp[i][j]
        
        if i > 0 and j > 0 and ground_truth[i - 1] == prediction[j - 1]:
            if cost == dp[i - 1][j - 1]:
                labels[j - 1] = 'match'
                i -= 1
                j -= 1
                continue
                
        if i > 0 and j > 0 and ground_truth[i - 1] != prediction[j - 1]:
            if cost == dp[i - 1][j - 1] + 1:
                labels[j - 1] = 'substitution'
                i -= 1
                j -= 1
                continue
                
        if j > 0 and cost == dp[i][j - 1] + 1:
            labels[j - 1] = 'insertion'
            j -= 1
            continue
            
        if i > 0 and cost == dp[i - 1][j] + 1:
            i -= 1
            continue
            
    return labels


def process_item(item, item_type, model_path, reconstructor_path, args, device):
    """Process a document, script, or full data_folder (item=None); return metrics."""
    
    print(f"\n{'='*60}")
    if item is None:
        print(f"Processing full dataset: {args.data_folder}")
    else:
        print(f"Processing {item_type}: {item}")
    print(f"{'='*60}")
    
    # Output dir: per-item subfolder, or output_dir root for full dataset
    if item is None:
        doc_output_dir = args.output_dir
    else:
        doc_output_dir = os.path.join(args.output_dir, item)
    if not os.path.exists(doc_output_dir):
        os.makedirs(doc_output_dir)
    
    # Dataset configuration for this item
    if item_type == "document":
        args.document = item
        args.script = None
    elif item_type == "script":
        args.script = item
        args.document = None
    else:
        args.document = None
        args.script = None
    
    # Build dataset
    try:
        # Preserve the line aspect ratio when generating bbox statistics / transcribe.json.
        # This avoids the width-compression behavior of ResizeToFixedHeightAndMaxWidth.
        args.preserve_line_aspect_ratio = True
        dataset = build_dataset(image_set='train', args=args)
        charset = dataset.charset
        print(f"Dataset built: {len(dataset)} examples, charset: {len(charset)} characters")
    except Exception as e:
        print(f"ERROR: Failed to build dataset: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Load detection model
    try:
        model, args_model = load_model(
            args.model_config_path, 
            device, 
            model_path, 
            build_model_main, 
            charset, 
            expand_bbox=args.skew, 
            resume=False, 
            num_fine_classes=2, 
            dataset_train=dataset, 
            init=args.init, 
            num_sprites_per_letter=args.num_sprites_per_letter
        )
        model.eval()
        model = model.to(device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Load reconstructor (sprite size inferred from checkpoint proto shape)
    try:
        sprite_size = sprite_size_from_checkpoint_path(reconstructor_path)
        reconstructor = Reconstructor(
            n_outputs=len(charset) * args.num_sprites_per_letter,
            sprite_size=sprite_size,
        ).to(device)

        checkpoint = torch.load(reconstructor_path, weights_only=True)
        weights = checkpoint["weights"] if isinstance(checkpoint, dict) and "weights" in checkpoint else checkpoint
        reconstructor.load_state_dict(weights)
        reconstructor.eval()
        print("Reconstructor loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load reconstructor: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Build dataloader
    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = torch.utils.data.BatchSampler(
        sampler, args.batch_size, drop_last=False
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=utils.collate_fn,
        num_workers=0,
    )
    
    print(f"Processing {len(dataset)} examples...")
    
    # Metrics accumulators
    total_cer = 0.0
    total_examples = 0
    document_results = []
    bbox_stats = {}
    sprite_counts = {}
    
    with torch.no_grad():
        for batch_idx, (samples, targets) in enumerate(tqdm(dataloader, desc=f"{item_type} {item}")):
            image, mask = samples.decompose()
            mask = mask.to(device)
            
            bboxes, scores, features_decoder, features_resnet, mask_resnet = get_bboxes_scores(
                samples.to(device), 
                model, 
                num_fine_classes=args.num_fine_classes
            )
            
            batch_scores = scores.detach().cpu().numpy()
            batch_bboxes = bboxes.detach().cpu().numpy()

            # Match reconstruction.py preprocessing:
            # - treat invalid boxes (cx/cy ~ 0) as blank by forcing scores to blank class
            # - clamp bbox sizes to avoid degenerate (near-zero) widths/heights
            cx = batch_bboxes[:, :, 0]
            cy = batch_bboxes[:, :, 1]
            valid_mask = (np.abs(cx) > 1e-3) & (np.abs(cy) > 1e-3)
            invalid_mask = ~valid_mask
            if invalid_mask.any():
                batch_scores = batch_scores.copy()
                batch_scores[invalid_mask, 0] = 1.0
                batch_scores[invalid_mask, 1:] = 1e-6

            batch_bboxes = np.maximum(batch_bboxes, 1.0)
            
            for example_idx in range(len(targets)):
                target = targets[example_idx]
                example_scores = batch_scores[example_idx]
                example_bboxes = batch_bboxes[example_idx]
                image_size = target['size'].detach().cpu().numpy()
                image_original_size = target['orig_size'].detach().cpu().numpy()
                r_height = image_original_size[0]/image_size[0]
                r_width = image_original_size[1]/image_size[1]

                idx = target['idx'].item()
                
                predictions = []
                predicted_sprites = np.argmax(example_scores, axis=1)
                predicted_chars = predicted_sprites[::2]
                predicted_accents = predicted_sprites[1::2]
                
                pairs = []
                
                for query_idx, pred_char in enumerate(predicted_chars):
                    pred_accent = predicted_accents[query_idx]
                    
                    if pred_char == 0 and pred_accent == 0:
                        continue
                        
                    char_query_idx = query_idx * 2
                    accent_query_idx = query_idx * 2 + 1
                    
                    char_pred = None
                    if pred_char != 0:
                        adjusted_sprite_char = pred_char - 1
                        bbox_char = example_bboxes[char_query_idx]
                        # bbox_stats: w/h in original image pixels (same as JSON predictions) for grid affine.
                        width_char = float(bbox_char[2]) * r_width
                        height_char = float(bbox_char[3]) * r_height
                        
                        # reconstruction.py stores bbox_stats under the *decremented* sprite index
                        # (i.e. adjusted_sprite_char in [0, total_sprites-1]).
                        if adjusted_sprite_char not in bbox_stats:
                            bbox_stats[adjusted_sprite_char] = {'widths': [], 'heights': []}
                        bbox_stats[adjusted_sprite_char]['widths'].append(width_char)
                        bbox_stats[adjusted_sprite_char]['heights'].append(height_char)
                        
                        if args.num_sprites_per_letter > 1:
                            char_idx = adjusted_sprite_char % len(charset)
                            sprite_idx_in_char = adjusted_sprite_char // len(charset)
                            if char_idx not in sprite_counts: sprite_counts[char_idx] = {}
                            if sprite_idx_in_char not in sprite_counts[char_idx]: sprite_counts[char_idx][sprite_idx_in_char] = 0
                            sprite_counts[char_idx][sprite_idx_in_char] += 1

                        char_pred = {
                            "sprite_index": int(adjusted_sprite_char),
                            "character": charset[int(adjusted_sprite_char) % len(charset)],
                            "bbox": {
                                "cx": float(bbox_char[0] * r_width),
                                "cy": float(bbox_char[1] * r_height),
                                "w": width_char,
                                "h": height_char,
                            },
                            "confidence": float(example_scores[char_query_idx, pred_char]),
                            "query_idx": char_query_idx
                        }

                    accent_pred = None
                    if pred_accent != 0:
                        adjusted_sprite_accent = pred_accent - 1
                        bbox_accent = example_bboxes[accent_query_idx]
                        width_accent = float(bbox_accent[2]) * r_width
                        height_accent = float(bbox_accent[3]) * r_height
                        
                        # reconstruction.py stores bbox_stats under the *decremented* sprite index
                        # (i.e. adjusted_sprite_accent in [0, total_sprites-1]).
                        if adjusted_sprite_accent not in bbox_stats:
                            bbox_stats[adjusted_sprite_accent] = {'widths': [], 'heights': []}
                        bbox_stats[adjusted_sprite_accent]['widths'].append(width_accent)
                        bbox_stats[adjusted_sprite_accent]['heights'].append(height_accent)
                        
                        if args.num_sprites_per_letter > 1:
                            accent_char_idx = adjusted_sprite_accent % len(charset)
                            accent_sprite_idx_in_char = adjusted_sprite_accent // len(charset)
                            if accent_char_idx not in sprite_counts: sprite_counts[accent_char_idx] = {}
                            if accent_sprite_idx_in_char not in sprite_counts[accent_char_idx]: sprite_counts[accent_char_idx][accent_sprite_idx_in_char] = 0
                            sprite_counts[accent_char_idx][accent_sprite_idx_in_char] += 1
                            
                        accent_pred = {
                            "sprite_index": int(adjusted_sprite_accent),
                            "character": charset[int(adjusted_sprite_accent) % len(charset)],
                            "bbox": {
                                "cx": float(bbox_accent[0] * r_width),
                                "cy": float(bbox_accent[1] * r_height),
                                "w": width_accent,
                                "h": height_accent,
                            },
                            "confidence": float(example_scores[accent_query_idx, pred_accent]),
                            "query_idx": accent_query_idx
                        }
                    
                    pairs.append((char_pred, accent_pred))
                # Order matches CTC: get_bboxes_scores sorts queries by pred_boxes[1] cx, then interleaves heads 1 & 2.
                # Do not re-sort pairs here (would diverge from training / CTC_loss).

                for char_p, accent_p in pairs:
                    if char_p:
                        predictions.append(char_p)
                    if accent_p:
                        predictions.append(accent_p)
                
                # Ground truth
                ground_truth = []
                if 'labels' in target:
                    labels = target['labels']
                    character_without_accent = labels[1]
                    accents = labels[2]
                    for char_idx in range(len(character_without_accent)):
                        char = character_without_accent[char_idx]
                        accent = accents[char_idx]
                        base_char = charset[char]
                        if accent != 2 * len(charset):
                            accent_char = charset[accent % len(charset)]
                            composed_char = unicodedata.normalize('NFC', base_char + accent_char)
                            ground_truth.append(composed_char)
                        else:
                            ground_truth.append(base_char)
                
                predicted_text = ''.join([pred["character"] for pred in predictions])
                ground_truth_text = ''.join(ground_truth)
                
                predicted_text_decomposed = unicodedata.normalize('NFD', predicted_text)
                ground_truth_text_decomposed = unicodedata.normalize('NFD', ground_truth_text)
                
                cer = compute_cer(predicted_text_decomposed, ground_truth_text_decomposed)
                alignment_labels = get_alignment_labels(ground_truth_text_decomposed, predicted_text_decomposed)
                
                for pred_idx, pred_item in enumerate(predictions):
                    if pred_idx < len(alignment_labels):
                        pred_item["error_label"] = alignment_labels[pred_idx]
                    else:
                        pred_item["error_label"] = "unknown"
                
                result = {
                    "predictions": predictions,
                    "ground_truth": ground_truth,
                    "predicted_text": predicted_text,
                    "ground_truth_text": ground_truth_text,
                    "cer": cer,
                    "num_predictions": len(predictions),
                    "num_ground_truth": len(ground_truth),
                    "path": target.get("path", f"example_{idx}")
                }
                
                idx_name = target.get("path", f"example_{idx}").split('/')[-1].replace('.png','').replace('.jpg','')
                json_filename = f"{idx_name}.json"
                json_path = os.path.join(doc_output_dir, json_filename)
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                total_cer += cer
                total_examples += 1
                document_results.append({
                    "example_idx": idx,
                    "cer": cer,
                    "predicted_text": predicted_text,
                    "ground_truth_text": ground_truth_text,
                    "num_predictions": len(predictions),
                    "num_ground_truth": len(ground_truth)
                })

    # Final stats
    avg_cer = total_cer / total_examples if total_examples > 0 else 0.0
    
    # Bbox stats
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
    
    # Per-character sprite selection stats
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
    
    # transcribe.json: one entry per sprite with that sprite's bbox_stats
    # (older code duplicated the loop and overwrote the mapping with the last sprite's stats).
    total_sprites = len(charset) * args.num_sprites_per_letter
    sprite_mapping = {}

    for sprite_i in range(total_sprites):
        char_idx = sprite_i % len(charset)
        char = charset[char_idx]

        if sprite_i in sprite_bbox_stats:
            bbox_stats_data = sprite_bbox_stats[sprite_i]
        else:
            bbox_stats_data = {
                "mean_width": 0.0,
                "mean_height": 0.0,
                "var_width": 0.0,
                "var_height": 0.0,
                "count": 0
            }

        if char_idx in sprite_selection_stats:
            selection_stats = sprite_selection_stats[char_idx]
        else:
            selection_stats = {
                "sprite_0_ratio": 0.0,
                "sprite_1_ratio": 0.0,
                "sprite_0_count": 0,
                "sprite_1_count": 0,
                "total_count": 0
            }

        sprite_mapping[sprite_i] = {
            "character": char,
            "bbox_stats": bbox_stats_data,
            "sprite_selection_stats": selection_stats
        }
    
    # Save transcribe.json
    transcribe_path = os.path.join(doc_output_dir, "transcribe.json")
    with open(transcribe_path, 'w', encoding='utf-8') as f:
        json.dump(sprite_mapping, f, indent=2, ensure_ascii=False)
    print(f"Saved transcribe.json: {transcribe_path}")
    
    # Generate grid if requested
    if args.generate_grid:
        print(f"\n{'='*60}")
        print("Generating grid and transformed sprites...")
        print(f"{'='*60}")
        
        sprites_folder = os.path.join(doc_output_dir, args.sprites_subdir)
        os.makedirs(sprites_folder, exist_ok=True)
        
        proto_pixel = reconstructor.generator()
        sprites = proto_pixel[1:]
        
        sprite_bbox_stats_for_grid = {k: v['bbox_stats'] for k, v in sprite_mapping.items()}
        
        padding = [args.left_padding, args.right_padding, args.top_padding, args.bottom_padding]
        
        from reconstruction.utils import create_manual_grid
        grid_with_labels, transformed_sprites = create_manual_grid(
            sprites,
            charset,
            num_sprites_per_letter=args.num_sprites_per_letter,
            sprite_bbox_stats=sprite_bbox_stats_for_grid,
            padding=padding
        )
        
        grid_path = os.path.join(doc_output_dir, "grid.jpg")
        grid_with_labels.save(grid_path)
        print(f"Saved grid: {grid_path}")
        
        print(f"Saving {len(transformed_sprites)} sprites to {sprites_folder}...")
        for i, sprite_img in enumerate(transformed_sprites):
            sprite_file = os.path.join(sprites_folder, f"{i}.png")
            sprite_img.save(sprite_file)
        
        print(f"Saved {len(transformed_sprites)} sprites to: {sprites_folder}")
    
    # Save document summary
    document_summary = {
        item_type: item,
        "total_examples": total_examples,
        "average_cer": avg_cer,
        "total_cer": total_cer,
        "results": document_results
    }
    
    summary_path = os.path.join(doc_output_dir, "document_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(document_summary, f, indent=2, ensure_ascii=False)
    
    if item is None:
        print(f"Dataset {args.data_folder} processed successfully")
    else:
        print(f"{item_type.capitalize()} {item} processed successfully")
    print(f"Average CER: {avg_cer:.4f} ({total_examples} examples)")
    print(f"Results saved to: {doc_output_dir}")
    
    return {
        item_type: item,
        "total_examples": total_examples,
        "average_cer": avg_cer,
        "total_cer": total_cer
    }


def main():
    parser = argparse.ArgumentParser(description="Prediction script for DTLR models")
    parser.add_argument("--dataset_file", type=str, default="dataset", help="Dataset module name")
    parser.add_argument("--data_folder", type=str, required=True, help="Dataset folder name under datasets_path")
    parser.add_argument("--documents", action="store_true", help="Process all documents (multi-document mode)")
    parser.add_argument("--single_model", action="store_true", help="Single model mode (model.pth and reconstructor in path_models)")
    parser.add_argument("--full_dataset", action="store_true", help="With --single_model: entire data_folder (like reconstruction step 1)")
    parser.add_argument("--document", type=str, help="Document name to process (single_model subset mode)")
    parser.add_argument("--sprites_subdir", type=str, default=None, help="Subfolder for grid.jpg + sprite PNGs (default: sprite_final if --full_dataset, else sprites)")
    parser.add_argument("--script", nargs='+', default=None, help="List of scripts to process")
    parser.add_argument("--annotation_file", type=str, help="Path to annotation JSON file (optional)")
    parser.add_argument("--split", type=str, default="all", choices=["train", "all"], help="Annotation split to use")
    parser.add_argument("--path_models", type=str, required=True, help="Folder containing model checkpoints")
    parser.add_argument("--model_config_path", type=str, default="config/Latin_CTC.py", help="Model config file path")
    parser.add_argument("--num_fine_classes", type=int, default=2, help="Number of fine classes")
    parser.add_argument("--num_sprites_per_letter", type=int, default=1, help="Number of sprites per letter")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--skew", action="store_true", help="Use skew bbox expansion")
    parser.add_argument("--init", action="store_true", help="Model initialization flag")
    parser.add_argument("--generate_grid", action="store_true", help="Generate grid and save transformed sprites")
    parser.add_argument("--left_padding", type=int, default=0, help="Left padding (default: 0)")
    parser.add_argument("--right_padding", type=int, default=0, help="Right padding (default: 0)")
    parser.add_argument("--top_padding", type=int, default=0, help="Top padding (default: 0)")
    parser.add_argument("--bottom_padding", type=int, default=0, help="Bottom padding (default: 0)")
    parser.add_argument("--old_data_augmentation", action="store_true", help="old data augmentation")
    parser.add_argument(
        "--line_resize_h_ref",
        type=int,
        default=90,
        help="Line image resize target height",
    )
    parser.add_argument(
        "--line_resize_max_width",
        type=int,
        default=1400,
        help="Line image max width before width compression",
    )
    args = parser.parse_args()
    
    # Exactly one mode must be selected
    if sum([args.documents, args.single_model, args.script is not None]) > 1:
        raise ValueError("Use only one of --documents, --single_model, or --script")
    
    if not args.documents and not args.single_model and args.script is None:
        raise ValueError("You must use --documents, --single_model, or --script")
    
    if args.full_dataset and not args.single_model:
        raise ValueError("--full_dataset requires --single_model")
    if args.full_dataset and args.document:
        raise ValueError("Do not combine --full_dataset and --document")
    if args.single_model and not args.document and not args.full_dataset:
        raise ValueError("In --single_model mode, specify --document <name> or --full_dataset")
    
    if args.sprites_subdir is None:
        args.sprites_subdir = "sprite_final" if args.full_dataset else "sprites"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    
    print(f"Device: {device}")
    print(f"Path models: {args.path_models}")
    print(f"Dataset: {args.dataset_file}")
    print(f"Data folder: {args.data_folder}")
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    # Global summary accumulators
    global_results = []
    global_total_cer = 0.0
    global_total_examples = 0
    
    # MODE SINGLE MODEL
    if args.single_model:
        print(f"\n{'='*60}")
        print("MODE: Single model")
        print(f"{'='*60}")
        
        model_path = os.path.join(args.path_models, "model.pth")
        reconstructor_path = find_reconstructor_file(args.path_models)
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model not found: {model_path}")
        if not reconstructor_path:
            raise ValueError(
                f"Reconstructor not found in: {args.path_models}\n"
                "Looked for: reconstructor_unfrozen.pth, reconstructor.pth, checkpoint.pth"
            )
        
        print("Found models:")
        print(f"   - {model_path}")
        print(f"   - {reconstructor_path}")
        
        if args.full_dataset:
            result = process_item(None, "dataset", model_path, reconstructor_path, args, device)
        else:
            result = process_item(args.document, "document", model_path, reconstructor_path, args, device)
        
        if result:
            global_results.append(result)
            global_total_cer = result["total_cer"]
            global_total_examples = result["total_examples"]
    
    # MODE SCRIPTS
    elif args.script is not None:
        print(f"\n{'='*60}")
        print("MODE: Scripts")
        print(f"{'='*60}")
        
        scripts = args.script
        print(f"Scripts to process: {len(scripts)}")
        for i, script in enumerate(scripts, 1):
            print(f"   {i}. {script}")
        
        for script in scripts:
            model_path = os.path.join(args.path_models, script, "model.pth")
            reconstructor_path = find_reconstructor_file(os.path.join(args.path_models, script))
            
            if not os.path.exists(model_path):
                print(f"WARNING: Model not found: {model_path}")
                continue
            if not reconstructor_path:
                print(f"WARNING: Reconstructor not found in: {os.path.join(args.path_models, script)}")
                continue
            
            result = process_item(script, "script", model_path, reconstructor_path, args, device)
            
            if result:
                global_results.append(result)
                global_total_cer += result["total_cer"]
                global_total_examples += result["total_examples"]
    
    # MODE MULTI DOCUMENTS
    else:
        print(f"\n{'='*60}")
        print("MODE: Multi-documents")
        print(f"{'='*60}")
        
        if args.annotation_file:
            print(f"Using annotation file: {args.annotation_file}")
            discovered = get_documents_from_annotation(args.annotation_file, args.split)
            documents = [doc for doc, _ in discovered]
        else:
            print(f"Scanning models folder: {args.path_models}")
            documents = get_documents_from_models_folder(args.path_models)
        
        print(f"Documents found: {len(documents)}")
        for i, doc in enumerate(documents, 1):
            print(f"   {i}. {doc}")
        
        for document in documents:
            model_path = os.path.join(args.path_models, document, "model.pth")
            reconstructor_path = find_reconstructor_file(os.path.join(args.path_models, document))
            
            if not os.path.exists(model_path):
                print(f"WARNING: Model not found: {model_path}")
                continue
            if not reconstructor_path:
                print(f"WARNING: Reconstructor not found in: {os.path.join(args.path_models, document)}")
                continue
            
            result = process_item(document, "document", model_path, reconstructor_path, args, device)
            
            if result:
                global_results.append(result)
                global_total_cer += result["total_cer"]
                global_total_examples += result["total_examples"]
    
    # Global CER
    global_avg_cer = global_total_cer / global_total_examples if global_total_examples > 0 else 0.0
    
    # Global summary
    global_summary = {
        "total_documents": len(global_results),
        "total_examples": global_total_examples,
        "global_average_cer": global_avg_cer,
        "global_total_cer": global_total_cer,
        "documents": global_results
    }
    
    # Save global summary
    global_summary_path = os.path.join(args.output_dir, "global_summary.json")
    with open(global_summary_path, 'w', encoding='utf-8') as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone. Results in: {args.output_dir}")
    print(f"Global CER: {global_avg_cer:.4f} ({global_total_examples} examples, {len(global_results)} documents)")
    print(f"Global summary saved: {global_summary_path}")


if __name__ == "__main__":
    main()