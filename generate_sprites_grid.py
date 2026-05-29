#!/usr/bin/env python3
"""
Generate transformed sprites and grid from a reconstructor checkpoint.

Modes:
  - Single document: --reconstructor, --charset, --results_folder (+ --transcribe for aspect ratio)
  - Finetune folder: --finetune_dir (one subfolder per document with reconstructor.pth)
"""

import argparse
import json
import os

import torch

from reconstruction.reconstructor import Reconstructor
from reconstruction.utils import create_manual_grid

FINETUNE_SKIP_DIRS = frozenset({"baseline", "predict", "__pycache__"})
RECONSTRUCTOR_CANDIDATES = (
    "reconstructor_unfrozen.pth",
    "reconstructor.pth",
)


def find_reconstructor(folder):
    """Return path to reconstructor checkpoint in folder, or None."""
    for name in RECONSTRUCTOR_CANDIDATES:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None


def find_transcribe(doc_dir, finetune_dir, doc_name):
    """transcribe.json in document folder, or under finetune_dir/predict/<doc>/."""
    candidates = (
        os.path.join(doc_dir, "transcribe.json"),
        os.path.join(finetune_dir, "predict", doc_name, "transcribe.json"),
    )
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def discover_finetune_documents(finetune_dir):
    """List (doc_name, doc_dir, reconstructor_path) for each finetuned document."""
    if not os.path.isdir(finetune_dir):
        raise FileNotFoundError(f"Finetune directory not found: {finetune_dir}")

    documents = []
    missing = []
    for name in sorted(os.listdir(finetune_dir)):
        if name in FINETUNE_SKIP_DIRS:
            continue
        doc_dir = os.path.join(finetune_dir, name)
        if not os.path.isdir(doc_dir):
            continue
        reconstructor_path = find_reconstructor(doc_dir)
        if reconstructor_path:
            documents.append((name, doc_dir, reconstructor_path))
        else:
            missing.append(name)

    if missing:
        print(f"WARNING: {len(missing)} folder(s) skipped (no reconstructor): {', '.join(missing[:5])}"
              + (" ..." if len(missing) > 5 else ""))

    return documents


def load_charset(charset_path):
    if not os.path.isfile(charset_path):
        raise FileNotFoundError(f"Charset file not found: {charset_path}")
    with open(charset_path, "r", encoding="utf-8") as f:
        charset = json.load(f)
    return charset


def load_bbox_stats_from_transcribe(transcribe_path):
    if not transcribe_path or not os.path.isfile(transcribe_path):
        return None
    with open(transcribe_path, "r", encoding="utf-8") as f:
        transcribe_data = json.load(f)
    sprite_bbox_stats = {}
    for sprite_idx_str, data in transcribe_data.items():
        if "bbox_stats" in data:
            sprite_bbox_stats[int(sprite_idx_str)] = data["bbox_stats"]
    return sprite_bbox_stats


def generate_sprites_grid(
    reconstructor_path,
    charset,
    results_folder,
    *,
    transcribe_path=None,
    without_aspect_ratio=False,
    num_sprites_per_letter=1,
    padding=None,
):
    """Load reconstructor, build grid + per-sprite PNGs under results_folder."""
    padding = padding or [0, 0, 0, 0]
    os.makedirs(results_folder, exist_ok=True)

    sprites_subdir = (
        "sprites_without_aspect_ratio" if without_aspect_ratio else "sprites"
    )
    sprites_folder = os.path.join(results_folder, sprites_subdir)
    os.makedirs(sprites_folder, exist_ok=True)

    print(f"Loading checkpoint from {reconstructor_path}...")
    checkpoint = torch.load(reconstructor_path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict) and "weights" in checkpoint:
        weights = checkpoint["weights"]
        epoch = checkpoint.get("epoch", "N/A")
        print(f"Checkpoint format: dict (epoch={epoch})")
    else:
        weights = checkpoint
        print("Checkpoint format: direct weights")

    if "generator.proto" not in weights:
        raise ValueError("Could not find generator.proto in checkpoint")

    proto = weights["generator.proto"]
    n_outputs, _, H, W = proto.shape
    sprite_size = (H, W)
    print(f"Found {n_outputs} sprites of size {sprite_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    reconstructor = Reconstructor(n_outputs=n_outputs, sprite_size=sprite_size).to(device)
    if isinstance(checkpoint, dict) and "weights" in checkpoint:
        reconstructor.load_state_dict(checkpoint["weights"])
    else:
        reconstructor.load_state_dict(checkpoint)
    reconstructor.eval()

    proto_pixel = reconstructor.generator()
    sprites = proto_pixel[1:]
    print(f"Extracted {len(sprites)} sprites (shape: {sprites.shape})")

    if without_aspect_ratio:
        sprite_bbox_stats = None
        print("Without aspect ratio: no bbox deformation")
    else:
        sprite_bbox_stats = load_bbox_stats_from_transcribe(transcribe_path)
        if sprite_bbox_stats:
            print(f"Bbox stats from {transcribe_path}: {len(sprite_bbox_stats)} sprites")
        else:
            print("No bbox stats (transcribe missing or empty)")

    print("Creating grid and transforming sprites...")
    grid_with_labels, transformed_sprites = create_manual_grid(
        sprites,
        charset,
        num_sprites_per_letter=num_sprites_per_letter,
        sprite_bbox_stats=sprite_bbox_stats,
        padding=padding,
    )

    grid_filename = "grid_without_AR.jpg" if without_aspect_ratio else "grid.jpg"
    grid_path = os.path.join(results_folder, grid_filename)
    grid_with_labels.save(grid_path)
    print(f"Grid saved: {grid_path}")

    for i, sprite_img in enumerate(transformed_sprites):
        sprite_img.save(os.path.join(sprites_folder, f"{i}.png"))
    print(f"Saved {len(transformed_sprites)} sprites to: {sprites_folder}")

    return grid_path, sprites_folder


def run_finetune_dir(finetune_dir, charset_path, args):
    """Generate grids without aspect ratio for every document in a finetune run."""
    charset = load_charset(charset_path)
    documents = discover_finetune_documents(finetune_dir)

    if not documents:
        raise RuntimeError(
            f"No document with {RECONSTRUCTOR_CANDIDATES} found under {finetune_dir}"
        )

    print(f"\n{'='*60}")
    print(f"Finetune batch: {finetune_dir}")
    print(f"Documents: {len(documents)} | without aspect ratio")
    print(f"{'='*60}\n")

    ok, failed = [], []
    for doc_name, doc_dir, reconstructor_path in documents:
        print(f"\n--- {doc_name} ---")
        try:
            generate_sprites_grid(
                reconstructor_path,
                charset,
                doc_dir,
                transcribe_path=find_transcribe(doc_dir, finetune_dir, doc_name),
                without_aspect_ratio=True,
                num_sprites_per_letter=args.num_sprites_per_letter,
                padding=[
                    args.left_padding,
                    args.right_padding,
                    args.top_padding,
                    args.bottom_padding,
                ],
            )
            ok.append(doc_name)
        except Exception as e:
            print(f"ERROR: Failed {doc_name}: {e}")
            failed.append((doc_name, str(e)))

    print(f"\n{'='*60}")
    print(f"Done: {len(ok)} OK, {len(failed)} failed")
    if failed:
        for name, err in failed[:10]:
            print(f"  - {name}: {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sprites and grid from reconstructor checkpoint(s)",
    )
    parser.add_argument(
        "--finetune_dir",
        type=str,
        help="Root folder of a finetune run (one subfolder per document with reconstructor.pth). "
        "Generates grid_without_AR.jpg + sprites_without_aspect_ratio/ for each document.",
    )
    parser.add_argument(
        "--reconstructor",
        type=str,
        help="Path to reconstructor checkpoint (single-document mode)",
    )
    parser.add_argument(
        "--charset",
        type=str,
        default="charset.json",
        help="Path to charset.json (default: charset.json)",
    )
    parser.add_argument(
        "--transcribe",
        type=str,
        help="Path to transcribe.json (single-document mode, required unless --without-aspect-ratio)",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        help="Output folder (single-document mode)",
    )
    parser.add_argument("--num_sprites_per_letter", type=int, default=1)
    parser.add_argument("--left_padding", type=int, default=0)
    parser.add_argument("--right_padding", type=int, default=0)
    parser.add_argument("--top_padding", type=int, default=0)
    parser.add_argument("--bottom_padding", type=int, default=0)
    parser.add_argument(
        "--without-aspect-ratio",
        action="store_true",
        help="No bbox deformation; grid_without_AR.jpg + sprites_without_aspect_ratio/",
    )

    args = parser.parse_args()
    padding = [
        args.left_padding,
        args.right_padding,
        args.top_padding,
        args.bottom_padding,
    ]

    if args.finetune_dir:
        if args.reconstructor or args.results_folder:
            parser.error(
                "Use either --finetune_dir or (--reconstructor + --results_folder), not both"
            )
        run_finetune_dir(os.path.abspath(args.finetune_dir), args.charset, args)
        return

    if not args.reconstructor or not args.results_folder:
        parser.error(
            "Single-document mode requires --reconstructor and --results_folder, "
            "or use --finetune_dir for batch finetune"
        )
    if not args.without_aspect_ratio and not args.transcribe:
        parser.error(
            "Single-document mode requires --transcribe unless --without-aspect-ratio"
        )

    charset = load_charset(args.charset)
    generate_sprites_grid(
        args.reconstructor,
        charset,
        args.results_folder,
        transcribe_path=args.transcribe,
        without_aspect_ratio=args.without_aspect_ratio,
        num_sprites_per_letter=args.num_sprites_per_letter,
        padding=padding,
    )

    print("\nGeneration complete!")


if __name__ == "__main__":
    main()
