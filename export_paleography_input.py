#!/usr/bin/env python3
"""
Export a paleography_input bundle after step-1 training and finetuning.

Runs (unless skipped):
  1. Step-1: sprites without aspect ratio (reconstructor_unfrozen.pth, no bbox deformation)
  2. Step-1: full-dataset predict + sprites with aspect ratio (same reconstructor_unfrozen.pth)
  3. Finetune: predict + grid (auto-detects document vs script subfolders)

Then organizes:
  <output_dir>/
    prototypes/
      baseline_with_ar/              # step-1 sprite_final (AR from inference)
      baseline_without_ar/           # step-1 sprites_without_aspect_ratio
      <item>/                        # finetune sprites (document or script name)
    characters_measurements/<item>/  # per-line bbox JSONs from finetune predict

Example (conda env ocrdino):
  conda activate ocrdino
  python export_paleography_input.py \\
    --step1_dir logs_reconstruction/IWCP_step_1_32x32 \\
    --finetune_dir logs_reconstruction/IWCP_finetune_32x32

  # or:
  ./scripts/export_paleography_input.sh \\
    logs_reconstruction/IWCP_step_1_32x32 \\
    logs_reconstruction/IWCP_finetune_32x32
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

RECONSTRUCTOR_CANDIDATES = (
    "reconstructor_unfrozen.pth",
    "reconstructor.pth",
)

STEP1_RECONSTRUCTOR = "reconstructor_unfrozen.pth"


def load_dataset_charset(data_folder: str) -> list[str]:
    """Same charset as predict.py / LineDataset (not project charset.json)."""
    base = os.path.join(load_datasets_path(), data_folder)
    with open(os.path.join(base, "charset_without_accent.json"), encoding="utf-8") as f:
        charset_without_accent = json.load(f)
    with open(os.path.join(base, "charset_accent.json"), encoding="utf-8") as f:
        charset_accent = json.load(f)
    return charset_without_accent + charset_accent


def run_step1_without_ar(step1_dir: str, args: argparse.Namespace) -> None:
    """
    Same reconstructor and grid pipeline as predict.py (with AR),
    but sprite_bbox_stats=None so prototypes keep native aspect ratio.
    """
    import torch
    from reconstruction.reconstructor import Reconstructor
    from reconstruction.utils import create_manual_grid, sprite_size_from_checkpoint_path

    reconstructor_path = find_step1_reconstructor(step1_dir)
    charset = load_dataset_charset(args.data_folder)
    print(f"Step-1 without AR: {reconstructor_path}")
    print(f"  charset: {len(charset)} characters (dataset {args.data_folder})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sprite_size = sprite_size_from_checkpoint_path(reconstructor_path)
    reconstructor = Reconstructor(
        n_outputs=len(charset) * args.num_sprites_per_letter,
        sprite_size=sprite_size,
    ).to(device)

    checkpoint = torch.load(reconstructor_path, map_location="cpu", weights_only=True)
    weights = (
        checkpoint["weights"]
        if isinstance(checkpoint, dict) and "weights" in checkpoint
        else checkpoint
    )
    reconstructor.load_state_dict(weights)
    reconstructor.eval()

    with torch.no_grad():
        proto_pixel = reconstructor.generator()
        sprites = proto_pixel[1:]

    grid_with_labels, transformed_sprites = create_manual_grid(
        sprites,
        charset,
        num_sprites_per_letter=args.num_sprites_per_letter,
        sprite_bbox_stats=None,
        padding=[0, 0, 0, 0],
    )

    sprites_folder = os.path.join(step1_dir, "sprites_without_aspect_ratio")
    os.makedirs(sprites_folder, exist_ok=True)
    grid_path = os.path.join(step1_dir, "grid_without_AR.jpg")
    grid_with_labels.save(grid_path)
    print(f"Grid saved: {grid_path}")

    for i, sprite_img in enumerate(transformed_sprites):
        sprite_img.save(os.path.join(sprites_folder, f"{i}.png"))
    print(f"Saved {len(transformed_sprites)} sprites to: {sprites_folder}")


def find_reconstructor(folder: str) -> str | None:
    for name in RECONSTRUCTOR_CANDIDATES:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None


def find_step1_reconstructor(step1_dir: str) -> str:
    """Same checkpoint as predict.py step-1 (--single_model --full_dataset)."""
    path = os.path.join(step1_dir, STEP1_RECONSTRUCTOR)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(
        f"Step-1 reconstructor not found: {path}\n"
        "Baseline with/without aspect ratio both require reconstructor_unfrozen.pth."
    )

SKIP_JSON_NAMES = frozenset({
    "transcribe.json",
    "document_summary.json",
    "global_summary.json",
})

FINETUNE_SKIP_DIRS = frozenset({"baseline", "predict", "__pycache__", "paleography_input"})


def resolve_path(path: str | None, *candidates: str) -> str | None:
    if path and os.path.exists(path):
        return os.path.abspath(path)
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(path) if path else None


def run_subprocess(cmd: list[str], *, cwd: str | None = None) -> None:
    print(f"\n{'='*60}")
    print(">>>", " ".join(cmd))
    print(f"{'='*60}")
    subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, check=True)


def run_predict_step1(args: argparse.Namespace, step1_dir: str) -> None:
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "predict.py"),
        "--dataset_file",
        args.dataset_file,
        "--data_folder",
        args.data_folder,
        "--single_model",
        "--full_dataset",
        "--path_models",
        step1_dir,
        "--output_dir",
        step1_dir,
        "--model_config_path",
        args.model_config_path,
        "--num_fine_classes",
        str(args.num_fine_classes),
        "--batch_size",
        str(args.step1_batch_size),
        "--generate_grid",
        "--line_resize_h_ref",
        str(args.line_resize_h_ref),
        "--line_resize_max_width",
        str(args.line_resize_max_width),
    ]
    if args.num_sprites_per_letter != 1:
        cmd.extend(["--num_sprites_per_letter", str(args.num_sprites_per_letter)])
    if args.old_data_augmentation:
        cmd.append("--old_data_augmentation")
    if args.skew:
        cmd.append("--skew")
    if args.init:
        cmd.append("--init")
    run_subprocess(cmd)


def load_datasets_path() -> str:
    config_path = os.path.join(PROJECT_ROOT, "datasets", "config.json")
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)["datasets_path"]


def resolve_annotation_path(data_folder: str, annotation_file: str | None) -> str:
    if annotation_file:
        return os.path.abspath(annotation_file)
    return os.path.join(load_datasets_path(), data_folder, "annotation.json")


def load_annotation(data_folder: str, annotation_file: str | None) -> dict:
    path = resolve_annotation_path(data_folder, annotation_file)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"annotation.json not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def discover_finetune_model_folders(finetune_dir: str) -> list[str]:
    """Subfolders under finetune_dir that contain model.pth + reconstructor."""
    items = []
    for name in sorted(os.listdir(finetune_dir)):
        if name in FINETUNE_SKIP_DIRS:
            continue
        item_dir = os.path.join(finetune_dir, name)
        if not os.path.isdir(item_dir):
            continue
        model_path = os.path.join(item_dir, "model.pth")
        if os.path.isfile(model_path) and find_reconstructor(item_dir):
            items.append(name)
    return items


def classify_finetune_item(name: str, annotation: dict) -> str:
    """
    Return 'document' or 'script' using the same rules as LineDataset filters.
    """
    document_matches = sum(1 for key in annotation if name in key)
    script_matches = sum(
        1
        for entry in annotation.values()
        if name in entry.get("script", "")
    )

    if document_matches > 0 and script_matches == 0:
        return "document"
    if script_matches > 0 and document_matches == 0:
        return "script"
    if document_matches > 0 and script_matches > 0:
        # Prefer script when both match (e.g. substring collision in line keys).
        return "script" if script_matches >= document_matches else "document"
    return "unknown"


def classify_finetune_items(
    finetune_dir: str,
    annotation: dict,
) -> tuple[list[str], list[str], list[str]]:
    documents: list[str] = []
    scripts: list[str] = []
    unknown: list[str] = []

    for name in discover_finetune_model_folders(finetune_dir):
        kind = classify_finetune_item(name, annotation)
        if kind == "document":
            documents.append(name)
        elif kind == "script":
            scripts.append(name)
        else:
            unknown.append(name)

    return documents, scripts, unknown


def _predict_common_args(
    args: argparse.Namespace,
    *,
    path_models: str,
    output_dir: str,
    batch_size: int,
) -> list[str]:
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "predict.py"),
        "--dataset_file",
        args.dataset_file,
        "--data_folder",
        args.data_folder,
        "--path_models",
        path_models,
        "--output_dir",
        output_dir,
        "--model_config_path",
        args.model_config_path,
        "--num_fine_classes",
        str(args.num_fine_classes),
        "--batch_size",
        str(batch_size),
        "--generate_grid",
        "--line_resize_h_ref",
        str(args.line_resize_h_ref),
        "--line_resize_max_width",
        str(args.line_resize_max_width),
    ]
    if args.annotation_file:
        cmd.extend(["--annotation_file", args.annotation_file])
    if args.num_sprites_per_letter != 1:
        cmd.extend(["--num_sprites_per_letter", str(args.num_sprites_per_letter)])
    if args.old_data_augmentation:
        cmd.append("--old_data_augmentation")
    if args.skew:
        cmd.append("--skew")
    if args.init:
        cmd.append("--init")
    return cmd


def run_predict_finetune(args: argparse.Namespace, finetune_dir: str, predict_dir: str) -> None:
    annotation = load_annotation(args.data_folder, args.annotation_file)
    documents, scripts, unknown = classify_finetune_items(finetune_dir, annotation)

    for name in unknown:
        print(
            f"WARNING: Skipping finetune folder {name!r}: "
            "not found as document key or script in annotation"
        )

    if not documents and not scripts:
        raise RuntimeError(
            f"No finetune items to predict under {finetune_dir} "
            "(need subfolders with model.pth + reconstructor)"
        )

    print("\nFinetune items detected:")
    if documents:
        print(f"  documents ({len(documents)}): {', '.join(documents)}")
    if scripts:
        print(f"  scripts ({len(scripts)}): {', '.join(scripts)}")

    mixed = bool(documents and scripts)

    if scripts:
        cmd = _predict_common_args(
            args,
            path_models=finetune_dir,
            output_dir=predict_dir,
            batch_size=args.finetune_batch_size,
        )
        cmd.extend(["--script", *scripts])
        run_subprocess(cmd)

    if documents and not mixed:
        cmd = _predict_common_args(
            args,
            path_models=finetune_dir,
            output_dir=predict_dir,
            batch_size=args.finetune_batch_size,
        )
        cmd.append("--documents")
        run_subprocess(cmd)
    elif documents and mixed:
        for document in documents:
            cmd = _predict_common_args(
                args,
                path_models=os.path.join(finetune_dir, document),
                output_dir=predict_dir,
                batch_size=args.finetune_batch_size,
            )
            cmd.extend(["--single_model", "--document", document])
            run_subprocess(cmd)


def copy_tree_contents(src_dir: str, dst_dir: str) -> int:
    """Copy all files from src_dir into dst_dir. Returns number of files copied."""
    if not os.path.isdir(src_dir):
        return 0
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for name in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isdir(src):
            continue
        shutil.copy2(src, dst)
        count += 1
    return count


def is_line_prediction_json(filename: str) -> bool:
    return filename.endswith(".json") and filename not in SKIP_JSON_NAMES


def discover_finetune_items(finetune_dir: str, predict_dir: str) -> list[str]:
    """Finetune item names (document or script) that have predict output with sprites."""
    if os.path.isdir(predict_dir):
        docs = []
        for name in sorted(os.listdir(predict_dir)):
            doc_predict = os.path.join(predict_dir, name)
            sprites_dir = os.path.join(doc_predict, "sprites")
            if os.path.isdir(doc_predict) and os.path.isdir(sprites_dir):
                docs.append(name)
        if docs:
            return docs

    docs = []
    for name in sorted(os.listdir(finetune_dir)):
        if name in FINETUNE_SKIP_DIRS:
            continue
        doc_dir = os.path.join(finetune_dir, name)
        if os.path.isdir(doc_dir) and find_reconstructor(doc_dir):
            docs.append(name)
    return docs


def _validate_baseline_sprite_sizes(baseline_with_ar: str, baseline_without_ar: str) -> None:
    """Warn if baseline folders disagree on sprite resolution."""
    from PIL import Image

    def sample_size(folder: str) -> tuple[int, int] | None:
        if not os.path.isdir(folder):
            return None
        for name in sorted(os.listdir(folder)):
            if name.endswith(".png"):
                with Image.open(os.path.join(folder, name)) as im:
                    return im.size
        return None

    with_size = sample_size(baseline_with_ar)
    without_size = sample_size(baseline_without_ar)
    if with_size and without_size and with_size != without_size:
        raise RuntimeError(
            "Baseline sprite size mismatch after organize: "
            f"baseline_with_ar={with_size}, baseline_without_ar={without_size}. "
            "Regenerate step-1 sprites_without_aspect_ratio from reconstructor_unfrozen.pth."
        )


def organize_paleography_input(
    output_dir: str,
    step1_dir: str,
    predict_dir: str,
    *,
    clean: bool = False,
) -> None:
    prototypes_root = os.path.join(output_dir, "prototypes")
    baseline_with_ar = os.path.join(prototypes_root, "baseline_with_ar")
    baseline_without_ar = os.path.join(prototypes_root, "baseline_without_ar")
    measurements_root = os.path.join(output_dir, "characters_measurements")

    if clean and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(prototypes_root, exist_ok=True)
    os.makedirs(measurements_root, exist_ok=True)

    # Always refresh baseline sprite folders (drop stale 32x32 exports, etc.)
    for baseline_dir in (baseline_with_ar, baseline_without_ar):
        if os.path.isdir(baseline_dir):
            shutil.rmtree(baseline_dir)

    os.makedirs(baseline_with_ar, exist_ok=True)
    os.makedirs(baseline_without_ar, exist_ok=True)

    step1_ar_src = os.path.join(step1_dir, "sprite_final")
    step1_no_ar_src = os.path.join(step1_dir, "sprites_without_aspect_ratio")

    n_ar = copy_tree_contents(step1_ar_src, baseline_with_ar)
    n_no_ar = copy_tree_contents(step1_no_ar_src, baseline_without_ar)

    if n_ar == 0:
        print(f"WARNING: prototypes/baseline_with_ar: no sprites in {step1_ar_src}")
    else:
        print(f"OK: prototypes/baseline_with_ar: {n_ar} sprites from {step1_ar_src}")

    if n_no_ar == 0:
        print(f"WARNING: prototypes/baseline_without_ar: no sprites in {step1_no_ar_src}")
    else:
        print(f"OK: prototypes/baseline_without_ar: {n_no_ar} sprites from {step1_no_ar_src}")

    _validate_baseline_sprite_sizes(baseline_with_ar, baseline_without_ar)

    items = discover_finetune_items(os.path.dirname(predict_dir), predict_dir)
    if not items:
        print(f"WARNING: No finetune items found under {predict_dir}")
        return

    for item in items:
        item_predict = os.path.join(predict_dir, item)
        if not os.path.isdir(item_predict):
            print(f"WARNING: predict output missing for {item}: {item_predict}")
            continue
        sprites_src = os.path.join(item_predict, "sprites")
        proto_dst = os.path.join(prototypes_root, item)
        meas_dst = os.path.join(measurements_root, item)

        os.makedirs(proto_dst, exist_ok=True)
        os.makedirs(meas_dst, exist_ok=True)

        n_proto = copy_tree_contents(sprites_src, proto_dst)
        if n_proto == 0:
            print(f"WARNING: prototypes/{item}: no sprites in {sprites_src}")
        else:
            print(f"OK: prototypes/{item}: {n_proto} sprites")

        n_json = 0
        for name in sorted(os.listdir(item_predict)):
            if not is_line_prediction_json(name):
                continue
            shutil.copy2(
                os.path.join(item_predict, name),
                os.path.join(meas_dst, name),
            )
            n_json += 1
        if n_json == 0:
            print(f"WARNING: characters_measurements/{item}: no line JSONs")
        else:
            print(f"OK: characters_measurements/{item}: {n_json} JSON files")

    print(f"\nDone: paleography_input ready at {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export paleography_input after step-1 training and finetuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--step1_dir",
        type=str,
        required=True,
        help="Step-1 training folder (model.pth + reconstructor_unfrozen.pth)",
    )
    parser.add_argument(
        "--finetune_dir",
        type=str,
        required=True,
        help="Finetune run folder (subfolders with model.pth, one per document or script)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Destination for paleography_input (default: <step1_dir>/paleography_input)",
    )
    parser.add_argument(
        "--predict_dir",
        type=str,
        default=None,
        help="Finetune predict output (default: <finetune_dir>/predict)",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default=None,
        help="Path to annotation.json (default: <datasets_path>/<data_folder>/annotation.json)",
    )
    parser.add_argument(
        "--charset",
        type=str,
        default=None,
        help="charset.json path (default: charset.json in project root)",
    )
    parser.add_argument("--dataset_file", type=str, default="dataset")
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="config/Latin_accent.py",
    )
    parser.add_argument("--num_fine_classes", type=int, default=2)
    parser.add_argument("--num_sprites_per_letter", type=int, default=1)
    parser.add_argument("--step1_batch_size", type=int, default=16)
    parser.add_argument("--finetune_batch_size", type=int, default=4)
    parser.add_argument("--skew", action="store_true")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--old_data_augmentation", action="store_true")
    parser.add_argument(
        "--line_resize_h_ref",
        type=int,
        default=90,
        help="Line image resize target height (passed to predict.py)",
    )
    parser.add_argument(
        "--line_resize_max_width",
        type=int,
        default=1400,
        help="Line image max width (passed to predict.py)",
    )

    parser.add_argument(
        "--skip_step1",
        action="store_true",
        help="Skip step-1 inference (use existing sprite_final / sprites_without_aspect_ratio)",
    )
    parser.add_argument(
        "--skip_finetune",
        action="store_true",
        help="Skip finetune predict (use existing finetune predict/ output)",
    )
    parser.add_argument(
        "--organize_only",
        action="store_true",
        help="Only copy files into paleography_input layout (no inference)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove output_dir before organizing",
    )

    args = parser.parse_args()

    step1_dir = os.path.abspath(args.step1_dir)
    finetune_dir = os.path.abspath(args.finetune_dir)
    output_dir = os.path.abspath(
        args.output_dir or os.path.join(step1_dir, "paleography_input")
    )
    predict_dir = os.path.abspath(
        args.predict_dir or os.path.join(finetune_dir, "predict")
    )

    charset_path = resolve_path(
        args.charset,
        os.path.join(PROJECT_ROOT, "charset.json"),
        os.path.join(step1_dir, "charset.json"),
    )
    if not charset_path or not os.path.isfile(charset_path):
        parser.error(f"charset.json not found (tried --charset, project root, {step1_dir})")

    if not os.path.isdir(step1_dir):
        parser.error(f"step1_dir not found: {step1_dir}")
    if not os.path.isdir(finetune_dir):
        parser.error(f"finetune_dir not found: {finetune_dir}")

    print(f"Step-1 dir:     {step1_dir}")
    print(f"Finetune dir:   {finetune_dir}")
    print(f"Predict dir:    {predict_dir}")
    print(f"Output dir:     {output_dir}")
    print(f"Charset:        {charset_path}")

    if args.organize_only:
        args.skip_step1 = True
        args.skip_finetune = True

    if not args.skip_step1:
        print("\n[1/3] Step-1: sprites without aspect ratio")
        run_step1_without_ar(step1_dir, args)

        print("\n[2/3] Step-1: full-dataset predict + sprites with aspect ratio")
        run_predict_step1(args, step1_dir)
    else:
        print("\n[1-2/3] Step-1: skipped")

    if not args.skip_finetune:
        print("\n[3/3] Finetune: predict + sprites + bbox JSONs (auto document/script)")
        os.makedirs(predict_dir, exist_ok=True)
        run_predict_finetune(args, finetune_dir, predict_dir)
    else:
        print("\n[3/3] Finetune: skipped")

    print("\n[organize] Building paleography_input layout")
    organize_paleography_input(
        output_dir,
        step1_dir,
        predict_dir,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
