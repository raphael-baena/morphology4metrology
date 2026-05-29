"""Line-image dataset with per-line annotations (annotation.json + images/)."""

import csv
import json
import os
import unicodedata

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import datasets.transforms as T

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = _MODULE_DIR if "dataset" in _MODULE_DIR else os.path.join(_MODULE_DIR, ".")
with open(os.path.join(_CONFIG_DIR, "config.json"), "r", encoding="utf-8") as f:
    datasets_path = json.load(f)["datasets_path"]

_DISAMBIGUATION_TABLE = os.path.join(_MODULE_DIR, "disambiguation_table.csv")


class LineDataset(Dataset):
  """PyTorch dataset for line OCR with accent decomposition."""

  def __init__(
      self,
      mode,
      transform=transforms.ToTensor(),
      target_transform=None,
      data_folder=None,
      document=None,
      split=None,
      script=None,
      line=None,
  ):
    if data_folder is None:
      raise ValueError("data_folder is None. Please provide a data_folder")

    self.mode = mode
    self._transforms = transform
    self.document = document
    self.split = split
    self.script = script
    self.line = line
    self.data_folder = data_folder
    self.transform = transform
    self.target_transform = target_transform

    if os.path.isfile(_DISAMBIGUATION_TABLE):
      with open(_DISAMBIGUATION_TABLE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        self.disambiguation_mapping = {
            row["char"]: row["replacement"] for row in reader
        }
    else:
      self.disambiguation_mapping = {}

    annotation_path = os.path.join(datasets_path, self.data_folder, "annotation.json")
    if not os.path.isfile(annotation_path):
      raise ValueError(f"annotation.json not found: {annotation_path}")
    with open(annotation_path, "r", encoding="utf-8") as f:
      self.data = json.load(f)

    charset_without_path = os.path.join(
        datasets_path, self.data_folder, "charset_without_accent.json"
    )
    if not os.path.isfile(charset_without_path):
      print("Creating the charset")
      self.create_charset()
    else:
      with open(charset_without_path, "r", encoding="utf-8") as f:
        self.charset_without_accent = json.load(f)
      with open(
          os.path.join(datasets_path, self.data_folder, "charset_accent.json"),
          "r",
          encoding="utf-8",
      ) as f:
        self.charset_accent = json.load(f)
      self.charset = self.charset_without_accent + self.charset_accent

    data_mode = []
    for key, entry in self.data.items():  # self.data is still a dict here
      if self.split == "all" or entry["split"] == self.mode:
        data_mode.append([key, entry])
    self.data = data_mode

    filters = sum(x is not None for x in (document, script, line))
    if filters > 1:
      raise ValueError("Only one of document, script, or line can be set")

    if document is not None:
      self.data = self._filter_by_document(document)
    elif script is not None:
      self.data = self._filter_by_script(script)
    elif line is not None:
      self.data = self._filter_by_line(line)

    self.img_labels = []
    self.img_idx = []

  def _filter_by_document(self, document):
    filtered = []
    for key, entry in self.data:
      if document not in key:
        continue
      if self.split is not None and self.split != "all" and entry["split"] != self.split:
        continue
      filtered.append([key, entry])
    if not filtered:
      raise ValueError(
          f"No data found for document {document!r} and split {self.split!r}"
      )
    return filtered

  def _filter_by_script(self, script):
    filtered = []
    for key, entry in self.data:
      if script not in entry["script"]:
        continue
      if self.split is not None and self.split != "all" and self.split not in entry["split"]:
        continue
      filtered.append([key, entry])
    if not filtered:
      raise ValueError(f"No data found for script {script!r} and split {self.split!r}")
    return filtered

  def _filter_by_line(self, line):
    filtered = []
    for key, entry in self.data:
      if entry.get("line") != line:
        continue
      if self.split is not None and self.split != "all" and entry["split"] != self.split:
        continue
      filtered.append([key, entry])
    if not filtered:
      raise ValueError(f"No data found for line {line!r} and split {self.split!r}")
    return filtered

  def create_charset(self):
    charset_without_accent = []
    charset_accent = []
    for _key, entry in self.data.items():
      text = self.convert_str_to_tensor(entry["label"])
      for cluster in text:
        cluster = unicodedata.normalize("NFD", cluster)
        if len(cluster) == 1:
          if cluster not in charset_without_accent:
            charset_without_accent.append(cluster)
        else:
          if cluster[0] not in charset_without_accent:
            charset_without_accent.append(cluster[0])
          if cluster[1] not in charset_accent:
            charset_accent.append(cluster[1])

    charset_without_accent.sort()
    charset_accent.sort()
    charset = charset_without_accent + charset_accent
    self.charset_without_accent = charset_without_accent
    self.charset_accent = charset_accent
    self.charset = charset
    base = os.path.join(datasets_path, self.data_folder)
    with open(os.path.join(base, "charset_without_accent.json"), "w", encoding="utf-8") as f:
      json.dump(charset_without_accent, f)
    with open(os.path.join(base, "charset_accent.json"), "w", encoding="utf-8") as f:
      json.dump(charset_accent, f)
    with open(os.path.join(base, "charset.json"), "w", encoding="utf-8") as f:
      json.dump(charset, f)

  def __len__(self):
    return len(self.data)

  def convert_str_to_tensor(self, text):
    decomposed = unicodedata.normalize("NFD", text)
    clusters = []
    for char in decomposed:
      if unicodedata.combining(char):
        clusters[-1] += char
      else:
        clusters.append(char)
    return clusters

  def generate_labels(self, norm_labels):
    with_accent = []
    without_accent = []
    general_labels = []

    for cluster in norm_labels:
      cluster = unicodedata.normalize("NFD", cluster)
      if len(cluster) == 1:
        ll_without_accent = self.charset.index(cluster)
        accent = 2 * len(self.charset)
      else:
        ll_without_accent = self.charset.index(cluster[0])
        accent = len(self.charset) + self.charset.index(cluster[1])

      with_accent.append(ll_without_accent)
      without_accent.append(accent)
      general_labels.append(ll_without_accent)

    return torch.tensor(
        [general_labels, with_accent, without_accent], dtype=torch.int64
    )

  def __getitem__(self, idx):
    key, example = self.data[idx]
    text = example["label"]

    images_root = os.path.join(datasets_path, self.data_folder, "images")
    folders = os.listdir(images_root)

    best_match = None
    best_match_length = 0
    for folder_name in folders:
      if key.startswith(folder_name) and len(folder_name) > best_match_length:
        test_path = os.path.join(images_root, folder_name, key)
        if os.path.exists(test_path):
          best_match = folder_name
          best_match_length = len(folder_name)

    if best_match is None:
      raise FileNotFoundError(f"Could not find folder containing file {key}")

    path_image = os.path.join(images_root, best_match, key)
    image = Image.open(path_image).convert("RGB")

    labels = {}
    labels_norm = self.convert_str_to_tensor(text)
    labels["labels"] = self.generate_labels(labels_norm)
    labels["orig_size"] = torch.tensor([image.size[1], image.size[0]], dtype=torch.int64)
    labels["size"] = torch.tensor([image.size[1], image.size[0]], dtype=torch.int64)
    labels["img_idx"] = torch.tensor([idx], dtype=torch.int64)
    labels["idx"] = torch.tensor([idx], dtype=torch.int64)
    labels["path"] = path_image

    dummy_boxes = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
    labels["boxes"] = dummy_boxes.repeat(labels["labels"].shape[0], 1)

    image, labels = self._transforms(image, labels)
    return image, labels


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):
  normalize = T.Compose([
      T.ToTensor(),
      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])

  scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
  max_size = 1333
  scales2_resize = [400, 500, 600]
  scales2_crop = [384, 600]

  scales = getattr(args, "data_aug_scales", scales)
  max_size = getattr(args, "data_aug_max_size", max_size)
  scales2_resize = getattr(args, "data_aug_scales2_resize", scales2_resize)
  scales2_crop = getattr(args, "data_aug_scales2_crop", scales2_crop)

  data_aug_scale_overlap = getattr(args, "data_aug_scale_overlap", None)
  if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
    data_aug_scale_overlap = float(data_aug_scale_overlap)
    scales = [int(i * data_aug_scale_overlap) for i in scales]
    max_size = int(max_size * data_aug_scale_overlap)
    scales2_resize = [int(i * data_aug_scale_overlap) for i in scales2_resize]
    scales2_crop = [int(i * data_aug_scale_overlap) for i in scales2_crop]

  if getattr(args, "old_data_augmentation", False):
    if image_set == "train":
      if fix_size:
        return T.Compose([
            T.RandomResize([(max_size, max(scales))]),
            normalize,
        ])

      if strong_aug:
        import datasets.sltransform as SLT

        return T.Compose([
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales, max_size=max_size),
                ]),
            ),
            SLT.RandomSelectMulti([
                SLT.LightingNoise(),
                SLT.AdjustBrightness(2),
                SLT.AdjustContrast(2),
            ]),
            normalize,
        ])

      return T.Compose([
          T.RandomSelect(
              T.RandomResize(scales, max_size=max_size),
              T.Compose([
                  T.RandomResize(scales, max_size=max_size),
              ]),
          ),
          normalize,
      ])

    if image_set in ["val", "eval_debug", "train_reg", "test"]:
      if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == "INFO":
        print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
        return T.Compose([
            T.ResizeDebug((1280, 800)),
            normalize,
        ])

      return T.Compose([
          T.RandomResize([max(scales)], max_size=max_size),
          normalize,
      ])

  h_ref = getattr(args, "line_resize_h_ref", 90)
  max_width = getattr(args, "line_resize_max_width", 1400)
  if getattr(args, "preserve_line_aspect_ratio", False):
    resize_tf = T.ResizeToFixedHeightPreserveAspectRatio(
        h_ref=h_ref, max_width=max_width
    )
  else:
    resize_tf = T.ResizeToFixedHeightAndMaxWidth(h_ref=h_ref, max_width=max_width)
  return T.Compose([resize_tf, normalize])


def build_line_dataset(image_set, args):
  transforms = make_coco_transforms(image_set, args=args)
  if not getattr(args, "data_folder", None):
    args.data_folder = None
  if not getattr(args, "document", None):
    args.document = None
  if not getattr(args, "split", None):
    args.split = None
  if not getattr(args, "script", None):
    args.script = None
  if not getattr(args, "line", None):
    args.line = None
  return LineDataset(
      image_set,
      transforms,
      data_folder=args.data_folder,
      document=args.document,
      split=args.split,
      script=args.script,
      line=args.line,
  )
