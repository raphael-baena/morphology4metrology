import os, sys

if __name__ == "__main__":
    sys.path.append(os.path.dirname(sys.path[0]))
import torch
import pickle
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont, ImageChops
from torch.utils.data import Dataset
from torchvision import transforms
import datasets.transforms as T
import random
import multiprocessing
from numpy.random import uniform, choice
from PIL import Image, ImageDraw, ImageFilter
from .generate_canva import generate_canva
from random import randint
import datasets.sltransform as SLT
import re
import unicodedata

current_dir = os.path.dirname(os.path.abspath(__file__))

if "dataset" not in current_dir:
    current_dir = os.path.join(current_dir, "dataset")
else:
    current_dir = os.path.join(current_dir, ".")


with open(current_dir + "/dictionnary_category_ability_paths.json", "r") as f:
    dictionnary_category_ability_paths = json.load(f)
NEG_ELEMENT_BLUR_RADIUS_RANGE = (0.2, 1.6)
POS_ELEMENT_OPACITY_RANGE = {
    "drawing": (200, 255),
    "glyph": (150, 255),
    "image": (150, 255),
    "table": (200, 255),
    "text": (200, 255),
}
TEXT_COLORED_FREQ = 0.5


## padding ratio
padding_left_ratio_min = 0.02
padding_left_ratio_max = 0.1
padding_right_ratio_min = 0.02
padding_right_ratio_max = 0.1
padding_top_ratio_min = 0.02
padding_top_ratio_max = 0.2
padding_bottom_ratio_min = 0.02
padding_bottom_ratio_max = 0.2

with open(current_dir + "/default_charset.json", "r") as f:
    charset = json.load(f)
with open(current_dir + "/default_charset_without_accent.json", "r") as f:
    charset_without_accent = json.load(f)
with open(current_dir + "/accent.json", "r") as f:
    charset_accent = json.load(f)
charset = charset + charset_accent
charset_de = ["ß" if item == "Þ" else item for item in charset]

def remove_accents(text: str) -> str:
    # # 1. Normalize the text to NFD form, decomposing accented characters
    # decomposed = unicodedata.normalize("NFD", text)
    
    # # 2. Filter out all combining diacritics (marks)
    # filtered = [
    #     char for char in decomposed
    #     if not unicodedata.combining(char)
    # ]
    
    # # 3. Join back into a string
    # return "".join(filtered)
    text_without_accent = [unicodedata.normalize('NFD', char)[0] for char in text]
    text_without_accent = "".join(text_without_accent)
    return text_without_accent
class Synthetic(Dataset):
    def __init__(
        self,
        mode,
        transform=transforms.ToTensor(),
        target_transform=None,
        language=None,
    ):
        """
        mode: train, valid, test
        """
        if mode == "val":
            mode = "valid"
        self.mode = mode
        self._transforms = transform
        self.charset = charset
        self.charset_without_accent = charset_without_accent
        self.charset_accent = charset_accent
        self.img_labels = []
        self.transform = transform
        self.target_transform = target_transform
        self.images = []

        self.prop = 10
        self.language = language
        if self.language == "de":
            self.charset = charset_de
        if self.mode == "train":
            self.num_samples = 5000#
        else:
            self.num_samples = 100
        self.create_synthetic_folder()

    def __len__(self):
        return self.num_samples * self.prop

    def create_synthetic_folder(self):
        if self.language is None:
            self.saving_path = "synthetic_images_symbols"
        elif self.language == "en":
            self.saving_path =  "synthetic_images_english"
        elif self.language == "de":
            self.saving_path = "synthetic_images_german"
        elif self.language == "fr":
            self.saving_path = "synthetic_images_french"
        self.saving_path = os.path.join(current_dir, self.saving_path)
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)
        if not os.path.exists(os.path.join(self.saving_path, self.mode)):
            os.makedirs(os.path.join(self.saving_path, self.mode))  

    def generates_synthetic(self, text, idx, font_paths):
        font_path = np.random.choice(font_paths)
        font_path = current_dir + "/" + font_path
    
        image, xy, bounding_boxes,labels_text= self.generate_textimage_with_bounding_boxes(
            text, font_path
        )
        image = generate_canva({"img": image, "position": (0, 0)})
        labels = {}
        labels["labels"] = labels_text
        # for char in text:
        #     if char in self.charset:
        #         labels["labels"].append(self.charset.index(char))
        #     else:
        #         decoded_char = char.encode().decode("unicode_escape")
        #         if decoded_char in self.charset:
        #             labels["labels"].append(self.charset.index(decoded_char))
        #         else:
        #             ## print unicode value of char

        #             raise ValueError(f"char {char} not in charset")
        labels["labels"] = labels["labels"]
        labels["boxes"] = bounding_boxes
        labels["orig_size"] = [image.size[1], image.size[0]]
        labels["size"] = labels["orig_size"]
        labels["image_id"] = idx
        labels["idx"] = idx
        labels["font_path"] = font_path
        return image, labels

    def __getitem__(self, idx):
        idx = idx // self.prop
        image = Image.open(
                os.path.join(
                    self.saving_path, self.mode, f"{idx}.jpg")).convert("RGB")

        with open(
            os.path.join(
               self.saving_path, self.mode, f"{idx}.json"),
            "r",
        ) as f:
            labels_json = json.load(f)

        labels = {}

        labels["labels"] = torch.tensor(labels_json["labels"], dtype=torch.int64)
        #convert -1 to len(self.charset) +1
        labels["labels"][labels["labels"] == -1] = len(self.charset) 
        labels_json["boxes"] = torch.tensor(labels_json["boxes"], dtype=torch.float32)
        labels["boxes"] = torch.tensor(labels_json["boxes"], dtype=torch.float32) 
        labels["orig_size"] = torch.tensor(labels_json["orig_size"], dtype=torch.int64)
        labels["size"] = labels["orig_size"]
        labels["image_id"] = torch.tensor(labels_json["idx"], dtype=torch.int64)
        labels["idx"] = torch.tensor(labels_json["idx"], dtype=torch.int64)

        image, labels = self._transforms(image, labels)
        return image, labels

    def random_text(self, charset):
        ## sample 1 or 2
        if random.randint(1, 2) == 1:
            charset = self.charset
            d_fonts = sample_d_fonts("fonts_letters_with_accent_and_symbols")
            nb_words = random.randint(1, 5)
        else:
            charset = self.charset_without_accent
            d_fonts = sample_d_fonts("fonts_letters_with_accent_and_numbers")
            nb_words = random.randint(1, 30)
        text = []
        for i in range(nb_words):
            length_word = random.randint(1, 15)
            for j in range(length_word):
                text.append(random.choice(charset))
            if i < nb_words - 1:
                text.append(" ")
        text = "".join(text)
        if len(text) > 100:
            text = text[0:100]
        return text, d_fonts

    def random_text_from_wikitext(self):
        if self.language == "en":
            if "val" in self.mode:
                with open(current_dir + "/resources/text/en/val.txt") as f:
                    text_set = f.readlines()
            else:
                i = random.choice(range(1, 6))
                with open(current_dir + f"/resources/text/en/train_split_{i}.txt") as f:
                    text_set = f.readlines()
        elif self.language == "de":
            if "val" in self.mode:
                with open(current_dir + "/resources/text/de/val.txt") as f:
                    text_set = f.readlines()
            else:
                i = random.choice(range(1, 6))
                with open(current_dir + f"/resources/text/de/train_split_{i}.txt") as f:
                    text_set = f.readlines()
        elif self.language == "fr":
            if "val" in self.mode:
                with open(current_dir + "/resources/text/fr/val.txt") as f:
                    text_set = f.readlines
            else:
                i = random.choice(range(1, 6))
                with open(current_dir + f"/resources/text/fr/train_split_{i}.txt") as f:
                    text_set = f.readlines()

        for _ in range(100):
            current_text = random.choice(text_set)
            if len(current_text) < 2:
                continue
            current_text = current_text.split("\n")[:-1]
            idx_line = random.randint(0, len(current_text) - 1)
            current_text = current_text[idx_line]
            if current_text.startswith(" = "):
                continue
            current_text = re.sub(
                r""" \.| ,|" | :| ;| '|""",
                lambda match: match.group().strip(),
                current_text,
            )
            current_text = re.sub(r"\( ", "(", current_text)
            current_text = re.sub(r" \)", ")", current_text)

            current_text = re.sub(r" @-@ ", "-", current_text)
            current_text = re.sub(r" @.@ ", ".", current_text)
            # print(current_text)
            break

        if len(current_text) > 100:

            words = current_text.split()
            for _ in range(10):
                end_index = random.randint(
                    min(1, len(words) - 1), min(len(words) - 1, 20)
                )
                current_text = " ".join(words[:end_index])
                if len(current_text) > 100:
                    end_index = random.randint(
                        50, 100
                    )  # could be an issue for the language model if we are cutting a word in the end
                    current_text = current_text[0:end_index]
                if len(current_text) > 1:
                    break

        return current_text
    
    def generate_image(self, k, current_dir, mode):
        print("\r", f"Generating synthetic image {k+1}/{self.num_samples}", end="")
        while True:
            try:
                if random.randint(1, 2) == 1 and self.language is not None:
                    text = self.random_text_from_wikitext()
                    text = clean_text(text)
                    d_fonts = sample_d_fonts("fonts_letters_with_accent_and_symbols")
                else:
                    text, d_fonts = self.random_text(self.charset)
                    text = clean_text(text)
                image, labels = self.generates_synthetic(text, k, d_fonts)
                im_path = os.path.join(self.saving_path, mode, f"{k}.jpg")

                image.save(im_path)
                label_path = os.path.join( self.saving_path, mode, f"{k}.json"
                    )
                with open(label_path, "w") as f:
                        json.dump(labels, f)
                break
            except Exception as e:
                print(e)
                continue
        
            ## if ctrl+c is pressed, stop the generation
            except KeyboardInterrupt:
                break

    def generates_synthetic_data(self):
        # check that folder synthetic_images_symbols exists


        pool = multiprocessing.Pool()
        results = [
            pool.apply_async(self.generate_image, args=(k, current_dir, self.mode))
            for k in range(self.num_samples)
        ]
        output = [p.get() for p in results]
        pool.close()
        # for k in range(self.num_samples):
        #     self.generate_image(k, current_dir, self.mode)
    def generate_textimage_with_bounding_boxes(self, text, font_path):
        # --- 1. Initialization ---
        font_size_min = 30
        font_size_max = 50
        font_size = int(torch.randint(font_size_min, font_size_max + 1, (1,)).item())
        font = ImageFont.truetype(font_path, size=font_size)

        # --- 2. Global dimensions ---
        full_bbox = font.getbbox(text)

        # Safety: if text is only spaces or empty, full_bbox may be None
        # Use the size of an "A" as default reference
        if not full_bbox:
            ref_bbox = font.getbbox("A")
            if not ref_bbox:
                ref_bbox = (0, -font_size, font_size, 0)  # Ultimate fallback
            full_top_rel = ref_bbox[1]
            full_bottom_rel = ref_bbox[3]
            text_width = font.getlength(text)
            text_height = ref_bbox[3] - ref_bbox[1]
            full_bbox = (0, full_top_rel, text_width, full_bottom_rel)  # Fake bbox
        else:
            # Store global vertical limits (top and bottom) relative to baseline
            full_top_rel = full_bbox[1]
            full_bottom_rel = full_bbox[3]
            text_width = full_bbox[2] - full_bbox[0]
            text_height = full_bottom_rel - full_top_rel

        # --- 3. Padding ---
        padding_left = random.randint(0, int(text_width * 0.1) + 1)
        padding_right = random.randint(0, int(text_width * 0.1) + 1)
        padding_top = random.randint(0, int(text_height * 0.4) + 1)
        padding_bottom = random.randint(0, int(text_height * 0.4) + 1)

        img_width = int(padding_left + text_width + padding_right)
        img_height = int(padding_top + text_height + padding_bottom)

        draw_x = random.randint(0, max(0, int(img_width - text_width)))
        draw_y = random.randint(0, max(0, int(img_height - text_height)))
        xy = (draw_x, draw_y)

        # Baseline position in the image
        baseline_x = draw_x - full_bbox[0]
        baseline_y = draw_y - full_bbox[1]

        bounding_boxes_char = []
        bounding_boxes_char_without_accent = []

        def get_bounding_boxes_char(char, full_text, font, char_idx):
            char_advance_x = font.getlength(full_text[:char_idx])
            
            if char == " ":
                space_width = font.getlength(" ")
                x_min = baseline_x + char_advance_x
                x_max = x_min + space_width
                y_min = baseline_y + full_top_rel
                y_max = baseline_y + full_bottom_rel
            else:
                char_bbox = font.getbbox(char)
                if not char_bbox:
                    char_bbox = (0, -font_size, font.getlength(char), 0)
                x_min = baseline_x + char_advance_x + char_bbox[0]
                y_min = baseline_y + char_bbox[1]
                x_max = baseline_x + char_advance_x + char_bbox[2]
                y_max = baseline_y + char_bbox[3]
            
            x_min = max(0, min(img_width - 1e-8, x_min))
            x_max = max(0, min(img_width - 1e-8, x_max))
            y_min = max(0, min(img_height - 1e-8, y_min))
            y_max = max(0, min(img_height - 1e-8, y_max))
            
            return x_min, y_min, x_max, y_max

        text_without_accent = remove_accents(text)
        for i, char in enumerate(text):
            char = unicodedata.normalize('NFD', char)
            x_min, y_min, x_max, y_max = get_bounding_boxes_char(char,text, font, i)
            bounding_boxes_char.append([x_min, y_min, x_max, y_max])
            #char_without_accent: eg "é" -> "e"
            char_without_accent = unicodedata.normalize('NFD', char)[0]
    
            
            x_min_without_accent, y_min_without_accent, x_max_without_accent, y_max_without_accent = get_bounding_boxes_char(
                char_without_accent,text_without_accent, font, i)
            bounding_boxes_char_without_accent.append([x_min_without_accent, y_min_without_accent, x_max_without_accent, y_max_without_accent])


        image = Image.new("RGBA", (img_width, img_height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)

        blur_radius = uniform(*NEG_ELEMENT_BLUR_RADIUS_RANGE)
        opacity = randint(*POS_ELEMENT_OPACITY_RANGE["text"])
        color_range = (0, 75)
        colored = choice([True, False], p=[TEXT_COLORED_FREQ, 1 - TEXT_COLORED_FREQ])
        colors = (
            tuple([randint(*color_range)] * 3)
            if not colored
            else tuple([randint(*color_range) for _ in range(3)])
        )
        colors_alpha = colors + (opacity,)

        draw.text(xy, text, font=font, fill=colors_alpha, anchor="lt")
        # resize the image
        image_without_filter = image.copy()
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))
        image = image.resize((img_width, img_height))
        
        # text_without_accent = remove_accents(text)

        image_without_accent = Image.new("RGBA", (img_width, img_height), (255, 255, 255, 0))
        # print('text_without_accent ',text_without_accent)
        draw = ImageDraw.Draw(image_without_accent)
        
        # Calculate the correct y position so it aligns on the same baseline
        bbox_without = font.getbbox(text_without_accent)
        if bbox_without:
            xy_without = (xy[0], baseline_y + bbox_without[1])
        else:
            xy_without = xy
            
        draw.text(xy_without, text_without_accent, font=font, fill=colors_alpha, anchor="lt")
        # # resize the image
        # image_without_accent = image_without_accent.filter(ImageFilter.GaussianBlur(blur_radius))
        # image_without_accent = image_without_accent.resize((img_width, img_height))
    
                    
        new_bounding_boxes_char = []
        labels_text = []
        for i in range(len(text)):
            char = text[i]
            char = unicodedata.normalize('NFD', char)
            char  = [c for c in char ]
            if len(char) ==1:
                char = char[0]
                new_bounding_boxes_char.append([bounding_boxes_char[i],bounding_boxes_char[i],[0,0,0,0]])
                if char in self.charset:
                    labels_text.append([self.charset.index(char),self.charset.index(char),-1])
                else:
                    decoded_char = char.encode().decode("unicode_escape")
                    if decoded_char in self.charset:
                        labels_text.append([self.charset.index(decoded_char),self.charset.index(decoded_char),-1])
                    else:
                        ## print unicode value of char

                        raise ValueError(f"char {char} not in charset")
            else:

                bbox_whole_char = bounding_boxes_char[i]
                cropped_char = image_without_filter.crop(bbox_whole_char)
                cropped_char_without_accent = image_without_accent.crop(bbox_whole_char)
                diff = ImageChops.difference(cropped_char, cropped_char_without_accent)
                # compare pixel between cropped_char and diff to find bbox of accent
                diff = diff.convert('L')
                diff_gray = diff.convert('L')
                accent_bbox = diff_gray.getbbox()
                # bbox_whole_char = bounding_boxes_char[i]
                # bbox_single_char = bounding_boxes_char_without_accent[i]

                # accent_bbox
                # #we need to shift accent_bbox to the original image
                if accent_bbox is not None:
                    accent_bbox = (accent_bbox[0] + bbox_whole_char[0], accent_bbox[1] + bbox_whole_char[1], accent_bbox[2] + bbox_whole_char[0], accent_bbox[3] + bbox_whole_char[1])
                #bounding_boxes_char.append(accent_bbox)
                    sub_labels_text = []
                    sub_bounding_boxes = []
                    for idx_char, c in enumerate(char):
                        if c in self.charset: #first char is always the main char
                                sub_labels_text.append(self.charset.index(c))
                        else:
                            decoded_char = c.encode().decode("unicode_escape")
                            if decoded_char in self.charset:
                                sub_labels_text.append(self.charset.index(decoded_char))
                            else:
                                ## print unicode value of char
                                    raise ValueError(f"char {c} not in charset, char {char}")
                        if idx_char == 0:
                            sub_bounding_boxes.append(bounding_boxes_char[i]) # first the whole char then without accent
                            sub_bounding_boxes.append(bounding_boxes_char_without_accent[i])
                        else:   
                            sub_bounding_boxes.append(accent_bbox)
                    #dupplicate sub_labels_text[0] to have the same size as sub_bounding_boxes and it at the first position
                    sub_labels_text.insert(0,sub_labels_text[0])
                    new_bounding_boxes_char.append(sub_bounding_boxes)
                    labels_text.append(sub_labels_text)
                else:
                    new_bounding_boxes_char.append([bounding_boxes_char[i],bounding_boxes_char[i],[0,0,0,0]])
                    c = char[0]
                    if c in self.charset:
                        labels_text.append([self.charset.index(c),self.charset.index(c),-1])
                    else:
                        decoded_char = c.encode().decode("unicode_escape")
                        if decoded_char in self.charset:
                            labels_text.append([self.charset.index(decoded_char),self.charset.index(decoded_char),-1])

        return image, xy, new_bounding_boxes_char, labels_text

def clean_text(text):
    new_text = []
    for char in text:
        if char in charset:
            new_text.append(char)
        else:
            decoded_char = char.encode().decode("unicode_escape")
            if decoded_char in charset:
                new_text.append(decoded_char)
    return "".join(new_text)


def sample_d_fonts(ability):
    if random.randint(1, 2) == 1:
        category = "HANDWRITING"
    else:
        category = random.choice(["SANS_SERIF", "MONOSPACE", "SERIF", "DISPLAY"])
    return dictionnary_category_ability_paths[category][ability]


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]

    # update args from config files
    scales = getattr(args, "data_aug_scales", scales)
    max_size = getattr(args, "data_aug_max_size", max_size)
    scales2_resize = getattr(args, "data_aug_scales2_resize", scales2_resize)
    scales2_crop = getattr(args, "data_aug_scales2_crop", scales2_crop)
    random_erasing = getattr(args, "random_erasing", False)

    # resize them
    data_aug_scale_overlap = getattr(args, "data_aug_scale_overlap", None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i * data_aug_scale_overlap) for i in scales]
        max_size = int(max_size * data_aug_scale_overlap)
        scales2_resize = [int(i * data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i * data_aug_scale_overlap) for i in scales2_crop]

    datadict_for_print = {
        "scales": scales,
        "max_size": max_size,
        "scales2_resize": scales2_resize,
        "scales2_crop": scales2_crop,
    }

    if fix_size:
        return T.Compose(
            [
                T.RandomResize([(max_size, max(scales))]),
                normalize,
            ]
        )

    

    if image_set == "train":
        if random_erasing:
            random_erasing_transforms = [
                T.RandomErasingFullVertical(p=0.5, scale=(0.02, 0.05), ratio=(3, 6))
                for _ in range(5)
            ]

            return T.Compose(
                [
                    T.RandomSelect(
                        T.RandomResize(scales, max_size=max_size),
                        T.Compose(
                            [
                                T.RandomResize(scales, max_size=max_size),
                            ]
                        ),
                    ),
                    normalize,
                    *random_erasing_transforms,
                T.RandomErasing(p=0.5, scale=(0.01, 0.02), ratio=(0.1, 1)),
                T.RandomErasing(p=0.5, scale=(0.01, 0.02), ratio=(0.1, 1)),
                T.RandomErasing(p=0.5, scale=(0.01, 0.02), ratio=(0.1, 1)),
                T.RandomErasing(p=0.5, scale=(0.01, 0.02), ratio=(0.1, 1)),
                ]
            )
        
        else:
            return T.Compose(
                [
                    T.RandomSelect(
                        T.RandomResize(scales, max_size=max_size),
                        T.Compose(
                            [
                                T.RandomResize(scales, max_size=max_size),
                            ]
                        ),
                    ),
                    T.GaussianBlur(kernel=(3, 3), sigma=(1, 1)),
                    normalize,
                ]
            )
    elif image_set == "val":

        return T.Compose(
            [
                T.RandomResize(scales[-1:], max_size=max_size),
                normalize,
            ]
        )






def build_synthetic_line_OCR_general(image_set, args):
    transforms = make_coco_transforms(image_set, args=args)
    # check if english is in args

    language = getattr(args, "language", None)
    if language is not None:
        print(f"USING language: {language} ")
    return Synthetic(image_set, transforms, language=language)