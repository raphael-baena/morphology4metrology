# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision



def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'synthetic_line_OCR_general':
        from .synthetic_lines_general import build_synthetic_line_OCR_general
        return build_synthetic_line_OCR_general(image_set, args)
    if args.dataset_file == 'google1000':
        from .google1000 import build_google1000
        return build_google1000(image_set, args)
    if args.dataset_file == 'IAM':
        from .IAM import build_iam
        return build_iam(image_set, args)
    if args.dataset_file == 'READ':
        from .READ import build_READ
        return build_READ(image_set, args)
    if args.dataset_file == 'RIMES':
        from .RIMES import build_RIMES
        return build_RIMES(image_set, args)
    # Ciphers
    if args.dataset_file =='borg':
        from .borg import build_borg
        return build_borg(image_set, args)
    if args.dataset_file == 'copiale':
        from .copiale import build_copiale
        return build_copiale(image_set, args)
    # Chinese
    if args.dataset_file =='HWDB_synth':
        from .HWDB_Synth import build_synthetic_HWDB
        return build_synthetic_HWDB(image_set, args)
    if args.dataset_file == 'HWDB':
        from .HWDB import build_HWDB
        return build_HWDB(image_set, args)
     #ICDAR COMPETITION
    if args.dataset_file == 'icdar':
        from .icdar import build_icdar
        return build_icdar(image_set, args)
    #ICDAR COMPETITION
    if args.dataset_file == 'icdar_antiqua':
        from .icdar import build_icdar_antiqua
        return build_icdar_antiqua(image_set, args)
    if args.dataset_file == 'icdar_italic':
        from .icdar import build_icdar_italic
        return build_icdar_italic(image_set, args)
    if args.dataset_file == "icdar_bastarda":
        from .icdar import  build_icdar_bastarda
        return build_icdar_bastarda(image_set, args)
    if args.dataset_file == "icdar_fraktur":
        from .icdar import  build_icdar_fraktur
        return build_icdar_fraktur(image_set, args)
    if args.dataset_file == "icdar_gotico_antiqua":
        from .icdar import  build_icdar_gotico_antiqua
        return build_icdar_gotico_antiqua(image_set, args)
    if args.dataset_file == "icdar_rotunda":
        from .icdar import  build_icdar_rotunda
        return build_icdar_rotunda(image_set, args)
    if args.dataset_file == "icdar_schwabacher":
        from .icdar import  build_icdar_schwabacher
        return build_icdar_schwabacher(image_set, args)
    if args.dataset_file == "icdar_textura":
        from .icdar import  build_icdar_textura
        return build_icdar_textura(image_set, args)
    if args.dataset_file == "icdar_multi":
        from .icdar import  build_icdar_multi
        return build_icdar_multi(image_set, args)
    if args.dataset_file == 'BNF':
        from .BNF import build_BNF
        return build_BNF(image_set, args)
    if args.dataset_file == 'ramanacoil':
        from .ramanacoil import build_ramanacoil
        return build_ramanacoil(image_set, args)
    if args.dataset_file == 'Borg':
        from .Borg import build_Borg
        return build_Borg(image_set, args)
    if args.dataset_file == 'dataset':
        from .dataset import build_line_dataset
        return build_line_dataset(image_set, args)


    if args.dataset_file == 'custom':
        from .custom_dataset import build_custom
        return build_custom(image_set, args)
    
    raise ValueError(f'dataset {args.dataset_file} not supported')
