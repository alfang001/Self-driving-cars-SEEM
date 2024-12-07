import copy
import logging

import numpy as np
import torch
from detectron2.data import transforms as T
from PIL import Image

__all__ = ["NuScenesSemanticSegmentationMapper"]


def build_transform_gen(cfg, is_train):
    """
    Build transformations for nuScenes dataset.
    """
    assert is_train, "Only support training augmentation"
    cfg_input = cfg['INPUT']
    image_size = cfg_input['IMAGE_SIZE']
    min_scale = cfg_input['MIN_SCALE']
    max_scale = cfg_input['MAX_SCALE']

    augmentations = []

    if cfg_input['RANDOM_FLIP'] != "none":
        augmentations.append(
            T.RandomFlip(
                horizontal=cfg_input['RANDOM_FLIP'] == "horizontal",
                vertical=cfg_input['RANDOM_FLIP'] == "vertical",
            )
        )

    augmentations.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentations


class NuScenesSemanticSegmentationMapper:
    """
    A callable that takes a nuScenes dataset dict and maps it into a Detectron2-compatible format.

    1. Reads the RGB image and corresponding semantic segmentation mask.
    2. Applies geometric transformations to the image and mask.
    3. Prepares the data for 2D semantic segmentation tasks.
    """

    def __init__(
        self,
        is_train=True,
        tfm_gens=None,
        image_format="jpg",
        semseg_format="jpg",
    ):
        self.is_train = is_train
        self.tfm_gens = tfm_gens
        self.image_format = image_format
        self.semseg_format = semseg_format  # Format of semantic segmentation files (e.g., "png" or "npz")
        logging.getLogger(__name__).info(
            "[NuScenesSemanticSegmentationMapper] Using TransformGens: {}".format(str(self.tfm_gens))
        )

    @classmethod
    def from_config(cls, cfg, is_train=True):
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg['INPUT']['FORMAT'],
            "semseg_format": cfg['DATASET']['SEMSEG_FORMAT'],
        }
        return ret

    def read_semseg(self, file_name):
        """
        Reads the semantic segmentation mask.
        """
        if self.semseg_format == "png":
            return np.asarray(Image.open(file_name))
        elif self.semseg_format == "npz":
            return np.load(file_name)['arr_0']
        elif self.semseg_format == "jpg":
            return np.asarray(Image.open(file_name))
        else:
            raise ValueError(f"Unsupported semantic segmentation format: {self.semseg_format}")

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one sample.

        Returns:
            dict: A Detectron2-compatible format.
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        file_name = dataset_dict["file_name"]
        semseg_name = dataset_dict["sem_seg_file_name"]
        image = Image.open(file_name).convert("RGB")
        dataset_dict["width"] = image.size[0]
        dataset_dict["height"] = image.size[1]
        image = torch.from_numpy(np.asarray(image).copy())
        image = image.permute(2,0,1)

        semseg = self.read_semseg(semseg_name)
        semseg = torch.from_numpy(semseg.astype(np.uint8))
        dataset_dict["image"] = image
        dataset_dict["sem_seg"] = semseg

        return dataset_dict
