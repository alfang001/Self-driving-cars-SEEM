import copy
import logging

import numpy as np
import torch
from PIL import Image
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

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
        *,
        tfm_gens,
        image_format,
        semseg_format="png",
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
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # Read the semantic segmentation mask
        semseg = self.read_semseg(dataset_dict["sem_seg_file_name"])

        # Apply transformations to the image and mask
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        semseg = transforms.apply_segmentation(semseg)

        # Convert to torch tensors
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["sem_seg"] = torch.as_tensor(semseg.astype("long"))

        return dataset_dict
