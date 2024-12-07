# import torch
# from modeling.utils import configurable

# __all__ = ["NuscenesDatasetMapper"]

# class NuscenesDatasetMapper:
#     @configurable
#     def __init__(self, is_train=True, tfm_gens=None, image_format=None, min_size_test=None, max_size_test=None, mean=None, std=None):
#         """
#         NOTE: this interface is experimental.
#         Args:
#             is_train: for training or inference
#             augmentations: a list of augmentations or deterministic transforms to apply
#             tfm_gens: data augmentation
#             image_format: an image format supported by :func:`detection_utils.read_image`.
#         """
#         self.tfm_gens = tfm_gens
#         self.img_format = image_format
#         self.is_train = is_train
#         self.min_size_test = min_size_test
#         self.max_size_test = max_size_test
#         self.pixel_mean = torch.tensor(mean)[:,None,None]
#         self.pixel_std = torch.tensor(std)[:,None,None]

#         # t = []
#         # t.append(T.ResizeShortestEdge(min_size_test, max_size=max_size_test))
#         # self.transform = transforms.Compose(t)        # self.transform = transforms.Compose(t)        # self.transform = transforms.Compose(t)        # self.transform = transforms.Compose(t)

#     @classmethod
#     def from_config(cls, cfg, is_train=True):
#         pass

import copy
import logging

import numpy as np
import torch
from detectron2.data import detection_utils as utils
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

        # image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        # utils.check_image_size(dataset_dict, image)

        # # Read the semantic segmentation mask
        # semseg = self.read_semseg(dataset_dict["sem_seg_file_name"])

        # # Apply transformations to the image and mask
        # image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # semseg = transforms.apply_segmentation(semseg)

        # # Convert to torch tensors
        # dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        # dataset_dict["sem_seg"] = torch.as_tensor(semseg.astype("long"))

        return dataset_dict
