import collections
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.file_io import PathManager
from nuscenes import NuScenes

# Panoptic labels for nuscenes
NUSC_ID_2_NAME = {0: 'noise',
 1: 'animal',
 2: 'human.pedestrian.adult',
 3: 'human.pedestrian.child',
 4: 'human.pedestrian.construction_worker',
 5: 'human.pedestrian.personal_mobility',
 6: 'human.pedestrian.police_officer',
 7: 'human.pedestrian.stroller',
 8: 'human.pedestrian.wheelchair',
 9: 'movable_object.barrier',
 10: 'movable_object.debris',
 11: 'movable_object.pushable_pullable',
 12: 'movable_object.trafficcone',
 13: 'static_object.bicycle_rack',
 14: 'vehicle.bicycle',
 15: 'vehicle.bus.bendy',
 16: 'vehicle.bus.rigid',
 17: 'vehicle.car',
 18: 'vehicle.construction',
 19: 'vehicle.emergency.ambulance',
 20: 'vehicle.emergency.police',
 21: 'vehicle.motorcycle',
 22: 'vehicle.trailer',
 23: 'vehicle.truck',
 24: 'flat.driveable_surface',
 25: 'flat.other',
 26: 'flat.sidewalk',
 27: 'flat.terrain',
 28: 'static.manmade',
 29: 'static.other',
 30: 'static.vegetation',
 31: 'vehicle.ego'}

NUSC_ID_2_NAME_LIST = []
# Go through each key-value pair in the dictionary and add a dictionary {id, name} to the list
for key, value in NUSC_ID_2_NAME.items():
    NUSC_ID_2_NAME_LIST.append({'id': key, 'name': value})

# Converting the nuscenes categories to coco categories
NUSCENES_2_COCO = {}

def get_nuscenes_full_meta():
    stuff_ids = [k["id"] for k in NUSC_ID_2_NAME_LIST]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in NUSC_ID_2_NAME_LIST]
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret

# TODO: Map nuscenes categories to detectron2 categories
def load_nuscenes_semantic_segmentation(name, dirname):
    IMG_FOLDER = os.path.join(dirname, 'samples', 'CAM_FRONT')
    # TODO: Get the labels
    nusc_dir = os.path.join(dirname, 'nuscenes')
    nusc = NuScenes(version='v1.0-mini', dataroot=dirname, verbose=True)
    ret = []
    return ret

def get_dataset_dict():
    dataset_dict = []
    nusc = NuScenes(version='v1.0-mini', dataroot='datasets/nuscenes', verbose=True)

    # Go through the images in the dataset
    for i in range(len(nusc.scene)):
        scene = nusc.scene[i]
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        sample_token = first_sample_token

        # iterate through the samples in the scene
        while sample_token != last_sample_token:
            sample = nusc.get('sample', sample_token)
            sensor_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])

            # TODO: Need to get the groundtruth semantic labels for the image
            # 1. Find the masks for the given image
            # 2. Get coco equivalent masks for the given image
            # 3. Make a grayscale image of the masks where each pixel is labeled with a coco category as an integer

            # Get the file name, id
            dataset_dict.append({
                'file_name': sensor_data['filename'],
                'image_id': sensor_data['token'],
            })
            sample_token = sample['next']
    return dataset_dict

def register_nuscenes_sem_seg(name):
    meta = get_nuscenes_full_meta()
    image_dir = 'datasets/nuscenes/samples/CAM_FRONT'
    gt_dir = 'datasets/nuscenes/sem_seg_samples_gt/CAM_FRONT'
    DatasetCatalog.register('nuscenes_mini_val_v1', lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='jpg', image_ext='jpg'))
    MetadataCatalog.get('nuscenes_mini_val_v1').set(
        stuff_classes=meta['stuff_classes'][:],
        image_root=image_dir,
        sem_seg_root=gt_dir,
        thing_dataset_id_to_contiguous_id={},
        evaluator_type="sem_seg",
        ignore_label=0,
        class_offset=0,
        keep_sem_bgd=False,
    )

_root = os.getenv("DATASET", "datasets")
register_nuscenes_sem_seg(_root)