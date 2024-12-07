import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.color_map import get_colormap
from PIL import Image

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


def generate_nuscenes_semantic_segmentation_gt():
    SENSOR_CHANNEL = 'CAM_FRONT'
    POINT_SENSOR_CHANNEL = 'LIDAR_TOP'
    OUTPUT_DIR = 'datasets/nuscenes/sem_seg_samples_gt/CAM_FRONT'
    nusc = NuScenes(version='v1.0-mini', dataroot='datasets/nuscenes', verbose=True)
    nuscenes_name2idx_map = nusc.lidarseg_name2idx_mapping
    nuscenes_idx2name_map = nusc.lidarseg_idx2name_mapping
     # Color map maps the labels to the colors in nuscenes
    color_map = get_colormap()
    inverse_color_map = {v: k for k, v in color_map.items()}

    for i in range(len(nusc.scene)):
        print("Processing scene: ", i)
        scene = nusc.scene[i]

        # First sample in the scene
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        sample_token = first_sample_token

        while sample_token != last_sample_token:
            sample = nusc.get('sample', sample_token)
            pointsensor_token = sample['data'][POINT_SENSOR_CHANNEL]
            camera_token = sample['data'][SENSOR_CHANNEL]
            sample_image_path = nusc.get('sample_data', camera_token)['filename']
            sample_image_filename = sample_image_path.split('/')[-1]

            # Map the point cloud to the image, im is the image without semantic labels on it
            points, coloring, im = nusc.explorer.map_pointcloud_to_image(pointsensor_token, camera_token,
                                                                    render_intensity=False,
                                                                    show_lidarseg=True,
                                                                    filter_lidarseg_labels=None,
                                                                    lidarseg_preds_bin_path=None,
                                                                    show_panoptic=False)
            
            # Make the coloring be in 0-255 format
            coloring = (coloring * 255).astype(np.uint8)

            # Assert that the min and max of the points are within the image size
            assert np.logical_and(np.all(points[0, :] >= 0), np.all(points[0, :] < im.size[0]))
            assert np.logical_and(np.all(points[1, :] >= 0), np.all(points[1, :] < im.size[1]))

            # Points[0,:] and points[1,:] are the x and y coordinates of the points in the image of where the semantic labels are
            cam_front_x, cam_front_y = points[1,:], points[0,:]

            # each entry in coloring corresponds to a point in the point cloud, so coloring[0,:] is the RGB (or BGR) value of the first point in points
            # Numpy array that is grayscale, all zeros to start. Each number is a label associated with the pixel.
            segmentation_ground_truths = np.zeros((im.size[1], im.size[0]), dtype=np.uint8)
            
            # For each entry in the coloring, we need to convert the RGB value to a label
            for i in range(coloring.shape[0]):
                # Find the label for the color
                current_color = tuple(coloring[i,:]) # Gives us a tuple
                # Look through the color map to find the label
                try:
                    current_label = inverse_color_map[current_color]
                except KeyError:
                    current_label = None
                if current_label is None:
                    print(f"Did not find the label for the color: {current_label}")
                
                # Get the nuScenes integer for the label
                current_name2idx = nuscenes_name2idx_map[current_label]

                # Set the label in the segmentation_ground_truths where the point is
                segmentation_ground_truths[int(cam_front_x[i]), int(cam_front_y[i])] = current_name2idx
            
            # Save the ground truth to the output directory
            gt_path = f"{OUTPUT_DIR}/{sample_image_filename}"
            segmentation_ground_truths = Image.fromarray(segmentation_ground_truths)
            segmentation_ground_truths.save(gt_path)
            sample_token = sample['next']
        

def main():
    generate_nuscenes_semantic_segmentation_gt()

if __name__ == '__main__':
    main()