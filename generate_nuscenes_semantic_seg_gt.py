from nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.color_map import get_colormap


def generate_nuscenes_semantic_segmentation_gt():
    SENSOR_CHANNEL = 'CAM_FRONT'
    POINT_SENSOR_CHANNEL = 'LIDAR_TOP'
    nusc = NuScenes(version='v1.0-mini', dataroot='datasets/nuscenes', verbose=True)
    for i in range(len(nusc.scene)):
        scene = nusc.scene[i]

        # First sample in the scene
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        sample_token = first_sample_token

        while sample_token != last_sample_token:
            sample = nusc.get('sample', sample_token)
            pointsensor_token = sample['data'][POINT_SENSOR_CHANNEL]
            camera_token = sample['data'][SENSOR_CHANNEL]

            # Map the point cloud to the image, im is the image without semantic labels on it
            points, coloring, im = nusc.explorer.map_pointcloud_to_image(pointsensor_token, camera_token,
                                                                    render_intensity=False,
                                                                    show_lidarseg=True,
                                                                    filter_lidarseg_labels=None,
                                                                    lidarseg_preds_bin_path=None,
                                                                    show_panoptic=False)
            # TODO: Need to take the coloring from get_colormap and convert it to the coco-stuffs format for each point
            # then, we need to convert the coloring to a grayscale image where each pixel is labeled with a coco category as an integer
            # finally, we need to save the image to the gt_path.

            # Points[0,:] and points[1,:] are the x and y coordinates of the points in the image of where the semantic labels are
            cam_front_x, cam_front_y = points[0,:], points[1,:]

            # each entry in coloring corresponds to a point in the point cloud, so coloring[0,:] is the RGB (or BGR) value of the first point in points
            
            import pdb; pdb.set_trace()
            sample_token = sample['next']
        

def main():
    generate_nuscenes_semantic_segmentation_gt()

if __name__ == '__main__':
    main()