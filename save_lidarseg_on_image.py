import os

from nuscenes import NuScenes


def main():
    nusc = NuScenes(version='v1.0-mini', dataroot='datasets/nuscenes', verbose=True)
    sample_indices = [30, 40, 55, 87]

    for index in sample_indices:
        sample = nusc.sample[index]
        nusc.render_pointcloud_in_image(
            sample['token'],
            pointsensor_channel='LIDAR_TOP',
            camera_channel='CAM_FRONT',
            render_intensity=False,
            show_lidarseg=True,
            show_lidarseg_legend=True,
        )
        # Render the corresponding image without labels
        cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        cam_front_filename = cam_front_data['filename']

        # Copy the image to the output directory
        os.system(f'cp datasets/nuscenes/{cam_front_filename} output/{index}.png')

if __name__ == '__main__':
    main()