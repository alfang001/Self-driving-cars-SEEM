from nuscenes import NuScenes


def main():
    SENSOR = 'CAM_FRONT'
    # Load the dataset
    nusc = NuScenes(version='v1.0-mini', dataroot='datasets/nuscenes', verbose=True)
    for i in range(len(nusc.scene)):
        # Get the current scene
        scene = nusc.scene[i]

        # Get the first sample in the scene
        first_sample_token = scene['first_sample_token']
        sample = nusc.get('sample', first_sample_token)
        sensor_data = nusc.get('sample_data', sample['data'][SENSOR])

if __name__ == '__main__':
    main()