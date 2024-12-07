import os
import sys

from nuscenes.nuscenes import NuScenes

sys.path.insert(0, 'Segment-Everything-Everywhere-All-At-Once')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torchvision import transforms
from utils.arguments import load_opt_command

from load_utils import NuScenes2DImageDataset, load_and_evaluate


# Make the main file accept arguments
def main(args=None):
    '''
    Make sure to run this as python load_seem.py evaluate --conf_files Segment-Everything-Everywhere-All-At-Once/configs/seem/focall_unicl_lang_v1.yaml
    '''
     # Parse arguments
    opt, cmdline_args = load_opt_command(args)
    opt['device'] = 'cuda'

    # Load the model into eval and cuda
    # model = BaseModel(opt, build_model(opt)).from_pretrained("seem_focall_v1.pt").eval().cuda()
    nusc = NuScenes(version='v1.0-mini', dataroot='/path/to/nuscenes', verbose=True)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = NuScenes2DImageDataset(nusc, camera="CAM_FRONT", transform=transform)
    results = load_and_evaluate(opt, test_dataset, checkpoint_path=None, batch_size=32)

    print(results)

if __name__ == '__main__':
    # Call the main function
    main()