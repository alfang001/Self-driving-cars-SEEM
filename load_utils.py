import os
import sys

import torch
from nuscenes.nuscenes import NuScenes
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.insert(0, 'Segment-Everything-Everywhere-All-At-Once')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trainer.default_trainer import DefaultTrainer
from trainer.utils_trainer import UtilsTrainer
from utils.arguments import load_opt_command


class NuScenes2DImageDataset(Dataset):
    def __init__(self, nusc, camera="CAM_FRONT", transform=None):
        """
        Initialize the dataset for 2D images.
        :param nusc: NuScenes object.
        :param camera: Camera channel (e.g., 'CAM_FRONT', 'CAM_BACK', etc.).
        :param transform: Transformations to apply to the images.
        """
        self.nusc = nusc
        self.camera = camera
        self.transform = transform
        self.samples = [s['data'][camera] for s in nusc.sample]  # Get all samples for the specified camera

    def __len__(self):
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a single sample.
        :param idx: Index of the sample.
        :return: A dictionary containing the image and metadata.
        """
        cam_token = self.samples[idx]
        cam_data = self.nusc.get('sample_data', cam_token)
        cam_path = self.nusc.get_sample_data_path(cam_token)
        
        # Load the image
        image = Image.open(cam_path).convert("RGB")
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        # Prepare the data dictionary
        data = {
            'image': image,
            'file_path': cam_path,
            'camera_token': cam_token,
            'timestamp': cam_data['timestamp']
        }

        return data


def load_and_evaluate(opt, checkpoint_path, test_dataset, batch_size=32):
    """
    Loads a saved model from the checkpoint and evaluates it on the test dataset.
    
    :param checkpoint_path: Path to the saved model checkpoint.
    :param test_dataset: A PyTorch Dataset object for the test set.
    :param batch_size: Batch size for testing.
    """
    # Load options and initialize trainer
    opt['RESUME_FROM'] = checkpoint_path
    opt['EVAL_AT_START'] = True
    opt['WEIGHT'] = True
    trainer = DefaultTrainer(opt)
    return trainer.eval()
    
    # Load the checkpoint
    trainer.load_checkpoint(checkpoint_path)
    print(f"Model loaded from checkpoint: {checkpoint_path}")
    
    # Set model to evaluation mode
    for module_name in trainer.model_names:
        trainer.models[module_name].eval()
    
    # Prepare the test dataloader
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=opt.get('NUM_WORKERS', 4),
        pin_memory=True
    )
    
    # Evaluate the model on the test dataset
    results = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            inputs = batch['input'].to(opt['device'])  
            targets = batch['target'].to(opt['device']) 
            
            outputs = {}
            for module_name in trainer.model_names:
                outputs[module_name] = trainer.models[module_name](inputs)
            
            # Compute metrics
            # results[module_name].append(custom_metric(outputs[module_name], targets))
            
            print(f"Processed batch {batch_idx + 1}/{len(test_dataloader)}")
    
    print("Evaluation completed.")
    return results


def main(args=None):
    opt, cmdline_args = load_opt_command(args)
    opt['device'] = 'cuda'
    checkpoint_path = "seem_focall_v1.pt"
    nusc = NuScenes(version='v1.0-mini', dataroot='datasets/nuscenes/', verbose=True)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = NuScenes2DImageDataset(nusc, camera="CAM_FRONT", transform=transform)
    # Load the model and evaluate on the test dataset
    results = load_and_evaluate(opt, checkpoint_path, test_dataset)
    print(results)

# This is an example on how to use it!! Not 100% sure right now
if __name__ == "__main__":
    main()
