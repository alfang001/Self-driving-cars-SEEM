import os
import torch
from utils.arguments import load_opt_command
from trainer.utils_trainer import UtilsTrainer
from trainer.default_trainer import DefaultTrainer
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from PIL import Image
import os

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
    opt, _ =  opt
    opt['RESUME_FROM'] = checkpoint_path
    opt['EVAL_AT_START'] = True
    trainer = DefaultTrainer(opt)
    
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


# This is an example on how to use it!! Not 100% sure right now
if __name__ == "__main__":
    
    checkpoint_path = "/path/to/saved/checkpoint"
    test_dataset = TestDataset()
    # Load the model and evaluate on the test dataset
    results = load_and_evaluate(checkpoint_path, test_dataset)
