import sys,os
import torch

# Add the path for the SEEM project to the sys path
sys.path.insert(0, 'Segment-Everything-Everywhere-All-At-Once')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from datasets.evaluation.classification_evaluation import ClassificationEvaluator
from modeling.BaseModel import BaseModel
from utils.arguments import load_opt_command
from modeling import build_model

from dataloader import build_detection_test_loader


# Make the main file accept arguments
def main(args=None):
    '''
    Make sure to run this as python load_seem.py evaluate --conf_files Segment-Everything-Everywhere-All-At-Once/configs/seem/focall_unicl_lang_v1.yaml
    '''
     # Parse arguments
    opt, cmdline_args = load_opt_command(args)
    opt['device'] = 'cuda'

    # Load the model into eval and cuda
    model = BaseModel(opt, build_model(opt)).from_pretrained("seem_focall_v1.pt").eval().cuda()
    

    # Load the eval dataset
    # eval_dataloader = build_detection_test_loader()

    # for images, labels in eval_dataloader:
    #     outputs = model(images)

    # TODO: can call the model with model(input) to get the output

if __name__ == '__main__':
    # Call the main function
    main()