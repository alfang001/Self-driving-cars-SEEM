import argparse

import torch


# Make the main file accept arguments
def main(kwargs):
    # Load the model
    model = torch.load('model.pth')
    print(model)
    # Turn the model into eval mode
    model.eval()
    # TODO: can call the model with model(input) to get the output

if __name__ == '__main__':
    # Parse the arguments from the commandline
    parser = argparse.ArgumentParser()
    # Add the arguments
    parser.add_argument('--input', type=str, help='The input file path')
    # Parse the arguments
    args = parser.parse_args()
    # Call the main function
    main(args)