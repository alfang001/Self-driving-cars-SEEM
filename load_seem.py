import torch


def main():
    # Load the model
    model = torch.load('model.pth')
    print(model)
    # Turn the model into eval mode
    model.eval()
    # TODO: can call the model with model(input) to get the output

if __name__ == '__main__':
    main()