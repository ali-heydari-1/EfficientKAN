from torchvision import transforms

def get_data_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (0.5,))
    ])
