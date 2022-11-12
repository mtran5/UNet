import numpy as np
from skimage.draw import random_shapes
import torch
import torchvision.transforms as transforms

def generate_images(size=224):
    image, labels = random_shapes((size, size), min_shapes=2, max_shapes=8, intensity_range=((0, 127),), shape="ellipse", min_size=20, allow_overlap=True)
    # Generate mask only for ellipses
    mask = np.sum(image, axis=-1) < 255*3

    image2, _ = random_shapes((size, size), min_shapes=2, max_shapes=8, intensity_range=((0, 127),), shape="triangle", min_size=20, allow_overlap=True)

    image += image2
    image = 255 - image
    noise = np.random.randint(0, 30, (224, 224, 1), dtype = np.uint8)
    image += noise
    return image, mask

class EllipseDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.transform = transforms.Compose([transforms.ToTensor()])
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        # Generate random image sample
        X, y = generate_images()
        # Convert the shape from HxWxC -> CxHxW
        X = self.transform(X)
        y = torch.as_tensor(y, dtype = torch.uint8)
        y = torch.unsqueeze(y, 0)
        return X, y