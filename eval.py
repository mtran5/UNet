import torch
from sklearn.metrics import jaccard_score
import numpy as np
from generate_images import EllipseDataset
from models import UNet
from tqdm import tqdm
import matplotlib.pyplot as plt

dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print('using device:', device)

if __name__ == "__main__":
    unet = UNet(nclass=1, in_chans=3, depth=3, skip=True, padding="same")

    unet.load_state_dict(torch.load("unet.pkl"))
    dataset = EllipseDataset(2000)

    # Make sample predictions
    unet.eval()

    unet.to(device)
    sigmoid = torch.nn.Sigmoid()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    total_scores = 0.0

    for t, (X, y) in enumerate(tqdm(loader)):
        y = y.numpy()
        scores = unet(X.to(device, dtype = dtype)).squeeze()
        
        # Convert scores to mask
        scores = sigmoid(scores)
        scores = scores.detach().cpu().numpy()
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        scores = scores.astype(np.uint8)
        
        # Calculate jacaard scores
        jscore = jaccard_score(y.flatten(), scores.flatten(), average="binary", pos_label=1)
        total_scores += jscore

        if t%500==0:
            plt.subplot(131)
            X = np.transpose(X.squeeze().numpy(), axes=(1,2,0))
            plt.imshow(X)
            plt.axis("off")
            plt.title("Image")

            plt.subplot(132)
            plt.imshow(y.squeeze())
            plt.axis("off")
            plt.title("Ground truth")

            plt.subplot(133)
            plt.imshow(scores, cmap="gray")
            plt.axis("off")
            plt.title("Prediction")

    print(f"Average Jaccard Score: {total_scores/(t+1)}")
