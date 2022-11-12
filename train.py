import torch
from torch.utils.data import DataLoader
from models import UNet
from generate_images import EllipseDataset
from tqdm import tqdm

dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print('using device:', device)

if __name__ == "__main__":
    model = UNet(nclass=1, in_chans=3, depth=3, skip=True, padding="same")
    model.to(device, dtype=dtype)

    batch_size = 32
    lr = 1e-3
    dataset = EllipseDataset(size=5000)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    criterion = torch.nn.BCEWithLogitsLoss()
    sigmoid = torch.nn.Sigmoid()

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    epochs = 10
    model.train()
    for e in range(epochs):
        print("Epoch {0} out of {1}".format(e+1, epochs))
        print("_"*10)
        epoch_loss = 0.0

        for t, (x, y) in enumerate(tqdm(loader)):
            x = x.to(device, dtype=dtype)
            scores = model(x)
            y = y.to(device).type_as(scores)
            loss = criterion(scores, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                epoch_loss += loss.item()
        print(f"epoch loss: {epoch_loss}")

    # save model
    torch.save(model.state_dict(), "unet.pkl")
