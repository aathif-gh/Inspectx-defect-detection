import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

IMG_SIZE = 256  # bumped up from 128 -- defects are small details that get
                # blurred away at lower resolution
DATA_DIR = "data/bottle/train/good"
MODEL_PATH = "models/autoencoder.pth"

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


class GoodImagesDataset(Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return transform(img)


class ConvAutoencoder(nn.Module):
    """
    Same encoder/decoder shape as before, one extra down/up-sampling stage
    added since the input is now 256x256 instead of 128x128.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),    # 256 -> 128
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 128 -> 64
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),  # 64 -> 32
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(), # 32 -> 16
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 16 -> 32
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),    # 32 -> 64
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),     # 64 -> 128
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),   # 128 -> 256
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def main():
    from pytorch_msssim import SSIM  # pip install pytorch-msssim

    os.makedirs("models", exist_ok=True)
    dataset = GoodImagesDataset(DATA_DIR)
    print(f"Training on {len(dataset)} non-defective images at {IMG_SIZE}x{IMG_SIZE}")

    loader = DataLoader(dataset, batch_size=8, shuffle=True)  # smaller batch, bigger images

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ConvAutoencoder().to(device)
    ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 40
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)

            # Combined loss: SSIM captures structural/textural differences
            # (what a scratch or dent actually looks like), MSE keeps
            # overall pixel values anchored. This combo is the standard
            # approach used in the original MVTec AD baseline work,
            # rather than plain MSE alone.
            loss = 0.7 * (1 - ssim_loss(reconstructed, batch)) + 0.3 * mse_loss(reconstructed, batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} — combined loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()