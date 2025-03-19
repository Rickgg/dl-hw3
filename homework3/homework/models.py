from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.tensor(INPUT_MEAN).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor(INPUT_STD).view(1, 3, 1, 1))

        layers = []

        layers.append(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Conv2d(256, 512, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU())


        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(512, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Normalize input
        x = (x - self.input_mean) / self.input_std
        
        logits = self.network(x)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w)

        Returns:
            pred (torch.LongTensor): class labels with shape (b,)
        """
        return self(x).argmax(dim=1)


class Detector(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD).view(1, 3, 1, 1))

        # Encoder (Downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False)  # Avoid in-place operation
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )

        # Decoder (Upsampling + Skip Connections)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False)
        )

        # Additional 1x1 convolution layers for skip connections (to match channel dimensions)
        self.skip1 = nn.Conv2d(16, 16, kernel_size=1)  # Skip connection for enc1
        self.skip2 = nn.Conv2d(32, 32, kernel_size=1)  # Skip connection for enc2

        # Output heads
        self.segmentation = nn.Conv2d(16, num_classes, kernel_size=1)
        self.depth = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = (x - self.input_mean) / self.input_std  # Normalize input

        # Encoder
        enc1_out = self.enc1(z)  # (b, 16, h/2, w/2)
        enc2_out = self.enc2(enc1_out)  # (b, 32, h/4, w/4)
        enc3_out = self.enc3(enc2_out)  # (b, 64, h/8, w/8)

        # Apply skip connections
        enc1_skip = self.skip1(enc1_out.clone())  # Match dimensions for addition
        enc2_skip = self.skip2(enc2_out.clone())

        # Decoder with skip connections
        dec3_out = self.dec3(enc3_out)  # (b, 32, h/4, w/4)
        dec3_out = dec3_out + nn.functional.interpolate(enc2_skip, size=dec3_out.shape[2:], mode="nearest")  # Resize and add

        dec2_out = self.dec2(torch.cat([dec3_out, nn.functional.interpolate(enc2_out, size=dec3_out.shape[2:], mode="nearest")], dim=1))
        dec2_out = dec2_out + nn.functional.interpolate(enc1_skip, size=dec2_out.shape[2:], mode="nearest")  # Resize and add

        dec1_out = self.dec1(torch.cat([dec2_out, nn.functional.interpolate(enc1_out, size=dec2_out.shape[2:], mode="nearest")], dim=1))

        logits = self.segmentation(dec1_out)
        raw_depth = self.depth(dec1_out).squeeze(1)

        return logits, raw_depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
