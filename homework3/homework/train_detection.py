import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    seg_loss_fn = torch.nn.CrossEntropyLoss().to(device)
    depth_loss_fn = torch.nn.MSELoss().to(device)

    optim = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    det_metric = DetectionMetric()

    global_step = 0

    # training loop
    for epoch in range(num_epoch):
        det_metric.reset()

        model.train()

        for batch in train_data:
            image, depth, track = batch["image"].to(device), batch["depth"].to(device), batch["track"].to(device)

            # TODO: implement training step
            seg_pred, depth_pred = model(image)

            seg_loss = seg_loss_fn(seg_pred, track)
            # depth_loss = depth_loss_fn(depth_pred, depth_target)
            # loss_val = seg_loss + depth_loss

            optim.zero_grad()
            seg_loss.backward()
            optim.step()

            seg_predictions, depth_predictions = model.predict(image)
            det_metric.add(seg_predictions, track, depth_predictions, depth)

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            # for batch in val_data:
            #     batch = batch.to(device)
            #     seg_pred, depth_pred = model(batch["image"])
            #     seg_predictions, depth_predictions = model.predict(img)
                
            #     det_metric.add(seg_predictions, seg_target, depth_predictions, depth_target)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"det_metrics={det_metric.compute()} "
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="detector")
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
