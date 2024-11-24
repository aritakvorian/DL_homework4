from datetime import datetime
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from models import load_model, save_model
from datasets.road_dataset import load_data

def train_planner(
    model_name: str,
    train_data,
    val_data,
    n_track: int = 10,
    n_waypoints: int = 3,
    lr: float = 1e-3,
    num_epochs: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):

    log_dir = Path("logs") / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = SummaryWriter(log_dir)


    model = load_model(model_name, n_track=n_track, n_waypoints=n_waypoints)
    model = model.to(device)
    model.train()

    # Loss
    loss_func = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for batch in train_data:

            track_left, track_right, targets = (
                batch["track_left"].to(device),
                batch["track_right"].to(device),
                batch["waypoints"].to(device),
            )

            outputs = model(track_left=track_left, track_right=track_right)
            loss = loss_func(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            global_step += 1

        # Train stats
        avg_train_loss = sum(train_losses) / len(train_losses)
        logger.add_scalar("train/loss", avg_train_loss, epoch)

        # Val stats
        model.eval()
        val_losses = []
        with torch.inference_mode():
            for batch in val_data:
                track_left, track_right, targets = (
                    batch["track_left"].to(device),
                    batch["track_right"].to(device),
                    batch["waypoints"].to(device),
                )

                outputs = model(track_left=track_left, track_right=track_right)
                loss = loss_func(outputs, targets)
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        logger.add_scalar("val/loss", avg_val_loss, epoch)


        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
        )

    logger.close()
    save_model(model)


def train_cnn_planner(
    train_data,
    val_data,
    model_name: str = "cnn_planner",
    lr: float = 1e-3,
    num_epochs: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):

    log_dir = Path("logs") / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = SummaryWriter(log_dir)


    model = load_model(model_name, n_waypoints=3)
    model = model.to(device)
    model.train()

    # Loss
    loss_func = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for batch in train_data:
            images, targets = batch["image"].to(device), batch["waypoints"].to(device)

            outputs = model(images)
            loss = loss_func(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            global_step += 1

        # Train stats
        avg_train_loss = sum(train_losses) / len(train_losses)
        logger.add_scalar("train/loss", avg_train_loss, epoch)

        # Val stats
        model.eval()
        val_losses = []
        with torch.inference_mode():
            for batch in val_data:
                images, targets = batch["image"].to(device), batch["waypoints"].to(device)

                outputs = model(images)
                loss = loss_func(outputs, targets)
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        logger.add_scalar("val/loss", avg_val_loss, epoch)

        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
        )

    logger.close()
    save_model(model)


# Example usage
if __name__ == "__main__":

    train_data = load_data("../drive_data/train", shuffle=True, num_workers=2)
    val_data = load_data("../drive_data/train", shuffle=False)

    train_cnn_planner(
        train_data=train_data,
        val_data=val_data,
        model_name="cnn_planner",
    )