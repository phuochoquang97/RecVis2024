import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from model_factory import ModelFactory
from transformers import get_linear_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a model for image classification")
    parser.add_argument("--data", type=str, default="data_sketches",
                        help="Folder containing train_images/ and val_images/")
    parser.add_argument("--model_name", type=str, default="basic_cnn",
                        help="Model name for instantiation (e.g., 'basic_cnn', 'dinov2')")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate (default: 0.1)")
    parser.add_argument("--momentum", type=float, default=0.5, help="Momentum for SGD (default: 0.5)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="How often to log training status (in batches)")
    parser.add_argument("--experiment", type=str, default="experiment",
                        help="Directory to save experiment outputs")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of workers for data loading")
    parser.add_argument("--num_classes", type=int, default=500,
                        help="Number of output classes (default: 500)")
    return parser.parse_args()


def train(model: nn.Module, optimizer: torch.optim.Optimizer,
          train_loader: torch.utils.data.DataLoader, device: torch.device,
          epoch: int, log_interval: int) -> None:
    """Training loop."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

    accuracy = 100. * correct / len(train_loader.dataset)
    print(f"\nTrain Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)\n")


def validate(model: nn.Module, val_loader: torch.utils.data.DataLoader,
             device: torch.device) -> float:
    """Validation loop."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="mean")
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100. * correct / len(val_loader.dataset)
    print(f"\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.2f}%)\n")
    return val_loss


def main() -> None:
    """Main function."""
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.experiment, exist_ok=True)

    # Model and transforms
    model_factory = ModelFactory(args.model_name, args.num_classes)
    model, transform = model_factory.get_all()
    model.to(device)

    # Data
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(args.data, "train_images"), transform=transform),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(args.data, "val_images"), transform=transform),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, train_loader, device, epoch, args.log_interval)
        val_loss = validate(model, val_loader, device)

        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.experiment, "model_best.pth")
            torch.save(model.state_dict(), best_path)

        # Save each epoch
        epoch_path = os.path.join(args.experiment, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), epoch_path)

        print(f"Model saved to {epoch_path}")
        print(f"To evaluate, run:\n  python evaluate.py --model_name {args.model_name} --model {best_path}\n")


if __name__ == "__main__":
    main()
