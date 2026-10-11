"""
Plot training and validation loss curves from loss_log.json.

Usage:
    uv run python plot_loss.py [--log loss_log.json] [--output loss_curve.png] [--show]
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def plot_loss(log_file: str = "loss_log.json", output_file: str = "loss_curve.png", show: bool = False):
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"Error: {log_file} not found. Please run training first to generate loss logs.")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        log = json.load(f)

    plt.figure(figsize=(10, 6))

    # Plot raw training loss
    if "steps" in log and "train" in log and log["train"]:
        plt.plot(log["steps"], log["train"], alpha=0.3, color="blue", label="train (raw)")

        # Optional smoothed rolling average if many steps exist
        if len(log["train"]) >= 50:
            window = 25
            train_smooth = [
                sum(log["train"][max(0, i - window):i + 1]) / len(log["train"][max(0, i - window):i + 1])
                for i in range(len(log["train"]))
            ]
            plt.plot(log["steps"], train_smooth, color="blue", linewidth=1.5, label=f"train (moving avg {window})")

    # Plot validation loss (recorded every eval interval)
    if "val" in log and log["val"]:
        val_steps = log.get("val_steps")
        if not val_steps:
            # Fallback for logs recording val at multiples of 100
            val_steps = [s for s in log["steps"] if s % 100 == 0][:len(log["val"])]
        
        plt.plot(val_steps, log["val"], "ro-", linewidth=2, label="val")

    plt.title("GPT Training & Validation Loss Curve", fontsize=14)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)
    print(f"Saved loss curve to {output_file}")

    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GPT training loss curves")
    parser.add_argument("--log", default="loss_log.json", help="Path to loss_log.json")
    parser.add_argument("--output", default="loss_curve.png", help="Path to output image file")
    parser.add_argument("--show", action="store_true", help="Display interactive plot window")
    args = parser.parse_args()

    plot_loss(log_file=args.log, output_file=args.output, show=args.show)
