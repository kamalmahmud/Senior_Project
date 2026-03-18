"""
Entry point for Phase 1 training on synthetic data.

Usage:
    python train.py
    python train.py --epochs 100 --patience 15 --device cuda
"""

import argparse
import torch

from training import train


def parse_args():
    parser = argparse.ArgumentParser(
        description="TSM-MobileNetV2 training — Phase 1 synthetic data"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=800,
                        help="Number of synthetic training samples")
    parser.add_argument("--n-val", type=int, default=200,
                        help="Number of synthetic validation samples")
    parser.add_argument("--pos-ratio", type=float, default=0.2,
                        help="Fraction of positive (infiltration) clips")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--save-path", type=str,
                        default="best_model.pth",
                        help="Where to save the best model weights")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 55)
    print("  Thermal Infiltration Detection — Phase 1 Training")
    print("=" * 55)
    print(f"  Device     : {args.device}")
    print(f"  Epochs     : {args.epochs}  (patience={args.patience})")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Train/Val  : {args.n_train} / {args.n_val} samples")
    print(f"  Pos ratio  : {args.pos_ratio:.0%}")
    print(f"  LR         : {args.lr}  wd={args.weight_decay}")
    print(f"  Save path  : {args.save_path}")
    print("=" * 55)

    model = train(
        n_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        n_train=args.n_train,
        n_val=args.n_val,
        pos_ratio=args.pos_ratio,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )

    torch.save(model.state_dict(), args.save_path)
    print(f"\nBest model weights saved to: {args.save_path}")


if __name__ == "__main__":
    main()
