import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Isolated fixed-step reward-ablation run')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--arm', choices=['baseline','posture','position','both'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    from .train import train_one
    train_one(args.manifest, args.arm, args.seed, args.output_dir)


if __name__ == '__main__':
    main()
