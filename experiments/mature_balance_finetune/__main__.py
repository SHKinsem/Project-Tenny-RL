import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='Mature policy true-XY reward fine-tuning')
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    options=parser.parse_args()
    from .train import train_one
    train_one(options.manifest,options.seed,options.output_dir)


if __name__=='__main__':
    main()
