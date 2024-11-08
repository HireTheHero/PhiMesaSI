import argparse

import wandb

from train import train, train_batch


def argparser():
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument(
        "--dataset", type=str, default="wikitext,wikitext-2-raw-v1", help="Dataset name"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--config-path", type=str, default="config.json", help="Config file path"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--is-batch", action="store_true", help="Use batch training")
    parser.add_argument("--log-wandb", action="store_true", help="Log to wandb")
    parser.add_argument(
        "--wandb-project", type=str, default="my-awesome-project", help="Wandb project"
    )
    parser.add_argument(
        "--output-dir", type=str, default="/path/to/dir", help="Output directory"
    )
    return parser


def main():
    parser = argparser()
    args = parser.parse_args()
    if args.log_wandb:
        wandb.finish()
        wandb.init(project=args.wandb_project, config=args, name="train-model")
    if args.is_batch:
        train_batch(args)
    else:
        train(args)
    if args.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
