"""
Main entry point for the PyCANUPO project.
Handles command-line interface and inline configuration.
"""

import argparse
from model import cmd_train, cmd_predict
from compare import cmd_compare


def build_parser():
    """Builds the CLI parser and registers subcommands."""
    parser = argparse.ArgumentParser(description="Pure-Python CANUPO-style classifier.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Train command
    train = sub.add_parser("train", help="Train a binary classifier from labeled point clouds.")
    # TODO: add CLI arguments for input files, radii, model type, etc.
    train.set_defaults(func=cmd_train)

    # Predict command
    predict = sub.add_parser("predict", help="Classify new point clouds using a trained model.")
    # TODO: add CLI arguments for model, input cloud, and output paths.
    predict.set_defaults(func=cmd_predict)

    # Compare command
    compare = sub.add_parser("compare", help="Compare qCANUPO vs Python model results.")
    # TODO: add CLI arguments for .prm, .pkl/.pyprm, and output.
    compare.set_defaults(func=cmd_compare)

    return parser


def main():
    """Entry point that runs inline config or CLI depending on setup."""
    # TODO: Implement inline configuration or CLI-based flow.
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
