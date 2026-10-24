"""The ``doc-motion`` command: ``train`` and ``predict``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .config import TrainConfig, load_config


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; return the process exit code."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "train":
            return _train(args)
        return _predict(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"doc-motion: {error}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doc-motion",
        description="Classify document frames as moving or still.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser(
        "train", help="cross-validate, train and evaluate; save a run folder"
    )
    train.add_argument("--config", help="YAML settings (default: 2022 values)")
    train.add_argument(
        "--data", required=True, help="folder with {train,test}/{moving,still}/"
    )
    train.add_argument("--backend", choices=["keras", "torch"], default="keras")
    train.add_argument("--out", default="runs", help="where run folders go")

    predict = commands.add_parser(
        "predict", help="classify a folder of images and write a CSV"
    )
    predict.add_argument("--model", required=True, help="a .keras or .pt file")
    predict.add_argument("--images", required=True, help="folder of images")
    predict.add_argument("--out", required=True, help="CSV file to write")
    predict.add_argument("--threshold", type=float, default=0.5)
    return parser


def _train(args: argparse.Namespace) -> int:
    from .backends import get_backend
    from .data import load_split
    from .training import run

    cfg = load_config(args.config) if args.config else TrainConfig()
    train = load_split(args.data, "train")
    test = load_split(args.data, "test")
    record = run(cfg, train, test, get_backend(args.backend), args.out)
    print(
        f"cross-validation accuracy {record.cv_accuracy_mean:.3f} "
        f"(std {record.cv_accuracy_std:.3f}); "
        f"test accuracy {record.test_report['accuracy']:.3f}"
    )
    print(f"run saved to {record.run_dir}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    from .predict import predict_folder, write_csv

    rows = predict_folder(args.model, args.images, threshold=args.threshold)
    write_csv(rows, args.out)
    print(f"wrote {len(rows)} predictions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
