"""Command-line entry point for the Stock Price Prediction project.

Examples
--------
    python main.py data            # create / cache the dataset CSV
    python main.py train           # full EDA + train + evaluate + figures
    python main.py forecast --steps 5
    python main.py all             # everything, in order
"""
from __future__ import annotations

import argparse
import json

from src import config, data, pipeline, predict, utils


def cmd_data(_args) -> None:
    df = data.build_dataset()
    print(f"Wrote {config.DATASET_CSV}  ({len(df)} rows)")
    print(df.tail(3).to_string())


def cmd_train(_args) -> None:
    utils.set_seed()
    results = pipeline.run_pipeline()
    print("\n=== Evaluation summary ===")
    print(json.dumps({"metrics": results["metrics"],
                      "best_model": results["best_model"]},
                     indent=2, default=float))
    print(f"\nFigures   -> {config.FIGURES_DIR}")
    print(f"Results   -> {config.RESULTS_DIR}")


def cmd_forecast(args) -> None:
    rows = predict.forecast_next_day(steps=args.steps)
    print(f"\nForward forecast (next {args.steps} trading day(s)):")
    for r in rows:
        print(f"  Day +{r['step']}: {r['forecast_close']:>9.2f}  "
              f"({r['change_%']:+.2f}%, {r['direction']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock Price Prediction")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("data")
    sub.add_parser("train")
    sub.add_parser("all")
    f = sub.add_parser("forecast")
    f.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    if args.command == "data" or args.command is None:
        cmd_data(args)
        if args.command is None:
            parser.print_help()
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "forecast":
        cmd_forecast(args)
    elif args.command == "all":
        cmd_data(args)
        cmd_train(args)
        cmd_forecast(argparse.Namespace(steps=5))


if __name__ == "__main__":
    main()
