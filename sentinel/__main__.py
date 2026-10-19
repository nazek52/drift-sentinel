import argparse
import json
from pathlib import Path
from sentinel.data import generate
from sentinel.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Reproducible delayed-label evaluation for synthetic incident escalation"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate synthetic data and run the experiment")
    demo.add_argument("--output", default="artifacts/demo")
    demo.add_argument("--seed", type=int, default=42)
    run = sub.add_parser("evaluate", help="Evaluate a CSV matching the documented schema")
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--as-of", type=float)
    args = parser.parse_args()
    try:
        path = (
            generate(Path(args.output) / "synthetic-incidents.csv", seed=args.seed)
            if args.command == "demo"
            else args.input
        )
        report = evaluate(path, args.output, getattr(args, "as_of", None))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Invalid experiment: {exc}\n")
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "test": report["test"],
                "review_features": report["drift_review_features"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
