"""
main.py
=======
Entry point for the Movie Genre Predictor.

Usage
-----
Train all models and print an evaluation report::

    python main.py

Interactive CLI prediction (requires a trained model)::

    python main.py --predict

Use as a library::

    from dataset import build_dataset
    from model import train_and_evaluate, predict_genre

    df = build_dataset()
    train_and_evaluate(df)

    result = predict_genre("A detective hunts a serial killer who leaves cryptic clues.")
    print(result["genre"])   # Thriller
    print(result["top3"])    # [('Thriller', 48.2), ('Action', 22.1), ...]
"""

import argparse

from dataset import build_dataset
from model import predict_genre, train_and_evaluate


def _run_training() -> None:
    """Load dataset, train all pipelines, print results, save best model."""
    df = build_dataset()
    train_and_evaluate(df)
    print("  Tip: run `python main.py --predict` to classify your own plot.\n")


def _run_interactive() -> None:
    """REPL: read a plot summary from stdin and print the predicted genre."""
    print("\n  Movie Genre Predictor — interactive mode")
    print("  Type a plot summary and press Enter.  Type 'quit' to exit.\n")

    while True:
        try:
            text = input("  Plot > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if text.lower() in ("quit", "exit", "q", ""):
            break

        result = predict_genre(text)
        print(f"\n  Predicted genre : {result['genre']}")
        print(f"  Model used      : {result['model']}")
        print("  Top 3 guesses   :")
        for genre, pct in result["top3"]:
            bar = "█" * int(pct / 5)
            print(f"    {genre:<20} {pct:5.1f}%  {bar}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Movie Genre Predictor — train or interactively predict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py              # train & evaluate all models\n"
            "  python main.py --predict    # interactive prediction mode\n"
        ),
    )
    parser.add_argument(
        "--predict", action="store_true",
        help="Run interactive CLI prediction (requires a trained model).",
    )
    args = parser.parse_args()

    if args.predict:
        _run_interactive()
    else:
        _run_training()


if __name__ == "__main__":
    main()
