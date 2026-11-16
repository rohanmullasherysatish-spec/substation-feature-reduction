"""Evaluate saved IEC61850 final conditions on the fixed test population."""

import argparse
from src import settings
from src.evaluate_test import evaluate as shared_evaluate, plan as shared_plan


def plan():
    return shared_plan("iec61850")


def evaluate(root=settings.PROJECT_ROOT, overwrite=False):
    return shared_evaluate(root, overwrite, "iec61850")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    evaluate(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
