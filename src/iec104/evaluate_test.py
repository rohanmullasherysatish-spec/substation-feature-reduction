"""Evaluate saved IEC104 final conditions on the fixed test population."""

import argparse
from src import settings
from src.evaluate_test import evaluate as shared_evaluate, plan as shared_plan
from src.evaluate_test import bootstrap, group_error_operator, scores_from_counts


def plan():
    return shared_plan("iec104")


def evaluate(root=settings.PROJECT_ROOT, overwrite=False):
    return shared_evaluate(root, overwrite, "iec104")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    evaluate(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
