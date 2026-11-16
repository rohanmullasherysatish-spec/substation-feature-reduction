"""Fit IEC104 reductions with the saved grouped development allocation."""

import argparse
from src import settings
from src.train_reduced_models import subset_processor, reduction_recipe as shared_recipe, train as shared_train


def reduction_recipe():
    return shared_recipe("iec104")


def train(root=settings.PROJECT_ROOT, overwrite=False):
    return shared_train(root, overwrite, "iec104")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
