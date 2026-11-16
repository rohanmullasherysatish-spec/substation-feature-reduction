"""Shared fixed allocation and group-separation checks for both protocols."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.checks import require


def grouped_assignments(data, groups, config):
    """Use a fixed outer fold and three rotating development-only validation folds."""
    require(config.OUTER_FOLDS >= 2 and 0 <= config.TEST_FOLD < config.OUTER_FOLDS,
            "Invalid outer-fold settings.")
    require(config.VALIDATION_FOLDS >= 2, "At least two validation folds are required.")
    groups = pd.Series(np.asarray(groups), index=data.index)
    require(len(groups) == len(data) and not groups.isna().any(), "Invalid prepared dependency groups.")
    y = data.class_label.to_numpy(dtype=str)
    dummy_x = np.zeros((len(data), 1), dtype=np.uint8)
    support = pd.DataFrame({"group_id": groups, "class_label": data.class_label}).drop_duplicates()
    class_groups = support.groupby("class_label").group_id.nunique()
    require(all(class_groups.get(label, 0) >= config.OUTER_FOLDS for label in config.CLASSES),
            "Insufficient distinct groups for the fixed outer allocation.")

    outer = StratifiedGroupKFold(n_splits=config.OUTER_FOLDS, shuffle=True,
                                random_state=config.OUTER_SEED)
    # Use the declared fold, not the allocation with the most favourable support or scores.
    development, test = next(
        pair for fold, pair in enumerate(outer.split(dummy_x, y, groups))
        if fold == config.TEST_FOLD
    )
    fraction = len(test) / len(data)
    require(abs(fraction - 1 / config.OUTER_FOLDS) <= config.MAX_TEST_FRACTION_DEVIATION,
            "Fixed outer allocation exceeds the declared test-fraction tolerance.")
    for name, positions in (("development", development), ("test", test)):
        require(set(y[positions]) == set(config.CLASSES), f"A class is absent from {name}.")

    assignments = data[["record_id", "dataset_row", "source_file", "source_row", "class_label"]].copy()
    assignments["group_id"] = groups
    assignments["partition"] = "development"
    assignments.loc[test, "partition"] = "test"
    # Test rows never participate in validation; -1 is an assignment marker, not an input.
    assignments["validation_fold"] = -1
    dev_groups = groups.iloc[development]
    dev_support = assignments.iloc[development].groupby("class_label").group_id.nunique()
    require(all(dev_support.get(label, 0) >= config.VALIDATION_FOLDS for label in config.CLASSES),
            "Insufficient development groups for the declared validation folds.")
    inner = StratifiedGroupKFold(n_splits=config.VALIDATION_FOLDS, shuffle=True,
                                random_state=config.VALIDATION_SEED)
    for fold, (train_local, valid_local) in enumerate(
        inner.split(dummy_x[development], y[development], dev_groups)
    ):
        training, validation = development[train_local], development[valid_local]
        require(set(y[training]) == set(config.CLASSES)
                and set(y[validation]) == set(config.CLASSES),
                f"A class is absent from training or validation in fold {fold}.")
        assignments.loc[validation, "validation_fold"] = fold
    validate_assignments(assignments, config)
    return assignments


def validate_assignments(assignments, config):
    """Enforce separation of groups and complete, unambiguous row assignments."""
    require(assignments.record_id.is_unique, "A record has more than one assignment.")
    require(set(assignments.partition) == {"development", "test"}, "Invalid outer partition labels.")
    require(assignments.groupby("group_id").partition.nunique().eq(1).all(),
            "A measurement group crosses development/test.")
    dev = assignments.loc[assignments.partition.eq("development")]
    test = assignments.loc[assignments.partition.eq("test")]
    require(set(dev.validation_fold) == set(range(config.VALIDATION_FOLDS)),
            "Development validation assignments are incomplete.")
    require(dev.groupby("group_id").validation_fold.nunique().eq(1).all(),
            "A measurement group crosses development validation folds.")
    require(test.validation_fold.eq(-1).all(), "Test rows were assigned to validation.")


def support_table(assignments, config):
    """Save class-level row/group counts for every fitting and evaluation population."""
    dev = assignments.loc[assignments.partition.eq("development")]
    populations = [("all", -1, assignments), ("development", -1, dev),
                   ("test", -1, assignments.loc[assignments.partition.eq("test")])]
    for fold in range(config.VALIDATION_FOLDS):
        populations.extend([
            ("training", fold, dev.loc[dev.validation_fold.ne(fold)]),
            ("validation", fold, dev.loc[dev.validation_fold.eq(fold)]),
        ])
    rows = []
    for population, fold, frame in populations:
        require(set(frame.class_label) == set(config.CLASSES), f"Missing class in {population}, fold {fold}.")
        for label in config.CLASSES:
            subset = frame.loc[frame.class_label.eq(label)]
            rows.append({"population": population, "validation_fold": fold, "class_label": label,
                         "records": len(subset), "groups": subset.group_id.nunique()})
    return pd.DataFrame(rows)


