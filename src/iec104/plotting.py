"""IEC 104 plotting interfaces; implementation is shared by both protocols."""

from src.plot_results import selection_figures, reduction_figures
from src.plot_results import test_figures as shared_test_figures


def test_figures(root):
    return shared_test_figures(root, "iec104")
