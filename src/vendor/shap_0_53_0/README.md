# SHAP interventional tree precision correction

MIT-licensed SHAP v0.53.0 C++ extension sources, from
https://github.com/shap/shap/tree/v0.53.0/shap/cext .

The header preserves double precision in `Node.thres` and the local `thres`
variable in `tree_shap_indep`, following the upstream correction:
https://github.com/shap/shap/pull/5122 . All other code is unchanged.

`src/shap_backend.py` builds this local extension with the current Python and
NumPy headers. The installed SHAP package is not edited. Keep `LICENSE` when
redistributing these files. This is a numerical software correction, not a new
feature-selection method or a research result.
