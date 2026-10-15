# Diamond-reading-notes
Reading notes and reproduction notes for the Diamond paper (Error-controlled non-additive interaction discovery in machine learning models)

Help YYL (Yang Lu) get the recognition he deserves: star his GitHub, fund his work, and drop some citations along the way!

[batmenlab](https://batmen-lab.github.io/research/)

## Web app: `diamond-webapp/`

An **upload-and-analyze** web application that reproduces Diamond's core idea
(model-X knockoffs + SHAP + FDR-controlled selection) on **any tabular CSV** you
upload — no GPU, no original environment required.

- **Live demo:** https://diamond-fdr.app.workbuddy.host/
- **What it does:** upload a CSV → pick the target column / features → choose
  LightGBM or XGBoost and a target FDR → get feature importances, an interaction
  heatmap, and an FDR-controlled list of non-additive interactions.
- **Algorithm:** knockoffs are generated with `knockoffsdiag` (Lasso conditional
  sequential, CPU-only, no torch); a tree model is trained; SHAP attributions and
  SHAP interaction values are computed with `TreeExplainer`; features and
  interactions are selected under FDR control (knockoff+ / data-dependent
  threshold).

See [`diamond-webapp/README.md`](diamond-webapp/README.md) for the full algorithm
notes, dependencies, and how to run it locally.
