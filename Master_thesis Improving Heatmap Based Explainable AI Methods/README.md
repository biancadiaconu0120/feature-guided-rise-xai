# Master_thesis Improving Heatmap Based Explainable AI Methods

## Running the Code

All required code is consolidated in the `run_classifier.py` file. To execute the project, only this file needs to be run.  
The remaining files are imported automatically and must not be executed separately.  
The only exceptions are the files `aggregate_ranking_general` and `anova_posthoc_analysis_general.py`, which need to be executed independently.

### Required Command-Line Arguments

When running `run_classifier.py`, the following command-line arguments must be provided:

- `--network`  
  Specifies the network architecture to be used.

- `--folder`  
  Defines the input data directory.

- `--results`  
  Defines the directory where the output results will be saved.

If these arguments are not provided, the script will not execute correctly.
