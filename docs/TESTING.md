# Testing Guide

Recommended checks before release:

- Run the notebook from start to finish.
- Confirm all Python source files import without syntax errors.
- Load each saved `.pkl` model and generate a test prediction.
- Launch `streamlit run app.py` and verify every tab and filter.
- Confirm report CSVs and images exist and are readable.
- Verify the final test metrics match the saved comparison report.
