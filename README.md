# Kitchain AI

Streamlit AI tool to match food brands with areas.

## Environment

The included `kitchain_match_model.joblib` was trained with:

- scikit-learn 1.1.3
- numpy 1.24.4

The provided Conda environment targets **Python 3.10**. Install it with:

```bash
conda env create -f environment.yml
```

Package versions are pinned in both `environment.yml` and `requirements.txt`
so the bundled model loads correctly.
