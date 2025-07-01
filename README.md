# Kitchain AI

Streamlit AI tool to match food brands with areas.

## Environment

The included `kitchain_match_model.joblib` was trained with:

- scikit-learn 1.1.3
- numpy 1.24.4
- xgboost 1.7.6

The provided Conda environment targets **Python 3.10**. Install it with:

```bash
conda env create -f environment.yml
```

Package versions are pinned in both `environment.yml` and `requirements.txt`
so the bundled model loads correctly.

## XGBoost Model and Feature Weights

`feature_weights_full.xlsx` defines the importance of each feature. The
`train_xgb_model.py` script reads this spreadsheet, normalizes the weights and
uses them to create a synthetic training target before fitting an XGBoost
regression model. Running the script will produce `xgb_model.json` which can be
loaded by the app.

To train the demo XGBoost model:

```bash
python train_xgb_model.py
```

When launching the Streamlit app you can choose **XGBoost** from the sidebar to
use the new model instead of the bundled RandomForest model.

## Running the App

Activate the Conda environment and start Streamlit:

```bash
conda activate kitchain-ai
streamlit run app.py
```

## License

This project is licensed under the [MIT License](LICENSE).
