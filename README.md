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
`train_xgb_model.py` script reads this spreadsheet and uses the weights to
create a synthetic training target before fitting an XGBoost
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

## Input File Format

Uploaded brand and area tables must contain the following columns. Any
Excel or CSV file that does not match these names will cause the app to stop.

**Brand file columns**

- `Brand`
- `Cuisine`
- `AOV`
- `AggregatorScore`
- `MonthlyOrders`

**Area file columns**

- `Area`
- `Population`
- `Households`
- `Top1Nationality`
- `Top2Nationality`
- `Top3Nationality`
- `Top1Cuisine`
- `Top2Cuisine`
- `Top3Cuisine`
- `AOV_area`
- `Frequency`
- `Competition1`
- `Competition2`
- `Competition3`

The provided `sample_brands.csv` and `sample_areas.csv` can be used as
templates when preparing your own data.

## License

This project is licensed under the [MIT License](LICENSE).
