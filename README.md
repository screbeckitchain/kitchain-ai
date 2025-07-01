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

For a minimal example that only uses AOV and cuisine scores you can run:

```bash
python train_xgb_two_feature.py
```

`feature_weights_full.xlsx` also contains threshold and score values used when
calculating AOV and cuisine alignment. The relevant keys are:

- `AOV GRADE 1` – first AOV difference threshold
- `AOV GRADE 2` – second AOV difference threshold
- `AOV GRADE 3` – third AOV difference threshold
- `AOV GRADE 3` (fourth entry) – highest AOV difference threshold
- `AOV GRADE 1 Score` (four rows) – scores applied for each grade
- `cuisine_match_score 1`, `cuisine_match_score 2`, `cuisine_match_score 3` –
  scores for cuisine matches

These keys are read by both the training scripts and the Streamlit app.

When launching the Streamlit app you can choose **XGBoost** from the sidebar to
use the new model instead of the bundled RandomForest model. This XGBoost model
was trained only on the AOV alignment and cuisine match scores, so the
application automatically limits the feature set to those two inputs whenever
this option is selected.

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
