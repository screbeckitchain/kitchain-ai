# Kitchain AI

Streamlit AI tool to match food brands with areas.

## Environment

The included `kitchain_match_model.joblib` was trained with:

- scikit-learn 1.1.3
- numpy 1.24.4
- xgboost 1.7.6
- openai

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

The Streamlit app now defaults to **XGBoost**, which relies on these feature
weights. You can still select the original RandomForest model from the sidebar
if desired. This XGBoost model was trained only on the AOV alignment and cuisine
match scores, so the application automatically limits the feature set to those
two inputs whenever this option is selected.

## Running the App

Activate the Conda environment and start Streamlit:

```bash
conda activate kitchain-ai
streamlit run app.py
```

The results table shows only the **Score (%)** column where the top
prediction is scaled to 100%.

## Score Calculation

Each prediction uses weighted features from both the brand and the area. The
RandomForest model considers area AOV, order frequency, three competition
cuisine scores, brand AOV, aggregator position and monthly orders. The
simplified XGBoost variant looks only at AOV alignment and cuisine match. After
the model outputs a raw value, the scores are normalized so that the highest
prediction equals 100 and every other result is scaled relative to that value.

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
- `AOV_area`
- `Top1Cuisine`
- `Top2Cuisine`
- `Top3Cuisine`
- `Frequency`
- `Competition1`
- `Competition2`
- `Competition3`

## OpenAI Explanations (Optional)

To generate short text explanations for the top matches the application
uses the OpenAI ChatCompletion API. The `openai` Python package must be
installed and you need to provide a valid API key. Add your key as
`OPENAI_API_KEY` in the app's Streamlit **Secrets** settings.

Run Streamlit once the key is configured:

```bash
streamlit run app.py
```

Enable the **Generate explanations** option in the sidebar and choose how many
top results to explain. The text is added to the **Explanation** column beneath
each score.

If the predicted score for a brand/area pair is below 60, the generated text
now explains why it is **not** a good match. The AOV difference is calculated as
`abs(brand AOV - area AOV) / brand AOV` and the cuisine is compared against the
area's top three cuisines. The prompt highlights whether a large AOV difference
or a cuisine mismatch is the main reason for the low score.

## Brand Analysis Page

The sidebar includes a **Brand analysis** link that opens the `/brand_analysis`
page. Upload brand and area tables (or rely on the provided samples) and choose
a brand from the dropdown. The page predicts the brand's score for every area
using the selected model, generates a brief explanation for each match and
displays a bar chart of the results.
