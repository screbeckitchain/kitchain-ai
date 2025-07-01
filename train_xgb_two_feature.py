from pathlib import Path
import pandas as pd
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_FILE = BASE_DIR / "feature_weights_full.xlsx"
BRANDS_FILE = BASE_DIR / "brands_example.xlsx"
AREAS_FILE = BASE_DIR / "areas_example.xlsx"


def load_params(path: Path = WEIGHTS_FILE):
    """Read threshold and score values from the Excel weights file."""
    df = pd.read_excel(path, header=None)
    thresholds = df.iloc[19:23, 1].tolist()
    aov_scores = df.iloc[23:27, 1].tolist()
    cuisine_scores = df.iloc[27:30, 1].tolist()
    return thresholds, aov_scores, cuisine_scores


def build_dataset(brands_path: Path = BRANDS_FILE, areas_path: Path = AREAS_FILE):
    brands = pd.read_excel(brands_path)
    areas = pd.read_excel(areas_path)
    thresholds, aov_scores, cuisine_scores = load_params()

    rows = []
    target = []
    for _, b in brands.iterrows():
        for _, a in areas.iterrows():
            diff_ratio = abs(b["AOV"] - a["AOV_area"]) / b["AOV"]
            if diff_ratio <= thresholds[0]:
                aov_score = aov_scores[0]
            elif diff_ratio <= thresholds[1]:
                aov_score = aov_scores[1]
            elif diff_ratio <= thresholds[2]:
                aov_score = aov_scores[2]
            else:
                aov_score = aov_scores[3]

            if b["Cuisine"] == a["Top1Cuisine"]:
                cuisine_score = cuisine_scores[0]
            elif b["Cuisine"] == a["Top2Cuisine"]:
                cuisine_score = cuisine_scores[1]
            elif b["Cuisine"] == a["Top3Cuisine"]:
                cuisine_score = cuisine_scores[2]
            else:
                cuisine_score = 0

            rows.append({
                "aov_alignment_score": aov_score,
                "cuisine_match_score": cuisine_score,
            })
            target.append(aov_score + cuisine_score)

    X = pd.DataFrame(rows)
    y = pd.Series(target)
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series):
    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    model.fit(X, y)
    return model


def main():
    X, y = build_dataset()
    model = train_model(X, y)
    model_path = BASE_DIR / "xgb_model.json"
    model.save_model(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
