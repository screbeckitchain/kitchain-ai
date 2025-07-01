from pathlib import Path
import pandas as pd
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_FILE = BASE_DIR / "feature_weights_full.xlsx"
BRANDS_FILE = BASE_DIR / "brands_example.xlsx"
AREAS_FILE = BASE_DIR / "areas_example.xlsx"


def load_params(path: Path = WEIGHTS_FILE):
    """Read threshold and score values from the Excel weights file."""
    df = pd.read_excel(path, header=None, names=["feature", "value"]).dropna()
    df = df[df["feature"] != "Feature"]
    thresholds = (
        df[
            df["feature"].str.startswith("AOV GRADE")
            & ~df["feature"].str.contains("Score")
        ]["value"].astype(float).tolist()
    )
    aov_scores = (
        df[
            df["feature"].str.startswith("AOV GRADE")
            & df["feature"].str.contains("Score")
        ]["value"].astype(float).tolist()
    )
    cuisine_scores = (
        df[df["feature"].str.startswith("cuisine_match_score")]["value"]
        .astype(float)
        .tolist()
    )
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
