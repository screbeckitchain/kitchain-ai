import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def build_dataset(n_random: int = 500):
    """Generate training data covering the full score range."""
    grid_aov = np.linspace(1, 10, 10)
    grid_cuisine = np.linspace(1, 10, 10)
    grid = np.array([(a, c) for a in grid_aov for c in grid_cuisine], dtype=float)
    grid_target = grid[:, 0] + grid[:, 1]

    rng = np.random.default_rng(42)
    random_data = rng.uniform(1, 10, size=(n_random, 2))
    random_target = random_data[:, 0] + random_data[:, 1]

    X = np.vstack([grid, random_data])
    y = np.concatenate([grid_target, random_target])

    df = pd.DataFrame(X, columns=["aov_score", "cuisine_score"])
    target = pd.Series(y)
    return df, target


def train_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=200,
    )
    model.fit(X, y)
    return model


def main() -> None:
    X, y = build_dataset()
    model = train_model(X, y)
    model_path = BASE_DIR / "xgb_sum_model.json"
    model.save_model(model_path)
    print(f"Saved model to {model_path}")

    # quick sanity checks
    extremes = pd.DataFrame(
        {
            "aov_score": [1, 10],
            "cuisine_score": [1, 10],
        }
    )
    preds = model.predict(extremes)
    for aov, cui, pred in zip(extremes["aov_score"], extremes["cuisine_score"], preds):
        print(f"{aov}+{cui} -> prediction: {pred:.2f}")


if __name__ == "__main__":
    main()
