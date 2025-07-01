from pathlib import Path
import pandas as pd
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_FILE = BASE_DIR / "feature_weights_full.xlsx"
BRANDS_FILE = BASE_DIR / "sample_brands.csv"
AREAS_FILE = BASE_DIR / "sample_areas.csv"


def load_weights(path: Path = WEIGHTS_FILE) -> dict:
    """Return normalized feature weights.

    Only keep features used by the training script.
    """
    df = pd.read_excel(path, header=None)
    # first row has column names 'Feature', 'Weight'
    df = df.dropna().iloc[1:]
    df.columns = ['feature', 'weight']
    weights = dict(zip(df['feature'], df['weight']))

    used_features = {
        'area_aov',
        'order_freq',
        'competition_cuisine_1',
        'competition_cuisine_2',
        'competition_cuisine_3',
        'brand_aov',
        'agg_position',
        'brand_orders',
    }
    weights = {k: v for k, v in weights.items() if k in used_features}

    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def build_dataset(brands_path: Path = BRANDS_FILE, areas_path: Path = AREAS_FILE):
    brands = pd.read_csv(brands_path)
    areas = pd.read_csv(areas_path)
    rows = []
    target = []
    weights = load_weights()
    for _, b in brands.iterrows():
        for _, a in areas.iterrows():
            features = {
                'area_aov': a['AOV_area'],
                'order_freq': a['Frequency'],
                'competition_cuisine_1': a['Competition1'],
                'competition_cuisine_2': a['Competition2'],
                'competition_cuisine_3': a['Competition3'],
                'brand_aov': b['AOV'],
                'agg_position': b['AggregatorScore'],
                'brand_orders': b['MonthlyOrders'],
            }
            rows.append(features)
            target.append(sum(features[k] * weights[k] for k in weights))
    X = pd.DataFrame(rows)
    y = pd.Series(target)
    return X, y


def train_model(X, y):
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X, y)
    return model


def main():
    X, y = build_dataset()
    model = train_model(X, y)
    model_path = BASE_DIR / "xgb_model.json"
    model.save_model(model_path)
    print(f"Saved model to {model_path}")


if __name__ == '__main__':
    main()
