import pandas as pd
import numpy as np
from features.encoder import FeatureEncoder
from similarity.similarity import SimilarityEngine
from forecasting.predictor import DemandPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# 1️⃣ Load dataset
# -------------------------------
df = pd.read_csv("processed_products.csv")

# Ensure numeric monthly_demand
df['monthly_demand'] = pd.to_numeric(df['monthly_demand'], errors='coerce')

# Split known vs unknown products
known_df = df[df['monthly_demand'].notna()].copy()
unknown_df = df[df['monthly_demand'].isna()].copy()

if len(known_df) < 1:
    raise ValueError("You must have at least 1 product with known demand")

# -------------------------------
# 2️⃣ Initialize components
# -------------------------------
encoder = FeatureEncoder()
sim_engine = SimilarityEngine(k=5)  # top-k similar products
predictor = DemandPredictor()

print("\n=== Cold-Start Demand Predictions ===\n")

# -------------------------------
# 3️⃣ Predict unknown/new products
# -------------------------------
for idx_new, new_product in unknown_df.iterrows():
    encoder.fit(known_df)
    X_existing = encoder.transform(known_df)
    X_new = encoder.transform(pd.DataFrame([new_product]))

    # Find top similar products
    top_idx, sims = sim_engine.find_similar(X_new, X_existing)
    demands = known_df.iloc[top_idx]["monthly_demand"].astype(float).values
    sims = np.maximum(sims, 0)

    # Clip extreme demands to reduce MAPE
    if len(demands) > 1:
        low, high = np.percentile(demands, [10, 90])
        demands = np.clip(demands, low, high)

    # Weighted prediction
    if sims.sum() == 0:
        weights = np.ones_like(sims) / len(sims)
    else:
        weights = sims / sims.sum()
    pred = float(np.dot(demands, weights))
    pred = max(pred, 0)

    # Confidence interval (95%)
    std = np.sqrt(np.dot(weights, (demands - pred)**2))
    ci_lower = max(pred - 1.96 * std, 0)
    ci_upper = pred + 1.96 * std

    # Display results
    print(f"Product ID: {new_product['product_id']}")
    print(f"Predicted Demand: {pred:.0f}")
    print(f"95% Confidence Interval: [{ci_lower:.0f}, {ci_upper:.0f}]")
    print("Top Similar Products:")
    for rank, sim_idx in enumerate(top_idx):
        sp = known_df.iloc[sim_idx]
        print(f"  {rank+1}. {sp['product_id']} | Demand: {sp['monthly_demand']} | Similarity: {sims[rank]:.2f}")
    print("-" * 50)

# -------------------------------
# 4️⃣ Evaluate on known products
# -------------------------------
if len(known_df) > 1:
    eval_true = []
    eval_pred = []

    for i in range(1, len(known_df)):
        existing = known_df.iloc[:i].copy()
        new_product = known_df.iloc[i].copy()

        encoder.fit(existing)
        X_existing = encoder.transform(existing)
        X_new = encoder.transform(pd.DataFrame([new_product]))

        top_idx, sims = sim_engine.find_similar(X_new, X_existing)
        demands = existing.iloc[top_idx]["monthly_demand"].astype(float).values
        sims = np.maximum(sims, 0)

        if len(demands) > 1:
            low, high = np.percentile(demands, [10, 90])
            demands = np.clip(demands, low, high)

        if sims.sum() == 0:
            weights = np.ones_like(sims) / len(sims)
        else:
            weights = sims / sims.sum()

        pred = float(np.dot(demands, weights))
        pred = max(pred, 0)

        eval_true.append(float(new_product["monthly_demand"]))
        eval_pred.append(pred)

    eval_true = np.array(eval_true, dtype=float)
    eval_pred = np.array(eval_pred, dtype=float)

    mae = mean_absolute_error(eval_true, eval_pred)
    rmse = np.sqrt(mean_squared_error(eval_true, eval_pred))
    mape = np.mean(np.abs((eval_true - eval_pred) / eval_true)) * 100

    print("\n=== Evaluation Metrics on Known Products ===")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
