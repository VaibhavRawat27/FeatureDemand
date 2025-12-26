import pandas as pd
import numpy as np
from features.encoder import FeatureEncoder
from similarity.similarity import SimilarityEngine
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# 1️⃣ Load dataset
# -------------------------------
df = pd.read_csv("data/products.csv")

df["monthly_units"] = pd.to_numeric(df["monthly_units"], errors="coerce")
df["true_demand"] = pd.to_numeric(df["true_demand"], errors="coerce")
df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce")

# -------------------------------
# 2️⃣ Split known vs cold-start
# -------------------------------
known_df = df[df["monthly_units"].notna()].copy()
unknown_df = df[df["monthly_units"].isna()].copy()

# Rename for encoder
for d in [known_df, unknown_df]:
    d.rename(columns={
        "monthly_units": "monthly_demand",
        "avg_price": "price",
        "product_name": "description"
    }, inplace=True)

# -------------------------------
# 3️⃣ Init components
# -------------------------------
encoder = FeatureEncoder()
sim_engine = SimilarityEngine(k=5)

print("\n=== Cold-Start Demand Predictions (Launch Month) ===\n")

predictions = []

# -------------------------------
# 4️⃣ Predict cold-start products
# -------------------------------
for _, new_prod in unknown_df.iterrows():

    # 🔒 CATEGORY FILTER (huge improvement)
    same_cat = known_df[known_df["category"] == new_prod["category"]]

    if len(same_cat) == 0:
        fallback = known_df["monthly_demand"].median()
        predictions.append(fallback)
        continue

    encoder.fit(same_cat)
    X_known = encoder.transform(same_cat)
    X_new = encoder.transform(pd.DataFrame([new_prod]))

    idxs, sims = sim_engine.find_similar(X_new, X_known)
    sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)

    demands = same_cat.iloc[idxs]["monthly_demand"].values.astype(float)
    prices = same_cat.iloc[idxs]["price"].values.astype(float)

    # 🔥 PRICE PENALTY (KEY)
    price_diff = np.abs(prices - new_prod["price"]) / new_prod["price"]
    price_weight = np.exp(-price_diff)

    sims = sims * price_weight

    # Robust clipping
    if len(demands) > 2:
        lo, hi = np.percentile(demands, [20, 80])
        demands = np.clip(demands, lo, hi)

    # 🚑 HARD FALLBACK (no NaN ever)
    if sims.sum() <= 1e-6:
        pred = np.median(demands)
        weights = np.ones_like(demands) / len(demands)
    else:
        weights = sims / sims.sum()
        pred = float(np.dot(demands, weights))

    pred = max(pred, 0)
    predictions.append(pred)

    # CI
    std = np.sqrt(np.dot(weights, (demands - pred) ** 2))
    ci_l = max(pred - 1.96 * std, 0)
    ci_u = pred + 1.96 * std

    # ---------------- PRINT ----------------
    print(f"Product ID: {new_prod['product_id']}")
    print(f"Predicted Demand: {pred:.1f}")
    print(f"95% CI: [{ci_l:.1f}, {ci_u:.1f}]")
    print("Matched Products:")

    for i, idx in enumerate(idxs):
        kp = same_cat.iloc[idx]
        print(
            f"  - {kp['product_id']} | {kp['description']} | "
            f"Demand={kp['monthly_demand']}"
        )
    print("-" * 55)

# -------------------------------
# 5️⃣ Evaluation (TRUE cold-start)
# -------------------------------
y_true = unknown_df["true_demand"].values.astype(float)
y_pred = np.array(predictions, dtype=float)

mask = np.isfinite(y_pred)
y_true = y_true[mask]
y_pred = y_pred[mask]

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("\n=== Cold-Start Evaluation (Launch Month) ===")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
