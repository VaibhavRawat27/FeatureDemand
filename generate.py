import pandas as pd
import numpy as np

np.random.seed(42)

# ---------------- CONFIG ----------------
categories = ["Earphones", "Mobile", "Television", "Laptop", "Smartwatch"]
brands = ["AlphaTech", "BetaGear", "GammaElectro", "DeltaTech", "OmegaWare"]
products_per_cat = 10  # known products per category
months = ["2024-01", "2024-02", "2024-03"]

rows = []
pid_counter = 1000

# ---------------- KNOWN PRODUCTS ----------------
for cat in categories:
    for i in range(products_per_cat):
        brand = np.random.choice(brands)
        name = f"{brand} {cat} Model {i+1}"
        price = np.random.randint(500, 5000)
        for m in months:
            monthly_units = np.random.randint(10, 50)
            rows.append([
                f"P{pid_counter}", name, "Electronics", m, price,
                monthly_units, monthly_units, False
            ])
        pid_counter += 1

# ---------------- COLD-START PRODUCT ----------------
for cat in categories:
    brand = np.random.choice(brands)
    name = f"{brand} {cat} New Launch"
    price = np.random.randint(500, 5000)
    for idx, m in enumerate(months):
        if idx == 2:  # launch month, leave monthly_units empty
            monthly_units = np.nan
            true_demand = np.random.randint(10, 50)
            is_launch = True
        else:
            monthly_units = np.random.randint(10, 50)
            true_demand = monthly_units
            is_launch = False
        rows.append([
            f"P{pid_counter}", name, "Electronics", m, price,
            monthly_units, true_demand, is_launch
        ])
    pid_counter += 1

# ---------------- CREATE DATAFRAME ----------------
df = pd.DataFrame(rows, columns=[
    "product_id", "product_name", "category", "month", 
    "avg_price", "monthly_units", "true_demand", "is_launch"
])

# ---------------- SAVE CSV ----------------
df.to_csv("electronics_single_coldstart.csv", index=False)
print("✅ electronics_single_coldstart.csv created with only 1 empty product unit")
print(df.tail(10))
