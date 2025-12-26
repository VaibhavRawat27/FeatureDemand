import pandas as pd
import numpy as np

# 1️⃣ Load the full Superstore dataset safely
df = pd.read_csv("Superstore.csv", encoding='ISO-8859-1')

# 2️⃣ Aggregate by Product ID to create a small dataset
agg = df.groupby(['Product ID', 'Category', 'Product Name'], as_index=False).agg({
    'Sales': 'sum',
    'Profit': 'sum'
})

# 3️⃣ Add monthly_demand (some known, some unknown)
# We'll mark a few products as unknown (NaN) for testing
np.random.seed(42)
agg['monthly_demand'] = np.random.randint(5, 20, size=len(agg))
# Make some unknown
agg.loc[agg.index[-3:], 'monthly_demand'] = np.nan

# 4️⃣ Add 'price' and 'description' columns for your main code
# For simplicity, price = Sales / random quantity, description = product name
agg['price'] = agg['Sales'] / np.random.randint(1, 5, size=len(agg))
agg['description'] = agg['Product Name']

# 5️⃣ Normalize column names (optional but recommended)
agg.columns = agg.columns.str.strip().str.lower()
agg.rename(columns={'product id':'product_id', 'product name':'description'}, inplace=True)

# 6️⃣ Save processed CSV for main.py
agg.to_csv("processed_products.csv", index=False)
print("✅ processed_products.csv generated with sample demand values.")
print(agg.head(10))
