import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset (skipping the first 3 title/blank rows)
file_path = 'FODS_Lab_Datasets_Q20_to_Q40.xlsx'
df = pd.read_excel(file_path, sheet_name='Q31_Ecom_Clustering', skiprows=3)

# 2. Select demographic & behavioral features
feature_cols = [
    'Age',
    'Browsing_Minutes_Per_Week',
    'Purchases_Last_6Mo',
    'Total_Spend_USD',
]
X = df[feature_cols]

# 3. Standardize features so scale differences (e.g., Spend vs. Purchases) do not skew distances
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Evaluate cluster counts using Inertia (Elbow Method) and Silhouette Score
print('--- Cluster Evaluation Metrics ---')
for k in range(2, 6):
  km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
  labels = km_test.fit_predict(X_scaled)
  sil = silhouette_score(X_scaled, labels)
  print(f'k = {k} | Inertia: {km_test.inertia_:6.2f} | Silhouette Score: {sil:.4f}')

# 5. Fit final K-Means model with k = 3
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
df['Segment'] = kmeans.fit_predict(X_scaled)

# 6. Summarize each segment's characteristics
segment_summary = df.groupby('Segment')[feature_cols].mean()
segment_summary['Customer_Count'] = df['Segment'].value_counts()

print('\n--- Customer Segment Profiles (Mean Values) ---')
print(segment_summary.round(2))

# 7. Visualize the customer segments
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#2b5c8f', '#d95f02', '#1b9e77']
segment_names = {
    0: 'Segment 0: Young Occasional Buyers',
    1: 'Segment 1: High-Value Power Shoppers',
    2: 'Segment 2: Mature Moderate Shoppers',
}

for seg in sorted(df['Segment'].unique()):
  subset = df[df['Segment'] == seg]
  axes[0].scatter(
      subset['Browsing_Minutes_Per_Week'],
      subset['Total_Spend_USD'],
      c=colors[seg],
      label=segment_names[seg],
      s=60,
      alpha=0.85,
      edgecolors='k',
  )
  axes[1].scatter(
      subset['Age'],
      subset['Purchases_Last_6Mo'],
      c=colors[seg],
      label=segment_names[seg],
      s=60,
      alpha=0.85,
      edgecolors='k',
  )

axes[0].set_title('Browsing Time vs. Total Spend')
axes[0].set_xlabel('Browsing Minutes Per Week')
axes[0].set_ylabel('Total Spend (USD)')
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend(fontsize=8)

axes[1].set_title('Customer Age vs. Purchase Frequency')
axes[1].set_xlabel('Age (Years)')
axes[1].set_ylabel('Purchases in Last 6 Months')
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()
