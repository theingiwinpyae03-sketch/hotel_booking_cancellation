# src/train_models.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.data_preprocessing import load_and_clean_data
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_and_clean_data
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
...
# 1. Load data
# To this (since the CSV is in your main folder):
df = load_and_clean_data('hotel_bookings.csv')
# 2. Prepare features and target
X = df.drop(['is_canceled', 'financial_loss'], axis=1)
y = df['is_canceled']

# 3. One‑hot encode categoricals
X = pd.get_dummies(X, drop_first=True)

# 4. Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Scale numeric columns
num_cols = ['lead_time', 'adr', 'total_guests', 'total_stays']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 6. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# 7. Train XGBoost with scale_pos_weight
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)

# 8. Evaluate both models (optional, just to print)
for name, model in [('Random Forest', rf), ('XGBoost', xgb)]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{name} -> Acc:{acc:.4f}, Prec:{prec:.4f}, Rec:{rec:.4f}, F1:{f1:.4f}")

# Change these lines:
joblib.dump(rf, 'random_forest.pkl') # Removed 'models/'
joblib.dump(xgb, 'xgboost.pkl')      # Removed 'models/'
joblib.dump(scaler, 'scaler.pkl')    # Removed 'models/'
joblib.dump(X_train.columns.tolist(), 'columns.pkl') # Removed 'models/'
# --- At the bottom of train_models.py ---
# ============================================================
# ASSOCIATION RULES FOR CANCELLATIONS (Integrated)
# ============================================================
from mlxtend.frequent_patterns import apriori, association_rules

print("--- Starting Descriptive Mining (Association Rules) ---")

# 1. Filter for only canceled bookings
# Note: Ensure 'df' is the name of your cleaned dataframe in this script
cancel_df = df[df['is_canceled'] == 1].copy()

# 2. Select columns
cat_cols_ar = ['hotel', 'market_segment', 'deposit_type', 'customer_type', 'arrival_date_month']

# 3. One-hot encode and run Apriori
transactions = pd.get_dummies(cancel_df[cat_cols_ar]).astype(bool)
frequent_itemsets = apriori(transactions, min_support=0.05, use_colnames=True)

# 4. Generate rules and sort by Lift
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values('lift', ascending=False)

# 5. Save to CSV (Streamlit will read this file)
rules.to_csv('cancellation_rules.csv', index=False)

print("Done! 'cancellation_rules.csv' has been generated for the UI.")
