
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =====================================================
# 1. Read the dataset
# =====================================================

df = pd.read_csv("zomato.csv")

print("=" * 60)
print("ORIGINAL DATA")
print("=" * 60)
print("Original shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum())


# =====================================================
# 2. Clean column names and values
# =====================================================

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Check that the required columns exist
required_columns = [
    "name",
    "address",
    "rate",
    "votes",
    "online_order",
    "book_table",
    "location",
    "rest_type",
    "cuisines",
    "approx_cost(for two people)",
    "listed_in(type)"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Convert rate values such as "4.1/5" or "3.4 /5" to numbers
df["rate"] = pd.to_numeric(
    df["rate"]
      .astype("string")
      .str.extract(r"(\d+(?:\.\d+)?)")[0],
    errors="coerce"
)

# Convert votes to numeric
df["votes"] = pd.to_numeric(
    df["votes"],
    errors="coerce"
)

# Convert cost to numeric after removing commas
cost_col = "approx_cost(for two people)"

df[cost_col] = pd.to_numeric(
    df[cost_col]
      .astype("string")
      .str.replace(",", "", regex=False),
    errors="coerce"
)

# Normalize Yes/No values
for col in ["online_order", "book_table"]:
    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({
            "yes": "Yes",
            "no": "No"
        })
    )

# Remove rows without a target value
df = df.dropna(subset=["rate"]).copy()

print("\nRows after removing missing rates:", len(df))


# =====================================================
# 3. Remove duplicate restaurant records
# =====================================================

# Normalize restaurant names and addresses for grouping
df["_name_key"] = (
    df["name"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df["_address_key"] = (
    df["address"]
    .astype("string")
    .str.strip()
    .str.lower()
)

# Do not combine rows whose name or address is missing:
# give each such row its own unique grouping key.
missing_key = (
    df["_name_key"].isna()
    | df["_address_key"].isna()
)

df["_row_id"] = np.arange(len(df))

df["_name_key"] = df["_name_key"].fillna(
    "__missing_name__"
    + df["_row_id"].astype(str)
)

df["_address_key"] = df["_address_key"].fillna(
    "__missing_address__"
    + df["_row_id"].astype(str)
)

# Sort so that the highest rate is kept first.
# If rates are equal, keep the row with the most votes.
df = df.sort_values(
    by=["rate", "votes"],
    ascending=[False, False],
    na_position="last"
)

# Consolidate records sharing name, address, and listing type.
# Other columns come from the selected highest-rate row.
group_keys = [
    "_name_key",
    "_address_key",
    "listed_in(type)"
]

before_grouping = len(df)

df = df.drop_duplicates(
    subset=group_keys,
    keep="first"
).copy()

print("Rows before grouping:", before_grouping)
print("Rows after grouping:", len(df))
print("Repeated records consolidated:", before_grouping - len(df))


# =====================================================
# 4. Prepare features and target
# =====================================================

# Target: restaurant rating
y = df["rate"].copy()

# Log-transform votes to reduce their skew
df["votes_log"] = np.log1p(
    df["votes"].clip(lower=0)
)

# Features used by the model
feature_columns = [
    "listed_in(type)",
    "online_order",
    "book_table",
    "location",
    "rest_type",
    "cuisines",
    "votes_log",
    cost_col
]

X = df[feature_columns].copy()

# Convert categorical columns to consistent strings.
# Keep missing values as missing so the imputer can handle them.
categorical_features = [
    "listed_in(type)",
    "online_order",
    "book_table",
    "location",
    "rest_type",
    "cuisines"
]

numeric_features = [
    "votes_log",
    cost_col
]

for col in categorical_features:
    X[col] = X[col].map(
        lambda value: str(value).strip()
        if pd.notna(value) else np.nan
    )

print("\nFeatures used:")
print(feature_columns)

print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("\nTarget summary:")
print(y.describe())


# =====================================================
# 5. Split data by restaurant
# =====================================================

# Keep the same restaurant out of both training and test sets.
# The grouping uses name + address, even if the restaurant
# appears under more than one listing type.

groups = (
    df["_name_key"].astype(str)
    + "||"
    + df["_address_key"].astype(str)
)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

shared_restaurants = set(groups_train) & set(groups_test)

print(
    "Restaurants shared between train/test:",
    len(shared_restaurants)
)


# =====================================================
# 6. Build preprocessing pipelines
# =====================================================

# Numeric columns:
# - Fill missing values with the median
# - Scale numeric values
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical columns:
# - Fill missing values with the most frequent value
# - One-Hot Encode categories
# - Combine infrequent categories to reduce dimensionality
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=20
        )
    )
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)


# =====================================================
# 7. Define regression models
# =====================================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=12,
        min_samples_leaf=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),

    "SVR": SVR(
        kernel="rbf",
        C=10,
        epsilon=0.1
    ),

    "KNN": KNeighborsRegressor(
        n_neighbors=10,
        weights="distance"
    )
}


# =====================================================
# 8. Train and evaluate models
# =====================================================

results = []
trained_models = {}

# Baseline: predict the average training rating for every row
dummy = DummyRegressor(strategy="mean")
dummy.fit(
    np.zeros((len(X_train), 1)),
    y_train
)

dummy_train_pred = dummy.predict(
    np.zeros((len(X_train), 1))
)

dummy_test_pred = dummy.predict(
    np.zeros((len(X_test), 1))
)

results.append({
    "Model": "Dummy Baseline",
    "Train R2": r2_score(y_train, dummy_train_pred),
    "Test R2": r2_score(y_test, dummy_test_pred),
    "Train MAE": mean_absolute_error(y_train, dummy_train_pred),
    "Test MAE": mean_absolute_error(y_test, dummy_test_pred),
    "Test RMSE": np.sqrt(
        mean_squared_error(y_test, dummy_test_pred)
    )
})

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    results.append({
        "Model": name,
        "Train R2": r2_score(y_train, train_pred),
        "Test R2": r2_score(y_test, test_pred),
        "Train MAE": mean_absolute_error(y_train, train_pred),
        "Test MAE": mean_absolute_error(y_test, test_pred),
        "Test RMSE": np.sqrt(
            mean_squared_error(y_test, test_pred)
        )
    })

    trained_models[name] = pipeline


# =====================================================
# 9. Compare all models
# =====================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Test MAE",
    ascending=True
).reset_index(drop=True)

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(results_df.round(4).to_string(index=False))


# =====================================================
# 10. Select the best model by Test MAE
# =====================================================

# Lower MAE means smaller average prediction error.
best_model_name = (
    results_df.loc[
        results_df["Model"] != "Dummy Baseline"
    ]
    .sort_values("Test MAE")
    .iloc[0]["Model"]
)

best_model = trained_models[best_model_name]

best_row = results_df[
    results_df["Model"] == best_model_name
].iloc[0]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Best model:", best_model_name)
print("Train R2:", round(best_row["Train R2"], 4))
print("Test R2:", round(best_row["Test R2"], 4))
print("Train MAE:", round(best_row["Train MAE"], 4))
print("Test MAE:", round(best_row["Test MAE"], 4))
print("Test RMSE:", round(best_row["Test RMSE"], 4))


# =====================================================
# 11. Compare actual and predicted ratings
# =====================================================

best_predictions = best_model.predict(X_test)

comparison = pd.DataFrame({
    "Actual Rate": y_test.to_numpy(),
    "Predicted Rate": best_predictions
})

comparison["Absolute Error"] = (
    comparison["Actual Rate"]
    - comparison["Predicted Rate"]
).abs()

print("\n" + "=" * 60)
print("ACTUAL VS PREDICTED")
print("=" * 60)

print(comparison.head(20).round(3).to_string(index=False))


# =====================================================
# 12. Final data checks
# =====================================================

print("\n" + "=" * 60)
print("FINAL CHECKS")
print("=" * 60)

print("Final dataset shape:", df.shape)
print("Missing values in X before preprocessing:")
print(X.isnull().sum())

print("\nMissing values in y:", y.isnull().sum())
print(
    "Duplicate grouping keys:",
    df.duplicated(subset=group_keys).sum()
)

print("\nDone!")
