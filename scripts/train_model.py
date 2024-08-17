import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load data
reference_data = pd.read_csv("../data/reference_data.csv")
new_data = pd.read_csv("../data/new_data.csv")

# Concatenate reference and new data
df = pd.concat([reference_data, new_data], ignore_index=True)

# Define features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Define numeric features (assuming all features are numeric in this example)
numeric_features = X.columns.tolist()

# Create a numeric transformer pipeline
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

# Create a preprocessor with the numeric transformer
preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, numeric_features)]
)

# Create the full pipeline with preprocessing and classifier
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# Fit the pipeline to the data
pipeline.fit(X, y)

# Save the pipeline to a file
with open("../models/pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)