"""
NFL Play Success Prediction Model

This program analyzes NFL play-by-play data from the 2023 season using the
nfl_data_py library. A play is labeled as "successful" if its Expected Points
Added (EPA) is greater than zero.

The script:
- Loads and preprocesses NFL play-by-play data
- Selects situational and play-type features (down, distance, field position,
  formation, and score differential)
- Visualizes feature correlations using a heatmap
- Trains a logistic regression model to predict play success
- Displays model coefficients to show feature influence on success

The goal of this project is to explore which game situations and play
characteristics most strongly contribute to positive play outcomes.
"""
# NFL Play Prediction Model
# importing necessary libraries and NFL data
import nfl_data_py as nfl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# importing necessary sklearn libraries
# for data preprocessing, model building, and evaluation
# using logistic regression as the model -> binary classification problem -> predicting play success (yes/no)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load NFL play-by-play data for the 2023 season
pbp = nfl.import_pbp_data([2023])

# Select relevant features for the model
pbp['success'] = (pbp['epa'] > 0).astype(int)
features = ['down', 
            'ydstogo', 
            'yardline_100', 
            'shotgun', 
            'no_huddle',
            'pass',
            'rush',
            'score_differential']

# Prepare the dataset
df = pbp[features + ['success']].dropna()
corr = df.corr()

# heat map to visualize feature correlations
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)
plt.title("Feature Correlation Heatmap")
plt.show()

# Split the data into training and testing sets
X = df[features]
y = df['success']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train the logistic regression model
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

# Evaluate the model
coef_df = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_[0]
}).set_index('feature')

# Visualize feature coefficients
sns.heatmap(coef_df, annot=True, cmap="coolwarm", center=0)
plt.title("Logistic Regression Feature Weights")
plt.show()


