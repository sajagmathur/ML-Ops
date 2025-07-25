# train.py
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load and prepare data
df = pd.read_csv("ice_cream.csv")
X = df[['temp']]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and test data for later
joblib.dump(model, "model.pkl")
joblib.dump((X_test, y_test), "test_data.pkl")

print(" Model trained and saved to model.pkl")
