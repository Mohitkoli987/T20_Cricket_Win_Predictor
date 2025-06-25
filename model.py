import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv('dataset.csv')
x = df.iloc[:, :-1]
y = df.iloc[:,-1]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=100)

# Preprocessing
trf = ColumnTransformer([
    ('trf', OneHotEncoder(drop='first'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

# Create Random Forest with 5 trees for demonstration
rf_model = RandomForestClassifier(n_estimators=5, random_state=42)

# Create pipeline
ra_pipe = Pipeline([
    ('step1', trf),
    ('step2', rf_model)
])

# Train the model
ra_pipe.fit(x_train, y_train)

# Make predictions
ra_y_pred = ra_pipe.predict(x_test)

# Print model information
print("\nRandom Forest Model Information:")
print(f"Number of trees: {len(rf_model.estimators_)}")
print(f"Accuracy: {accuracy_score(y_test, ra_y_pred):.2f}")

# Show how individual trees make predictions
sample_data = x_test.iloc[0:1]  # Take first test sample
print("\nSample Prediction Demonstration:")
print("Test case features:")
print(sample_data)
print("\nActual value:", y_test.iloc[0])
print("Random Forest prediction:", ra_pipe.predict(sample_data)[0])

# Show individual tree predictions
print("\nIndividual Tree Predictions:")
transformed_sample = ra_pipe.named_steps['step1'].transform(sample_data)
for i, tree in enumerate(rf_model.estimators_):
    tree_pred = tree.predict(transformed_sample)
    print(f"Tree {i+1} prediction: {tree_pred[0]}")

# Save the model
pickle.dump(ra_pipe, open('ra_pipe.pkl', 'wb'))