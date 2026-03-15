import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load your study data
data = pd.read_csv("study_data.csv")

# Convert priority to numeric
priority_map = {'High':3, 'Medium':2, 'Low':1}
data['priority_num'] = data['priority'].map(priority_map)

# Features: priority_num and estimated_hours
X = data[['priority_num']]
y = data['estimated_hours']  # Here we predict study hours

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save the trained model
with open("planner_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("planner_model.pkl created successfully!")