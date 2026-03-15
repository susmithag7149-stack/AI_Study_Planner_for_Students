from flask import Flask, render_template, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load study data
DATA_FILE = "study_data.csv"
data = pd.read_csv(DATA_FILE)

# Optional: Load trained AI model
MODEL_FILE = "planner_model.pkl"
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
else:
    model = None

@app.route('/')
def dashboard():
    # Send study data to HTML via template
    study_list = data.to_dict(orient='records')
    return render_template('index.html', study_data=study_list)

@app.route('/predict/<priority>')
def predict(priority):
    """
    Optional endpoint to predict study hours based on priority
    """
    if model:
        # Map priority to numeric
        priority_map = {'High':3, 'Medium':2, 'Low':1}
        pred_hours = model.predict([[priority_map.get(priority, 1)]])
        return jsonify({"predicted_hours": round(pred_hours[0], 2)})
    else:
        return jsonify({"error":"Model not found"})

if __name__ == '__main__':
    app.run(debug=True)