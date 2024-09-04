from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV

# Load the dataset and prepare the model as done before
df = pd.read_csv("titanic/train.csv")
df.drop(columns=['Name', "PassengerId"], inplace=True)
df.dropna(inplace=True)

X = df.drop(columns=['Survived'])  # Predictors
y = df['Survived']  # Target variable

X = pd.get_dummies(X)
X = X.astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform RFE (Recursive Feature Elimination) to select top features
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=10)  # Selecting top 10 features
rfe = rfe.fit(X_scaled, y)

# List the selected features
selected_features = X.columns[rfe.support_]

X_train, X_test, y_train, y_test = train_test_split(X_scaled[:, rfe.support_], y, test_size=0.3, random_state=42)

# Perform GridSearchCV to find the best model
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear']
}

grid_model = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_model.fit(X_train, y_train)

# Use the best model for prediction
best_model = grid_model.best_estimator_

# Flask app
app = Flask(__name__)

# Function to predict based on the new fields
def predict_recession(features):
    try:
        # Create a DataFrame with all possible feature columns, initializing them to 0
        all_features = pd.DataFrame(columns=X.columns)
        all_features.loc[0] = 0  # Initialize the first (and only) row to zeros
        
        # Fill in the user's input data for the selected features
        for feature_name, feature_value in features.items():
            if feature_name in all_features.columns:
                all_features.at[0, feature_name] = feature_value

        # Ensure the DataFrame has all the columns in the correct order
        all_features = all_features.reindex(columns=X.columns, fill_value=0)
        
        # Scale the features
        features_scaled = scaler.transform(all_features)

        # Predict using the best logistic model
        prediction = best_model.predict(features_scaled[:, rfe.support_])

        return int(prediction[0])
    
    except Exception as e:
        # Print the exception to the console
        print(f"Error in prediction: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML template file

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    features = {
        'Age': float(data['age']),
        'Sex_female': int(data['sex_female']),
        'Sex_male': int(data['sex_male']),
        'Ticket_113760': int(data['ticket_113760']),
        'Ticket_113781': int(data['ticket_113781']),
        'Ticket_347054': int(data['ticket_347054']),
        'Ticket_PC 17572': int(data['ticket_pc_17572']),
        'Cabin_C49': int(data['cabin_c49']),
        'Cabin_E24': int(data['cabin_e24']),
        'Cabin_E77': int(data['cabin_e77']),
    }
    
    # Call your Python function with the input data
    result = predict_recession(features)
    
    if result is not None:
        return jsonify({'prediction': result})
    else:
        return jsonify({'prediction': 'Error in prediction.'})

if __name__ == '__main__':
    app.run(debug=True)
