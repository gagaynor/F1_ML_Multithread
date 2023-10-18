import sqlite3
import pandas as pd
from sklearn import tree

# Connect to the SQLite database
conn = sqlite3.connect('F1_Prediction.db')

# Read the data from the database into pandas dataframes
race_results = pd.read_sql_query("SELECT * FROM RACE_RESULTS", conn)
circuits = pd.read_sql_query("SELECT DISTINCT CIRCUIT_NAME, CIRCUIT_TYPE FROM CIRCUITS", conn)

# Merge the dataframes based on the circuit name
data = pd.merge(race_results, circuits, left_on='CIRCUIT', right_on='CIRCUIT_NAME')

# Preprocess your data if necessary, encoding categorical variables?
# Have not done any of this yet

# Define features and target variable
X = data[['DRIVER', 'TEAM', 'CIRCUIT_TYPE']]
y = data['POSITION']

# Convert categorical variables into numerical values using one-hot encoding
X = pd.get_dummies(X)

model = tree.DecisionTreeClassifier()

# Fit the model to your data
model.fit(X, y)

# use the model to make predictions on new data
# Have not done this either

# Preprocess the input data for the upcoming race in Las Vegas
upcoming_race_data = {'DRIVER': ['L HAMILTON'], 'TEAM': ['Mercedes'], 'CIRCUIT_TYPE': ['Street']}
upcoming_race_df = pd.DataFrame(upcoming_race_data)
upcoming_race_df = pd.get_dummies(upcoming_race_df)

# Make predictions using the trained model
predicted_positions = model.predict(upcoming_race_df)

# Print the predicted positions for the upcoming race
print("Predicted positions for the upcoming race in Las Vegas:")
print(predicted_positions)
