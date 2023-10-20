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

model = tree.DecisionTreeClassifier(n_jobs=-1)

# Fit the model to your data
model.fit(X, y)

# mark a 1 for the values we would like to predict, refer to the table to know which circuit type to use. 
# for example Albon = 1, Williams = 1 and Street type circuit = 1 for las vegas. Alex Albon would be projected 14th.
upcoming_race_df = pd.DataFrame({
    'DRIVER_A ALBON': [1], 'DRIVER_A GIOVINAZZI': [0],
    'DRIVER_C LECLERC': [0], 'DRIVER_C SAINZ': [0], 'DRIVER_D KVYAT': [0],
    'DRIVER_D RICCIARDO': [0], 'DRIVER_E OCON': [0], 'DRIVER_F ALONSO': [0],
    'DRIVER_G RUSSELL': [0], 'DRIVER_G ZHOU': [0], 'DRIVER_K MAGNUSSEN': [0],
    'DRIVER_K RAIKKONEN': [0],'DRIVER_L HAMILTON': [0], 'DRIVER_L LAWSON': [0], 'DRIVER_L NORRIS': [0],
    'DRIVER_L SARGEANT': [0], 'DRIVER_L STROLL': [0], 'DRIVER_M SCHUMACHER': [0],
    'DRIVER_M VERSTAPPEN': [0], 'DRIVER_N DE VRIES': [0], 'DRIVER_N HULKENBERG': [0],
    'DRIVER_N LATIFI': [0], 'DRIVER_N MAZEPIN': [0], 'DRIVER_O PIASTRI': [0],
    'DRIVER_P FITTIPALDI': [0], 'DRIVER_P GASLY': [0], 'DRIVER_R GROSJEAN': [0],
    'DRIVER_R KUBICA': [0], 'DRIVER_S PEREZ': [0], 'DRIVER_S VETTEL': [0],
    'DRIVER_V BOTTAS': [0],'DRIVER_Y TSUNODA': [0], 'DRIVER_nan': [0],
    'TEAM_Alfa Romeo': [0], 'TEAM_Alfa Romeo Racing': [0], 'TEAM_AlphaTauri':[1],
    'TEAM_Alpine':[0], 'TEAM_Aston Martin': [0], 'TEAM_Ferrari': [0], 'TEAM_Haas F1 Team': [0],
    'TEAM_McLaren': [0], 'TEAM_Mercedes': [0], 'TEAM_Racing Point': [0], 'TEAM_Red Bull Racing': [1], 
    'TEAM_Renault': [0], 'TEAM_Williams': [1], 'TEAM_nan': [0],
    'CIRCUIT_TYPE_High Speed': [0], 'CIRCUIT_TYPE_High-Elevation': [0],
    'CIRCUIT_TYPE_Intermediate': [0], 'CIRCUIT_TYPE_Low-Speed': [0],
    'CIRCUIT_TYPE_Other': [0], 'CIRCUIT_TYPE_Street': [1]
})

# Make predictions using the trained model
predicted_positions = model.predict(upcoming_race_df)

# Print the predicted positions for the upcoming race
print("Predicted positions for the upcoming race in Las Vegas:")
print(predicted_positions)
