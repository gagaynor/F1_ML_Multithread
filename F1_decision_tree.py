import sqlite3
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_score

from time import perf_counter
import os

from concurrent.futures import ThreadPoolExecutor


def main():
    cpu_count = os.cpu_count()

    print (f'Number of cpus in system: {cpu_count}')

    # Connect to the SQLite database
    conn = sqlite3.connect('F1_Prediction.db')

    # Read the data from the database into pandas dataframes
    race_results = pd.read_sql_query("SELECT * FROM RACE_RESULTS", conn)
    circuits = pd.read_sql_query("SELECT DISTINCT CIRCUIT_NAME, CIRCUIT_TYPE FROM CIRCUITS", conn)

    # Merge the dataframes based on the circuit name
    data = pd.merge(race_results, circuits, left_on='CIRCUIT', right_on='CIRCUIT_NAME')

    # Positions have not been filled in for Mexico City Grand Prix yet so drop from the dataset
    data.dropna(inplace=True)

    # Define features and target variable
    X = data[['DRIVER', 'TEAM', 'CIRCUIT_TYPE']]
    y = data['POSITION']

    # Convert categorical variables into numerical values using one-hot encoding
    X = pd.get_dummies(X)

    # Change n_jobs to modify how many CPU's are being utilized
    # model = tree.DecisionTreeClassifier()

    # Loop through using 1 to total number of cpus in the RandomForestClassifier(njobs) parameter

    # Create empty dataframe to store the execution times of each loop

    results_df = pd.DataFrame(columns=['CPUs_used', 'Execution_time', 'Prediction', 'Score'])

    # print ('Results using RandomForestClassifier() model.')    
    # print ('Results using KNeighborsClassifier() model.')
    print ('Results using SGDClassifier() model.')

    for number_cpus in range(1, cpu_count):

        models = [
            RandomForestClassifier(n_jobs=number_cpus),
            KNeighborsClassifier(n_neighbors=5,n_jobs=number_cpus),
            SGDClassifier(n_jobs=number_cpus)
        ]

        # model = RandomForestClassifier(n_jobs=number_cpus)

        # model = KNeighborsClassifier(n_jobs=number_cpus)

        # model = SGDClassifier(n_jobs=number_cpus)

        for model_inx, model in enumerate(models, start=1):
            # start counter to measure execution time
            t1_start = perf_counter() 

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
            # print("Predicted positions for the upcoming race in Las Vegas:")
            # print(predicted_positions)  

            scores = cross_val_score(estimator=model, X=X, y=y, cv=7)
            # print (f'Classification model average cross val score: {scores.mean():.04f}')


            # Stop the stopwatch / counter
            t1_stop = perf_counter()

            # create new row to append
            new_row = { 'Model': f'Model {model_inx}',
                        'CPUs_used':number_cpus ,
                        'Execution_time': round((t1_stop-t1_start), 4), 
                        'Prediction':predicted_positions[0],
                        'Score': round(scores.mean(), 4)}

            # add row to dataframe
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

            print (f'Execution with Model {model_inx} and {number_cpus} cpus complete.')

    # results_df.style.hide()

    results_df.set_index('CPUs_used', inplace=True)

    print (results_df)

    # print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()