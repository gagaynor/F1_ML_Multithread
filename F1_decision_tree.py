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

import matplotlib.pyplot as plt

def process_model(args):

    number_cpus, model_name, model, X, y, upcoming_race_df = args
    
    t1_start = perf_counter()
    
    model.fit(X, y)
    predicted_positions = model.predict(upcoming_race_df)
    scores = cross_val_score(estimator=model, X=X, y=y, cv=7)
    t1_stop = perf_counter()

    new_row = {
        'Model': f'{model_name}',
        'CPUs_used': number_cpus,
        'Execution_time': round((t1_stop - t1_start), 4),
        'Prediction': predicted_positions[0],
        'Score': round(scores.mean(), 4)
    }

    return new_row


def main():
    '''
    Loop through a range of 1 up to the max number of cpus on a machine.
    In each loop, call three models using an increading number of workers (one worker for each cpu)

    The n_jobs parameter is set to 1 for each model.
    '''

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

    # Create empty dataframe to store the execution times of each loop
    results_df = pd.DataFrame(columns=['Model','CPUs_used', 'Execution_time', 'Prediction', 'Score'])

        # Create empty dataframe to store the execution times of each loop
    summary_results_df = pd.DataFrame(columns=['Model','CPUs_used', 'Execution_time'])

    # Initialize untrained models to pass into the process_model function
    # Set n_jobs to 1. This ensures the thread pool is directing traffic.
    models = [{'name':"Random_Forest", 'model': RandomForestClassifier(n_jobs=1)},
        {'name':"KNeighbors", 'model': KNeighborsClassifier(n_neighbors=5,n_jobs=1)},
        {'name':"SGD", 'model': SGDClassifier(n_jobs=1)}
    ]

    # Create results list for all output
    results = []

    # Loop through using 1 to total number of cpus. Add 1 since range (1, 12) is not inclusive of the 12
    for number_cpus in range(1, cpu_count + 1): 


        # Keep track of how much time it took to do all three models
        total_time_start = perf_counter()

        for named_model in models:

            model_name = named_model.get('name')
            model = named_model.get('model')

            # Test data to use for prediction:
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



            # Create a pool with an increasing number or workers
            with ThreadPoolExecutor(max_workers=number_cpus) as executor:

                    # futures.append(
                    future = executor.submit(process_model, (number_cpus, model_name, model, X, y, upcoming_race_df))
                    # )

                    result = future.result()

                    results.append(result)

        total_time_stop = perf_counter()

        # Add a row for the total time to train and predict with all three models
        summary_row = {
        'Model': 'All three models',
        'CPUs_used': number_cpus,
        'Execution_time': round((total_time_stop - total_time_start), 4),
        'Prediction': '',
        'Score': ''
        }

        summary_results_df = pd.concat([summary_results_df, pd.DataFrame([summary_row])], ignore_index=True)



    for result in results:
        results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)
        print(f'Execution with {result["Model"]} and {result["CPUs_used"]} cpus complete.')
                


    results_df.set_index('CPUs_used', inplace=True)

    print (results_df)

    ## Uncomment this next section if you want to plot results

    results_df['Execution_time'] = pd.to_numeric(results_df['Execution_time'])

    # Group by 'CPUs_used' and sum up the 'Execution_time' for each CPU count
    sum_exec_time = results_df.groupby('CPUs_used')['Execution_time'].sum()

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.bar(sum_exec_time.index, sum_exec_time.values, color='black')
    plt.xlabel('CPUs_used')
    plt.ylabel('Sum of Execution Times')
    plt.title('Sum of Execution Times vs CPUs_used')
    plt.xticks(sum_exec_time.index)
    plt.grid(axis='y')

    # Show plot
    # plt.show()
    
    # Save the plot to an image file
    plt.savefig('sum_exec_time.png', bbox_inches="tight")


    sum_exec_time_models = results_df.groupby('Model')['Execution_time'].sum()

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.bar(sum_exec_time_models.index, sum_exec_time_models.values, color='red')
    plt.xlabel('Models')
    plt.ylabel('Sum of Execution Times')
    plt.title('Sum of Execution Times for Different Models')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(axis='y')

    # Show plot
    # plt.tight_layout()
    # plt.show()

    # Save the plot to an image file
    plt.savefig('sum_exec_time_models.png', bbox_inches="tight")

    # Group by 'Model' and 'CPUs_used' and sum up the 'Execution_time' for each combination
    sum_exec_time_models_cpus = results_df.groupby(['Model', 'CPUs_used'])['Execution_time'].sum()

    # Unstack the DataFrame to prepare for plotting
    sum_exec_time_models_cpus = sum_exec_time_models_cpus.unstack(level=0)

    # Plotting
    plt.figure(figsize=(16, 8))
    sum_exec_time_models_cpus.plot(kind='bar', stacked=False)
    plt.xlabel('CPUs Used')
    plt.ylabel('Sum of Execution Times')
    plt.title('Sum of Execution Times for Each Model Across CPU Usages')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show plot
    # plt.tight_layout()
    # plt.show()

    # Save the plot to an image file
    plt.savefig('sum_exec_time_models_cpus.png', bbox_inches="tight")


    print(summary_results_df)
    print()

    results_df.sort_values(by=['Model', 'CPUs_used'], inplace=True)

    print (results_df)

if __name__ == "__main__":
    main()