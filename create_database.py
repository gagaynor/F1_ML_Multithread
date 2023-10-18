import sqlite3
import fastf1 as f1

def load_session_results(yr, circuit):
    try:
        session = f1.get_session(yr, circuit, 'R')
        if session is not None:
            session.load()
            results = session.results
            return results[['BroadcastName', 'TeamName', 'Position']]
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def categorize_circuit(circuit_name):
    high_speed_circuits = ['Italian Grand Prix', 'Belgian Grand Prix', 'British Grand Prix', 'Azerbaijan Grand Prix']
    street_circuits = ['Monaco Grand Prix', 'Singapore Grand Prix', 'Miami Grand Prix', 'Azerbaijan Grand Prix', 'Las Vegas Grand Prix']
    intermediate_circuits = ['Hungarian Grand Prix', 'Spanish Grand Prix','São Paulo Grand Prix', 'Japanese Grand Prix', 'Austrian Grand Prix', 'Qatar Grand Prix']
    low_speed_circuits = ['Monaco Grand Prix', 'Bahrain Grand Prix','Singapore Grand Prix','Abu Dhabi Grand Prix', 'Hungarian Grand Prix', 'Dutch Grand Prix',]
    high_elevation_circuits = ['Mexico City Grand Prix','São Paulo Grand Prix',]
    abrasive_circuits = ['Circuit of the Americas', 'Monaco Grand Prix','Azerbaijan Grand Prix',]
    low_downforce_circuits = ['Italian Grand Prix', 'Belgian Grand Prix']

    if circuit_name in high_speed_circuits:
        return 'High Speed'
    elif circuit_name in street_circuits:
        return 'Street'
    elif circuit_name in intermediate_circuits:
        return 'Intermediate'
    elif circuit_name in low_speed_circuits:
        return 'Low-Speed'
    elif circuit_name in high_elevation_circuits:
        return 'High-Elevation'
    elif circuit_name in abrasive_circuits:
        return 'Abrasive'
    elif circuit_name in low_downforce_circuits:
        return 'Low-Downforce'
    else:
        return 'Other'


circuits = ['Bahrain Grand Prix',
'Saudi Arabian Grand Prix',
'Australian Grand Prix',
'Azerbaijan Grand Prix',
'Miami Grand Prix',
'Monaco Grand Prix',
'Spanish Grand Prix',
'Canadian Grand Prix',
'Austrian Grand Prix',
'British Grand Prix',
'Hungarian Grand Prix',
'Belgian Grand Prix',
'Dutch Grand Prix',
'Italian Grand Prix',
'Singapore Grand Prix',
'Japanese Grand Prix',
'Qatar Grand Prix',
'Mexico City Grand Prix',
'São Paulo Grand Prix',
'Las Vegas Grand Prix',
'Abu Dhabi Grand Prix']

years = [2023,2022,2021,2020]


db_name = 'F1_Prediction.db'

conn = sqlite3.connect(db_name)
cur = conn.cursor()
# Create RACE_RESULTS table
cur.execute('''CREATE TABLE IF NOT EXISTS RACE_RESULTS
            (PK INTEGER PRIMARY KEY,
            DRIVER TEXT,
            TEAM TEXT,
            POSITION INTEGER,
            CIRCUIT TEXT,
            YEAR INTEGER)''')

# Create CIRCUITS table
cur.execute('''CREATE TABLE IF NOT EXISTS CIRCUITS
            (PK INTEGER PRIMARY KEY,
            CIRCUIT_NAME TEXT,
            CIRCUIT_TYPE TEXT)''')

# Insert data into the tables
for yr in years:
    for circuit in circuits:
        results = load_session_results(yr, circuit)
        circuit_type = categorize_circuit(circuit)
        if results is not None:
            for idx, result in results.iterrows():
                print(idx, "result:", result)
                try:
                    cur.execute("INSERT INTO RACE_RESULTS (DRIVER, TEAM, POSITION, CIRCUIT, YEAR) VALUES (?, ?, ?, ?, ?)", 
                                (result['BroadcastName'], result['TeamName'], result['Position'], circuit, yr))
                    conn.commit()
                except sqlite3.Error as e:
                    print(f"An error occurred: {e}")
        else: 
            continue
        cur.execute("INSERT INTO CIRCUITS (CIRCUIT_NAME, CIRCUIT_TYPE) VALUES (?, ?)", (circuit, circuit_type))
        conn.commit()
        print("Added ",circuit, yr, " to DB.")

conn.commit()
conn.close()

conn.close()

print("Connection closed, all data loaded to database.")
