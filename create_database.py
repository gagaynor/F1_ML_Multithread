## NOT READY YET DONT RUN THIS!!!

import sqlite3
import fastf1 as f1

def load_session_results(yr,circuit):
    session = f1.get_session(yr, circuit, 'R')
    results = session.load_results()

    # Print the final position of each driver
    print(results[['Driver', 'Team', 'Pos']])


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
'Italian Grand Prix''Singapore Grand Prix'
'Japanese Grand Prix',
'Qatar Grand Prix',
'Mexico City Grand Prix',
'São Paulo Grand Prix',
'Las Vegas Grand Prix',
'Abu Dhabi Grand Prix']

years = [2023,2022,2021,2020]


db_name = 'F1_Prediction.db'

conn = sqlite3.connect(db_name)

conn.close()
