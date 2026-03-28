import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "malaria_diagnoses.db"

def init_db():
    """Initializes the SQLite database if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            diagnosis TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_diagnosis(patient_id, diagnosis, confidence, severity):
    """Saves a single diagnosis to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO diagnoses (patient_id, diagnosis, confidence, severity)
        VALUES (?, ?, ?, ?)
    ''', (patient_id, diagnosis, confidence, severity))
    conn.commit()
    conn.close()

def get_all_diagnoses():
    """Retrieves all diagnoses as a pandas DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM diagnoses ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def clear_all_data():
    """Clears all data from the diagnoses table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM diagnoses")
    conn.commit()
    conn.close()
