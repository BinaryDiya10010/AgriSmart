import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'agrismart.db')


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            first_name TEXT,
            last_name TEXT,
            state TEXT,
            district TEXT,
            farm_size REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table 2: Crop Recommendations - Store ML-generated crop suggestions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS crop_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            soil_data TEXT,           -- JSON: NPK levels, pH
            climate_data TEXT,        -- JSON: temperature, rainfall, humidity
            farm_data TEXT,           -- JSON: farm size, irrigation type
            recommendations TEXT,     -- JSON: recommended crops with scores
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS disease_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_path TEXT,
            crop_type TEXT,
            disease_name TEXT,
            confidence REAL,
            severity TEXT,
            treatment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS storage_batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            crop_type TEXT,
            quantity REAL,
            harvest_date DATE,
            storage_date DATE,
            temperature REAL,
            humidity REAL,
            storage_type TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS crops (
            crop_id TEXT PRIMARY KEY,
            crop_name TEXT NOT NULL,
            scientific_name TEXT,
            description TEXT,
            soil_requirements TEXT,
            climate_requirements TEXT,
            cultivation_data TEXT,
            economic_data TEXT
        )
    ''')
    
    conn.commit()
    conn.close()


def insert_crop_recommendation(user_id, soil_data, climate_data, farm_data, recommendations):
    conn = get_connection()
    cursor = conn.cursor()
    
    import json
    cursor.execute('''
        INSERT INTO crop_recommendations (user_id, soil_data, climate_data, farm_data, recommendations)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, json.dumps(soil_data), json.dumps(climate_data), 
          json.dumps(farm_data), json.dumps(recommendations)))
    
    conn.commit()
    recommendation_id = cursor.lastrowid
    conn.close()
    
    return recommendation_id


def insert_disease_analysis(user_id, image_path, crop_type, disease_name, confidence, severity, treatment):
    conn = get_connection()
    cursor = conn.cursor()
    
    import json
    cursor.execute('''
        INSERT INTO disease_analyses (user_id, image_path, crop_type, disease_name, confidence, severity, treatment)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, image_path, crop_type, disease_name, confidence, severity, json.dumps(treatment)))
    
    conn.commit()
    analysis_id = cursor.lastrowid
    conn.close()
    
    return analysis_id


def insert_storage_batch(user_id, crop_type, quantity, harvest_date, storage_date, 
                        temperature, humidity, storage_type, status='GREEN'):
    """
    Register a new crop storage batch in the database.
    
    Args:
        user_id (int): ID of the farmer
        crop_type (str): Type of crop being stored
        quantity (float): Quantity in quintals
        harvest_date (str): Date of harvest (YYYY-MM-DD)
        storage_date (str): Date storage began (YYYY-MM-DD)
        temperature (float): Storage temperature in Celsius
        humidity (float): Storage humidity percentage
        storage_type (str): Type of storage facility
        status (str): Initial status (default: 'GREEN')
    
    Returns:
        int: ID of the newly created batch
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO storage_batches 
        (user_id, crop_type, quantity, harvest_date, storage_date, temperature, humidity, storage_type, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, crop_type, quantity, harvest_date, storage_date, temperature, humidity, storage_type, status))
    
    conn.commit()
    batch_id = cursor.lastrowid
    conn.close()
    
    return batch_id


def get_storage_batch(batch_id):
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM storage_batches WHERE id = ?', (batch_id,))
    batch = cursor.fetchone()
    
    conn.close()
    return dict(batch) if batch else None


def get_all_storage_batches(user_id=None):
    conn = get_connection()
    cursor = conn.cursor()
    
    if user_id:
        # Get batches for specific user
        cursor.execute('SELECT * FROM storage_batches WHERE user_id = ? ORDER BY created_at DESC', (user_id,))
    else:
        # Get all batches
        cursor.execute('SELECT * FROM storage_batches ORDER BY created_at DESC')
    
    batches = cursor.fetchall()
    conn.close()
    
    # Convert SQLite Row objects to dictionaries for easier use
    return [dict(batch) for batch in batches]


def delete_storage_batch(batch_id):
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM storage_batches WHERE id = ?', (batch_id,))
    conn.commit()
    deleted = cursor.rowcount  # Number of rows affected
    
    conn.close()
    return deleted > 0  # Return True if at least one row was deleted


def update_storage_batch(batch_id, **kwargs):
    conn = get_connection()
    cursor = conn.cursor()
    
    allowed_fields = ['crop_type', 'quantity', 'harvest_date', 'storage_date', 
                     'temperature', 'humidity', 'storage_type', 'status']
    
    updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
    
    if not updates:
        return False
    
    set_clause = ', '.join([f'{k} = ?' for k in updates.keys()])
    values = list(updates.values()) + [batch_id]
    
    cursor.execute(f'UPDATE storage_batches SET {set_clause} WHERE id = ?', values)
    conn.commit()
    updated = cursor.rowcount
    
    conn.close()
    return updated > 0


if __name__ == '__main__':
    # Initialize database when run directly
    init_db()
