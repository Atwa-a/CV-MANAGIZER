import sqlite3

def add_column():
    conn = sqlite3.connect('cvmanagizer.db')
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE pro_settings ADD COLUMN use_custom_keywords INTEGER DEFAULT 0")
        print("Column 'use_custom_keywords' added successfully.")
    except sqlite3.OperationalError as e:
        print(f"Error adding column: {e}")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    add_column()
