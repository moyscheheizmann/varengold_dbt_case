import duckdb
import os
from pathlib import Path
# Set up paths
MODULE_DIR = Path(__file__).resolve().parent
REPO_DIR = MODULE_DIR.parent.parent
TRANS_DIR = REPO_DIR / 'transformation'
if not TRANS_DIR.exists():
    raise FileNotFoundError(f"Data directory {TRANS_DIR} does not exist. Please create it and add the required data files.")

DB_PATH = TRANS_DIR / "casestudy.duckdb"

if __name__ == "__main__":
    conn = duckdb.connect(DB_PATH)
    # List all schemas
    print("--- Schemas ---")
    schemas = conn.execute("SELECT schema_name FROM information_schema.schemata").fetchall()
    print(schemas)

    # List tables in the staging schema
    print("\n--- Tables in staging schema ---")
    try:
        tables = conn.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'staging'
        """).fetchall()
        print(tables)
    except Exception as e:
        print(f"Error querying staging tables: {e}")

    # List tables in the main schema
    print("\n--- Tables in main schema ---")
    try:
        tables = conn.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main'
        """).fetchall()
        print(tables)
    except Exception as e:
        print(f"Error querying main tables: {e}")
