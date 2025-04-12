"""
Utility functions for data exploration.
"""

import pandas as pd
import duckdb
from typing import Optional, Any, Dict, List, Union


def execute_query(conn: duckdb.DuckDBPyConnection, query: str) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a pandas DataFrame.
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        DuckDB connection object
    query : str
        SQL query to execute
        
    Returns:
    --------
    pd.DataFrame
        Query results as a pandas DataFrame
    """
    return conn.execute(query).fetchdf()


def get_table_schema(conn: duckdb.DuckDBPyConnection, schema: str, table: str) -> pd.DataFrame:
    """
    Get schema information for a table.
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        DuckDB connection object
    schema : str
        Schema name
    table : str
        Table name
        
    Returns:
    --------
    pd.DataFrame
        Schema information
    """
    query = f"""SELECT column_name, data_type, is_nullable 
                FROM information_schema.columns 
                WHERE table_schema = '{schema}' AND table_name = '{table}'
                ORDER BY ordinal_position"""
    return execute_query(conn, query)


def get_table_data(conn: duckdb.DuckDBPyConnection, schema: str, table: str, 
                   limit: Optional[int] = None) -> pd.DataFrame:
    """
    Get data from a table with optional limit.
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        DuckDB connection object
    schema : str
        Schema name
    table : str
        Table name
    limit : int, optional
        Limit the number of rows returned
        
    Returns:
    --------
    pd.DataFrame
        Table data as a pandas DataFrame
    """
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"SELECT * FROM {schema}.{table} {limit_clause}"
    return execute_query(conn, query)


def get_row_count(conn: duckdb.DuckDBPyConnection, schema: str, table: str) -> int:
    """
    Get the number of rows in a table.
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        DuckDB connection object
    schema : str
        Schema name
    table : str
        Table name
        
    Returns:
    --------
    int
        Number of rows in the table
    """
    query = f"SELECT COUNT(*) as row_count FROM {schema}.{table}"
    result = execute_query(conn, query)
    return result['row_count'][0]


def get_column_samples(df: pd.DataFrame, n_samples: int = 5, 
                       dtype_filter: Optional[Union[str, List[str]]] = None) -> Dict[str, List[Any]]:
    """
    Returns sample entries for each column in a pandas DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to extract samples from
    n_samples : int, default=5
        Number of sample values to return for each column
    dtype_filter : str or list, default=None
        Filter columns by data type(s). Can be a string or list of strings.
        Common values: 'object', 'int64', 'float64', 'bool', 'datetime64'
        If None, samples from all columns are returned.
    
    Returns:
    --------
    dict
        A dictionary where keys are column names and values are lists of sample values
    """
    samples = {}
    
    # Convert dtype_filter to list if it's a string
    if isinstance(dtype_filter, str):
        dtype_filter = [dtype_filter]
    
    # Get column data types
    dtypes = df.dtypes
    
    # Filter columns by dtype if specified
    if dtype_filter is not None:
        filtered_columns = [col for col in df.columns if dtypes[col].name in dtype_filter]
    else:
        filtered_columns = df.columns
    
    # Get samples for each column
    for col in filtered_columns:
        # Handle empty dataframes
        if len(df) == 0:
            samples[col] = []
            continue
            
        # Get unique values if there are fewer than n_samples unique values
        unique_values = df[col].dropna().unique()
        if len(unique_values) <= n_samples:
            samples[col] = unique_values.tolist()
        else:
            # Try to get a representative sample
            samples[col] = df[col].sample(n_samples, random_state=42).tolist()
    
    return samples
