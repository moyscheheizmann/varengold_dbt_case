"""
Data cleaning module for handling formatting issues in dataframes.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Union


class DataCleaner:
    """
    Class for cleaning data in a DataFrame.
    
    This class provides methods to clean various data issues such as
    German number format, mixed date formats, etc.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the data cleaner with a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame to clean
        """
        self.df = df
    
    def clean_dataframe(self, numeric_columns: Optional[List[str]] = None, 
                       date_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Cleans a DataFrame by converting German number notation in numeric columns
        and standardizing date formats in date columns.
        
        Parameters:
        -----------
        numeric_columns : list, optional
            Column names that should be treated as numeric
        date_columns : list, optional
            Column names that should be treated as dates
        
        Returns:
        --------
        pandas.DataFrame
            A cleaned copy of the original DataFrame
        """
        # Create a copy to avoid modifying the original
        cleaned_df = self.df.copy()
        
        # Auto-detect numeric columns if not specified
        if numeric_columns is None:
            # Look for columns that might contain numbers with German notation
            numeric_columns = []
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    # Check a sample of values
                    sample = self.df[col].dropna().sample(min(10, len(self.df[col].dropna())))
                    # If most values look like they could be German numbers, include the column
                    german_number_pattern = r'^[0-9]+(\.[0-9]{3})*(,[0-9]+)?$'
                    if sum(sample.astype(str).str.match(german_number_pattern)) > len(sample) * 0.5:
                        numeric_columns.append(col)
        
        # Convert numeric columns
        for col in numeric_columns:
            if col in self.df.columns:
                cleaned_df[col] = self.df[col].apply(self._convert_german_number)
        
        # Auto-detect date columns if not specified
        if date_columns is None:
            # Look for columns that might contain dates
            date_columns = []
            date_patterns = [
                r'\d{1,2}/\d{1,2}/\d{4}',  # DD/MM/YYYY or MM/DD/YYYY
                r'\d{1,2}\.\d{1,2}\.\d{4}',  # DD.MM.YYYY
                r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
            ]
            
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    # Check a sample of values
                    sample = self.df[col].dropna().sample(min(10, len(self.df[col].dropna())))
                    # If most values match a date pattern, include the column
                    has_date_pattern = sample.astype(str).apply(
                        lambda x: any(re.search(pattern, x) for pattern in date_patterns)
                    )
                    if has_date_pattern.mean() > 0.5:
                        date_columns.append(col)
        
        # Convert date columns
        for col in date_columns:
            if col in self.df.columns:
                cleaned_df[col] = self._parse_mixed_dates(self.df[col])
        
        return cleaned_df
    
    def _convert_german_number(self, value: Any) -> Union[float, Any]:
        """
        Converts a German formatted number (with comma as decimal separator and dot as thousands separator)
        to a float.
        
        Examples:
        - '1.234,56' -> 1234.56
        - '0,56' -> 0.56
        - '1.000' -> 1000.0
        
        Parameters:
        -----------
        value : Any
            The value to convert
            
        Returns:
        --------
        float or Any
            Converted float if possible, original value otherwise
        """
        if pd.isna(value) or not isinstance(value, str):
            return value
            
        # Remove any spaces
        value = value.strip()
        
        # Check if it's already a valid number
        try:
            return float(value)
        except ValueError:
            pass
        
        # If it contains both a dot and a comma, assume German notation
        if '.' in value and ',' in value:
            # Replace dots first (thousands), then commas (decimal)
            return float(value.replace('.', '').replace(',', '.'))
        # If it only contains a comma, treat it as a decimal point
        elif ',' in value:
            return float(value.replace(',', '.'))
        # If it only has dots (and passes the above), it's likely a thousands separator
        elif '.' in value:
            # Check if it's a reasonable pattern for thousands separators
            if re.match(r'^\d{1,3}(\.\d{3})+$', value):
                return float(value.replace('.', ''))
        
        # Return original value if conversion fails
        try:
            return float(value)
        except ValueError:
            return np.nan
    
    def _detect_date_format(self, date_str: Any) -> Optional[str]:
        """
        Detects the format of a date string.
        Returns a format string for datetime.strptime
        
        Parameters:
        -----------
        date_str : Any
            The date string to analyze
            
        Returns:
        --------
        str or None
            Format string for datetime.strptime if detected, None otherwise
        """
        if pd.isna(date_str) or not isinstance(date_str, str):
            return None
        
        date_str = date_str.strip()
        
        # Common formats to try
        formats = [
            # DD/MM/YYYY
            {'pattern': r'^\d{1,2}/\d{1,2}/\d{4}$', 'format': '%d/%m/%Y'},
            # MM/DD/YYYY
            {'pattern': r'^\d{1,2}/\d{1,2}/\d{4}$', 'format': '%m/%d/%Y'},
            # DD.MM.YYYY
            {'pattern': r'^\d{1,2}\.\d{1,2}\.\d{4}$', 'format': '%d.%m.%Y'},
            # YYYY-MM-DD
            {'pattern': r'^\d{4}-\d{1,2}-\d{1,2}$', 'format': '%Y-%m-%d'},
            # DD-MM-YYYY
            {'pattern': r'^\d{1,2}-\d{1,2}-\d{4}$', 'format': '%d-%m-%Y'},
            # YYYY/MM/DD
            {'pattern': r'^\d{4}/\d{1,2}/\d{1,2}$', 'format': '%Y/%m/%d'},
        ]
        
        for format_info in formats:
            if re.match(format_info['pattern'], date_str):
                try:
                    # Try to parse with this format
                    datetime.strptime(date_str, format_info['format'])
                    return format_info['format']
                except ValueError:
                    continue
        
        return None
    
    def _parse_mixed_dates(self, date_series: pd.Series) -> pd.Series:
        """
        Parses a pandas Series containing dates in different formats.
        Returns a Series of datetime objects.
        
        Parameters:
        -----------
        date_series : pd.Series
            Series containing date strings
            
        Returns:
        --------
        pd.Series
            Series of datetime objects
        """
        result = pd.Series(index=date_series.index, dtype='datetime64[ns]')
        
        for idx, date_str in date_series.items():
            if pd.isna(date_str) or not isinstance(date_str, str):
                result[idx] = pd.NaT
                continue
                
            date_str = date_str.strip()
            date_format = self._detect_date_format(date_str)
            
            if date_format:
                try:
                    result[idx] = datetime.strptime(date_str, date_format)
                except ValueError:
                    result[idx] = pd.NaT
            else:
                # Try pandas' flexible parser as fallback
                try:
                    result[idx] = pd.to_datetime(date_str)
                except:
                    result[idx] = pd.NaT
        
        return result
