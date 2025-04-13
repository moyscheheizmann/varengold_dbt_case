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
    German number format, mixed date formats, whitespace issues, etc.
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
                       date_columns: Optional[List[str]] = None,
                       whitespace_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Cleans a DataFrame by converting German number notation in numeric columns,
        standardizing date formats in date columns, and removing leading/trailing
        whitespace in specified columns.
        
        Parameters:
        -----------
        numeric_columns : list, optional
            Column names that should be treated as numeric
        date_columns : list, optional
            Column names that should be treated as dates
        whitespace_columns : list, optional
            Column names that should have whitespace cleaned. If None but 
            clean_whitespace=True, all string columns will be cleaned
        
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
        
        # Auto-detect whitespace columns if not specified
        if whitespace_columns is None:
            # Check all object/string columns for whitespace issues
            whitespace_columns = []
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    # Sample some values to check for leading/trailing whitespace
                    sample = self.df[col].dropna().sample(min(50, len(self.df[col].dropna())))
                    # Check if any values have leading or trailing whitespace
                    whitespace_found = False
                    for val in sample:
                        if isinstance(val, str) and (val != val.strip()):
                            whitespace_found = True
                            break
                    
                    if whitespace_found:
                        whitespace_columns.append(col)
        
        # Clean whitespace in columns
        for col in whitespace_columns:
            if col in self.df.columns:
                cleaned_df[col] = self._clean_whitespace(self.df[col])
        
        return cleaned_df
    
    def clean_whitespace(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Removes leading and trailing whitespace from string columns.
        
        Parameters:
        -----------
        columns : list, optional
            List of column names to clean. If None, cleans all string columns.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with cleaned whitespace
        """
        # Create a copy to avoid modifying the original
        cleaned_df = self.df.copy()
        
        # If no columns specified, find all string columns
        if columns is None:
            columns = [col for col in cleaned_df.columns if cleaned_df[col].dtype == 'object']
        
        # Clean whitespace in specified columns
        for col in columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = self._clean_whitespace(cleaned_df[col])
        
        return cleaned_df
    
    def _clean_whitespace(self, series: pd.Series) -> pd.Series:
        """
        Cleans whitespace in a pandas Series.
        
        Parameters:
        -----------
        series : pd.Series
            Series to clean
            
        Returns:
        --------
        pd.Series
            Cleaned series
        """
        return series.apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )
    
    def detect_whitespace_issues(self) -> Dict[str, Dict[str, Any]]:
        """
        Detects whitespace issues in string columns.
        
        Returns:
        --------
        dict
            Dictionary with detailed information about whitespace issues
        """
        results = {}
        
        # Loop through all columns
        for col in self.df.columns:
            # Only check string/object columns
            if self.df[col].dtype != 'object':
                continue
            
            # Initialize result dictionary for this column
            results[col] = {
                'total_values': len(self.df),
                'non_null_values': self.df[col].notnull().sum(),
                'leading_whitespace_count': 0,
                'trailing_whitespace_count': 0,
                'both_sides_whitespace_count': 0,
                'examples': []
            }
            
            # Check each non-null value
            for i, value in enumerate(self.df[col]):
                if pd.isna(value) or not isinstance(value, str):
                    continue
                
                has_leading = value.startswith(' ') or value.startswith('\t') or value.startswith('\n')
                has_trailing = value.endswith(' ') or value.endswith('\t') or value.endswith('\n')
                
                # Update counts
                if has_leading and has_trailing:
                    results[col]['both_sides_whitespace_count'] += 1
                elif has_leading:
                    results[col]['leading_whitespace_count'] += 1
                elif has_trailing:
                    results[col]['trailing_whitespace_count'] += 1
                
                # If it has whitespace issues, add to examples (up to 5)
                if (has_leading or has_trailing) and len(results[col]['examples']) < 5:
                    results[col]['examples'].append({
                        'index': i,
                        'original': repr(value),
                        'stripped': repr(value.strip()),
                        'issue': 'both' if (has_leading and has_trailing) else ('leading' if has_leading else 'trailing')
                    })
            
            # Calculate totals
            results[col]['total_whitespace_issues'] = (
                results[col]['leading_whitespace_count'] + 
                results[col]['trailing_whitespace_count'] + 
                results[col]['both_sides_whitespace_count']
            )
            
            # Calculate percentage
            non_null_count = results[col]['non_null_values']
            if non_null_count > 0:
                results[col]['whitespace_percentage'] = (results[col]['total_whitespace_issues'] / non_null_count) * 100
            else:
                results[col]['whitespace_percentage'] = 0
            
            # Remove columns with no issues
            if results[col]['total_whitespace_issues'] == 0:
                del results[col]
        
        return results
    
    def print_whitespace_analysis(self, results: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Prints the whitespace analysis results in a readable format.
        
        Parameters:
        -----------
        results : dict, optional
            Results from detect_whitespace_issues function. If None, detect issues first.
        """
        if results is None:
            results = self.detect_whitespace_issues()
            
        if not results:
            print("No whitespace issues found in any string columns.")
            return
        
        print("Whitespace Issues Analysis")
        print("=========================")
        
        for col, data in results.items():
            print(f"\nColumn: {col}")
            print(f"  Total values: {data['total_values']}")
            print(f"  Non-null values: {data['non_null_values']}")
            print(f"  Total whitespace issues: {data['total_whitespace_issues']} ({data['whitespace_percentage']:.2f}%)")
            print(f"    Leading whitespace only: {data['leading_whitespace_count']}")
            print(f"    Trailing whitespace only: {data['trailing_whitespace_count']}")
            print(f"    Both leading and trailing: {data['both_sides_whitespace_count']}")
            
            if data['examples']:
                print("\n  Examples:")
                for ex in data['examples']:
                    print(f"    Index {ex['index']}: {ex['original']} → {ex['stripped']} ({ex['issue']} whitespace)")
    
    def show_cleaning_summary(self, original_df: Optional[pd.DataFrame] = None, cleaned_df: Optional[pd.DataFrame] = None,
                             numeric_columns: Optional[List[str]] = None, date_columns: Optional[List[str]] = None,
                             whitespace_columns: Optional[List[str]] = None, n_samples: int = 3) -> None:
        """
        Prints a summary of cleaning changes.
        
        Parameters:
        -----------
        original_df : pd.DataFrame, optional
            Original DataFrame. If None, uses self.df
        cleaned_df : pd.DataFrame, optional
            Cleaned DataFrame. If None, generates a cleaned version
        numeric_columns : list, optional
            Column names treated as numeric
        date_columns : list, optional
            Column names treated as dates
        whitespace_columns : list, optional
            Column names with whitespace cleaning
        n_samples : int, default=3
            Number of sample changes to show per column
        """
        original_df = self.df if original_df is None else original_df
        
        if cleaned_df is None:
            cleaned_df = self.clean_dataframe(
                numeric_columns=numeric_columns,
                date_columns=date_columns,
                whitespace_columns=whitespace_columns
            )
        
        # Auto-detect columns if not specified
        if numeric_columns is None:
            numeric_columns = []
            for col in original_df.columns:
                if original_df[col].dtype == 'object' and cleaned_df[col].dtype in ['float64', 'int64']:
                    numeric_columns.append(col)
        
        if date_columns is None:
            date_columns = []
            for col in original_df.columns:
                if original_df[col].dtype == 'object' and pd.api.types.is_datetime64_dtype(cleaned_df[col]):
                    date_columns.append(col)
        
        if whitespace_columns is None:
            whitespace_columns = []
            for col in original_df.columns:
                if original_df[col].dtype == 'object' and cleaned_df[col].dtype == 'object':
                    # Sample some values to check for whitespace differences
                    sample_indices = original_df[col].dropna().sample(min(50, len(original_df[col].dropna()))).index
                    for idx in sample_indices:
                        orig_val = original_df.loc[idx, col]
                        clean_val = cleaned_df.loc[idx, col]
                        if isinstance(orig_val, str) and isinstance(clean_val, str) and orig_val != clean_val:
                            whitespace_columns.append(col)
                            break
        
        print("Data Cleaning Summary")
        print("====================")
        print(f"Original DataFrame: {original_df.shape[0]} rows, {original_df.shape[1]} columns")
        print(f"Cleaned DataFrame: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")
        
        if numeric_columns:
            print("\nNumeric Columns Cleaned:")
            for col in numeric_columns:
                print(f"\n  {col}:")
                # Find some changed values
                changes = []
                for i, (orig, clean) in enumerate(zip(original_df[col], cleaned_df[col])):
                    if pd.notna(orig) and pd.notna(clean) and str(orig) != str(clean):
                        changes.append((i, orig, clean))
                        if len(changes) >= n_samples:
                            break
                
                if changes:
                    for idx, orig, clean in changes:
                        print(f"    Index {idx}: {orig} → {clean}")
                else:
                    print("    No changes found in samples")
        
        if date_columns:
            print("\nDate Columns Cleaned:")
            for col in date_columns:
                print(f"\n  {col}:")
                # Find some changed values
                changes = []
                for i, (orig, clean) in enumerate(zip(original_df[col], cleaned_df[col])):
                    if pd.notna(orig) and pd.notna(clean) and str(orig) != str(clean):
                        changes.append((i, orig, clean))
                        if len(changes) >= n_samples:
                            break
                
                if changes:
                    for idx, orig, clean in changes:
                        print(f"    Index {idx}: {orig} → {clean}")
                else:
                    print("    No changes found in samples")
        
        if whitespace_columns:
            print("\nWhitespace Cleaned:")
            for col in whitespace_columns:
                print(f"\n  {col}:")
                # Find some changed values
                changes = []
                for i, (orig, clean) in enumerate(zip(original_df[col], cleaned_df[col])):
                    if isinstance(orig, str) and isinstance(clean, str) and orig != clean:
                        changes.append((i, orig, clean))
                        if len(changes) >= n_samples:
                            break
                
                if changes:
                    for idx, orig, clean in changes:
                        print(f"    Index {idx}: {repr(orig)} → {repr(clean)}")
                else:
                    print("    No changes found in samples")
        
        if not numeric_columns and not date_columns and not whitespace_columns:
            print("\nNo cleaning operations were performed or no changes were detected.")
    
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
