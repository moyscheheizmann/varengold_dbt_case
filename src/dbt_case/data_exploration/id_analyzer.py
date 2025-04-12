"""
ID Analysis module for examining and validating ID columns in dataframes.
"""

import pandas as pd
from typing import List, Dict, Optional, Any, Tuple, Union


class IDAnalyzer:
    """
    Class for analyzing ID columns in a DataFrame.
    
    This class provides methods to check for duplicates in ID columns,
    analyze ID column properties, and display the results.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the ID analyzer with a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame to analyze
        """
        self.df = df
    
    def check_id_duplicates(self, id_columns: Optional[List[str]] = None, 
                           return_duplicates: bool = True) -> Dict[str, Any]:
        """
        Checks for duplicates in specified ID columns or detects ID columns automatically.
        
        Parameters:
        -----------
        id_columns : list, optional
            List of column names to check for duplicates. If None, will try to auto-detect ID columns.
        return_duplicates : bool, default=True
            Whether to return the duplicate records
        
        Returns:
        --------
        dict
            A dictionary containing duplicate analysis results
        """
        results = {}
        
        # Auto-detect ID columns if not specified
        if id_columns is None:
            # Look for columns with 'id', 'key', 'code', or 'num' in their name (case insensitive)
            possible_id_cols = [col for col in self.df.columns if any(id_term in col.lower() 
                                                                      for id_term in ['id', 'key', 'code', 'num'])]
            
            # Also consider columns that are unique or nearly unique (>95% unique values)
            for col in self.df.columns:
                if col not in possible_id_cols:
                    unique_ratio = self.df[col].nunique() / len(self.df)
                    if unique_ratio > 0.95 and self.df[col].nunique() > 1:  # Avoid constant columns
                        possible_id_cols.append(col)
            
            id_columns = possible_id_cols
            results['auto_detected_id_columns'] = id_columns
        
        if not id_columns:
            results['status'] = "No ID columns found or specified"
            return results
        
        # Check each ID column
        for col in id_columns:
            if col not in self.df.columns:
                results[col] = {"error": f"Column '{col}' not found in DataFrame"}
                continue
            
            # Check for missing values
            missing_count = self.df[col].isna().sum()
            missing_percentage = (missing_count / len(self.df)) * 100
            
            # Check for duplicates
            duplicate_values = self.df[col].value_counts()
            duplicate_values = duplicate_values[duplicate_values > 1]
            duplicate_count = len(duplicate_values)
            duplicate_records = self.df[self.df[col].isin(duplicate_values.index)].sort_values(by=col)
            
            col_results = {
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'has_duplicates': duplicate_count > 0,
                'duplicate_values_count': duplicate_count,
                'records_with_duplicates': len(duplicate_records),
                'duplicate_values': duplicate_values.to_dict() 
            }
            
            # For small duplicate sets, include the actual records
            if return_duplicates and duplicate_count > 0:
                if len(duplicate_records) <= 100:  # Only return if not too many
                    col_results['duplicate_records'] = duplicate_records
                else:
                    col_results['duplicate_records_notice'] = f"Too many duplicate records ({len(duplicate_records)}) to display"
            
            results[col] = col_results
        
        # Check for composite keys
        if len(id_columns) > 1:
            # Check if multiple columns together form a unique key
            composite_key_dupes = self.df.duplicated(subset=id_columns, keep=False)
            composite_dupes_count = composite_key_dupes.sum()
            
            results['composite_key'] = {
                'columns': id_columns,
                'has_duplicates': composite_dupes_count > 0,
                'duplicate_records_count': composite_dupes_count
            }
            
            if composite_dupes_count > 0 and return_duplicates:
                composite_dupe_records = self.df[composite_key_dupes].sort_values(by=id_columns)
                if len(composite_dupe_records) <= 100:
                    results['composite_key']['duplicate_records'] = composite_dupe_records
                else:
                    results['composite_key']['duplicate_records_notice'] = f"Too many duplicate records ({composite_dupes_count}) to display"
        
        return results
    
    def display_duplicate_analysis(self, id_columns: Optional[List[str]] = None, 
                                  analysis_results: Optional[Dict[str, Any]] = None) -> None:
        """
        Displays the results of duplicate analysis in a readable format.
        
        Parameters:
        -----------
        id_columns : list, optional
            List of column names to check for duplicates
        analysis_results : dict, optional
            Results from check_id_duplicates function. If None, it will run the analysis.
        """
        if analysis_results is None:
            analysis_results = self.check_id_duplicates(id_columns)
        
        print(f"DataFrame Shape: {self.df.shape}")
        print("-" * 80)
        
        if 'status' in analysis_results and analysis_results['status'] == "No ID columns found or specified":
            print("No ID columns found or specified")
            return
        
        if 'auto_detected_id_columns' in analysis_results:
            print(f"Auto-detected ID columns: {', '.join(analysis_results['auto_detected_id_columns'])}")
            print("-" * 80)
        
        # Skip metadata keys when iterating through column results
        skip_keys = ['status', 'auto_detected_id_columns', 'composite_key']
        
        for col, results in analysis_results.items():
            if col in skip_keys:
                continue
                
            if 'error' in results:
                print(f"\n===== COLUMN: {col} =====")
                print(results['error'])
                continue
                
            print(f"\n===== COLUMN: {col} =====")
            print(f"Missing values: {results['missing_count']} ({results['missing_percentage']:.2f}%)")
            
            if results['has_duplicates']:
                print(f"DUPLICATES FOUND: {results['duplicate_values_count']} distinct values are duplicated")
                print(f"Total records with duplicate values: {results['records_with_duplicates']}")
                
                # Print the duplicate values and their counts
                if isinstance(results['duplicate_values'], dict):
                    print("\nDuplicate values (value: count):")
                    for val, count in results['duplicate_values'].items():
                        print(f"  - {val}: {count} occurrences")
                else:
                    print(f"\n{results['duplicate_values']}")
                
                # Print the actual records if available
                if 'duplicate_records' in results:
                    print("\nDuplicate records:")
                    print(results['duplicate_records'])
                elif 'duplicate_records_notice' in results:
                    print(f"\n{results['duplicate_records_notice']}")
            else:
                print("No duplicates found. This column contains unique values.")
            
            print("-" * 80)
        
        # Check composite key results
        if 'composite_key' in analysis_results:
            composite_results = analysis_results['composite_key']
            print("\n===== COMPOSITE KEY ANALYSIS =====")
            print(f"Columns analyzed as composite key: {', '.join(composite_results['columns'])}")
            
            if composite_results['has_duplicates']:
                print(f"DUPLICATES FOUND: {composite_results['duplicate_records_count']} records have duplicate composite key values")
                
                if 'duplicate_records' in composite_results:
                    print("\nRecords with duplicate composite keys:")
                    print(composite_results['duplicate_records'])
                elif 'duplicate_records_notice' in composite_results:
                    print(f"\n{composite_results['duplicate_records_notice']}")
            else:
                print("No duplicates found. These columns together form a unique composite key.")
            
            print("-" * 80)
    
    def check_id_properties(self, id_column: str) -> Dict[str, bool]:
        """
        Checks properties of a specified ID column in a DataFrame.
        
        Parameters:
        -----------
        id_column : str
            The name of the ID column to check
            
        Returns:
        --------
        dict
            A dictionary containing properties of the ID column
        """
        if id_column not in self.df.columns:
            raise ValueError(f"Column '{id_column}' not found in DataFrame")
        
        results = {}
        
        # Check uniqueness
        results['is_unique'] = self.df[id_column].is_unique
        
        # Check monotonicity
        results['is_monotonic_inc'] = self.df[id_column].is_monotonic_increasing
        results['is_monotonic_dec'] = self.df[id_column].is_monotonic_decreasing
        results['is_monotonic'] = results['is_monotonic_inc'] or results['is_monotonic_dec']
        
        return results
    
    def count_digits_in_column(self, column_name: str) -> pd.Series:
        """
        Counts the number of digits in each entry of a specified column.
        
        Parameters:
        -----------
        column_name : str
            The name of the column to analyze
            
        Returns:
        --------
        pandas.Series
            A Series with the count of digits for each entry in the specified column
        """
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")
        
        return self.df[column_name].astype(str).str.count(r'\d')
