"""
Outlier detection module for analyzing dataframe columns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List, Union


class OutlierDetector:
    """
    Class for detecting outliers in a DataFrame.
    
    This class provides methods to detect outliers in each column
    of a DataFrame using various methods (IQR, Z-score, etc.).
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the outlier detector with a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame to analyze
        """
        self.df = df
    
    def analyze_dataframe(self, outlier_threshold: float = 1.5) -> Dict[str, Dict[str, Any]]:
        """
        Analyzes each column in a pandas DataFrame, detecting outliers and providing summary statistics.
        
        Parameters:
        -----------
        outlier_threshold : float, default=1.5
            The threshold for IQR method (typically 1.5 or 3)
        
        Returns:
        --------
        dict
            A dictionary containing analysis results for each column
        """
        results = {}
        
        # Analyze each column
        for column in self.df.columns:
            col_data = self.df[column]
            col_type = col_data.dtype
            
            # Initialize column results
            col_results = {
                'data_type': str(col_type),
                'count': len(col_data),
                'missing': col_data.isna().sum(),
                'missing_percentage': (col_data.isna().sum() / len(col_data)) * 100
            }
            
            # Skip if all values are null
            if col_results['missing'] == col_results['count']:
                col_results['summary'] = "All values are missing"
                results[column] = col_results
                continue
            
            # For numeric columns
            if pd.api.types.is_numeric_dtype(col_type) and not pd.api.types.is_bool_dtype(col_type):
                # Basic statistics
                col_results.update({
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'skew': col_data.skew()
                })
                
                # Find outliers using IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - outlier_threshold * IQR
                upper_bound = Q3 + outlier_threshold * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                col_results.update({
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outliers_count': len(outliers),
                    'outliers_percentage': (len(outliers) / col_results['count']) * 100,
                    'outliers_values': outliers.value_counts().to_dict() if len(outliers) <= 10 else f"{len(outliers)} outliers found"
                })
                
                # Calculate Z-scores
                z_scores = np.abs(stats.zscore(col_data.dropna()))
                z_outliers = col_data[z_scores > 3]
                
                col_results.update({
                    'z_score_outliers_count': len(z_outliers),
                    'z_score_outliers_percentage': (len(z_outliers) / col_results['count']) * 100
                })
                
            # For categorical/object columns
            elif pd.api.types.is_object_dtype(col_type) or pd.api.types.is_bool_dtype(col_type) or pd.api.types.is_categorical_dtype(col_type):
                value_counts = col_data.value_counts()
                unique_count = len(value_counts)
                
                col_results.update({
                    'unique_values': unique_count,
                    'unique_percentage': (unique_count / col_results['count']) * 100,
                    'most_common': value_counts.index[0] if not value_counts.empty else None,
                    'most_common_count': value_counts.iloc[0] if not value_counts.empty else 0,
                    'most_common_percentage': (value_counts.iloc[0] / col_results['count']) * 100 if not value_counts.empty else 0
                })
                
                # For categorical columns, consider outliers as categories with very few occurrences
                if unique_count > 1:
                    rare_threshold = max(0.01, 1 / unique_count)  # Either 1% or 1/unique_values, whichever is higher
                    rare_categories = value_counts[value_counts / len(col_data.dropna()) < rare_threshold]
                    
                    col_results.update({
                        'rare_categories_count': len(rare_categories),
                        'rare_categories': rare_categories.to_dict() if len(rare_categories) <= 10 else f"{len(rare_categories)} rare categories found"
                    })
            
            # For datetime columns
            elif pd.api.types.is_datetime64_dtype(col_type):
                col_results.update({
                    'min_date': col_data.min(),
                    'max_date': col_data.max(),
                    'range_days': (col_data.max() - col_data.min()).days if not pd.isna(col_data.min()) and not pd.isna(col_data.max()) else None
                })
                
                # Check for datetime outliers using modified Z-score on Unix timestamps
                if not col_data.dropna().empty:
                    # Get non-null values and their indices
                    non_null_data = col_data.dropna()
                    timestamps = non_null_data.astype(np.int64) // 10**9  # Convert to Unix timestamps
                    median = np.median(timestamps)
                    mad = np.median(np.abs(timestamps - median))  # Median Absolute Deviation
                    
                    # MAD standardization (modified Z-score)
                    if mad > 0:
                        modified_z_scores = 0.6745 * (timestamps - median) / mad
                        # Create a boolean mask and filter only the non-null data
                        outlier_mask = np.abs(modified_z_scores) > 3.5
                        time_outliers = non_null_data[outlier_mask]
                        
                        col_results.update({
                            'datetime_outliers_count': len(time_outliers),
                            'datetime_outliers_percentage': (len(time_outliers) / len(non_null_data)) * 100,
                            'datetime_outliers': time_outliers.tolist() if len(time_outliers) <= 5 else f"{len(time_outliers)} date outliers found"
                        })
            
            results[column] = col_results
        
        return results
    
    def display_outlier_analysis(self, analysis_results: Optional[Dict[str, Dict[str, Any]]] = None, 
                                plot: bool = True, figsize: Tuple[int, int] = (12, 5)) -> None:
        """
        Displays the outlier analysis and summary statistics for each column.
        
        Parameters:
        -----------
        analysis_results : dict, optional
            Results from analyze_dataframe function. If None, it will run the analysis.
        plot : bool, default=True
            Whether to generate visualizations for columns with outliers
        figsize : tuple, default=(12, 5)
            Figure size for plots
        """
        if analysis_results is None:
            analysis_results = self.analyze_dataframe()
        
        print(f"DataFrame Shape: {self.df.shape}")
        print(f"Total Missing Values: {self.df.isna().sum().sum()}")
        print("-" * 80)
        
        for column, results in analysis_results.items():
            print(f"\n===== COLUMN: {column} ({results['data_type']}) =====")
            print(f"Count: {results['count']}, Missing: {results['missing']} ({results['missing_percentage']:.2f}%)")
            
            # If all values are missing, skip to next column
            if 'summary' in results and results['summary'] == "All values are missing":
                print("All values are missing")
                continue
            
            # For numeric columns
            if pd.api.types.is_numeric_dtype(self.df[column].dtype) and not pd.api.types.is_bool_dtype(self.df[column].dtype):
                print(f"Min: {results['min']}, Max: {results['max']}")
                print(f"Mean: {results['mean']:.2f}, Median: {results['median']:.2f}, Std: {results['std']:.2f}")
                print(f"Skewness: {results['skew']:.2f}")
                print(f"Outliers (IQR method): {results['outliers_count']} ({results['outliers_percentage']:.2f}%)")
                print(f"Outliers (Z-score method): {results['z_score_outliers_count']} ({results['z_score_outliers_percentage']:.2f}%)")
                
                if results['outliers_count'] > 0:
                    if isinstance(results['outliers_values'], dict):
                        print("Outlier values:")
                        for val, count in results['outliers_values'].items():
                            print(f"  - {val}: {count} occurrences")
                    else:
                        print(results['outliers_values'])
                
                if plot and results['outliers_count'] > 0:
                    plt.figure(figsize=figsize)
                    
                    # Box plot
                    plt.subplot(1, 2, 1)
                    sns.boxplot(x=self.df[column].dropna())
                    plt.title(f'Box Plot: {column}')
                    plt.tight_layout()
                    
                    # Histogram with KDE
                    plt.subplot(1, 2, 2)
                    sns.histplot(self.df[column].dropna(), kde=True)
                    
                    # Add vertical lines for outlier boundaries
                    plt.axvline(x=results['lower_bound'], color='r', linestyle='--', alpha=0.7)
                    plt.axvline(x=results['upper_bound'], color='r', linestyle='--', alpha=0.7)
                    
                    plt.title(f'Distribution: {column}')
                    plt.tight_layout()
                    plt.show()
            
            # For categorical columns
            elif pd.api.types.is_object_dtype(self.df[column].dtype) or pd.api.types.is_bool_dtype(self.df[column].dtype) or pd.api.types.is_categorical_dtype(self.df[column].dtype):
                print(f"Unique Values: {results['unique_values']} ({results['unique_percentage']:.2f}% of total)")
                print(f"Most Common: {results['most_common']} ({results['most_common_count']} occurrences, {results['most_common_percentage']:.2f}%)")
                
                if 'rare_categories_count' in results and results['rare_categories_count'] > 0:
                    print(f"Rare Categories: {results['rare_categories_count']}")
                    if isinstance(results['rare_categories'], dict):
                        print("Rare category values:")
                        for val, count in results['rare_categories'].items():
                            print(f"  - {val}: {count} occurrences")
                    else:
                        print(results['rare_categories'])
                
                if plot and results['unique_values'] <= 30:  # Only plot if not too many categories
                    plt.figure(figsize=figsize)
                    value_counts = self.df[column].value_counts().sort_values(ascending=False)
                    
                    if len(value_counts) > 15:  # If too many categories, show only top ones
                        value_counts = value_counts.head(15)
                        plt.title(f'Top 15 Categories: {column}')
                    else:
                        plt.title(f'Category Distribution: {column}')
                    
                    sns.barplot(x=value_counts.index, y=value_counts.values)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.show()
            
            # For datetime columns
            elif pd.api.types.is_datetime64_dtype(self.df[column].dtype):
                print(f"Min Date: {results['min_date']}")
                print(f"Max Date: {results['max_date']}")
                print(f"Range: {results['range_days']} days")
                
                if 'datetime_outliers_count' in results and results['datetime_outliers_count'] > 0:
                    print(f"Date Outliers: {results['datetime_outliers_count']} ({results['datetime_outliers_percentage']:.2f}%)")
                    if isinstance(results['datetime_outliers'], list):
                        print("Outlier dates:")
                        for date in results['datetime_outliers']:
                            print(f"  - {date}")
                    else:
                        print(results['datetime_outliers'])
                
                if plot:
                    plt.figure(figsize=figsize)
                    plt.hist(self.df[column].dropna(), bins=20)
                    plt.title(f'Date Distribution: {column}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()
            
            print("-" * 80)
