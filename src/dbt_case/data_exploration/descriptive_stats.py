"""
Descriptive statistics module for analyzing dataframe columns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple


class DescriptiveStats:
    """
    Class for computing descriptive statistics on a DataFrame.
    
    This class provides methods to analyze columns in a DataFrame
    and generate summary statistics.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the descriptive statistics analyzer with a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame to analyze
        """
        self.df = df
    
    def analyze_dataframe(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyzes basic statistics for each column in the DataFrame.
        
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
                'missing_percentage': (col_data.isna().sum() / len(col_data)) * 100,
                'unique_values': col_data.nunique(),
                'unique_percentage': (col_data.nunique() / len(col_data)) * 100
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
                    'skew': col_data.skew(),
                    'quartiles': {
                        'q1': col_data.quantile(0.25),
                        'q2': col_data.quantile(0.5),
                        'q3': col_data.quantile(0.75)
                    }
                })
            
            # For categorical/object columns
            elif pd.api.types.is_object_dtype(col_type) or pd.api.types.is_bool_dtype(col_type) or pd.api.types.is_categorical_dtype(col_type):
                value_counts = col_data.value_counts()
                
                col_results.update({
                    'most_common': value_counts.index[0] if not value_counts.empty else None,
                    'most_common_count': value_counts.iloc[0] if not value_counts.empty else 0,
                    'most_common_percentage': (value_counts.iloc[0] / col_results['count']) * 100 if not value_counts.empty else 0,
                    'top_values': value_counts.head(5).to_dict() if not value_counts.empty else {}
                })
            
            # For datetime columns
            elif pd.api.types.is_datetime64_dtype(col_type):
                col_results.update({
                    'min_date': col_data.min(),
                    'max_date': col_data.max(),
                    'range_days': (col_data.max() - col_data.min()).days if not pd.isna(col_data.min()) and not pd.isna(col_data.max()) else None
                })
            
            results[column] = col_results
        
        return results
    
    def print_descriptive_stats(self, stats: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Prints descriptive statistics for each column in a readable format.
        
        Parameters:
        -----------
        stats : dict, optional
            Results from analyze_dataframe method. If None, it will run the analysis.
        """
        if stats is None:
            stats = self.analyze_dataframe()
        
        print(f"DataFrame Shape: {self.df.shape}")
        print(f"Total Missing Values: {self.df.isna().sum().sum()}")
        print("-" * 80)
        
        for column, results in stats.items():
            print(f"\n===== COLUMN: {column} ({results['data_type']}) =====")
            print(f"Count: {results['count']}, Missing: {results['missing']} ({results['missing_percentage']:.2f}%)")
            print(f"Unique Values: {results['unique_values']} ({results['unique_percentage']:.2f}%)")
            
            # If all values are missing, skip to next column
            if 'summary' in results and results['summary'] == "All values are missing":
                print("All values are missing")
                continue
            
            # For numeric columns
            if pd.api.types.is_numeric_dtype(self.df[column].dtype) and not pd.api.types.is_bool_dtype(self.df[column].dtype):
                print(f"Min: {results['min']}, Max: {results['max']}")
                print(f"Mean: {results['mean']:.2f}, Median: {results['median']:.2f}, Std: {results['std']:.2f}")
                print(f"Skewness: {results['skew']:.2f}")
                print(f"Quartiles: Q1={results['quartiles']['q1']:.2f}, Q2={results['quartiles']['q2']:.2f}, Q3={results['quartiles']['q3']:.2f}")
            
            # For categorical columns
            elif pd.api.types.is_object_dtype(self.df[column].dtype) or pd.api.types.is_bool_dtype(self.df[column].dtype) or pd.api.types.is_categorical_dtype(self.df[column].dtype):
                print(f"Most Common: {results['most_common']} ({results['most_common_count']} occurrences, {results['most_common_percentage']:.2f}%)")
                
                if results['top_values']:
                    print("\nTop 5 values:")
                    for val, count in results['top_values'].items():
                        print(f"  - {val}: {count} occurrences")
            
            # For datetime columns
            elif pd.api.types.is_datetime64_dtype(self.df[column].dtype):
                print(f"Min Date: {results['min_date']}")
                print(f"Max Date: {results['max_date']}")
                print(f"Range: {results['range_days']} days")
            
            print("-" * 80)
    
    def analyze_datetime_column(self, date_column: str, plot: bool = True, 
                               figsize: Tuple[int, int] = (14, 10)) -> pd.Series:
        """
        Analyze the distribution of dates in a DataFrame column that is already in datetime format.
        
        Parameters:
        -----------
        date_column : str
            Name of the column containing dates
        plot : bool, default=True
            Whether to create visualizations
        figsize : tuple, default=(14, 10)
            Figure size for plots
            
        Returns:
        --------
        pd.Series
            Series of valid dates
        """
        if date_column not in self.df.columns:
            raise ValueError(f"Column '{date_column}' not found in DataFrame")
            
        print(f"Analyzing datetime column: {date_column}")
        
        # Get the date series
        dates = self.df[date_column]
        
        # Check if it's already a datetime series
        if not pd.api.types.is_datetime64_any_dtype(dates):
            print(f"Warning: Column '{date_column}' is not in datetime format. Current type: {dates.dtype}")
            print("Converting to datetime...")
            try:
                dates = pd.to_datetime(dates)
            except Exception as e:
                print(f"Error converting to datetime: {str(e)}")
                return pd.Series(dtype='datetime64[ns]')
        
        # Check for null values
        null_count = dates.isna().sum()
        if null_count > 0:
            print(f"Note: {null_count} values ({null_count/len(dates):.2%}) are null/NaT.")
        
        # Remove null values for analysis
        valid_dates = dates.dropna()
        
        if len(valid_dates) == 0:
            print("No valid dates found for analysis.")
            return pd.Series(dtype='datetime64[ns]')
        
        # Check if all dates are in the past
        today = pd.Timestamp.now().normalize()
        future_dates = valid_dates[valid_dates > today]
        
        if len(future_dates) > 0:
            print(f"\nWarning: {len(future_dates)} dates ({len(future_dates)/len(valid_dates):.2%}) are in the future!")
            print("Examples of future dates:")
            for i, (idx, date) in enumerate(future_dates.iloc[:5].items(), 1):
                print(f"  {i}. Row {idx}: {date.strftime('%Y-%m-%d')}")
        else:
            print("\nAll dates are in the past.")
        
        # Calculate summary statistics
        date_min = valid_dates.min()
        date_max = valid_dates.max()
        date_range = (date_max - date_min).days
        
        # Convert to numeric for statistical calculations (days since epoch)
        numeric_dates = valid_dates.astype(np.int64) // 10**9 // 86400
        
        q1 = np.percentile(numeric_dates, 25)
        q3 = np.percentile(numeric_dates, 75)
        iqr = q3 - q1
        
        # Convert back to dates for display
        q1_date = pd.Timestamp(q1 * 86400 * 10**9)
        q3_date = pd.Timestamp(q3 * 86400 * 10**9)
        
        # Calculate mean date
        mean_days = numeric_dates.mean()
        mean_date = pd.Timestamp(mean_days * 86400 * 10**9)
        
        # Calculate median date
        median_days = np.median(numeric_dates)
        median_date = pd.Timestamp(median_days * 86400 * 10**9)
        
        # Calculate standard deviation in days
        std_days = numeric_dates.std()
        
        # Summary table
        print("\nDate Distribution Summary:")
        print("--------------------------")
        print(f"Count:                {len(valid_dates)}")
        print(f"Minimum date:         {date_min.strftime('%Y-%m-%d')}")
        print(f"Maximum date:         {date_max.strftime('%Y-%m-%d')}")
        print(f"Range:                {date_range} days")
        print(f"Mean date:            {mean_date.strftime('%Y-%m-%d')}")
        print(f"Median date:          {median_date.strftime('%Y-%m-%d')}")
        print(f"Standard deviation:   {std_days:.2f} days")
        print(f"1st quartile (Q1):    {q1_date.strftime('%Y-%m-%d')}")
        print(f"3rd quartile (Q3):    {q3_date.strftime('%Y-%m-%d')}")
        print(f"Interquartile range:  {iqr:.2f} days")
        
        # Calculate temporal distribution by year and month
        valid_dates_df = pd.DataFrame({'date': valid_dates})
        valid_dates_df['year'] = valid_dates_df['date'].dt.year
        valid_dates_df['month'] = valid_dates_df['date'].dt.month
        
        # Year distribution
        year_counts = valid_dates_df['year'].value_counts().sort_index()
        print("\nDistribution by Year:")
        for year, count in year_counts.items():
            print(f"  {year}: {count} ({count/len(valid_dates):.2%})")
        
        # Month distribution (across all years)
        month_counts = valid_dates_df['month'].value_counts().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        print("\nDistribution by Month (all years):")
        for month, count in month_counts.items():
            print(f"  {month_names[month-1]}: {count} ({count/len(valid_dates):.2%})")
        
        # Create visualizations
        if plot:
            plt.figure(figsize=figsize)
            
            # Histogram of dates
            plt.subplot(2, 2, 1)
            plt.hist(valid_dates, bins=min(50, date_range//30 + 1), edgecolor='black')
            plt.title('Histogram of Dates')
            plt.xlabel('Date')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            
            # Boxplot of dates
            plt.subplot(2, 2, 2)
            plt.boxplot(numeric_dates)
            plt.title('Boxplot of Dates')
            plt.ylabel('Days since epoch')
            # Add specific date labels
            box_positions = [1]
            plt.xticks([])
            
            # Add label annotations
            plt.annotate(f"Min: {date_min.strftime('%Y-%m-%d')}", xy=(1, numeric_dates.min()), 
                         xytext=(1.1, numeric_dates.min()), fontsize=9)
            plt.annotate(f"Q1: {q1_date.strftime('%Y-%m-%d')}", xy=(1, q1), 
                         xytext=(1.1, q1), fontsize=9)
            plt.annotate(f"Median: {median_date.strftime('%Y-%m-%d')}", xy=(1, median_days), 
                         xytext=(1.1, median_days), fontsize=9)
            plt.annotate(f"Q3: {q3_date.strftime('%Y-%m-%d')}", xy=(1, q3), 
                         xytext=(1.1, q3), fontsize=9)
            plt.annotate(f"Max: {date_max.strftime('%Y-%m-%d')}", xy=(1, numeric_dates.max()), 
                         xytext=(1.1, numeric_dates.max()), fontsize=9)
            
            # Bar chart of years
            plt.subplot(2, 2, 3)
            plt.bar(year_counts.index.astype(str), year_counts.values)
            plt.title('Distribution by Year')
            plt.xlabel('Year')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            # Bar chart of months
            plt.subplot(2, 2, 4)
            plt.bar(month_names, [month_counts.get(i, 0) for i in range(1, 13)])
            plt.title('Distribution by Month')
            plt.xlabel('Month')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
        
        return valid_dates
