"""
Plotting module for data visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any, Tuple, Union


class Plotter:
    """
    Class for creating data visualizations from a DataFrame.
    
    This class provides methods to create various plots for data exploration.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the plotter with a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame to visualize
        """
        self.df = df
        
        # Set up visualization style
        plt.style.use('ggplot')
        sns.set(style="whitegrid")
    
    def plot_numeric_distribution(self, column: str, bins: int = 30, 
                                figsize: Tuple[int, int] = (12, 5),
                                kde: bool = True) -> None:
        """
        Plot the distribution of a numeric column.
        
        Parameters:
        -----------
        column : str
            The column to plot
        bins : int, default=30
            Number of bins for histogram
        figsize : tuple, default=(12, 5)
            Figure size
        kde : bool, default=True
            Whether to plot kernel density estimate
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        plt.figure(figsize=figsize)
        sns.histplot(self.df[column].dropna(), bins=bins, kde=kde)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
    
    def plot_categorical_distribution(self, column: str, top_n: Optional[int] = None,
                                    figsize: Tuple[int, int] = (12, 6),
                                    horizontal: bool = True) -> None:
        """
        Plot the distribution of a categorical column.
        
        Parameters:
        -----------
        column : str
            The column to plot
        top_n : int, optional
            If provided, only plot the top N categories
        figsize : tuple, default=(12, 6)
            Figure size
        horizontal : bool, default=True
            If True, creates a horizontal bar chart, otherwise vertical
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        plt.figure(figsize=figsize)
        value_counts = self.df[column].value_counts()
        
        if top_n is not None and top_n < len(value_counts):
            value_counts = value_counts.head(top_n)
            title = f'Top {top_n} Categories in {column}'
        else:
            title = f'Distribution of {column}'
        
        if horizontal:
            value_counts.plot(kind='barh')
            plt.xlabel('Count')
            plt.ylabel(column)
        else:
            value_counts.plot(kind='bar')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
        
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_time_series(self, date_column: str, value_column: Optional[str] = None,
                       freq: str = 'M', figsize: Tuple[int, int] = (12, 5)) -> None:
        """
        Plot a time series from a date column and optionally a value column.
        
        Parameters:
        -----------
        date_column : str
            The column containing dates
        value_column : str, optional
            If provided, aggregate this column. If None, count occurrences.
        freq : str, default='M'
            Frequency to group by: 'D' for daily, 'W' for weekly, 'M' for monthly, 'Y' for yearly
        figsize : tuple, default=(12, 5)
            Figure size
        """
        if date_column not in self.df.columns:
            raise ValueError(f"Column '{date_column}' not found in DataFrame")
        
        if value_column is not None and value_column not in self.df.columns:
            raise ValueError(f"Column '{value_column}' not found in DataFrame")
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_column]):
            print(f"Converting {date_column} to datetime format...")
            date_series = pd.to_datetime(self.df[date_column], errors='coerce')
        else:
            date_series = self.df[date_column]
        
        # Create a copy with proper datetime
        temp_df = self.df.copy()
        temp_df['_date'] = date_series
        
        # Group by date
        freq_map = {
            'D': 'Day',
            'W': 'Week',
            'M': 'Month',
            'Q': 'Quarter',
            'Y': 'Year'
        }
        
        if value_column is None:
            # Count occurrences by date
            time_series = temp_df.groupby(temp_df['_date'].dt.to_period(freq)).size()
            ylabel = 'Count'
            title = f'Counts by {freq_map.get(freq, freq)}'
        else:
            # Aggregate value column by date
            time_series = temp_df.groupby(temp_df['_date'].dt.to_period(freq))[value_column].sum()
            ylabel = value_column
            title = f'{value_column} by {freq_map.get(freq, freq)}'
        
        plt.figure(figsize=figsize)
        time_series.plot()
        plt.title(title)
        plt.xlabel(f'{freq_map.get(freq, freq)}')
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_boxplot(self, numeric_column: str, group_by: Optional[str] = None,
                   figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Create a boxplot for a numeric column, optionally grouped by a categorical column.
        
        Parameters:
        -----------
        numeric_column : str
            The numeric column to plot
        group_by : str, optional
            If provided, group the boxplot by this categorical column
        figsize : tuple, default=(12, 6)
            Figure size
        """
        if numeric_column not in self.df.columns:
            raise ValueError(f"Column '{numeric_column}' not found in DataFrame")
        
        if group_by is not None and group_by not in self.df.columns:
            raise ValueError(f"Column '{group_by}' not found in DataFrame")
        
        plt.figure(figsize=figsize)
        
        if group_by is None:
            # Single boxplot
            sns.boxplot(y=self.df[numeric_column].dropna())
            plt.title(f'Boxplot of {numeric_column}')
            plt.ylabel(numeric_column)
        else:
            # Grouped boxplot
            sns.boxplot(x=group_by, y=numeric_column, data=self.df)
            plt.title(f'Boxplot of {numeric_column} by {group_by}')
            plt.xlabel(group_by)
            plt.ylabel(numeric_column)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, columns: Optional[List[str]] = None, 
                               figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Create a correlation heatmap for numeric columns.
        
        Parameters:
        -----------
        columns : list, optional
            List of columns to include. If None, use all numeric columns.
        figsize : tuple, default=(12, 10)
            Figure size
        """
        # Use only numeric columns if not specified
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        else:
            # Filter to ensure all columns exist and are numeric
            numeric_cols = []
            for col in columns:
                if col not in self.df.columns:
                    print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
                    continue
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    print(f"Warning: Column '{col}' is not numeric. Skipping.")
                    continue
                numeric_cols.append(col)
        
        if len(numeric_cols) < 2:
            print("Not enough numeric columns for correlation analysis.")
            return
        
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', mask=mask, cmap=cmap, 
                    vmax=1.0, vmin=-1.0, center=0, square=True, linewidths=.5)
        
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def plot_pie_chart(self, column: str, figsize: Tuple[int, int] = (10, 10),
                     top_n: Optional[int] = None, title: Optional[str] = None) -> None:
        """
        Create a pie chart for a categorical column.
        
        Parameters:
        -----------
        column : str
            The categorical column to plot
        figsize : tuple, default=(10, 10)
            Figure size
        top_n : int, optional
            If provided, only plot the top N categories and group others as 'Other'
        title : str, optional
            Custom title for the chart
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        plt.figure(figsize=figsize)
        
        value_counts = self.df[column].value_counts()
        
        if top_n is not None and top_n < len(value_counts):
            # Group categories beyond top N into 'Other'
            top_values = value_counts.head(top_n)
            other_sum = value_counts.iloc[top_n:].sum()
            if other_sum > 0:
                top_values['Other'] = other_sum
            values = top_values
            chart_title = f'Top {top_n} Categories in {column}'
        else:
            values = value_counts
            chart_title = f'Distribution of {column}'
        
        if title:
            chart_title = title
        
        values.plot.pie(autopct='%1.1f%%', shadow=False, startangle=90)
        plt.title(chart_title)
        plt.ylabel('')  # Hide 'None' ylabel
        plt.tight_layout()
        plt.show()
    
    def plot_scatter(self, x_column: str, y_column: str, color_by: Optional[str] = None,
                   figsize: Tuple[int, int] = (12, 8), alpha: float = 0.7) -> None:
        """
        Create a scatter plot of two numeric columns.
        
        Parameters:
        -----------
        x_column : str
            Column for the x-axis
        y_column : str
            Column for the y-axis
        color_by : str, optional
            If provided, color points by this column
        figsize : tuple, default=(12, 8)
            Figure size
        alpha : float, default=0.7
            Transparency of points
        """
        if x_column not in self.df.columns:
            raise ValueError(f"Column '{x_column}' not found in DataFrame")
        
        if y_column not in self.df.columns:
            raise ValueError(f"Column '{y_column}' not found in DataFrame")
        
        if color_by is not None and color_by not in self.df.columns:
            raise ValueError(f"Column '{color_by}' not found in DataFrame")
        
        plt.figure(figsize=figsize)
        
        if color_by is None:
            plt.scatter(self.df[x_column], self.df[y_column], alpha=alpha)
        else:
            scatter = plt.scatter(self.df[x_column], self.df[y_column], 
                                 c=self.df[color_by].astype('category').cat.codes,
                                 alpha=alpha, cmap='viridis')
            # Add legend if categorical column
            if pd.api.types.is_object_dtype(self.df[color_by]) or pd.api.types.is_categorical_dtype(self.df[color_by]):
                categories = self.df[color_by].unique()
                if len(categories) <= 20:  # Only show legend if not too many categories
                    legend1 = plt.legend(scatter.legend_elements()[0], categories,
                                        loc="upper right", title=color_by)
                    plt.gca().add_artist(legend1)
        
        plt.title(f'Scatter Plot: {y_column} vs {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
