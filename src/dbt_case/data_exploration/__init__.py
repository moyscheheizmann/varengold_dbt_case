"""
Data exploration package for DBT case study.

This package contains modules for exploring DuckDB tables including:
- ID analysis
- Descriptive statistics
- Outlier detection
- Data quality analysis
- Data cleaning
- Visualization
"""

from dbt_case.data_exploration.id_analyzer import IDAnalyzer
from dbt_case.data_exploration.descriptive_stats import DescriptiveStats
from dbt_case.data_exploration.outlier_detection import OutlierDetector
from dbt_case.data_exploration.data_quality import DataQualityAnalyzer
from dbt_case.data_exploration.data_cleaner import DataCleaner
from dbt_case.data_exploration.plotting import Plotter
from dbt_case.data_exploration.utils import execute_query

__all__ = [
    'IDAnalyzer',
    'DescriptiveStats',
    'OutlierDetector',
    'DataQualityAnalyzer',
    'DataCleaner',
    'Plotter',
    'execute_query'
]

