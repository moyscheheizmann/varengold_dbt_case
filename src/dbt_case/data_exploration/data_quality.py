"""
Data quality analysis module for evaluating data issues.
"""

import pandas as pd
import re
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

class DataQualityAnalyzer:
    """
    Class for analyzing data quality issues in a DataFrame.
    
    This class provides methods to detect and analyze data quality issues
    such as incorrect formats, invalid values, whitespace issues, etc.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the data quality analyzer with a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame to analyze
        """
        self.df = df
    
    def analyze_data_quality(self) -> Dict[str, List[str]]:
        """
        Analyzes a DataFrame to detect columns that might need cleaning.
        Returns recommendations for cleaning.
        
        Returns:
        --------
        dict
            A dictionary containing recommendations for cleaning
        """
        recommendations = {
            'numeric_columns': [],
            'date_columns': [],
            'whitespace_columns': []  # New category for whitespace issues
        }
        
        # Patterns for detection
        german_number_pattern = r'^[0-9]+(\.[0-9]{3})*(,[0-9]+)?$'
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',  # DD/MM/YYYY or MM/DD/YYYY
            r'\d{1,2}\.\d{1,2}\.\d{4}',  # DD.MM.YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
        ]
        
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Get a sample of non-null values
                sample_size = min(50, self.df[col].dropna().shape[0])
                if sample_size == 0:
                    continue
                    
                sample = self.df[col].dropna().sample(sample_size).astype(str)
                
                # Check for German number pattern
                german_number_count = sum(sample.str.match(german_number_pattern))
                german_number_ratio = german_number_count / sample_size
                
                # Check for date patterns
                date_pattern_count = sum(sample.apply(
                    lambda x: any(re.search(pattern, x) for pattern in date_patterns)
                ))
                date_pattern_ratio = date_pattern_count / sample_size
                
                # Check for whitespace issues
                whitespace_count = sum(sample.apply(
                    lambda x: (x != x.strip()) if isinstance(x, str) else False
                ))
                whitespace_ratio = whitespace_count / sample_size
                
                # Make recommendations
                if german_number_ratio > 0.5:
                    recommendations['numeric_columns'].append(col)
                if date_pattern_ratio > 0.5:
                    recommendations['date_columns'].append(col)
                if whitespace_ratio > 0:  # Even a single instance is worth flagging
                    recommendations['whitespace_columns'].append(col)
        
        return recommendations
    
    def check_whitespace_issues(self) -> Dict[str, Dict[str, Any]]:
        """
        Checks all string columns for leading and trailing whitespace issues.
        
        Returns:
        --------
        dict
            Dictionary with analysis results for each string column
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
    
    def print_whitespace_analysis(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Prints the whitespace analysis results in a readable format.
        
        Parameters:
        -----------
        results : dict
            Results from check_whitespace_issues function
        """
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
    
    def clean_whitespace(self, columns: List[str] = None) -> pd.DataFrame:
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
        result_df = self.df.copy()
        
        # If no columns specified, find all string columns
        if columns is None:
            columns = [col for col in result_df.columns if result_df[col].dtype == 'object']
        
        # Clean whitespace in specified columns
        for col in columns:
            if col in result_df.columns and result_df[col].dtype == 'object':
                result_df[col] = result_df[col].apply(
                    lambda x: x.strip() if isinstance(x, str) else x
                )
        
        return result_df
    
    def show_samples_with_cleaning(self, n_samples: int = 5, clean_numeric: bool = True, 
                                 clean_dates: bool = True, clean_whitespace: bool = True) -> None:
        """
        Displays sample values for each column, along with suggested cleaned values.
        
        Parameters:
        -----------
        n_samples : int, default=5
            Number of sample values to show
        clean_numeric : bool, default=True
            Whether to suggest cleaning for numeric values
        clean_dates : bool, default=True
            Whether to suggest cleaning for date values
        clean_whitespace : bool, default=True
            Whether to suggest cleaning for whitespace issues
        """
        # Analyze data quality
        recommendations = self.analyze_data_quality()
        numeric_columns = recommendations['numeric_columns'] if clean_numeric else []
        date_columns = recommendations['date_columns'] if clean_dates else []
        whitespace_columns = recommendations['whitespace_columns'] if clean_whitespace else []
        
        # Get samples
        samples = {}
        for col in self.df.columns:
            if len(self.df) == 0:
                samples[col] = []
                continue
                
            unique_values = self.df[col].dropna().unique()
            if len(unique_values) <= n_samples:
                samples[col] = unique_values.tolist()
            else:
                samples[col] = self.df[col].sample(n_samples, random_state=42).tolist()
        
        # Print results
        print(f"DataFrame shape: {self.df.shape}")
        print(f"Data types:\n{self.df.dtypes}\n")
        print("Sample values by column (with suggested cleaning):")
        print("--------------------------------------------------")
        
        for col, values in samples.items():
            dtype = self.df[col].dtype
            
            # Determine if column needs cleaning
            needs_numeric_cleaning = col in numeric_columns
            needs_date_cleaning = col in date_columns
            needs_whitespace_cleaning = col in whitespace_columns
            
            print(f"{col} ({dtype}):")
            
            for i, val in enumerate(values, 1):
                print(f"  {i}. Original: {repr(val)}")
                
                # Show cleaned numeric value
                if needs_numeric_cleaning and isinstance(val, str):
                    cleaned_value = self._convert_german_number(val)
                    print(f"     Cleaned numeric: {cleaned_value}")
                
                # Show cleaned date value
                if needs_date_cleaning and isinstance(val, str):
                    date_format = self._detect_date_format(val)
                    if date_format:
                        try:
                            cleaned_date = datetime.strptime(val, date_format)
                            print(f"     Cleaned date: {cleaned_date.isoformat()[:10]} (format: {date_format})")
                        except:
                            print(f"     Cleaned date: [conversion failed]")
                
                # Show cleaned whitespace value
                if needs_whitespace_cleaning and isinstance(val, str):
                    if val != val.strip():
                        print(f"     Cleaned whitespace: {repr(val.strip())}")
            
            print()
        
        # Print recommendations
        print("\nRecommendations:")
        print("---------------")
        
        if numeric_columns:
            print(f"Columns that should be converted from German number format: {numeric_columns}")
        
        if date_columns:
            print(f"Columns that should be converted from mixed date formats: {date_columns}")
            
        if whitespace_columns:
            print(f"Columns with leading/trailing whitespace issues: {whitespace_columns}")
    
    def validate_email(self, email: Any) -> Tuple[bool, Optional[str]]:
        """
        Validates if a string is a properly formatted email address.
        
        Parameters:
        -----------
        email : str
            The email address to validate
            
        Returns:
        --------
        tuple
            (is_valid, error_message)
            is_valid: True if the email is valid, False otherwise
            error_message: Description of the validation error, or None if valid
        """
        if pd.isna(email) or not isinstance(email, str):
            return False, "Email is not a string or is null"
        
        email = email.strip()
        
        # Check for empty string
        if not email:
            return False, "Email is empty"
        
        # Check length constraints
        if len(email) > 254:  # Maximum allowed length for an email address
            return False, "Email is too long (>254 characters)"
        
        # Basic structure check with regex
        # This pattern checks for the basic structure: something@something.something
        basic_pattern = r'^[^@\s]+@[^@\s]+\.[^@\s]+$'
        if not re.match(basic_pattern, email):
            return False, "Email doesn't match basic pattern (user@domain.tld)"
        
        # Split into local and domain parts
        try:
            local, domain = email.rsplit('@', 1)
        except ValueError:
            return False, "Email doesn't contain @ symbol"
        
        # Check local part (before @)
        if not local:
            return False, "Username part before @ is empty"
        
        if len(local) > 64:  # Maximum allowed length for local part
            return False, "Username part is too long (>64 characters)"
        
        # Check for invalid characters in local part
        if not re.match(r'^[a-zA-Z0-9!#$%&\'*+\-/=?^_`{|}~.]+$', local):
            # Check if it's a quoted string, which allows more characters
            if not (local.startswith('"') and local.endswith('"')):
                return False, "Username contains invalid characters"
        
        # Check for consecutive dots in local part
        if '..' in local:
            return False, "Username contains consecutive dots"
        
        # Check domain part (after @)
        if not domain:
            return False, "Domain part after @ is empty"
        
        # Check domain length
        if len(domain) > 253:  # Maximum allowed length for domain
            return False, "Domain is too long (>253 characters)"
        
        # Check domain format - must be something.something
        if not re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', domain):
            return False, "Domain format is invalid"
        
        # Check for consecutive dots in domain
        if '..' in domain:
            return False, "Domain contains consecutive dots"
        
        # Check for hyphens at the beginning or end of domain parts
        domain_parts = domain.split('.')
        for part in domain_parts:
            if part.startswith('-') or part.endswith('-'):
                return False, "Domain part starts or ends with hyphen"
        
        # Check TLD (Top-Level Domain)
        tld = domain_parts[-1]
        if len(tld) < 2:
            return False, "Top-level domain is too short"
        
        if not tld.isalpha():
            return False, "Top-level domain should only contain letters"
        
        # All checks passed
        return True, None
    
    def analyze_emails(self, email_column: str) -> Dict[str, Any]:
        """
        Analyzes email addresses in a DataFrame column.
        
        Parameters:
        -----------
        email_column : str
            Name of the column containing email addresses
            
        Returns:
        --------
        dict
            Dictionary with analysis results
        """
        if email_column not in self.df.columns:
            return {"error": f"Column '{email_column}' not found in DataFrame"}
        
        emails = self.df[email_column]
        total_emails = len(emails)
        null_emails = emails.isna().sum()
        
        results = {
            "total_emails": total_emails,
            "null_emails": null_emails,
            "valid_emails": 0,
            "invalid_emails": 0,
            "unique_domains": set(),
            "common_domains": {},
            "error_types": {},
            "invalid_examples": []
        }
        
        # Validate each email
        for i, email in enumerate(emails):
            if pd.isna(email):
                continue
                
            is_valid, error = self.validate_email(email)
            
            if is_valid:
                results["valid_emails"] += 1
                # Extract domain for domain analysis
                domain = email.split('@')[1]
                results["unique_domains"].add(domain)
                
                # Count domain occurrences
                if domain in results["common_domains"]:
                    results["common_domains"][domain] += 1
                else:
                    results["common_domains"][domain] = 1
            else:
                results["invalid_emails"] += 1
                
                # Track error types
                if error in results["error_types"]:
                    results["error_types"][error] += 1
                else:
                    results["error_types"][error] = 1
                    
                # Store examples of invalid emails (up to 10)
                if len(results["invalid_examples"]) < 10:
                    results["invalid_examples"].append({
                        "index": i,
                        "email": email,
                        "error": error
                    })
        
        # Calculate percentages
        results["valid_percentage"] = (results["valid_emails"] / (total_emails - null_emails) * 100) if (total_emails - null_emails) > 0 else 0
        
        # Convert unique_domains to count
        results["unique_domain_count"] = len(results["unique_domains"])
        results["unique_domains"] = list(results["unique_domains"])
        
        # Sort common domains by frequency
        results["common_domains"] = dict(sorted(results["common_domains"].items(), key=lambda x: x[1], reverse=True)[:10])
        
        return results
    
    def print_email_analysis(self, results: Dict[str, Any]) -> None:
        """
        Prints the email analysis results in a readable format.
        
        Parameters:
        -----------
        results : dict
            Results from analyze_emails function
        """
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        print("Email Validation Results")
        print("=======================")
        print(f"Total emails analyzed: {results['total_emails']}")
        print(f"Null/missing emails: {results['null_emails']}")
        print(f"Valid emails: {results['valid_emails']} ({results['valid_percentage']:.2f}%)")
        print(f"Invalid emails: {results['invalid_emails']}")
        print(f"Unique domains found: {results['unique_domain_count']}")
        
        print("\nTop 10 Most Common Domains:")
        for domain, count in results['common_domains'].items():
            print(f"  {domain}: {count} emails")
        
        if results['invalid_emails'] > 0:
            print("\nCommon Error Types:")
            for error, count in results['error_types'].items():
                print(f"  {error}: {count} emails")
            
            print("\nExamples of Invalid Emails:")
            for example in results['invalid_examples']:
                print(f"  Index {example['index']}: {example['email']} - {example['error']}")
    
    def validate_emails_in_dataframe(self, email_column: str, add_validation_column: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates all emails in a DataFrame column and optionally adds a validation column.
        
        Parameters:
        -----------
        email_column : str
            Name of the column containing email addresses
        add_validation_column : bool, default=True
            If True, adds a new column with validation results
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added validation column if requested
        dict
            Analysis results
        """
        # Create a copy to avoid modifying the original
        result_df = self.df.copy()
        
        # Analyze emails
        analysis = self.analyze_emails(email_column)
        
        if add_validation_column:
            # Add validation column
            validation_column = f"{email_column}_is_valid"
            result_df[validation_column] = False
            
            # Validate each email
            for i, email in enumerate(result_df[email_column]):
                if pd.isna(email):
                    result_df.loc[i, validation_column] = None
                else:
                    is_valid, _ = self.validate_email(email)
                    result_df.loc[i, validation_column] = is_valid
        
        return result_df, analysis
    
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
            if re.match(r'^\d{1,3}(\.\d{3})+', value):
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
            {'pattern': r'^\d{1,2}/\d{1,2}/\d{4}', 'format': '%d/%m/%Y'},
            # MM/DD/YYYY
            {'pattern': r'^\d{1,2}/\d{1,2}/\d{4}', 'format': '%m/%d/%Y'},
            # DD.MM.YYYY
            {'pattern': r'^\d{1,2}\.\d{1,2}\.\d{4}', 'format': '%d.%m.%Y'},
            # YYYY-MM-DD
            {'pattern': r'^\d{4}-\d{1,2}-\d{1,2}', 'format': '%Y-%m-%d'},
            # DD-MM-YYYY
            {'pattern': r'^\d{1,2}-\d{1,2}-\d{4}', 'format': '%d-%m-%Y'},
            # YYYY/MM/DD
            {'pattern': r'^\d{4}/\d{1,2}/\d{1,2}', 'format': '%Y/%m/%d'},
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
