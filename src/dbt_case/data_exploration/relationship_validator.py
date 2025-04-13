"""
Module for validating relationships between tables in the database schema.
"""

import pandas as pd
import duckdb
from typing import Dict, List, Tuple, Optional, Any


class RelationshipValidator:
    """
    Class for validating relationships between tables in a database schema.
    """
    
    def __init__(self, conn: duckdb.DuckDBPyConnection, schema: str = 'raw'):
        """
        Initialize the relationship validator.
        
        Parameters:
        -----------
        conn : duckdb.DuckDBPyConnection
            DuckDB connection object
        schema : str, default='raw'
            Schema name to validate
        """
        self.conn = conn
        self.schema = schema
        
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a pandas DataFrame.
        
        Parameters:
        -----------
        query : str
            SQL query to execute
            
        Returns:
        --------
        pd.DataFrame
            Query results as a pandas DataFrame
        """
        return self.conn.execute(query).fetchdf()
    
    def get_table_count(self, table_name: str) -> int:
        """
        Get the number of rows in a table.
        
        Parameters:
        -----------
        table_name : str
            Name of the table
            
        Returns:
        --------
        int
            Number of rows in the table
        """
        query = f"SELECT COUNT(*) as count FROM {self.schema}.{table_name}"
        result = self.execute_query(query)
        return result['count'][0]
    
    def check_foreign_key_relationship(self, 
                                      parent_table: str, 
                                      parent_key: str, 
                                      child_table: str, 
                                      foreign_key: str) -> Dict[str, Any]:
        """
        Check foreign key relationship between two tables.
        
        Parameters:
        -----------
        parent_table : str
            Name of the parent table (with primary key)
        parent_key : str
            Name of the primary key column in parent table
        child_table : str
            Name of the child table (with foreign key)
        foreign_key : str
            Name of the foreign key column in child table
            
        Returns:
        --------
        dict
            Dictionary with analysis results
        """
        # Get the count of records in both tables
        parent_count = self.get_table_count(parent_table)
        child_count = self.get_table_count(child_table)
        
        # Count non-null foreign keys in child table
        non_null_query = f"""
            SELECT COUNT(*) as count
            FROM {self.schema}.{child_table}
            WHERE {foreign_key} IS NOT NULL
        """
        non_null_result = self.execute_query(non_null_query)
        non_null_count = non_null_result['count'][0]
        
        # Check for orphan records (foreign keys without matching primary keys)
        orphan_query = f"""
            SELECT COUNT(*) as count
            FROM {self.schema}.{child_table} c
            LEFT JOIN {self.schema}.{parent_table} p
                ON c.{foreign_key} = p.{parent_key}
            WHERE c.{foreign_key} IS NOT NULL
                AND p.{parent_key} IS NULL
        """
        orphan_result = self.execute_query(orphan_query)
        orphan_count = orphan_result['count'][0]
        
            # Get examples of orphaned records if any
        examples = []
        if orphan_count > 0:
            examples_query = f"""
                SELECT c.*
                FROM {self.schema}.{child_table} c
                LEFT JOIN {self.schema}.{parent_table} p
                    ON c.{foreign_key} = p.{parent_key}
                WHERE c.{foreign_key} IS NOT NULL
                    AND p.{parent_key} IS NULL
                LIMIT 5
            """
            examples = examples.append(self.execute_query(examples_query))
                
            # Calculate percentage of orphaned records
        orphan_percentage = (orphan_count / non_null_count * 100) if non_null_count > 0 else 0
            
            # Prepare the result
        result = {
            "parent_table": parent_table,
            "parent_key": parent_key,
            "parent_record_count": parent_count,
            "child_table": child_table,
            "foreign_key": foreign_key,
            "child_record_count": child_count,
            "non_null_foreign_keys": non_null_count,
            "null_foreign_keys": child_count - non_null_count,
            "null_percentage": ((child_count - non_null_count) / child_count * 100) if child_count > 0 else 0,
            "orphaned_records": orphan_count,
            "orphaned_percentage": orphan_percentage,
            "relationship_valid": orphan_count == 0,
            "examples": examples
        }
        
        return result
    
    def check_all_relationships(self) -> Dict[str, Dict[str, Any]]:
        """
        Check all predefined relationships in the schema.
        
        Returns:
        --------
        dict
            Dictionary with analysis results for all relationships
        """
        # Define the relationships to check
        relationships = [
            # customers to accounts
            {
                "parent_table": "customers",
                "parent_key": "customer_id",
                "child_table": "accounts",
                "foreign_key": "customer_id",
                "name": "customers_accounts"
            },
            # customers to loans
            {
                "parent_table": "customers",
                "parent_key": "customer_id",
                "child_table": "loans",
                "foreign_key": "customer_id",
                "name": "customers_loans"
            },
            # accounts to transactions
            {
                "parent_table": "accounts",
                "parent_key": "account_id",
                "child_table": "transactions",
                "foreign_key": "account_id",
                "name": "accounts_transactions"
            }
            # Commented out as currencies table doesn't exist
            # currencies to fx_rates
            # {
            #     "parent_table": "currencies",
            #     "parent_key": "currency_iso_code",
            #     "child_table": "fx_rates",
            #     "foreign_key": "currency_iso_code",
            #     "name": "currencies_fx_rates"
            # }
        ]
        
        # Check each relationship
        results = {}
        for rel in relationships:
            try:
                # Check if both tables exist before proceeding
                parent_exists_query = f"""
                    SELECT COUNT(*) as count FROM information_schema.tables 
                    WHERE table_schema = '{self.schema}' AND table_name = '{rel["parent_table"]}'
                """
                parent_exists = self.execute_query(parent_exists_query)['count'][0] > 0
                
                child_exists_query = f"""
                    SELECT COUNT(*) as count FROM information_schema.tables 
                    WHERE table_schema = '{self.schema}' AND table_name = '{rel["child_table"]}'
                """
                child_exists = self.execute_query(child_exists_query)['count'][0] > 0
                
                if not parent_exists:
                    results[rel["name"]] = {
                        "error": f"Table '{rel['parent_table']}' does not exist in schema '{self.schema}'",
                        "relationship_valid": False
                    }
                    continue
                    
                if not child_exists:
                    results[rel["name"]] = {
                        "error": f"Table '{rel['child_table']}' does not exist in schema '{self.schema}'",
                        "relationship_valid": False
                    }
                    continue
                
                # Check if columns exist in the tables
                parent_col_query = f"""
                    SELECT COUNT(*) as count FROM information_schema.columns
                    WHERE table_schema = '{self.schema}' AND table_name = '{rel["parent_table"]}'
                    AND column_name = '{rel["parent_key"]}'
                """
                parent_col_exists = self.execute_query(parent_col_query)['count'][0] > 0
                
                child_col_query = f"""
                    SELECT COUNT(*) as count FROM information_schema.columns
                    WHERE table_schema = '{self.schema}' AND table_name = '{rel["child_table"]}'
                    AND column_name = '{rel["foreign_key"]}'
                """
                child_col_exists = self.execute_query(child_col_query)['count'][0] > 0
                
                if not parent_col_exists:
                    results[rel["name"]] = {
                        "error": f"Column '{rel['parent_key']}' does not exist in table '{self.schema}.{rel['parent_table']}'",
                        "relationship_valid": False
                    }
                    continue
                    
                if not child_col_exists:
                    results[rel["name"]] = {
                        "error": f"Column '{rel['foreign_key']}' does not exist in table '{self.schema}.{rel['child_table']}'",
                        "relationship_valid": False
                    }
                    continue
                
                # If all checks pass, validate the relationship
                result = self.check_foreign_key_relationship(
                    rel["parent_table"],
                    rel["parent_key"],
                    rel["child_table"],
                    rel["foreign_key"]
                )
                results[rel["name"]] = result
            except Exception as e:
                results[rel["name"]] = {
                    "error": str(e),
                    "relationship_valid": False
                }
        
        return results
    
    def check_transaction_currencies(self) -> Dict[str, Any]:
        """
        Check if all transaction currencies have corresponding FX rates.
        
        Returns:
        --------
        dict
            Dictionary with analysis results
        """
        # First check if both tables exist
        transactions_exists_query = f"""
            SELECT COUNT(*) as count FROM information_schema.tables 
            WHERE table_schema = '{self.schema}' AND table_name = 'transactions'
        """
        transactions_exists = self.execute_query(transactions_exists_query)['count'][0] > 0
        
        fx_rates_exists_query = f"""
            SELECT COUNT(*) as count FROM information_schema.tables 
            WHERE table_schema = '{self.schema}' AND table_name = 'fx_rates'
        """
        fx_rates_exists = self.execute_query(fx_rates_exists_query)['count'][0] > 0
        
        if not transactions_exists:
            return {
                "error": f"Table 'transactions' does not exist in schema '{self.schema}'",
                "currency_coverage_valid": False
            }
            
        if not fx_rates_exists:
            return {
                "error": f"Table 'fx_rates' does not exist in schema '{self.schema}'",
                "currency_coverage_valid": False
            }
        
        # Check if transaction_currency column exists
        tx_currency_col_query = f"""
            SELECT COUNT(*) as count FROM information_schema.columns
            WHERE table_schema = '{self.schema}' AND table_name = 'transactions'
            AND column_name = 'transaction_currency'
        """
        tx_currency_col_exists = self.execute_query(tx_currency_col_query)['count'][0] > 0
        
        if not tx_currency_col_exists:
            return {
                "error": f"Column 'transaction_currency' does not exist in table '{self.schema}.transactions'",
                "currency_coverage_valid": False
            }
        
        # Check if currency_iso_code column exists in fx_rates
        fx_currency_col_query = f"""
            SELECT COUNT(*) as count FROM information_schema.columns
            WHERE table_schema = '{self.schema}' AND table_name = 'fx_rates'
            AND column_name = 'currency_iso_code'
        """
        fx_currency_col_exists = self.execute_query(fx_currency_col_query)['count'][0] > 0
        
        if not fx_currency_col_exists:
            return {
                "error": f"Column 'currency_iso_code' does not exist in table '{self.schema}.fx_rates'",
                "currency_coverage_valid": False
            }
        
        # Get distinct currencies from transactions
        tx_currency_query = f"""
            SELECT DISTINCT transaction_currency
            FROM {self.schema}.transactions
            WHERE transaction_currency IS NOT NULL
        """
        tx_currencies = self.execute_query(tx_currency_query)
        
        # Get distinct currencies from fx_rates
        fx_currency_query = f"""
            SELECT DISTINCT currency_iso_code
            FROM {self.schema}.fx_rates
        """
        fx_currencies = self.execute_query(fx_currency_query)
        
        # Check for missing currencies in fx_rates
        missing_currencies = []
        for _, row in tx_currencies.iterrows():
            tx_currency = row['transaction_currency']
            if tx_currency not in fx_currencies['currency_iso_code'].values:
                missing_currencies.append(tx_currency)
                
        # Get transaction count by currency
        currency_count_query = f"""
            SELECT transaction_currency, COUNT(*) as count
            FROM {self.schema}.transactions
            WHERE transaction_currency IS NOT NULL
            GROUP BY transaction_currency
        """
        currency_counts = self.execute_query(currency_count_query)
        
        # Calculate totals
        total_transactions = self.get_table_count('transactions')
        transactions_with_missing_fx = 0
        
        for currency in missing_currencies:
            currency_row = currency_counts[currency_counts['transaction_currency'] == currency]
            if not currency_row.empty:
                transactions_with_missing_fx += currency_row['count'].iloc[0]
        
        # Prepare result
        result = {
            "distinct_transaction_currencies": len(tx_currencies),
            "distinct_fx_currencies": len(fx_currencies),
            "missing_currencies": missing_currencies,
            "missing_currency_count": len(missing_currencies),
            "total_transactions": total_transactions,
            "transactions_with_missing_fx": transactions_with_missing_fx,
            "missing_fx_percentage": (transactions_with_missing_fx / total_transactions * 100) if total_transactions > 0 else 0,
            "currency_coverage_valid": len(missing_currencies) == 0,
            "currency_counts": currency_counts.to_dict('records')
        }
        
        return result
    
    def print_relationship_results(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Print the results of relationship validation in a readable format.
        
        Parameters:
        -----------
        results : dict
            Results from check_all_relationships method
        """
        print("\n==== DATABASE RELATIONSHIP VALIDATION ====\n")
        
        all_valid = True
        
        for rel_name, result in results.items():
            print(f"Relationship: {rel_name}")
            print("-" * 50)
            
            if "error" in result:
                print(f"Error: {result['error']}")
                all_valid = False
                print("-" * 50)
                continue
            
            print(f"Parent table: {result['parent_table']} ({result['parent_record_count']} records)")
            print(f"Child table: {result['child_table']} ({result['child_record_count']} records)")
            print(f"Relationship: {result['child_table']}.{result['foreign_key']} → {result['parent_table']}.{result['parent_key']}")
            print(f"Non-null foreign keys: {result['non_null_foreign_keys']} ({100 - result['null_percentage']:.2f}%)")
            print(f"Null foreign keys: {result['null_foreign_keys']} ({result['null_percentage']:.2f}%)")
            
            if result['relationship_valid']:
                print("\nVALID: All non-null foreign keys have matching primary keys")
            else:
                all_valid = False
                print(f"\nINVALID: Found {result['orphaned_records']} orphaned records ({result['orphaned_percentage']:.2f}%)")
                
                if result['examples']:
                    print("\nExamples of orphaned records:")
                    for i, example in enumerate(result['examples'], 1):
                        print(f"  Example {i}:")
                        for k, v in example.items():
                            print(f"    {k}: {v}")
            
            print("-" * 50)
        
        if all_valid:
            print("\nALL RELATIONSHIPS VALID: Referential integrity is maintained across all tables")
        else:
            print("\nSOME RELATIONSHIPS INVALID: There are referential integrity issues that need to be addressed")
    
    def print_currency_results(self, results: Dict[str, Any]) -> None:
        """
        Print the results of currency validation in a readable format.
        
        Parameters:
        -----------
        results : dict
            Results from check_transaction_currencies method
        """
        print("\n==== CURRENCY COVERAGE VALIDATION ====\n")
        
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        print(f"Distinct currencies in transactions: {results['distinct_transaction_currencies']}")
        print(f"Distinct currencies in fx_rates: {results['distinct_fx_currencies']}")
        
        if results['currency_coverage_valid']:
            print("\nVALID: All transaction currencies have corresponding FX rates")
        else:
            print(f"\nINVALID: Found {results['missing_currency_count']} transaction currencies without FX rates")
            print(f"Transactions affected: {results['transactions_with_missing_fx']} ({results['missing_fx_percentage']:.2f}%)")
            
            print("\nMissing currencies:")
            for currency in results['missing_currencies']:
                print(f"  - {currency}")
        
        print("\nTransaction count by currency:")
        for currency_data in results['currency_counts']:
            print(f"  {currency_data['transaction_currency']}: {currency_data['count']} transactions")
