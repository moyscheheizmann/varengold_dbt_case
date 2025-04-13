# Varengold Case Study Victor Staack

This project provides a comprehensive framework for banking data analysis using Python for data exploration and dbt (data build tool) for transformation. It's built with uv for Python package management.

## Project Setup

### Prerequisites

- Python 3.11+
- uv (Python package installer and environment manager)
- DuckDB (for data storage)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/moyscheheizmann/varengold_dbt_case.git 
cd dbt_case
```

2. Initialize and activate a virtual environment with uv:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies using uv:
```bash
uv pip install -r requirements.txt
```

4. Install project in development mode:
```bash
uv pip install -e .
```

## Data Exploration

### Overview

The data exploration components are located in the `src/dbt_case/data_exploration` directory. This package includes modules for:

- Data cleaning (`data_cleaner.py`)
- Data quality checks (`data_quality.py`)
- Descriptive statistics (`descriptive_stats.py`)
- ID analysis (`id_analyzer.py`)
- Outlier detection (`outlier_detection.py`)
- Visualization (`plotting.py`)
- Relationship validation (`relationship_validator.py`)
- Utility functions (`utils.py`)

### Running Data Exploration

The data exploration notebook can be found at:
```
notebooks/data_exploration.ipynb
```

You can run the notebook using Jupyter:
```bash
jupyter notebook notebooks/data_exploration.ipynb
```

Key findings from the data exploration are documented in:
```
docs/findings_explo.md
```

## dbt Pipeline

### Configuration

The dbt project is configured in the `transformation` directory. The main configuration file is `dbt_project.yml`.

### Running the Pipeline

1. Navigate to the transformation directory:
```bash
cd transformation
```

2. Run the dbt pipeline:
```bash
dbt run
```

3. Run tests:
```bash
dbt test
```

4. Generate documentation:
```bash
dbt docs generate
dbt docs serve
```

### Project Structure

- **Staging Models**: Located in `transformation/models/staging/`, these models perform initial data cleaning and type casting
- **Intermediate Models**: Located in `transformation/models/intermediate/`, these models join and transform the staged data
- **Reporting Models**: Located in `transformation/models/reporting/`, these models create final views for business users

## Database Schemas and Tables

The project uses the following schemas and tables:

### Raw Schema
- `accounts`: Raw account data
- `customers`: Raw customer information
- `fx_rates`: Foreign exchange rates
- `loans`: Loan information
- `transactions`: Transaction records

### Staging Schema
- `currencies`: Currency reference data
- `dbt_project_evaluator_exceptions`: Exceptions for dbt project evaluation
- `stg_raw_staging__accounts`: Staged account data
- `stg_raw_staging__customers`: Staged customer data
- `stg_raw_staging__fx_rates`: Staged foreign exchange rates
- `stg_raw_staging__loans`: Staged loan data
- `stg_raw_staging__transactions`: Staged transaction data

### Intermediate Schema
- `stg_staging_intermediate__accounts`: Processed account data
- `stg_staging_intermediate__currencies`: Processed currency data
- `stg_staging_intermediate__customers`: Processed customer data
- `stg_staging_intermediate__fx_rates`: Processed foreign exchange rates
- `stg_staging_intermediate__loans`: Processed loan data
- `stg_staging_intermediate__transactions`: Processed transaction data

### Reporting Schema
- `customer_transactions_summary`: Aggregated customer transaction data

## Additional Resources

- Entity Relationship Diagram: `docs/erd.png`
- Data quality testing macros: `transformation/macros/`
- Source definitions: `transformation/models/sources.yml`
      