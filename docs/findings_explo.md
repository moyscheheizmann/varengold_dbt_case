# Findings from Data Exoplo

## Description

- record of all issue I encountered

## Issues
### General

### Table Relas

- __Todo__: check relationships

### Column Formatting

- column names in `customers` -and `fx_rates` table not snakecased:

| table_name | column_name    |
|------------|----------------|
| customers  | Age            |
| customers  | Gender         |
| customers  | Address        |
| customers  | City           |
| customers  | Contact Number |
| customers  | Email          |
| fx_rates   | Column4        |

### Value Formatting

- Date columns have different formats
- German number formatting (comma instead of decimal point)

### Issue in Inditvidual Tables

#### Accounts

- `-` value in `account_type`
    - either `savings` or `current`

#### Customers

- duplicated ids:
    - 50% of customer ids are duplicated
- 1082 invalid email adresses found: contain German umlauts and `ß`
- 56 duplicate customers with different `customer_id`
    - same `email` e.g `kenneth.rudolf@gmx.net`, `firstname` and `lastname`, `city`
    - different `address`, `age`, `contact_number`
    - could be ID-fraud
- Marc Gebauer is not from the US ;)












