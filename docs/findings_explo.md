# Findings from Data Exoplo

## Description

- record of all issue I encountered

## Issues
### General
### Table Relas

- ALL RELATIONSHIPS VALID: Referential integrity is maintained across all tables
- CURRENCY COVERAGE INVALID
    - Distinct currencies in transactions: 15
    - Distinct currencies in fx_rates: 15
    - __INVALID__: Found 1 transaction currencies without FX rates
    - Transactions affected: 1 (0.00%)
    - Missing currencies:
        - RON1

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

#### Loans

- typo in column name `loan_type` (`loanT_type`)


#### Transactions

- RON1 currency code does not exist. Should be `RON` for Romanian leu
- shouldn't transaction_amounts for `withdrawal` and `transfer` be negative?
- Columns with leading/trailing whitespace issues: ['transaction_currency']


#### Currencies

Duplicate records:
                       currency currency_iso_code
45          Cook Islands dollar            (none)
60                Faroese króna            (none)
68               Guernsey pound            (none)
79                   Manx pound            (none)
83                 Jersey pound            (none)
87           Kiribati dollar[E]            (none)
115              Niue dollar[E]            (none)
126  Pitcairn Islands dollar[E]            (none)
131              Sahrawi peseta            (none)
140         Somaliland shilling            (none)
152         Transnistrian ruble            (none)
156             Tuvaluan dollar            (none)
24         United States dollar               USD
29         United States dollar               USD

- no currency code for (none) and USD duplicated
- sometimes no ISO code (unrecognized republic [e.g Somaliland], sometimes currency does not exist [Guernsey is Pound Sterling, Cook Island is NZD, Faroe is DKK]). Here's a table with comments:

I'll convert this data into a markdown table for you.

| # | Currency | Currency ISO Code / Comment |
|---|----------|-------------------|
| 45 | Cook Islands dollar | NZD |
| 60 | Faroese króna | DKK |
| 68 | Guernsey pound | GBP |
| 79 | Manx pound | Isle of Man [No ISO, (GBP unofficially)] |
| 83 | Jersey pound | (GBP, JEP unofficially) |
| 87 | Kiribati dollar[E] | No ISO, KID unofficially |
| 115 | Niue dollar[E] | NZD |
| 126 | Pitcairn Islands dollar[E] | NZD [However, the territory has issued commemorative Pitcairn Islands dollar coins since 1988] |
| 131 | Sahrawi peseta | No ISO, (EHP is used in commerce) |
| 140 | Somaliland shilling | No ISO, Currency symbol SLSH |
| 152 | Transnistrian ruble |  unrecognized state|
| 156 | Tuvaluan dollar | unrecognized currency, just issues commemorative coins|

#### FX Rates

- stray column: `Column4`
- currency codes have to be stripped (e.g ' USD')









