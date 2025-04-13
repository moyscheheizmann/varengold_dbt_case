with transactions as (
    select * from {{ ref('stg_staging_intermediate__transactions') }}
),

accounts as (
    select * from {{ ref('stg_staging_intermediate__accounts') }}
),

fx_rates as (
    select * from {{ ref('stg_staging_intermediate__fx_rates') }}
),
customers as (
    select * from {{ ref('stg_staging_intermediate__customers') }}
),


-- Join transactions with account information
transactions_with_account as (
    select
        t.transaction_id,
        t.account_id,
        a.customer_id,
        t.transaction_date,
        t.transaction_amount,
        t.transaction_currency as currency_iso_code,
        t.transaction_type,
    from transactions t
    inner join accounts a on t.account_id = a.account_id
),

-- Add currency exchange rates
transactions_with_currency as (
    select
        twa.*,
        coalesce(fxr.fx_rate, 1.0) as exchange_rate_to_eur,
        case
            when twa.currency_iso_code = 'EUR' then twa.transaction_amount
            else twa.transaction_amount * coalesce(fxr.fx_rate, 1.0)
        end as amount_eur
    from transactions_with_account twa
    left join fx_rates fxr on twa.currency_iso_code=fxr.currency_iso_code
),
 transactions_with_branch as (
    select
        twc.*,
        coalesce(c.branch_id, 'Unknown') as branch_id
    from transactions_with_currency twc
    left join customers c on twc.customer_id = c.customer_id
),
transaction_summary as (
    select
        transaction_date,
        customer_id,
        account_id,
        branch_id,
        sum(amount_eur) as total_amount_eur,
        count(*) as total_transactions
    from transactions_with_branch
    group by
        transaction_date,
        customer_id,
        account_id,
        branch_id
)

select * from transaction_summary