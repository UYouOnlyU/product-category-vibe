from typing import List, Dict, Any
import logging

from google.cloud import bigquery


def query_invoices_by_month(
    bq_client: bigquery.Client,
    table_id: str,
    month_str: str,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Query BigQuery rows for a given month (MM-YYYY) when `check_invoice_date` is stored as STRING.
    Attempts to parse the string into DATE using common formats, then filters by month.

    Returns a list of dict rows containing at least: check_invoice_date, item_description, and all columns selected.
    """
    # Simplified for STRING month-year values like 'MM-YYYY' (or 'MM/YYYY').
    # We treat the value as the first day of that month and filter by the requested month bounds.
    query = f"""
    WITH src AS (
      SELECT
        *,
        SAFE.PARSE_DATE('%m-%Y-%d', CONCAT(REPLACE(check_invoice_date, '/', '-'), '-01')) AS parsed_month_start
      FROM `{table_id}`
    ), bounds AS (
      SELECT
        SAFE.PARSE_DATE('%m-%Y-%d', CONCAT(@month_str, '-01')) AS start_month,
        DATE_ADD(SAFE.PARSE_DATE('%m-%Y-%d', CONCAT(@month_str, '-01')), INTERVAL 1 MONTH) AS next_month
    )
    SELECT s.* EXCEPT(parsed_month_start)
    FROM src s, bounds b
    WHERE s.parsed_month_start IS NOT NULL
      AND s.parsed_month_start >= b.start_month
      AND s.parsed_month_start < b.next_month
    ORDER BY s.parsed_month_start
    {"LIMIT @limit" if limit is not None else ""}
    """

    params: List[bigquery.ScalarQueryParameter] = [
        bigquery.ScalarQueryParameter("month_str", "STRING", month_str),
    ]
    if limit is not None:
        params.append(bigquery.ScalarQueryParameter("limit", "INT64", int(limit)))

    job_config = bigquery.QueryJobConfig(query_parameters=params)
    log = logging.getLogger("bq")
    log.debug("Submitting BigQuery job")
    query_job = bq_client.query(query, job_config=job_config)
    results = list(query_job.result())
    # Convert Row to dict
    rows: List[Dict[str, Any]] = [dict(row) for row in results]
    return rows
