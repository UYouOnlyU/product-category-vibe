from __future__ import annotations

import csv
import json
import re
from datetime import datetime
import logging
from pathlib import Path
from typing import List, Dict, Any

from google.cloud import bigquery

from .config import Config
from .bq import query_invoices_by_month
from .classifier import GeminiClassifier
from .storage import upload_to_gcs


MONTH_RE = re.compile(r"^(0[1-9]|1[0-2])-(19|20)\d\d$")

# Heuristic post-correction for obvious food/meat/seafood disambiguation
MEAT_KEYWORDS = {
    "pork", "bacon", "beef", "chicken", "lamb", "ham", "sausage", "duck", "turkey", "meat",
}
SEAFOOD_KEYWORDS = {
    "fish", "shrimp", "prawn", "squid", "octopus", "crab", "lobster", "salmon", "tuna", "mackerel",
}
FOOD_PRODUCE = {"Fruit and Vegetables"}
MEAT_CATEGORY = "Meat and Poultry"
SEAFOOD_CATEGORY = "Seafood"


def _is_like(text: str, keys: set[str]) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keys)


def adjust_food_prediction(desc: str, c1: str, s1: float, c2: str, s2: float) -> tuple[str, float, str, float]:
    # Meat rule: if meat-like but predicted produce, switch to meat
    if _is_like(desc, MEAT_KEYWORDS) and c1 in FOOD_PRODUCE:
        return MEAT_CATEGORY, max(s1, 0.9), c1, min(s2, 0.1)
    # Seafood rule: if seafood-like but predicted produce, switch to seafood
    if _is_like(desc, SEAFOOD_KEYWORDS) and c1 in FOOD_PRODUCE:
        return SEAFOOD_CATEGORY, max(s1, 0.9), c1, min(s2, 0.1)
    return c1, s1, c2, s2


def run_pipeline(
    cfg: Config,
    month: str,
    limit: int | None = None,
    dry_run: bool = False,
    progress_every: int = 1,
    batch_size: int = 8,
    concurrency: int = 4,
    deduplicate: bool = True,
) -> Dict[str, Any]:
    log = logging.getLogger("pipeline")
    log.info(f"Starting pipeline | month={month} | limit={limit} | dry_run={dry_run}")
    if not MONTH_RE.match(month):
        raise ValueError("month must be MM-YYYY")

    # Load allowed categories
    with open(cfg.categories_path, "r", encoding="utf-8") as f:
        categories: List[str] = json.load(f)
        if not isinstance(categories, list) or not all(isinstance(x, str) for x in categories):
            raise ValueError("allowed_categories.json must be a JSON array of strings")
        if not categories:
            raise ValueError("allowed_categories.json must not be empty")
    log.info(f"Loaded allowed categories | count={len(categories)}")

    # BigQuery client
    bq_client = bigquery.Client(project=cfg.gcp_project_id)
    log.info(f"Querying BigQuery | table={cfg.table_id} | month={month} | limit={limit}")
    rows = query_invoices_by_month(bq_client, cfg.table_id, month, limit)
    log.info(f"BigQuery returned rows | rows={len(rows)}")

    # Prepare descriptions for classification
    descriptions: List[str] = []
    for r in rows:
        val = str(r.get("item_description") or "").strip()
        descriptions.append(val)

    # Classify
    log.info(f"Initializing Gemini classifier | model={cfg.gemini_model} | location={cfg.gcp_location}")
    classifier = GeminiClassifier(
        project=cfg.gcp_project_id,
        location=cfg.gcp_location,
        model_name=cfg.gemini_model,
        categories=categories,
    )

    log.info(
        f"Classifying descriptions | count={len(descriptions)} | batch_size={batch_size} | concurrency={concurrency} | dedupe={deduplicate}"
    )
    predictions = classifier.classify_batch(
        descriptions,
        progress_every=progress_every,
        batch_size=batch_size,
        concurrency=concurrency,
        deduplicate=deduplicate,
    )
    log.info("Classification finished")

    # Merge predictions back to records (no heuristic override; model-only values)
    enriched: List[Dict[str, Any]] = []
    missing_score_count = 0
    missing_any_count = 0
    for r, pred in zip(rows, predictions):
        rr = dict(r)
        # add product_description from item_description
        product_desc = str(r.get("item_description") or "").strip()
        rr["product_description"] = product_desc
        # take model output directly; if model failed, values may be None
        c1 = pred.get("c1")
        s1 = pred.get("s1")
        c2 = pred.get("c2")
        s2 = pred.get("s2")
        if s1 is None or s2 is None:
            missing_score_count += 1
        if c1 is None or c2 is None or s1 is None or s2 is None:
            missing_any_count += 1
        rr["predicted_category"] = c1
        rr["relevance_score"] = s1
        rr["second_category"] = c2
        rr["second_relevance_score"] = s2
        enriched.append(rr)

    log.info(
        f"Missing counters | missing_scores={missing_score_count} | missing_any_field={missing_any_count}"
    )

    # Write CSV locally
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_name = f"product-category_{month}_{ts}.csv"
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = out_dir / file_name

    if enriched:
        # Ensure product_description is placed before the prediction columns
        pred_cols = [
            "predicted_category",
            "relevance_score",
            "second_category",
            "second_relevance_score",
        ]
        base_fields = [k for k in enriched[0].keys() if k not in pred_cols]
        if "product_description" in base_fields:
            base_fields = [k for k in base_fields if k != "product_description"] + [
                "product_description"
            ]
        fieldnames = base_fields + pred_cols
    else:
        fieldnames = [
            "product_description",
            "predicted_category",
            "relevance_score",
            "second_category",
            "second_relevance_score",
        ]

    log.info(f"Writing CSV | path={str(local_path)}")
    with open(local_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        total = len(enriched)
        if total == 0:
            pass
        for idx, row in enumerate(enriched, start=1):
            writer.writerow(row)
            if idx % max(1, progress_every) == 0 or idx == total:
                log.info(f"CSV progress | written={idx}/{total}")
    log.info(f"CSV written | rows={len(enriched)}")

    # Upload to GCS unless dry_run
    gcs_uri = None
    if not dry_run:
        blob_path = f"{cfg.gcs_output_prefix.rstrip('/')}/{month}/{file_name}"
        log.info(f"Uploading to GCS | bucket={cfg.gcs_bucket} | blob={blob_path}")
        gcs_uri = upload_to_gcs(cfg.gcs_bucket, blob_path, str(local_path))
        log.info(f"Uploaded to GCS | gcs_uri={gcs_uri}")

    return {
        "month": month,
        "total_rows": len(rows),
        "processed": len(enriched),
        "local_csv": str(local_path),
        "gcs_uri": gcs_uri,
    }
