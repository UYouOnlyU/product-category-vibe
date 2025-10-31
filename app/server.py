from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import load_config
from .pipeline import run_pipeline


log = logging.getLogger("server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


class RunRequest(BaseModel):
    month: str
    limit: int | None = None
    dry_run: bool = False
    progress_every: int | None = None
    batch_size: int | None = None
    concurrency: int | None = None
    deduplicate: bool = True


app = FastAPI(title="Product Category Vibe API")


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
def run(req: RunRequest) -> Dict[str, Any]:
    try:
        cfg = load_config()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        batch_size = req.batch_size if req.batch_size is not None else (cfg.classify_batch_size or 8)
        concurrency = req.concurrency if req.concurrency is not None else (cfg.classify_concurrency or 4)
        progress_every = req.progress_every if req.progress_every is not None else (cfg.classify_progress_every or 1)

        log.info(
            "API run | month=%s | limit=%s | dry_run=%s | batch_size=%s | concurrency=%s | progress_every=%s | dedupe=%s",
            req.month,
            req.limit,
            req.dry_run,
            batch_size,
            concurrency,
            progress_every,
            req.deduplicate,
        )
        result = run_pipeline(
            cfg,
            month=req.month,
            limit=req.limit,
            dry_run=req.dry_run,
            progress_every=max(1, progress_every),
            batch_size=max(1, batch_size),
            concurrency=max(1, concurrency),
            deduplicate=req.deduplicate,
        )
        return result
    except ValueError as e:
        # Validation errors (e.g., month format)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        log.exception("Pipeline error")
        raise HTTPException(status_code=500, detail="Internal server error")

