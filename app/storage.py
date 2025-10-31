from __future__ import annotations

from google.cloud import storage
import logging


def upload_to_gcs(bucket: str, blob_path: str, local_file: str) -> str:
    log = logging.getLogger("gcs")
    client = storage.Client()
    b = client.bucket(bucket)
    blob = b.blob(blob_path)
    blob.upload_from_filename(local_file)
    return f"gs://{bucket}/{blob_path}"
