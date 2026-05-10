"""Upload stage — ingest data from CSV, Excel, Parquet or JSON."""

from stages.upload.loader import apply_sentinels, optimise_dtypes, read_file
from stages.upload.models import DatasetMeta, UploadConfig

__all__ = [
    "read_file",
    "optimise_dtypes",
    "apply_sentinels",
    "UploadConfig",
    "DatasetMeta",
]
