"""flightlake - aerospace telemetry data platform: ingest, clean, index and query flight logs."""
from .lake import Lake
from .schema import CHANNEL_NAMES, FAULT_TYPES, PHASES

__all__ = ["Lake", "CHANNEL_NAMES", "FAULT_TYPES", "PHASES"]
__version__ = "0.1.0"
