"""Source adapters: each turns one raw log format into a :class:`RawFlight`."""
from pathlib import Path

from .csvdir import load_csvdir
from .simlog import load_simlog
from .ulog import load_ulog

ADAPTERS = {"sim": load_simlog, "ulog": load_ulog, "csvdir": load_csvdir}


def detect_adapter(path: str | Path) -> str:
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".ulg":
        return "ulog"
    if p.is_dir() and (p / "topics").is_dir() and (p / "meta.json").exists():
        return "sim"
    if p.is_dir():
        return "csvdir"
    raise ValueError(f"Cannot detect log format for {p}")


def load(path: str | Path, adapter: str | None = None, **kw):
    adapter = adapter or detect_adapter(path)
    return ADAPTERS[adapter](path, **kw)
