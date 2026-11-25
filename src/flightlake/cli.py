"""Command line interface: ``flightlake --help``."""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Aerospace telemetry data platform")


@app.command()
def simulate(n: int = 50, out: Path = Path("data/raw"), seed: int = 7, fault_rate: float = 0.65,
             start_index: int = 0, workers: int = 2):
    """Generate N simulated hybrid-electric STOL flights with injected faults."""
    from .sim.writer import generate_fleet

    paths = generate_fleet(n, out, seed=seed, fault_rate=fault_rate, start_index=start_index, workers=workers)
    typer.echo(f"wrote {len(paths)} flights to {out}")


@app.command()
def ingest(src: Path, lake: Path = Path("data/lake"), adapter: str = typer.Option(None, help="sim|ulog|csvdir"),
           mapping: Path = typer.Option(None, help="YAML mapping for csvdir"), rate_hz: float = 10.0,
           workers: int = 2, overwrite: bool = False):
    """Ingest one log (file/dir) or a directory of logs into the lake."""
    from .lake import Lake

    single = src.is_file() or (src / "meta.json").exists() or (adapter == "csvdir" and any(src.glob("*.csv")))
    paths = [src] if single else sorted(p for p in src.iterdir() if p.is_dir() or p.suffix == ".ulg")
    res = Lake(lake).ingest_many(paths, workers=workers, adapter=adapter, mapping=mapping, rate_hz=rate_hz,
                                 overwrite=overwrite)
    ok = sum(r["status"] == "ingested" for r in res)
    failed = [r for r in res if r["status"] == "failed"]
    typer.echo(f"ingested={ok} skipped={len(res) - ok - len(failed)} failed={len(failed)}")
    for f in failed:
        typer.echo(f"  FAILED {f['flight_id']}: {f['error']}")


@app.command()
def query(sql: str, lake: Path = Path("data/lake"), max_rows: int = 50):
    """Run a read-only SQL query (tables: flights, telemetry, messages, labels)."""
    from .lake import Lake
    from .sqlguard import guard

    typer.echo(Lake(lake).sql(guard(sql, max_rows)).to_string())


@app.command()
def show(flight_id: str, lake: Path = Path("data/lake")):
    """Print a flight's summary (QC, metrics, phases)."""
    from .lake import Lake

    s = Lake(lake).summary(flight_id)
    s["qc"] = {k: v for k, v in s["qc"].items() if k not in ("topics", "channels")}
    typer.echo(json.dumps(s, indent=2, default=str))


@app.command()
def serve(lake: Path = Path("data/lake"), host: str = "0.0.0.0", port: int = 8000):
    """Start the REST API."""
    import os

    import uvicorn

    os.environ["FLIGHTLAKE_ROOT"] = str(lake)
    uvicorn.run("flightlake.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
