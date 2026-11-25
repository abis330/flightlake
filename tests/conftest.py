import pytest

from flightlake.lake import Lake
from flightlake.sim.writer import generate_fleet


@pytest.fixture(scope="session")
def fleet(tmp_path_factory):
    root = tmp_path_factory.mktemp("fl")
    paths = generate_fleet(4, root / "raw", seed=11, fault_rate=1.0)
    lake = Lake(root / "lake")
    lake.ingest_many(paths, workers=1)
    return {"root": root, "paths": paths, "lake": lake}
