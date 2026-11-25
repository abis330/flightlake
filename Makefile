.PHONY: install test lint data serve docker
install:
	pip install -e ".[dev]"
test:
	pytest -q
lint:
	ruff check src tests
data:
	flightlake simulate --n 300 --out data/raw --workers 4
	flightlake ingest data/raw --lake data/lake --workers 4
serve:
	flightlake serve --lake data/lake
docker:
	docker compose up --build
