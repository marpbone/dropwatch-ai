.PHONY: install fixture demo backend frontend test lint clean

install:
	pip install -e ".[dev]"
	cd frontend && npm install

fixture:
	python scripts/make_test_track.py

demo: fixture
	python scripts/seed_demo.py

backend:
	uvicorn djprep.api.main:app --reload --app-dir backend

frontend:
	cd frontend && npm run dev

test:
	pytest -q

lint:
	ruff check backend/djprep scripts
	cd frontend && npx tsc --noEmit

clean:
	rm -rf backend/tests/fixtures/*.wav frontend/dist .pytest_cache
	find . -name __pycache__ -prune -exec rm -rf {} +
