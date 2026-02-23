.PHONY: run test lint docker-build docker-run

run:
	python app.py

test:
	pytest -q

lint:
	ruff check .

docker-build:
	docker build -t power-theft-demo .

docker-run:
	docker run -p 5000:5000 power-theft-demo
