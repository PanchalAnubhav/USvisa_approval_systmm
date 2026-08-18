.PHONY: install train run test docker-build docker-run clean

install:
	pip install -r requirements.txt

train:
	python -c "from us_visa.pipeline.training_pipeline import TrainPipeline; TrainPipeline().run_pipeline()"

run:
	uvicorn app:app --host 0.0.0.0 --port 8080 --reload

test:
	pytest tests/ -v --cov=us_visa --cov-report=term-missing

docker-build:
	docker build -t usvisa-app -f DockerFile .

docker-run:
	docker-compose up -d

clean:
	rm -rf artifact/ logs/ final_model/ __pycache__/ .pytest_cache/ us_visa.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	