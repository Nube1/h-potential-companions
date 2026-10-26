.PHONY: install test demo clean
install:
	python -m pip install -e ".[dev]"
test:
	pytest
demo:
	hpotential generate-synthetic --config configs/study.yaml --output data/sample/synthetic.csv
	hpotential run --config configs/study.yaml --data data/sample/synthetic.csv
clean:
	find . -type d -name __pycache__ -prune -exec rm -r {} +

