# Makefile for Fractal Frontier

.PHONY: help install lint format test clean

help:
	@echo "Available targets:" \
	&& echo "  install   – Install all dependencies (runtime & dev)" \
	&& echo "  lint      – Run flake8 linting" \
	&& echo "  format    – Auto‑format code with black and isort" \
	&& echo "  test      – Run pytest unit tests" \
	&& echo "  clean     – Remove build artifacts (pycache, .coverage)"

install:
	pip install -r requirements.txt

lint:
	flake8 .

format:
	# Black and isort handle most formatting
	black .
	isort .

test:
	pytest

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} + ; \
	find . -type f -name '*.pyc' -delete ; \
	find . -type f -name '.coverage' -delete

check:
	make format
	make lint

run:
	python Fractal_Frontier.py