install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	# python -m pytest -vv --cov=main --cov=src tests/test_*.py
	python -m pytest --nbval src/*.ipynb

format:	
	black src/lib/*.py src/*.py tests/*.py

lint:
	pylint --disable=R,C,unnecessary-pass --ignore-patterns=test_.*?py src/lib/*.py src/*.py

container-lint:
	docker run --rm -i hadolint/hadolint < Dockerfile

refactor: format lint

deploy:
	#deploy goes here
		
all: install lint test format deploy