.PHONY: help clean clean-pyc clean-build list test test-all coverage docs release sdist

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "release - package and upload a release"
	@echo "sdist - package"

clean: clean-build clean-pyc clean-misc clean-pycache

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

clean-misc:
	find . -name '.benchmarks' -exec rm -rf {} +
	find . -name '.pytest-cache' -exec rm -rf {} +
	find . -name '.pytest_cache' -exec rm -rf {} +

clean-pycache:
	find . -name '__pycache__' -exec rm -rf {} +

gzip-jsons:
	find segram -name '*.json' -exec gzip {} +
	find tests -name '*.json' -exec gzip {} +

gunzip-jsons:
	find segram -name '*.json.gz' -exec gunzip {} +
	find tests -name '*.json.gz' -exec gunzip {} +

list-deps:
	find segram tests -name "*.py" -exec cat {} + | grep "^import \| import " | grep -v "\"\|'" | grep -v "\(import\|from\) \+\." | sed 's/^\s*import\s*//g' | sed 's/\s*import.*//g' | sed 's/\s*from\s*//g' | grep -v "\..*" | sort | uniq

lint:
	py.test --pylint -m pylint

test:
	python -m pytest

test-all:
	tox

cov-run:
	coverage run
cov-report:
	coverage report
	coverage html
	xdg-open htmlcov/index.html
	# open htmlcov/index.html

coverage: cov-run cov-report

docs:
	rm -rf docs/_build/
	rm -rf docs/api/
	mkdir -p docs/_static
	sphinx-apidoc --module-first --separate -o docs/api/ segram
	sed -i 's/:undoc-members:/:no-undoc-members:/g' docs/api/*.rst
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	xdg-open docs/_build/html/index.html || open docs/_build/html/index.html

build: gzip-jsons clean
	git diff-index --quiet HEAD -- || echo "\n\033[1;31mThere are untracked/uncommited files!!!\033[0m\n" && exit 1
	python -m build
	twine check dist/*

release: clean
	python setup.py sdist upload
	python setup.py bdist_wheel upload

sdist: clean
	python setup.py sdist
	python setup.py bdist_wheel upload
	ls -l dist
