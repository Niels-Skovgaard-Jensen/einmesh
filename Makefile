.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync --group dev-all
	@uv run pre-commit install

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Static type checking: Running basedpyright"
	@OUTPUT=$$(uv run basedpyright --level error); \\\
	EXIT_CODE=$$?; \\\
	echo "$$OUTPUT"; \\\
	if [ $$EXIT_CODE -ne 0 ]; then \\\
		if echo "$$OUTPUT" | grep -q "0 errors"; then \\\
			echo "Basedpyright exited non-zero but reported 0 errors. Overriding exit code to 0."; \\\
			exit 0; \\\
		else \\\
			echo "Basedpyright failed with errors (exit code $$EXIT_CODE)."; \\\
			exit $$EXIT_CODE; \\\
		fi \\\
	fi

.PHONY: check-dep
check-dep: ## Test the code with pytest
	@echo "🚀 Testing dependencies"
	@uv sync --only-group numpy
	@uv run basedpyright --level error
	@uv sync --only-group torch
	@uv run basedpyright --level error
	@uv sync --only-group jax
	@uv run basedpyright --level error
	@uv sync dev-all

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: publish
publish: ## Publish a release to PyPI.
	@echo "🚀 Publishing."
	@uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs
docs: notebook_docs## Build and serve the documentation
	@uv run mkdocs serve -s

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.PHONY: notebook_docs
notebook_docs:
	@uv run python -m nbconvert --to markdown --execute docs/notebooks/*.ipynb

.DEFAULT_GOAL := help
