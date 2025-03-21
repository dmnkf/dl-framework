# Development Tools and Practices

This guide covers the development tools and configurations used in the Aether project.

## Code Quality Tools

### Pre-commit Hooks

We use pre-commit hooks to maintain code quality and consistency. The configuration is in `.pre-commit-config.yaml`:

```yaml
repos:
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.8.5
  hooks:
    - id: ruff-format
```

To set up pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Install the hooks
pre-commit install
```

### Code Formatting

We use Ruff for Python code formatting. The formatter is configured through pre-commit and runs automatically on git commits.

## Environment Configuration

### Environment Variables

We use a `.env` file for local development. Copy `sample.env` to `.env` and configure:

```bash
# API Keys for Experiment Tracking
NEPTUNE_API_TOKEN="YOUR_API_TOKEN"
WANDB_API_KEY="YOUR_API_TOKEN"

# AWS Configuration
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=sa-east-1

# Development Settings
HYDRA_FULL_ERROR=1
```

## Documentation

### MkDocs Configuration

The documentation is built using MkDocs with the Material theme. Configuration in `mkdocs.yml`:

```yaml
theme:
  name: material
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
```

### Python API Documentation

We use mkdocstrings for automatic Python API documentation. Configuration in `mkdocs.yml`:

```yaml
plugins:
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          paths: [src/, .]
          options:
            show_source: true
            show_root_heading: true
            heading_level: 2
```

### Markdown Extensions

The documentation supports various Markdown extensions for enhanced content:

```yaml
markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.superfences
  - admonition
  - pymdownx.details
  - pymdownx.tabbed:
      alternate_style: true
  - tables
  - toc:
      permalink: true
```