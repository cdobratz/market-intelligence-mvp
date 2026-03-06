# Contributing

Contributions are welcome! Please follow the workflow below.

## Development Workflow

1. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make changes and test**
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run linting
ruff check .

# Format code
black .
```

3. **Commit and push**
```bash
git add .
git commit -m "Add feature: description"
git push origin feature/your-feature-name
```

4. **Create Pull Request** on GitHub

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_ingestion.py

# Run integration tests
pytest tests/integration/
```

## Code Style

- Use `black` for formatting
- Use `ruff` for linting
- Follow existing patterns in the codebase
