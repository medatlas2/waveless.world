# Contributing

Thank you for helping defend thought and signal privacy!

## Ways to contribute
- File **issues** with reproducible steps and data samples
- Improve **docs** and **examples**
- Add **detectors** and tests under `/tools`
- Join **RF mapping** & **dataset** efforts

## Development setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r tools/requirements.txt
pytest
```
We use conventional commits and PR reviews. Large changes should start with an **RFC** issue.

## Datasets & ethics
Only collect data you have the right to capture. Anonymize and document consent. See `docs/ethics.md`.

## Licensing
By contributing, you agree your contributions are licensed under the MIT License.
