# Contributing to PyCausalSim

Thanks for your interest in contributing! PyCausalSim welcomes bug reports,
feature requests, documentation improvements, and code contributions.

## Reporting bugs

Open an issue at https://github.com/Bodhi8/pycausalsim/issues and include:

- What you did (a minimal code example is ideal)
- What you expected to happen
- What actually happened (full traceback if there was an error)
- Your environment: OS, Python version, and PyCausalSim version

## Requesting features

Open an issue describing the use case you're trying to solve. Explaining
*why* you need the feature (the analytical question you're trying to answer)
helps more than describing an implementation.

## Contributing code

1. Fork the repository and create a branch from `main`.
2. Set up a development environment:

   ```bash
   git clone https://github.com/<your-username>/pycausalsim.git
   cd pycausalsim
   pip install -e ".[dev]"
   ```

3. Make your changes. Please:
   - Follow the existing code style (the project uses `black` for
     formatting and `flake8` for linting: run `black pycausalsim tests`
     and `flake8 pycausalsim` before committing)
   - Add or update tests in `tests/` for any behavior you change
   - Add docstrings to any new public classes or functions

4. Run the test suite and make sure everything passes:

   ```bash
   pytest tests/
   ```

5. Open a pull request against `main` with a clear description of the
   change and the motivation for it.

## Questions and discussion

For questions that aren't bug reports or feature requests, open a GitHub
Discussion or reach out at brian@vector1.ai.

## Code of conduct

Be respectful and constructive. We follow the spirit of the
[Contributor Covenant](https://www.contributor-covenant.org/): harassment
or exclusionary behavior of any kind is not tolerated.
