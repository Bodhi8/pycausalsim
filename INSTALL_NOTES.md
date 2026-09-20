# Applying this to the repo

Copy `pycausalsim/futures/` into the package, `examples/postlabor_futures.py`
into examples, and `tests/test_futures.py` into tests. Nothing in the existing
package is modified, so the futures module is purely additive and cannot break
the discovery or attribution paths.

Then add to `pycausalsim/__init__.py`:

```python
from . import futures
```

And to `pyproject.toml`, under `[project.optional-dependencies]`:

```toml
futures = []          # numpy only; already a core dependency
```

`README_SECTION.md` contains the five edits for the main README.

Tests require pytest only for `raises` and `approx`; everything else is plain
asserts. Run with:

```bash
pytest tests/test_futures.py -q
```

The final test asserts that `examples/postlabor_futures.py` reproduces the
numbers published in "Simulating the Note from the Future" exactly, so it will
fail loudly if a change to the engine silently alters results.
