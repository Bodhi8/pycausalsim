"""
PyCausalSim: Causal Discovery and Inference Through Simulation

A Python framework for discovering and validating causal relationships
using counterfactual simulation, designed for digital metrics optimization.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Core dependencies
INSTALL_REQUIRES = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
        "isort>=5.10.0",
    ],
    "viz": [
        "matplotlib>=3.5.0",
        "networkx>=2.6.0",
        "seaborn>=0.11.0",
    ],
    "neural": [
        "torch>=1.10.0",
    ],
    "integrations": [
        "dowhy>=0.8",
        "econml>=0.12.0",
    ],
    "all": [
        "matplotlib>=3.5.0",
        "networkx>=2.6.0",
        "seaborn>=0.11.0",
        "dowhy>=0.8",
        "econml>=0.12.0",
    ],
}

setup(
    name="pycausalsim",
    version="0.1.0",
    author="Brian Curry",
    author_email="brian@vector1.ai",
    description="Causal Discovery and Inference Through Simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bodhi8/pycausalsim",
    project_urls={
        "Bug Tracker": "https://github.com/Bodhi8/pycausalsim/issues",
        "Documentation": "https://pycausalsim.readthedocs.io",
        "Source Code": "https://github.com/Bodhi8/pycausalsim",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "causal inference",
        "causal discovery",
        "counterfactual",
        "simulation",
        "machine learning",
        "data science",
        "a/b testing",
        "marketing attribution",
        "uplift modeling",
    ],
)
