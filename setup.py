"""Setup script for PyCausalSim."""
from setuptools import setup, find_packages
import os

# Read the long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pycausalsim",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Causal discovery and inference through simulation",
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
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "graphviz>=0.19.0",
        ],
        "agents": [
            "mesa>=0.9.0",
        ],
        "gpu": [
            "cupy>=9.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "causal inference",
        "causality",
        "simulation",
        "counterfactual",
        "causal discovery",
        "structural causal models",
        "uplift modeling",
        "treatment effects",
    ],
)
