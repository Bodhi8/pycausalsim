# Changelog

All notable changes to PyCausalSim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-XX

### Added
- Initial release of PyCausalSim
- Core `CausalSimulator` class
- Structural Causal Model (SCM) implementation
- Basic graph discovery (PC algorithm)
- Intervention simulator with counterfactual generation
- Uplift modeling stub
- Agent-based simulation stub
- Validation and sensitivity analysis framework
- Integration adapters (DoWhy, EconML, Papilon)
- Comprehensive documentation and examples
- Full test suite

### Features
- Simulation-based causal discovery
- Multiple discovery algorithms (PC, GES, LiNGAM, NOTEARS)
- Counterfactual reasoning
- Treatment effect estimation
- Visualization tools
- Validation and refutation tests

### Documentation
- README with quick start guide
- ARCHITECTURE.md with technical details
- Example scripts and notebooks
- API documentation (coming soon)

### Known Limitations
- Some advanced features are placeholders for v0.2.0
- Limited scalability for very large datasets (will improve)
- Neural causal models not yet implemented

## [Unreleased]

### Planned for v0.2.0
- Full uplift modeling implementation
- Neural causal models (NOTEARS, DAG-GNN)
- Enhanced visualization
- Performance optimization
- Comprehensive documentation site

### Planned for v0.3.0
- Agent-based simulation (full implementation)
- Time-series causal discovery
- Real-time inference
- GPU acceleration

### Planned for v1.0.0
- Production-ready release
- Full feature set
- Comprehensive benchmarks
- Industry case studies
