# Business Process Mining

A comprehensive Python package for business process mining and optimization, designed for research and educational purposes.

## ⚠️ DISCLAIMER

**This software is for research and educational purposes only.**

It should not be used for automated decision-making without human review. All results and recommendations should be validated by domain experts before implementing any process changes.

## Overview

Business process mining involves analyzing event logs to discover, monitor, and improve real business processes. This package provides:

- **Process Discovery**: Extract process models from event logs using multiple algorithms
- **Conformance Checking**: Compare actual vs. expected process behavior
- **Performance Analysis**: Identify bottlenecks, resource utilization, and efficiency metrics
- **Visualization**: Interactive dashboards and process flow diagrams
- **Synthetic Data Generation**: Create realistic event logs for testing and demonstration

## Features

### Process Discovery Algorithms
- **Simple Graph**: Basic transition-based process discovery
- **Alpha Miner**: Classic process discovery algorithm
- **Heuristics Miner**: Noise-tolerant process discovery
- **Inductive Miner**: Divide-and-conquer approach for complex processes

### Analysis Capabilities
- **Conformance Checking**: Fitness, precision, generalization, and simplicity metrics
- **Bottleneck Analysis**: Identify process inefficiencies and delays
- **Resource Efficiency**: Analyze resource utilization and workload distribution
- **Performance KPIs**: Throughput, quality, and cost metrics
- **Model Comparison**: Leaderboard for comparing different process models

### Visualization
- **Process Flow Diagrams**: Interactive process models with transition frequencies
- **Performance Dashboards**: Comprehensive metrics and KPI visualization
- **Bottleneck Analysis**: Visual identification of process constraints
- **Resource Utilization**: Workload and efficiency analysis

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager

### Install Dependencies

```bash
# Using pip
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Business-Process-Mining.git
cd Business-Process-Mining

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Quick Start

### 1. Generate Synthetic Data

```python
from business_process_mining import EventLogGenerator
from datetime import datetime

# Initialize generator
generator = EventLogGenerator(seed=42)

# Generate simple approval process
event_log = generator.generate_simple_approval_process(
    num_cases=100,
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31)
)

print(f"Generated {len(event_log.case_ids)} cases with {len(event_log.events)} events")
```

### 2. Discover Process Models

```python
from business_process_mining import ProcessDiscovery

# Initialize discovery
discovery = ProcessDiscovery(seed=42)

# Discover process models
simple_model = discovery.discover_simple_graph(event_log)
alpha_model = discovery.discover_alpha_miner(event_log)
heuristics_model = discovery.discover_heuristics_miner(event_log)
```

### 3. Check Conformance

```python
from business_process_mining import ConformanceChecker

# Initialize conformance checker
checker = ConformanceChecker(seed=42)

# Check conformance for each model
for model in [simple_model, alpha_model, heuristics_model]:
    result = checker.check_conformance(event_log, model)
    print(f"{model.name}: Fitness={result.fitness:.3f}, Precision={result.precision:.3f}")
```

### 4. Analyze Performance

```python
from business_process_mining import ProcessMetrics

# Initialize metrics calculator
metrics = ProcessMetrics(seed=42)

# Calculate process KPIs
kpis = metrics.calculate_process_kpis(event_log)
print(f"Mean case duration: {kpis.mean_case_duration:.2f} hours")
print(f"Completion rate: {kpis.completion_rate:.1%}")

# Analyze bottlenecks
bottlenecks = metrics.calculate_bottleneck_analysis(event_log)
print(f"Top bottlenecks: {bottlenecks['top_bottlenecks']}")
```

### 5. Visualize Results

```python
from business_process_mining import ProcessVisualizer

# Initialize visualizer
visualizer = ProcessVisualizer()

# Create visualizations
fig1 = visualizer.plot_process_model(simple_model, title="Simple Process Model")
fig2 = visualizer.plot_case_duration_distribution(event_log)
fig3 = visualizer.plot_activity_frequency(event_log)
```

## Interactive Demo

Run the Streamlit demo application:

```bash
streamlit run demo/app.py
```

The demo provides:
- Interactive process data generation
- Real-time process discovery and analysis
- Comprehensive visualization dashboards
- Model comparison and leaderboards
- Bottleneck and resource efficiency analysis

## Data Schema

### Event Log Format

The package expects event logs in the following format:

| Column | Type | Description |
|--------|------|-------------|
| case_id | string | Unique identifier for process instance |
| activity | string | Name of the activity performed |
| timestamp | datetime | When the activity occurred |
| resource | string | Resource (person/system) that performed the activity |
| cost | float | Cost associated with this activity |
| duration | float | Duration of the activity in hours |

### Supported Process Types

1. **Simple Approval Process**
   - Flow: Start → Check Form → [Approve|Reject] → End
   - Use case: Document approval workflows

2. **Loan Processing**
   - Flow: Application → Credit Check → [Auto-approve|Manual Review] → [Approve|Reject] → [Disburse|End]
   - Use case: Financial service processes

3. **Manufacturing Process**
   - Flow: Order → Plan → Material Check → [Produce|Wait] → Quality Check → Ship
   - Use case: Production and supply chain processes

## Configuration

Configuration is managed through YAML files in the `configs/` directory:

```yaml
# configs/default.yaml
data:
  seed: 42
  num_cases: 100
  start_date: "2023-01-01"
  end_date: "2023-12-31"

discovery:
  algorithms:
    - "simple_graph"
    - "alpha_miner"
    - "heuristics_miner"
    - "inductive_miner"

evaluation:
  calculate_kpis: true
  calculate_bottlenecks: true
  calculate_resource_efficiency: true
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_event_log.py
```

## Evaluation Metrics

### Process Mining Metrics
- **Fitness**: How well the model explains the event log (0-1)
- **Precision**: How precise the model is (0-1)
- **Generalization**: How well the model generalizes (0-1)
- **Simplicity**: How simple the model is (0-1)

### Business KPIs
- **Throughput**: Cases per day/week, mean case duration
- **Quality**: Completion rate, error rate, rework rate
- **Cost**: Cost per case, total process cost
- **Efficiency**: Resource utilization, bottleneck identification

### Bottleneck Analysis
- **Activity Duration**: Mean, median, standard deviation
- **Bottleneck Score**: Duration × frequency
- **Resource Utilization**: Workload distribution
- **Process Variants**: Number of different execution paths

## Architecture

```
src/business_process_mining/
├── data/           # Event log data structures and generation
├── mining/         # Process discovery and conformance checking
├── eval/           # Evaluation metrics and analysis
├── viz/            # Visualization utilities
└── utils/          # Utility functions and helpers
```

## Privacy and Compliance

- **Data Anonymization**: All synthetic data is generated without real personal information
- **PII Minimization**: No real customer or employee data is processed
- **Audit Trail**: All analysis results include data lineage information
- **Retention Policy**: Results are stored locally and can be deleted at any time

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for all classes and methods
- Write tests for new functionality
- Update documentation as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PM4Py](https://pm4py.fit.fraunhofer.de/) for process mining algorithms
- [NetworkX](https://networkx.org/) for graph analysis
- [Streamlit](https://streamlit.io/) for interactive dashboards
- [Plotly](https://plotly.com/) for visualization

## References

1. van der Aalst, W. M. P. (2016). Process Mining: Data Science in Action. Springer.
2. van der Aalst, W. M. P. (2011). Process Mining: Discovery, Conformance and Enhancement of Business Processes. Springer.
3. Rozinat, A., & van der Aalst, W. M. P. (2008). Conformance checking of processes based on monitoring real behavior. Information Systems, 33(1), 64-95.

## Support

For questions, issues, or contributions:
- Create an issue in the GitHub repository
- Check the documentation and examples
- Review the test cases for usage examples

---

**Remember**: This software is for research and educational purposes only. Always validate results with domain experts before making business decisions.
# Business-Process-Mining
