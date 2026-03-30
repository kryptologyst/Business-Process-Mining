"""Tests for process mining functionality."""

import pytest
import networkx as nx
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from business_process_mining.data.event_log import EventLog, EventLogGenerator, Event
from business_process_mining.mining.discovery import ProcessDiscovery, ProcessModel
from business_process_mining.mining.conformance import ConformanceChecker, ConformanceResult


class TestProcessDiscovery:
    """Test ProcessDiscovery class."""
    
    def test_discovery_initialization(self):
        """Test discovery initialization."""
        discovery = ProcessDiscovery(seed=42)
        assert discovery.seed == 42
    
    def test_discover_simple_graph(self):
        """Test simple graph discovery."""
        # Create a simple event log
        events = [
            Event(
                case_id="Case_001",
                activity="Start",
                timestamp=datetime(2023, 1, 1, 9, 0, 0)
            ),
            Event(
                case_id="Case_001",
                activity="Process",
                timestamp=datetime(2023, 1, 1, 9, 30, 0)
            ),
            Event(
                case_id="Case_001",
                activity="End",
                timestamp=datetime(2023, 1, 1, 10, 0, 0)
            ),
            Event(
                case_id="Case_002",
                activity="Start",
                timestamp=datetime(2023, 1, 2, 9, 0, 0)
            ),
            Event(
                case_id="Case_002",
                activity="End",
                timestamp=datetime(2023, 1, 2, 10, 0, 0)
            )
        ]
        
        event_log = EventLog(events)
        discovery = ProcessDiscovery(seed=42)
        
        model = discovery.discover_simple_graph(event_log)
        
        assert isinstance(model, ProcessModel)
        assert model.name == "Simple Graph"
        assert model.algorithm == "Simple Transition Graph"
        assert isinstance(model.graph, nx.DiGraph)
        assert model.graph.number_of_nodes() > 0
        assert model.graph.number_of_edges() > 0
    
    def test_discover_bottlenecks(self):
        """Test bottleneck discovery."""
        # Create event log with duration data
        events = [
            Event(
                case_id="Case_001",
                activity="Start",
                timestamp=datetime(2023, 1, 1, 9, 0, 0),
                duration=0.1
            ),
            Event(
                case_id="Case_001",
                activity="Slow Process",
                timestamp=datetime(2023, 1, 1, 9, 30, 0),
                duration=5.0  # Long duration
            ),
            Event(
                case_id="Case_001",
                activity="End",
                timestamp=datetime(2023, 1, 1, 10, 0, 0),
                duration=0.1
            )
        ]
        
        event_log = EventLog(events)
        discovery = ProcessDiscovery(seed=42)
        
        bottlenecks = discovery.discover_bottlenecks(event_log)
        
        assert isinstance(bottlenecks, dict)
        assert 'activity_durations' in bottlenecks
        assert 'resource_utilization' in bottlenecks
        assert 'case_durations' in bottlenecks
        
        # Check that Slow Process has the longest duration
        if 'Slow Process' in bottlenecks['activity_durations']:
            slow_duration = bottlenecks['activity_durations']['Slow Process']['mean']
            assert slow_duration > 1.0  # Should be longer than other activities


class TestProcessModel:
    """Test ProcessModel class."""
    
    def test_process_model_creation(self):
        """Test process model creation."""
        graph = nx.DiGraph()
        graph.add_node("Start", label="Start")
        graph.add_node("End", label="End")
        graph.add_edge("Start", "End", weight=1)
        
        model = ProcessModel(
            name="Test Model",
            algorithm="Test Algorithm",
            graph=graph,
            fitness=0.9,
            precision=0.8
        )
        
        assert model.name == "Test Model"
        assert model.algorithm == "Test Algorithm"
        assert model.fitness == 0.9
        assert model.precision == 0.8
        assert isinstance(model.graph, nx.DiGraph)


class TestConformanceChecker:
    """Test ConformanceChecker class."""
    
    def test_conformance_initialization(self):
        """Test conformance checker initialization."""
        checker = ConformanceChecker(seed=42)
        assert checker.seed == 42
    
    def test_check_conformance(self):
        """Test conformance checking."""
        # Create event log
        events = [
            Event(
                case_id="Case_001",
                activity="Start",
                timestamp=datetime(2023, 1, 1, 9, 0, 0)
            ),
            Event(
                case_id="Case_001",
                activity="Process",
                timestamp=datetime(2023, 1, 1, 9, 30, 0)
            ),
            Event(
                case_id="Case_001",
                activity="End",
                timestamp=datetime(2023, 1, 1, 10, 0, 0)
            )
        ]
        
        event_log = EventLog(events)
        
        # Create simple process model
        graph = nx.DiGraph()
        graph.add_node("Start", label="Start")
        graph.add_node("Process", label="Process")
        graph.add_node("End", label="End")
        graph.add_edge("Start", "Process", weight=1)
        graph.add_edge("Process", "End", weight=1)
        
        model = ProcessModel(
            name="Test Model",
            algorithm="Test Algorithm",
            graph=graph
        )
        
        checker = ConformanceChecker(seed=42)
        result = checker.check_conformance(event_log, model)
        
        assert isinstance(result, ConformanceResult)
        assert 0 <= result.fitness <= 1
        assert 0 <= result.precision <= 1
        assert 0 <= result.generalization <= 1
        assert 0 <= result.simplicity <= 1
        assert isinstance(result.case_results, dict)
        assert isinstance(result.overall_metrics, dict)
    
    def test_find_deviations(self):
        """Test deviation finding."""
        # Create event log with extra activity
        events = [
            Event(
                case_id="Case_001",
                activity="Start",
                timestamp=datetime(2023, 1, 1, 9, 0, 0)
            ),
            Event(
                case_id="Case_001",
                activity="Extra Activity",  # Not in model
                timestamp=datetime(2023, 1, 1, 9, 30, 0)
            ),
            Event(
                case_id="Case_001",
                activity="End",
                timestamp=datetime(2023, 1, 1, 10, 0, 0)
            )
        ]
        
        event_log = EventLog(events)
        
        # Create process model without the extra activity
        graph = nx.DiGraph()
        graph.add_node("Start", label="Start")
        graph.add_node("End", label="End")
        graph.add_edge("Start", "End", weight=1)
        
        model = ProcessModel(
            name="Test Model",
            algorithm="Test Algorithm",
            graph=graph
        )
        
        checker = ConformanceChecker(seed=42)
        deviations = checker.find_deviations(event_log, model)
        
        assert isinstance(deviations, dict)
        assert 'missing_activities' in deviations
        assert 'extra_activities' in deviations
        assert 'wrong_order' in deviations
        assert 'missing_cases' in deviations
        
        # Should find the extra activity
        extra_activities = [d['activity'] for d in deviations['extra_activities']]
        assert 'Extra Activity' in extra_activities


class TestConformanceResult:
    """Test ConformanceResult class."""
    
    def test_conformance_result_creation(self):
        """Test conformance result creation."""
        result = ConformanceResult(
            fitness=0.9,
            precision=0.8,
            generalization=0.7,
            simplicity=0.6,
            alignment_cost=0.1,
            token_replay_fitness=0.85,
            case_results={"Case_001": {"fitness": 0.9}},
            overall_metrics={"fitness": 0.9, "precision": 0.8}
        )
        
        assert result.fitness == 0.9
        assert result.precision == 0.8
        assert result.generalization == 0.7
        assert result.simplicity == 0.6
        assert result.alignment_cost == 0.1
        assert result.token_replay_fitness == 0.85
        assert isinstance(result.case_results, dict)
        assert isinstance(result.overall_metrics, dict)
