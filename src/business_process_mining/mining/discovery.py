"""Process discovery algorithms for extracting process models from event logs."""

import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Any
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass
# PM4Py imports (optional - will fallback if not available)
try:
    from pm4py.objects.log.obj import EventLog as PM4PyEventLog
    from pm4py.objects.petri_net.obj import PetriNet
    from pm4py.objects.petri_net.utils import petri_utils
    from pm4py.algo.discovery.alpha import algorithm as alpha_miner
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.objects.conversion.process_tree import converter as pt_converter
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    PM4PyEventLog = None
    PetriNet = None

from ..data.event_log import EventLog


@dataclass
class ProcessModel:
    """Represents a discovered process model."""
    
    name: str
    algorithm: str
    graph: nx.DiGraph
    petri_net: Optional[PetriNet] = None
    fitness: Optional[float] = None
    precision: Optional[float] = None
    generalization: Optional[float] = None
    simplicity: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None


class ProcessDiscovery:
    """Process discovery algorithms for extracting models from event logs."""
    
    def __init__(self, seed: int = 42) -> None:
        """Initialize process discovery with random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
    
    def discover_simple_graph(self, event_log: EventLog) -> ProcessModel:
        """Discover a simple process graph using activity transitions.
        
        This is a basic approach that creates a graph from observed activity
        transitions in the event log.
        
        Args:
            event_log: The event log to analyze
            
        Returns:
            ProcessModel with discovered graph
        """
        # Create directed graph
        graph = nx.DiGraph()
        
        # Extract activity transitions for each case
        transitions = []
        for case_id in event_log.case_ids:
            trace = event_log.get_case_trace(case_id)
            for i in range(len(trace) - 1):
                transitions.append((trace[i], trace[i + 1]))
        
        # Count transition frequencies
        transition_counts = Counter(transitions)
        
        # Add nodes and edges to graph
        for (source, target), count in transition_counts.items():
            if not graph.has_node(source):
                graph.add_node(source, label=source, frequency=0)
            if not graph.has_node(target):
                graph.add_node(target, label=target, frequency=0)
            
            graph.add_edge(source, target, weight=count, label=f"{count}x")
            
            # Update node frequencies
            graph.nodes[source]['frequency'] += count
            graph.nodes[target]['frequency'] += count
        
        return ProcessModel(
            name="Simple Graph",
            algorithm="Simple Transition Graph",
            graph=graph,
            parameters={"transition_threshold": 1}
        )
    
    def discover_alpha_miner(self, event_log: EventLog) -> ProcessModel:
        """Discover process model using Alpha Miner algorithm.
        
        The Alpha Miner is a classic process discovery algorithm that
        constructs a Petri net based on the directly-follows relations
        in the event log.
        
        Args:
            event_log: The event log to analyze
            
        Returns:
            ProcessModel with discovered Petri net
        """
        if not PM4PY_AVAILABLE:
            print("PM4Py not available. Falling back to simple graph.")
            return self.discover_simple_graph(event_log)
        
        try:
            # Convert to PM4Py format
            pm4py_log = self._convert_to_pm4py(event_log)
            
            # Apply Alpha Miner
            net, initial_marking, final_marking = alpha_miner.apply(pm4py_log)
            
            # Convert Petri net to NetworkX graph for visualization
            graph = self._petri_net_to_graph(net)
            
            return ProcessModel(
                name="Alpha Miner Model",
                algorithm="Alpha Miner",
                graph=graph,
                petri_net=net,
                parameters={
                    "initial_marking": initial_marking,
                    "final_marking": final_marking
                }
            )
        except Exception as e:
            # Fallback to simple graph if Alpha Miner fails
            print(f"Alpha Miner failed: {e}. Falling back to simple graph.")
            return self.discover_simple_graph(event_log)
    
    def discover_heuristics_miner(
        self, 
        event_log: EventLog,
        dependency_threshold: float = 0.5,
        and_threshold: float = 0.1,
        loop_two_threshold: float = 0.5
    ) -> ProcessModel:
        """Discover process model using Heuristics Miner algorithm.
        
        The Heuristics Miner uses frequency information to handle noise
        and infrequent behavior better than the Alpha Miner.
        
        Args:
            event_log: The event log to analyze
            dependency_threshold: Threshold for dependency relations
            and_threshold: Threshold for AND relations
            loop_two_threshold: Threshold for length-two loops
            
        Returns:
            ProcessModel with discovered Petri net
        """
        if not PM4PY_AVAILABLE:
            print("PM4Py not available. Falling back to simple graph.")
            return self.discover_simple_graph(event_log)
        
        try:
            # Convert to PM4Py format
            pm4py_log = self._convert_to_pm4py(event_log)
            
            # Apply Heuristics Miner
            net, initial_marking, final_marking = heuristics_miner.apply(
                pm4py_log,
                parameters={
                    "dependency_thresh": dependency_threshold,
                    "and_measure_thresh": and_threshold,
                    "loop_two_thresh": loop_two_threshold
                }
            )
            
            # Convert Petri net to NetworkX graph for visualization
            graph = self._petri_net_to_graph(net)
            
            return ProcessModel(
                name="Heuristics Miner Model",
                algorithm="Heuristics Miner",
                graph=graph,
                petri_net=net,
                parameters={
                    "dependency_threshold": dependency_threshold,
                    "and_threshold": and_threshold,
                    "loop_two_threshold": loop_two_threshold,
                    "initial_marking": initial_marking,
                    "final_marking": final_marking
                }
            )
        except Exception as e:
            # Fallback to simple graph if Heuristics Miner fails
            print(f"Heuristics Miner failed: {e}. Falling back to simple graph.")
            return self.discover_simple_graph(event_log)
    
    def discover_inductive_miner(self, event_log: EventLog) -> ProcessModel:
        """Discover process model using Inductive Miner algorithm.
        
        The Inductive Miner uses a divide-and-conquer approach to
        construct process trees that can handle complex process structures.
        
        Args:
            event_log: The event log to analyze
            
        Returns:
            ProcessModel with discovered process tree
        """
        if not PM4PY_AVAILABLE:
            print("PM4Py not available. Falling back to simple graph.")
            return self.discover_simple_graph(event_log)
        
        try:
            # Convert to PM4Py format
            pm4py_log = self._convert_to_pm4py(event_log)
            
            # Apply Inductive Miner
            process_tree = inductive_miner.apply(pm4py_log)
            
            # Convert process tree to Petri net
            net, initial_marking, final_marking = pt_converter.apply(process_tree)
            
            # Convert Petri net to NetworkX graph for visualization
            graph = self._petri_net_to_graph(net)
            
            return ProcessModel(
                name="Inductive Miner Model",
                algorithm="Inductive Miner",
                graph=graph,
                petri_net=net,
                parameters={
                    "process_tree": process_tree,
                    "initial_marking": initial_marking,
                    "final_marking": final_marking
                }
            )
        except Exception as e:
            # Fallback to simple graph if Inductive Miner fails
            print(f"Inductive Miner failed: {e}. Falling back to simple graph.")
            return self.discover_simple_graph(event_log)
    
    def discover_bottlenecks(self, event_log: EventLog) -> Dict[str, Any]:
        """Discover process bottlenecks and performance issues.
        
        Args:
            event_log: The event log to analyze
            
        Returns:
            Dictionary with bottleneck analysis results
        """
        bottlenecks = {
            "activity_durations": {},
            "waiting_times": {},
            "resource_utilization": {},
            "case_durations": {},
            "throughput": {}
        }
        
        # Analyze activity durations
        activity_durations = defaultdict(list)
        for event in event_log.events:
            if event.duration is not None:
                activity_durations[event.activity].append(event.duration)
        
        for activity, durations in activity_durations.items():
            bottlenecks["activity_durations"][activity] = {
                "mean": np.mean(durations),
                "median": np.median(durations),
                "std": np.std(durations),
                "min": np.min(durations),
                "max": np.max(durations),
                "count": len(durations)
            }
        
        # Analyze case durations
        case_durations = []
        for case_id in event_log.case_ids:
            duration = event_log.get_case_duration(case_id)
            if duration is not None:
                case_durations.append(duration)
                bottlenecks["case_durations"][case_id] = duration
        
        if case_durations:
            bottlenecks["throughput"]["mean_case_duration"] = np.mean(case_durations)
            bottlenecks["throughput"]["median_case_duration"] = np.median(case_durations)
            bottlenecks["throughput"]["std_case_duration"] = np.std(case_durations)
        
        # Analyze resource utilization
        resource_work = defaultdict(float)
        for event in event_log.events:
            if event.resource and event.duration:
                resource_work[event.resource] += event.duration
        
        total_work = sum(resource_work.values())
        for resource, work in resource_work.items():
            bottlenecks["resource_utilization"][resource] = {
                "total_work": work,
                "utilization": work / total_work if total_work > 0 else 0
            }
        
        return bottlenecks
    
    def _convert_to_pm4py(self, event_log: EventLog) -> PM4PyEventLog:
        """Convert EventLog to PM4Py format.
        
        Args:
            event_log: EventLog to convert
            
        Returns:
            PM4Py EventLog object
        """
        if not PM4PY_AVAILABLE:
            raise ImportError("PM4Py not available")
        
        df = event_log.to_pm4py_format()
        return log_converter.apply(df)
    
    def _petri_net_to_graph(self, petri_net: PetriNet) -> nx.DiGraph:
        """Convert Petri net to NetworkX graph for visualization.
        
        Args:
            petri_net: Petri net to convert
            
        Returns:
            NetworkX directed graph
        """
        if not PM4PY_AVAILABLE:
            raise ImportError("PM4Py not available")
        
        graph = nx.DiGraph()
        
        # Add places as nodes
        for place in petri_net.places:
            graph.add_node(
                f"P_{place.name}",
                label=place.name,
                type="place",
                marking=place.properties.get("token_count", 0)
            )
        
        # Add transitions as nodes
        for transition in petri_net.transitions:
            graph.add_node(
                f"T_{transition.name}",
                label=transition.name,
                type="transition"
            )
        
        # Add arcs as edges
        for arc in petri_net.arcs:
            source = f"P_{arc.source.name}" if hasattr(arc.source, 'name') else f"T_{arc.source.name}"
            target = f"P_{arc.target.name}" if hasattr(arc.target, 'name') else f"T_{arc.target.name}"
            
            graph.add_edge(source, target, weight=1)
        
        return graph
