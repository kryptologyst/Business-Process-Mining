"""Conformance checking algorithms for comparing event logs with process models."""

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
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    from pm4py.algo.conformance.alignments import algorithm as alignments
    from pm4py.objects.conversion.log import converter as log_converter
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    PM4PyEventLog = None
    PetriNet = None

from ..data.event_log import EventLog
from .discovery import ProcessModel


@dataclass
class ConformanceResult:
    """Results of conformance checking analysis."""
    
    fitness: float
    precision: float
    generalization: float
    simplicity: float
    alignment_cost: float
    token_replay_fitness: float
    case_results: Dict[str, Dict[str, Any]]
    overall_metrics: Dict[str, float]


class ConformanceChecker:
    """Conformance checking algorithms for process models."""
    
    def __init__(self, seed: int = 42) -> None:
        """Initialize conformance checker with random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
    
    def check_conformance(
        self, 
        event_log: EventLog, 
        process_model: ProcessModel
    ) -> ConformanceResult:
        """Check conformance between event log and process model.
        
        Args:
            event_log: The event log to check
            process_model: The process model to check against
            
        Returns:
            ConformanceResult with detailed analysis
        """
        # Calculate basic conformance metrics
        fitness = self._calculate_fitness(event_log, process_model)
        precision = self._calculate_precision(event_log, process_model)
        generalization = self._calculate_generalization(event_log, process_model)
        simplicity = self._calculate_simplicity(process_model)
        
        # Calculate alignment-based metrics if Petri net is available
        alignment_cost = 0.0
        token_replay_fitness = fitness
        
        if process_model.petri_net is not None:
            try:
                alignment_cost, token_replay_fitness = self._calculate_alignment_metrics(
                    event_log, process_model
                )
            except Exception as e:
                print(f"Alignment calculation failed: {e}. Using basic metrics.")
        
        # Analyze individual cases
        case_results = self._analyze_cases(event_log, process_model)
        
        # Calculate overall metrics
        overall_metrics = {
            "fitness": fitness,
            "precision": precision,
            "generalization": generalization,
            "simplicity": simplicity,
            "alignment_cost": alignment_cost,
            "token_replay_fitness": token_replay_fitness
        }
        
        return ConformanceResult(
            fitness=fitness,
            precision=precision,
            generalization=generalization,
            simplicity=simplicity,
            alignment_cost=alignment_cost,
            token_replay_fitness=token_replay_fitness,
            case_results=case_results,
            overall_metrics=overall_metrics
        )
    
    def _calculate_fitness(self, event_log: EventLog, process_model: ProcessModel) -> float:
        """Calculate fitness metric (how well the model explains the log).
        
        Args:
            event_log: The event log
            process_model: The process model
            
        Returns:
            Fitness score between 0 and 1
        """
        if not process_model.graph.nodes:
            return 0.0
        
        # Get all activities in the log
        log_activities = set(event_log.activities)
        
        # Get all activities in the model
        model_activities = set()
        for node in process_model.graph.nodes:
            if process_model.graph.nodes[node].get('type') != 'place':
                model_activities.add(process_model.graph.nodes[node].get('label', node))
        
        # Calculate coverage
        if not log_activities:
            return 0.0
        
        covered_activities = log_activities.intersection(model_activities)
        fitness = len(covered_activities) / len(log_activities)
        
        return fitness
    
    def _calculate_precision(self, event_log: EventLog, process_model: ProcessModel) -> float:
        """Calculate precision metric (how precise the model is).
        
        Args:
            event_log: The event log
            process_model: The process model
            
        Returns:
            Precision score between 0 and 1
        """
        if not process_model.graph.nodes:
            return 0.0
        
        # Get all activities in the log
        log_activities = set(event_log.activities)
        
        # Get all activities in the model
        model_activities = set()
        for node in process_model.graph.nodes:
            if process_model.graph.nodes[node].get('type') != 'place':
                model_activities.add(process_model.graph.nodes[node].get('label', node))
        
        # Calculate precision
        if not model_activities:
            return 0.0
        
        used_activities = log_activities.intersection(model_activities)
        precision = len(used_activities) / len(model_activities)
        
        return precision
    
    def _calculate_generalization(self, event_log: EventLog, process_model: ProcessModel) -> float:
        """Calculate generalization metric (how well the model generalizes).
        
        Args:
            event_log: The event log
            process_model: The process model
            
        Returns:
            Generalization score between 0 and 1
        """
        # Simple generalization based on model complexity vs log diversity
        if not process_model.graph.nodes:
            return 0.0
        
        # Count unique traces in log
        unique_traces = set()
        for case_id in event_log.case_ids:
            trace = tuple(event_log.get_case_trace(case_id))
            unique_traces.add(trace)
        
        # Count model complexity (number of edges)
        model_complexity = process_model.graph.number_of_edges()
        
        # Simple generalization metric
        if model_complexity == 0:
            return 0.0
        
        # Higher ratio of unique traces to model complexity suggests better generalization
        generalization = min(1.0, len(unique_traces) / model_complexity)
        
        return generalization
    
    def _calculate_simplicity(self, process_model: ProcessModel) -> float:
        """Calculate simplicity metric (how simple the model is).
        
        Args:
            process_model: The process model
            
        Returns:
            Simplicity score between 0 and 1
        """
        if not process_model.graph.nodes:
            return 0.0
        
        # Count nodes and edges
        num_nodes = process_model.graph.number_of_nodes()
        num_edges = process_model.graph.number_of_edges()
        
        # Simple complexity metric
        if num_nodes == 0:
            return 0.0
        
        # Lower edge-to-node ratio suggests simpler model
        complexity_ratio = num_edges / num_nodes
        
        # Convert to simplicity (inverse of complexity)
        simplicity = 1.0 / (1.0 + complexity_ratio)
        
        return simplicity
    
    def _calculate_alignment_metrics(
        self, 
        event_log: EventLog, 
        process_model: ProcessModel
    ) -> Tuple[float, float]:
        """Calculate alignment-based conformance metrics.
        
        Args:
            event_log: The event log
            process_model: The process model with Petri net
            
        Returns:
            Tuple of (alignment_cost, token_replay_fitness)
        """
        if not PM4PY_AVAILABLE:
            return 0.0, 0.0
        
        try:
            # Convert to PM4Py format
            pm4py_log = self._convert_to_pm4py(event_log)
            
            # Get Petri net components
            net = process_model.petri_net
            initial_marking = process_model.parameters.get("initial_marking")
            final_marking = process_model.parameters.get("final_marking")
            
            if initial_marking is None or final_marking is None:
                return 0.0, 0.0
            
            # Calculate alignments
            alignments_result = alignments.apply_log(
                pm4py_log, net, initial_marking, final_marking
            )
            
            # Calculate average alignment cost
            total_cost = sum(alignment['cost'] for alignment in alignments_result)
            avg_cost = total_cost / len(alignments_result) if alignments_result else 0.0
            
            # Calculate token replay fitness
            token_replay_result = token_replay.apply(
                pm4py_log, net, initial_marking, final_marking
            )
            
            fitness = token_replay_result['log_fitness']
            
            return avg_cost, fitness
            
        except Exception as e:
            print(f"Alignment calculation failed: {e}")
            return 0.0, 0.0
    
    def _analyze_cases(
        self, 
        event_log: EventLog, 
        process_model: ProcessModel
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze conformance for individual cases.
        
        Args:
            event_log: The event log
            process_model: The process model
            
        Returns:
            Dictionary with case-level analysis results
        """
        case_results = {}
        
        for case_id in event_log.case_ids:
            trace = event_log.get_case_trace(case_id)
            duration = event_log.get_case_duration(case_id)
            
            # Check if trace can be replayed on the model
            can_replay = self._can_replay_trace(trace, process_model)
            
            # Calculate trace-specific metrics
            trace_fitness = 1.0 if can_replay else 0.0
            
            case_results[case_id] = {
                "trace": trace,
                "duration": duration,
                "can_replay": can_replay,
                "fitness": trace_fitness,
                "length": len(trace)
            }
        
        return case_results
    
    def _can_replay_trace(self, trace: List[str], process_model: ProcessModel) -> bool:
        """Check if a trace can be replayed on the process model.
        
        Args:
            trace: Sequence of activities
            process_model: The process model
            
        Returns:
            True if trace can be replayed, False otherwise
        """
        if not trace or not process_model.graph.nodes:
            return False
        
        # Simple replay check: verify all activities in trace exist in model
        model_activities = set()
        for node in process_model.graph.nodes:
            if process_model.graph.nodes[node].get('type') != 'place':
                model_activities.add(process_model.graph.nodes[node].get('label', node))
        
        trace_activities = set(trace)
        return trace_activities.issubset(model_activities)
    
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
    
    def find_deviations(
        self, 
        event_log: EventLog, 
        process_model: ProcessModel
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Find deviations between event log and process model.
        
        Args:
            event_log: The event log
            process_model: The process model
            
        Returns:
            Dictionary with different types of deviations
        """
        deviations = {
            "missing_activities": [],
            "extra_activities": [],
            "wrong_order": [],
            "missing_cases": []
        }
        
        # Get activities in log and model
        log_activities = set(event_log.activities)
        model_activities = set()
        for node in process_model.graph.nodes:
            if process_model.graph.nodes[node].get('type') != 'place':
                model_activities.add(process_model.graph.nodes[node].get('label', node))
        
        # Find missing activities (in model but not in log)
        missing_activities = model_activities - log_activities
        for activity in missing_activities:
            deviations["missing_activities"].append({
                "activity": activity,
                "type": "missing_in_log"
            })
        
        # Find extra activities (in log but not in model)
        extra_activities = log_activities - model_activities
        for activity in extra_activities:
            deviations["extra_activities"].append({
                "activity": activity,
                "type": "extra_in_log"
            })
        
        # Find cases that cannot be replayed
        for case_id in event_log.case_ids:
            trace = event_log.get_case_trace(case_id)
            if not self._can_replay_trace(trace, process_model):
                deviations["missing_cases"].append({
                    "case_id": case_id,
                    "trace": trace,
                    "type": "cannot_replay"
                })
        
        return deviations
