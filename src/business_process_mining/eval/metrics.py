"""Comprehensive evaluation metrics for process mining."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import networkx as nx

from ..data.event_log import EventLog
from ..mining.discovery import ProcessModel
from ..mining.conformance import ConformanceResult


@dataclass
class ProcessKPIs:
    """Key Performance Indicators for business processes."""
    
    # Throughput metrics
    mean_case_duration: float
    median_case_duration: float
    case_duration_std: float
    cases_per_day: float
    cases_per_week: float
    
    # Quality metrics
    completion_rate: float
    error_rate: float
    rework_rate: float
    
    # Resource metrics
    resource_utilization: Dict[str, float]
    bottleneck_activities: List[str]
    cost_per_case: float
    
    # Process metrics
    process_variants: int
    most_common_path: List[str]
    path_frequency: Dict[Tuple[str, ...], int]


@dataclass
class ModelQualityMetrics:
    """Quality metrics for process models."""
    
    # Conformance metrics
    fitness: float
    precision: float
    generalization: float
    simplicity: float
    
    # Model complexity
    num_nodes: int
    num_edges: int
    cyclomatic_complexity: int
    model_size: int
    
    # Coverage metrics
    activity_coverage: float
    trace_coverage: float
    case_coverage: float


class ProcessMetrics:
    """Comprehensive evaluation metrics for process mining."""
    
    def __init__(self, seed: int = 42) -> None:
        """Initialize metrics calculator with random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
    
    def calculate_process_kpis(self, event_log: EventLog) -> ProcessKPIs:
        """Calculate key performance indicators for the process.
        
        Args:
            event_log: The event log to analyze
            
        Returns:
            ProcessKPIs with calculated metrics
        """
        # Calculate case durations
        case_durations = []
        for case_id in event_log.case_ids:
            duration = event_log.get_case_duration(case_id)
            if duration is not None:
                case_durations.append(duration)
        
        mean_case_duration = np.mean(case_durations) if case_durations else 0.0
        median_case_duration = np.median(case_durations) if case_durations else 0.0
        case_duration_std = np.std(case_durations) if case_durations else 0.0
        
        # Calculate throughput
        if event_log.events:
            time_span = (max(e.timestamp for e in event_log.events) - 
                        min(e.timestamp for e in event_log.events)).days
            if time_span > 0:
                cases_per_day = len(event_log.case_ids) / time_span
                cases_per_week = cases_per_day * 7
            else:
                cases_per_day = 0.0
                cases_per_week = 0.0
        else:
            cases_per_day = 0.0
            cases_per_week = 0.0
        
        # Calculate completion rate
        completed_cases = 0
        for case_id in event_log.case_ids:
            trace = event_log.get_case_trace(case_id)
            if trace and trace[-1] in ['End', 'Complete', 'Finished']:
                completed_cases += 1
        
        completion_rate = completed_cases / len(event_log.case_ids) if event_log.case_ids else 0.0
        
        # Calculate error rate (cases with 'Reject' or 'Error' activities)
        error_cases = 0
        for case_id in event_log.case_ids:
            trace = event_log.get_case_trace(case_id)
            if any(activity in ['Reject', 'Error', 'Failed'] for activity in trace):
                error_cases += 1
        
        error_rate = error_cases / len(event_log.case_ids) if event_log.case_ids else 0.0
        
        # Calculate rework rate (cases with repeated activities)
        rework_cases = 0
        for case_id in event_log.case_ids:
            trace = event_log.get_case_trace(case_id)
            if len(trace) != len(set(trace)):
                rework_cases += 1
        
        rework_rate = rework_cases / len(event_log.case_ids) if event_log.case_ids else 0.0
        
        # Calculate resource utilization
        resource_work = defaultdict(float)
        for event in event_log.events:
            if event.resource and event.duration:
                resource_work[event.resource] += event.duration
        
        total_work = sum(resource_work.values())
        resource_utilization = {
            resource: work / total_work if total_work > 0 else 0
            for resource, work in resource_work.items()
        }
        
        # Find bottleneck activities
        activity_durations = defaultdict(list)
        for event in event_log.events:
            if event.duration:
                activity_durations[event.activity].append(event.duration)
        
        bottleneck_activities = []
        for activity, durations in activity_durations.items():
            if len(durations) > 1:
                mean_duration = np.mean(durations)
                if mean_duration > np.percentile([np.mean(durs) for durs in activity_durations.values()], 75):
                    bottleneck_activities.append(activity)
        
        # Calculate cost per case
        total_cost = sum(event.cost for event in event_log.events if event.cost)
        cost_per_case = total_cost / len(event_log.case_ids) if event_log.case_ids else 0.0
        
        # Calculate process variants
        unique_traces = set()
        for case_id in event_log.case_ids:
            trace = tuple(event_log.get_case_trace(case_id))
            unique_traces.add(trace)
        
        process_variants = len(unique_traces)
        
        # Find most common path
        trace_counts = Counter(tuple(event_log.get_case_trace(case_id)) for case_id in event_log.case_ids)
        most_common_path = list(trace_counts.most_common(1)[0][0]) if trace_counts else []
        
        # Path frequency
        path_frequency = {path: count for path, count in trace_counts.items()}
        
        return ProcessKPIs(
            mean_case_duration=mean_case_duration,
            median_case_duration=median_case_duration,
            case_duration_std=case_duration_std,
            cases_per_day=cases_per_day,
            cases_per_week=cases_per_week,
            completion_rate=completion_rate,
            error_rate=error_rate,
            rework_rate=rework_rate,
            resource_utilization=resource_utilization,
            bottleneck_activities=bottleneck_activities,
            cost_per_case=cost_per_case,
            process_variants=process_variants,
            most_common_path=most_common_path,
            path_frequency=path_frequency
        )
    
    def calculate_model_quality(self, process_model: ProcessModel) -> ModelQualityMetrics:
        """Calculate quality metrics for a process model.
        
        Args:
            process_model: The process model to evaluate
            
        Returns:
            ModelQualityMetrics with calculated metrics
        """
        # Basic model metrics
        num_nodes = process_model.graph.number_of_nodes()
        num_edges = process_model.graph.number_of_edges()
        
        # Cyclomatic complexity
        try:
            cyclomatic_complexity = num_edges - num_nodes + 2
        except:
            cyclomatic_complexity = 0
        
        # Model size (total number of elements)
        model_size = num_nodes + num_edges
        
        # Get conformance metrics if available
        fitness = process_model.fitness or 0.0
        precision = process_model.precision or 0.0
        generalization = process_model.generalization or 0.0
        simplicity = process_model.simplicity or 0.0
        
        # Coverage metrics (simplified)
        activity_coverage = 1.0  # Placeholder - would need log comparison
        trace_coverage = 1.0     # Placeholder - would need log comparison
        case_coverage = 1.0      # Placeholder - would need log comparison
        
        return ModelQualityMetrics(
            fitness=fitness,
            precision=precision,
            generalization=generalization,
            simplicity=simplicity,
            num_nodes=num_nodes,
            num_edges=num_edges,
            cyclomatic_complexity=cyclomatic_complexity,
            model_size=model_size,
            activity_coverage=activity_coverage,
            trace_coverage=trace_coverage,
            case_coverage=case_coverage
        )
    
    def create_leaderboard(
        self, 
        models: List[ProcessModel], 
        event_log: EventLog
    ) -> pd.DataFrame:
        """Create a leaderboard comparing different process models.
        
        Args:
            models: List of process models to compare
            event_log: Event log for evaluation
            
        Returns:
            DataFrame with model comparison results
        """
        results = []
        
        for model in models:
            # Calculate basic metrics
            quality_metrics = self.calculate_model_quality(model)
            process_kpis = self.calculate_process_kpis(event_log)
            
            # Calculate additional metrics
            model_complexity = model.graph.number_of_edges() / max(1, model.graph.number_of_nodes())
            
            results.append({
                'Model': model.name,
                'Algorithm': model.algorithm,
                'Fitness': quality_metrics.fitness,
                'Precision': quality_metrics.precision,
                'Generalization': quality_metrics.generalization,
                'Simplicity': quality_metrics.simplicity,
                'Nodes': quality_metrics.num_nodes,
                'Edges': quality_metrics.num_edges,
                'Complexity': model_complexity,
                'Cyclomatic_Complexity': quality_metrics.cyclomatic_complexity,
                'Model_Size': quality_metrics.model_size
            })
        
        df = pd.DataFrame(results)
        
        # Calculate composite score
        df['Composite_Score'] = (
            df['Fitness'] * 0.3 +
            df['Precision'] * 0.3 +
            df['Generalization'] * 0.2 +
            df['Simplicity'] * 0.2
        )
        
        # Sort by composite score
        df = df.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
        
        return df
    
    def calculate_bottleneck_analysis(self, event_log: EventLog) -> Dict[str, Any]:
        """Perform detailed bottleneck analysis.
        
        Args:
            event_log: The event log to analyze
            
        Returns:
            Dictionary with bottleneck analysis results
        """
        # Activity-level analysis
        activity_stats = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0.0,
            'durations': [],
            'resources': set(),
            'cases': set()
        })
        
        for event in event_log.events:
            activity = event.activity
            activity_stats[activity]['count'] += 1
            activity_stats[activity]['cases'].add(event.case_id)
            
            if event.duration:
                activity_stats[activity]['total_duration'] += event.duration
                activity_stats[activity]['durations'].append(event.duration)
            
            if event.resource:
                activity_stats[activity]['resources'].add(event.resource)
        
        # Calculate bottleneck metrics
        bottleneck_analysis = {}
        for activity, stats in activity_stats.items():
            if stats['durations']:
                mean_duration = np.mean(stats['durations'])
                std_duration = np.std(stats['durations'])
                median_duration = np.median(stats['durations'])
                
                # Bottleneck score based on duration and frequency
                bottleneck_score = mean_duration * stats['count']
                
                bottleneck_analysis[activity] = {
                    'count': stats['count'],
                    'mean_duration': mean_duration,
                    'median_duration': median_duration,
                    'std_duration': std_duration,
                    'total_duration': stats['total_duration'],
                    'bottleneck_score': bottleneck_score,
                    'resource_count': len(stats['resources']),
                    'case_count': len(stats['cases']),
                    'avg_duration_per_case': stats['total_duration'] / len(stats['cases']) if stats['cases'] else 0
                }
        
        # Sort by bottleneck score
        sorted_bottlenecks = sorted(
            bottleneck_analysis.items(),
            key=lambda x: x[1]['bottleneck_score'],
            reverse=True
        )
        
        return {
            'activity_analysis': dict(sorted_bottlenecks),
            'top_bottlenecks': [item[0] for item in sorted_bottlenecks[:5]],
            'total_activities': len(bottleneck_analysis),
            'avg_activity_duration': np.mean([stats['mean_duration'] for stats in bottleneck_analysis.values()])
        }
    
    def calculate_resource_efficiency(self, event_log: EventLog) -> Dict[str, Any]:
        """Calculate resource efficiency metrics.
        
        Args:
            event_log: The event log to analyze
            
        Returns:
            Dictionary with resource efficiency analysis
        """
        resource_stats = defaultdict(lambda: {
            'total_work': 0.0,
            'activity_count': 0,
            'cases': set(),
            'activities': set(),
            'cost': 0.0
        })
        
        for event in event_log.events:
            if event.resource:
                resource = event.resource
                resource_stats[resource]['cases'].add(event.case_id)
                resource_stats[resource]['activities'].add(event.activity)
                resource_stats[resource]['activity_count'] += 1
                
                if event.duration:
                    resource_stats[resource]['total_work'] += event.duration
                
                if event.cost:
                    resource_stats[resource]['cost'] += event.cost
        
        # Calculate efficiency metrics
        efficiency_analysis = {}
        total_work = sum(stats['total_work'] for stats in resource_stats.values())
        total_cost = sum(stats['cost'] for stats in resource_stats.values())
        
        for resource, stats in resource_stats.items():
            utilization = stats['total_work'] / total_work if total_work > 0 else 0
            cost_efficiency = stats['total_work'] / stats['cost'] if stats['cost'] > 0 else 0
            case_efficiency = stats['total_work'] / len(stats['cases']) if stats['cases'] else 0
            
            efficiency_analysis[resource] = {
                'total_work': stats['total_work'],
                'utilization': utilization,
                'cost_efficiency': cost_efficiency,
                'case_efficiency': case_efficiency,
                'activity_count': stats['activity_count'],
                'case_count': len(stats['cases']),
                'activity_diversity': len(stats['activities']),
                'total_cost': stats['cost']
            }
        
        # Sort by utilization
        sorted_resources = sorted(
            efficiency_analysis.items(),
            key=lambda x: x[1]['utilization'],
            reverse=True
        )
        
        return {
            'resource_analysis': dict(sorted_resources),
            'top_resources': [item[0] for item in sorted_resources[:5]],
            'total_resources': len(efficiency_analysis),
            'avg_utilization': np.mean([stats['utilization'] for stats in efficiency_analysis.values()]),
            'total_work': total_work,
            'total_cost': total_cost
        }
