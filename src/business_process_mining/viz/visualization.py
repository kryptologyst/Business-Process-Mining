"""Visualization utilities for process mining results."""

import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import seaborn as sns
from collections import Counter, defaultdict

from ..data.event_log import EventLog
from ..mining.discovery import ProcessModel
from ..mining.conformance import ConformanceResult
from ..eval.metrics import ProcessKPIs, ModelQualityMetrics


class ProcessVisualizer:
    """Visualization utilities for process mining results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = "whitegrid") -> None:
        """Initialize visualizer with default settings.
        
        Args:
            figsize: Default figure size for matplotlib plots
            style: Seaborn style for plots
        """
        self.figsize = figsize
        plt.style.use('default')
        sns.set_style(style)
        
        # Set color palette
        self.colors = px.colors.qualitative.Set3
    
    def plot_process_model(
        self, 
        process_model: ProcessModel, 
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot a process model as a directed graph.
        
        Args:
            process_model: The process model to visualize
            title: Title for the plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create layout
        pos = nx.spring_layout(process_model.graph, seed=42, k=3, iterations=50)
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        
        for node in process_model.graph.nodes():
            node_type = process_model.graph.nodes[node].get('type', 'activity')
            if node_type == 'place':
                node_colors.append('#FFB6C1')  # Light pink for places
                node_sizes.append(300)
            else:
                node_colors.append('#87CEEB')  # Light blue for transitions
                node_sizes.append(500)
        
        nx.draw_networkx_nodes(
            process_model.graph, pos, 
            node_color=node_colors, 
            node_size=node_sizes,
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            process_model.graph, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            alpha=0.6
        )
        
        # Draw labels
        labels = {}
        for node in process_model.graph.nodes():
            label = process_model.graph.nodes[node].get('label', node)
            labels[node] = label
        
        nx.draw_networkx_labels(process_model.graph, pos, labels, font_size=10)
        
        # Draw edge labels (weights)
        edge_labels = nx.get_edge_attributes(process_model.graph, 'label')
        if edge_labels:
            nx.draw_networkx_edge_labels(process_model.graph, pos, edge_labels, font_size=8)
        
        ax.set_title(title or f"Process Model: {process_model.name}", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_case_duration_distribution(
        self, 
        event_log: EventLog, 
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot distribution of case durations.
        
        Args:
            event_log: The event log to analyze
            title: Title for the plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calculate case durations
        case_durations = []
        case_ids = []
        
        for case_id in event_log.case_ids:
            duration = event_log.get_case_duration(case_id)
            if duration is not None:
                case_durations.append(duration)
                case_ids.append(case_id)
        
        if case_durations:
            # Histogram
            ax1.hist(case_durations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Case Duration (hours)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Case Duration Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(case_durations, vert=True)
            ax2.set_ylabel('Case Duration (hours)')
            ax2.set_title('Case Duration Box Plot')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            mean_duration = np.mean(case_durations)
            median_duration = np.median(case_durations)
            std_duration = np.std(case_durations)
            
            ax1.axvline(mean_duration, color='red', linestyle='--', label=f'Mean: {mean_duration:.2f}h')
            ax1.axvline(median_duration, color='green', linestyle='--', label=f'Median: {median_duration:.2f}h')
            ax1.legend()
            
            ax2.text(1.1, mean_duration, f'Mean: {mean_duration:.2f}h', 
                    verticalalignment='center', fontsize=10)
            ax2.text(1.1, median_duration, f'Median: {median_duration:.2f}h', 
                    verticalalignment='center', fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'No duration data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=14)
            ax2.text(0.5, 0.5, 'No duration data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=14)
        
        plt.suptitle(title or 'Case Duration Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_activity_frequency(
        self, 
        event_log: EventLog, 
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot frequency of activities in the event log.
        
        Args:
            event_log: The event log to analyze
            title: Title for the plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count activity frequencies
        activity_counts = Counter()
        for event in event_log.events:
            activity_counts[event.activity] += 1
        
        if activity_counts:
            activities = list(activity_counts.keys())
            counts = list(activity_counts.values())
            
            # Bar plot
            bars = ax1.bar(activities, counts, color='lightcoral', alpha=0.7)
            ax1.set_xlabel('Activity')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Activity Frequency')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom')
            
            # Pie chart
            ax2.pie(counts, labels=activities, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Activity Distribution')
        else:
            ax1.text(0.5, 0.5, 'No activity data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=14)
            ax2.text(0.5, 0.5, 'No activity data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=14)
        
        plt.suptitle(title or 'Activity Frequency Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_resource_utilization(
        self, 
        event_log: EventLog, 
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot resource utilization analysis.
        
        Args:
            event_log: The event log to analyze
            title: Title for the plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calculate resource utilization
        resource_work = defaultdict(float)
        resource_activities = defaultdict(set)
        
        for event in event_log.events:
            if event.resource and event.duration:
                resource_work[event.resource] += event.duration
                resource_activities[event.resource].add(event.activity)
        
        if resource_work:
            resources = list(resource_work.keys())
            work_hours = list(resource_work.values())
            activity_counts = [len(resource_activities[res]) for res in resources]
            
            # Work hours bar plot
            bars1 = ax1.bar(resources, work_hours, color='lightgreen', alpha=0.7)
            ax1.set_xlabel('Resource')
            ax1.set_ylabel('Total Work Hours')
            ax1.set_title('Resource Work Hours')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, hours in zip(bars1, work_hours):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{hours:.1f}h', ha='center', va='bottom')
            
            # Activity diversity bar plot
            bars2 = ax2.bar(resources, activity_counts, color='lightblue', alpha=0.7)
            ax2.set_xlabel('Resource')
            ax2.set_ylabel('Number of Different Activities')
            ax2.set_title('Resource Activity Diversity')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, count in zip(bars2, activity_counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom')
        else:
            ax1.text(0.5, 0.5, 'No resource data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=14)
            ax2.text(0.5, 0.5, 'No resource data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=14)
        
        plt.suptitle(title or 'Resource Utilization Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_conformance_metrics(
        self, 
        conformance_result: ConformanceResult, 
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot conformance checking metrics.
        
        Args:
            conformance_result: The conformance checking results
            title: Title for the plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Main conformance metrics
        metrics = ['Fitness', 'Precision', 'Generalization', 'Simplicity']
        values = [
            conformance_result.fitness,
            conformance_result.precision,
            conformance_result.generalization,
            conformance_result.simplicity
        ]
        
        bars = ax1.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.7)
        ax1.set_ylabel('Score')
        ax1.set_title('Conformance Metrics')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Case fitness distribution
        case_fitness = [result['fitness'] for result in conformance_result.case_results.values()]
        if case_fitness:
            ax2.hist(case_fitness, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('Case Fitness')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Case Fitness Distribution')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No case data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=14)
        
        # Case duration vs fitness scatter
        case_durations = []
        case_fitness_values = []
        
        for case_id, result in conformance_result.case_results.items():
            if result['duration'] is not None:
                case_durations.append(result['duration'])
                case_fitness_values.append(result['fitness'])
        
        if case_durations and case_fitness_values:
            ax3.scatter(case_durations, case_fitness_values, alpha=0.6, color='purple')
            ax3.set_xlabel('Case Duration (hours)')
            ax3.set_ylabel('Case Fitness')
            ax3.set_title('Case Duration vs Fitness')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No duration/fitness data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes, fontsize=14)
        
        # Overall metrics summary
        overall_metrics = conformance_result.overall_metrics
        metric_names = list(overall_metrics.keys())
        metric_values = list(overall_metrics.values())
        
        bars = ax4.bar(metric_names, metric_values, color='lightcoral', alpha=0.7)
        ax4.set_ylabel('Score')
        ax4.set_title('Overall Metrics Summary')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle(title or 'Conformance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_process_kpis(
        self, 
        process_kpis: ProcessKPIs, 
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot key performance indicators.
        
        Args:
            process_kpis: The process KPIs to visualize
            title: Title for the plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Throughput metrics
        throughput_metrics = ['Mean Duration', 'Median Duration', 'Cases/Day', 'Cases/Week']
        throughput_values = [
            process_kpis.mean_case_duration,
            process_kpis.median_case_duration,
            process_kpis.cases_per_day,
            process_kpis.cases_per_week
        ]
        
        bars = ax1.bar(throughput_metrics, throughput_values, color='lightblue', alpha=0.7)
        ax1.set_ylabel('Value')
        ax1.set_title('Throughput Metrics')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, throughput_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Quality metrics
        quality_metrics = ['Completion Rate', 'Error Rate', 'Rework Rate']
        quality_values = [
            process_kpis.completion_rate,
            process_kpis.error_rate,
            process_kpis.rework_rate
        ]
        
        bars = ax2.bar(quality_metrics, quality_values, color='lightgreen', alpha=0.7)
        ax2.set_ylabel('Rate')
        ax2.set_title('Quality Metrics')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, quality_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Resource utilization
        if process_kpis.resource_utilization:
            resources = list(process_kpis.resource_utilization.keys())
            utilizations = list(process_kpis.resource_utilization.values())
            
            bars = ax3.bar(resources, utilizations, color='lightcoral', alpha=0.7)
            ax3.set_ylabel('Utilization')
            ax3.set_title('Resource Utilization')
            ax3.tick_params(axis='x', rotation=45)
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, utilizations):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, 'No resource data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes, fontsize=14)
        
        # Process complexity
        complexity_metrics = ['Process Variants', 'Cost per Case']
        complexity_values = [
            process_kpis.process_variants,
            process_kpis.cost_per_case
        ]
        
        bars = ax4.bar(complexity_metrics, complexity_values, color='lightyellow', alpha=0.7)
        ax4.set_ylabel('Value')
        ax4.set_title('Process Complexity')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, complexity_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.suptitle(title or 'Process Key Performance Indicators', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self, event_log: EventLog) -> go.Figure:
        """Create an interactive dashboard using Plotly.
        
        Args:
            event_log: The event log to analyze
            
        Returns:
            Plotly figure with multiple subplots
        """
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Activity Frequency', 'Case Duration Distribution', 
                          'Resource Utilization', 'Process Flow'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Activity frequency
        activity_counts = Counter(event.activity for event in event_log.events)
        if activity_counts:
            activities = list(activity_counts.keys())
            counts = list(activity_counts.values())
            
            fig.add_trace(
                go.Bar(x=activities, y=counts, name="Activity Frequency"),
                row=1, col=1
            )
        
        # Case duration distribution
        case_durations = []
        for case_id in event_log.case_ids:
            duration = event_log.get_case_duration(case_id)
            if duration is not None:
                case_durations.append(duration)
        
        if case_durations:
            fig.add_trace(
                go.Histogram(x=case_durations, name="Case Duration", nbinsx=20),
                row=1, col=2
            )
        
        # Resource utilization
        resource_work = defaultdict(float)
        for event in event_log.events:
            if event.resource and event.duration:
                resource_work[event.resource] += event.duration
        
        if resource_work:
            resources = list(resource_work.keys())
            work_hours = list(resource_work.values())
            
            fig.add_trace(
                go.Bar(x=resources, y=work_hours, name="Resource Work Hours"),
                row=2, col=1
            )
        
        # Process flow (simplified)
        transitions = []
        for case_id in event_log.case_ids:
            trace = event_log.get_case_trace(case_id)
            for i in range(len(trace) - 1):
                transitions.append((trace[i], trace[i + 1]))
        
        if transitions:
            transition_counts = Counter(transitions)
            sources = [t[0] for t in transition_counts.keys()]
            targets = [t[1] for t in transition_counts.keys()]
            values = list(transition_counts.values())
            
            fig.add_trace(
                go.Scatter(x=sources, y=targets, mode='markers+lines',
                          marker=dict(size=values, sizemode='diameter'),
                          name="Process Flow"),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Process Mining Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig
