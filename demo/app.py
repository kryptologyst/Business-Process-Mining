"""Streamlit demo application for Business Process Mining."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from business_process_mining import (
    EventLog, EventLogGenerator, ProcessDiscovery, 
    ConformanceChecker, ProcessMetrics, ProcessVisualizer
)

# Page configuration
st.set_page_config(
    page_title="Business Process Mining Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Business Process Mining Demo</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <h4>⚠️ DISCLAIMER</h4>
        <p><strong>This software is for research and educational purposes only.</strong></p>
        <p>It should not be used for automated decision-making without human review. 
        All results and recommendations should be validated by domain experts before 
        implementing any process changes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Process type selection
    process_type = st.sidebar.selectbox(
        "Select Process Type",
        ["Simple Approval", "Loan Processing", "Manufacturing"],
        help="Choose the type of business process to analyze"
    )
    
    # Parameters
    num_cases = st.sidebar.slider(
        "Number of Cases",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Number of process instances to generate"
    )
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime(2023, 1, 1).date(),
        help="Start date for event generation"
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime(2023, 12, 31).date(),
        help="End date for event generation"
    )
    
    # Algorithm selection
    st.sidebar.subheader("Process Discovery")
    use_alpha = st.sidebar.checkbox("Alpha Miner", value=True)
    use_heuristics = st.sidebar.checkbox("Heuristics Miner", value=True)
    use_inductive = st.sidebar.checkbox("Inductive Miner", value=True)
    
    # Generate data button
    if st.sidebar.button("Generate Process Data", type="primary"):
        generate_and_analyze(process_type, num_cases, start_date, end_date, 
                           use_alpha, use_heuristics, use_inductive)
    
    # Default analysis
    if 'event_log' not in st.session_state:
        st.info("👈 Configure parameters in the sidebar and click 'Generate Process Data' to start the analysis.")
        show_sample_data()
    else:
        show_analysis_results()

def generate_and_analyze(process_type, num_cases, start_date, end_date, 
                        use_alpha, use_heuristics, use_inductive):
    """Generate event log and perform analysis."""
    
    with st.spinner("Generating process data and performing analysis..."):
        # Initialize components
        generator = EventLogGenerator(seed=42)
        discovery = ProcessDiscovery(seed=42)
        conformance = ConformanceChecker(seed=42)
        metrics = ProcessMetrics(seed=42)
        visualizer = ProcessVisualizer()
        
        # Generate event log
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time())
        
        if process_type == "Simple Approval":
            event_log = generator.generate_simple_approval_process(
                num_cases, start_datetime, end_datetime
            )
        elif process_type == "Loan Processing":
            event_log = generator.generate_complex_loan_process(
                num_cases, start_datetime, end_datetime
            )
        else:  # Manufacturing
            event_log = generator.generate_manufacturing_process(
                num_cases, start_datetime, end_datetime
            )
        
        # Store in session state
        st.session_state.event_log = event_log
        st.session_state.process_type = process_type
        st.session_state.num_cases = num_cases
        
        # Discover process models
        models = []
        if use_alpha:
            try:
                alpha_model = discovery.discover_alpha_miner(event_log)
                models.append(alpha_model)
            except Exception as e:
                st.warning(f"Alpha Miner failed: {e}")
        
        if use_heuristics:
            try:
                heuristics_model = discovery.discover_heuristics_miner(event_log)
                models.append(heuristics_model)
            except Exception as e:
                st.warning(f"Heuristics Miner failed: {e}")
        
        if use_inductive:
            try:
                inductive_model = discovery.discover_inductive_miner(event_log)
                models.append(inductive_model)
            except Exception as e:
                st.warning(f"Inductive Miner failed: {e}")
        
        # If no advanced models, use simple graph
        if not models:
            simple_model = discovery.discover_simple_graph(event_log)
            models.append(simple_model)
        
        st.session_state.models = models
        
        # Perform conformance checking
        conformance_results = {}
        for model in models:
            result = conformance.check_conformance(event_log, model)
            conformance_results[model.name] = result
        
        st.session_state.conformance_results = conformance_results
        
        # Calculate metrics
        process_kpis = metrics.calculate_process_kpis(event_log)
        st.session_state.process_kpis = process_kpis
        
        # Create leaderboard
        leaderboard = metrics.create_leaderboard(models, event_log)
        st.session_state.leaderboard = leaderboard
        
        # Bottleneck analysis
        bottleneck_analysis = metrics.calculate_bottleneck_analysis(event_log)
        st.session_state.bottleneck_analysis = bottleneck_analysis
        
        # Resource efficiency
        resource_efficiency = metrics.calculate_resource_efficiency(event_log)
        st.session_state.resource_efficiency = resource_efficiency
        
        st.success("✅ Analysis completed successfully!")

def show_sample_data():
    """Show sample data and information."""
    
    st.subheader("📊 Sample Process Data")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Case ID': ['Case_0001', 'Case_0001', 'Case_0001', 'Case_0002', 'Case_0002'],
        'Activity': ['Start', 'Check Form', 'Approve', 'Start', 'Reject'],
        'Timestamp': ['2023-01-15 09:00:00', '2023-01-15 10:30:00', '2023-01-15 14:00:00', 
                     '2023-01-16 08:00:00', '2023-01-16 11:00:00'],
        'Resource': ['System', 'Clerk_A', 'Manager_A', 'System', 'Manager_B'],
        'Duration (hours)': [0.1, 1.5, 0.5, 0.1, 0.3],
        'Cost': [0.0, 25.0, 50.0, 0.0, 30.0]
    })
    
    st.dataframe(sample_data, use_container_width=True)
    
    st.subheader("🎯 What is Process Mining?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔍 Process Discovery**
        - Extract process models from event logs
        - Identify actual vs. designed processes
        - Discover process variants and patterns
        """)
    
    with col2:
        st.markdown("""
        **📊 Conformance Checking**
        - Compare actual vs. expected behavior
        - Identify deviations and violations
        - Measure process compliance
        """)
    
    with col3:
        st.markdown("""
        **⚡ Process Enhancement**
        - Identify bottlenecks and inefficiencies
        - Optimize resource allocation
        - Improve process performance
        """)
    
    st.subheader("🚀 Features")
    
    features = [
        "Multiple process discovery algorithms (Alpha, Heuristics, Inductive Miner)",
        "Comprehensive conformance checking and metrics",
        "Bottleneck and resource efficiency analysis",
        "Interactive visualizations and dashboards",
        "Process performance KPIs and benchmarking",
        "Synthetic data generation for testing"
    ]
    
    for feature in features:
        st.markdown(f"✅ {feature}")

def show_analysis_results():
    """Show the analysis results."""
    
    event_log = st.session_state.event_log
    models = st.session_state.models
    conformance_results = st.session_state.conformance_results
    process_kpis = st.session_state.process_kpis
    leaderboard = st.session_state.leaderboard
    bottleneck_analysis = st.session_state.bottleneck_analysis
    resource_efficiency = st.session_state.resource_efficiency
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "🔍 Process Models", "📈 Performance", 
        "🚧 Bottlenecks", "👥 Resources", "🏆 Leaderboard"
    ])
    
    with tab1:
        show_overview(event_log, process_kpis)
    
    with tab2:
        show_process_models(models, conformance_results)
    
    with tab3:
        show_performance_metrics(process_kpis, event_log)
    
    with tab4:
        show_bottleneck_analysis(bottleneck_analysis)
    
    with tab5:
        show_resource_analysis(resource_efficiency)
    
    with tab6:
        show_leaderboard(leaderboard)

def show_overview(event_log, process_kpis):
    """Show overview metrics and summary."""
    
    st.subheader("📊 Process Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Cases",
            value=len(event_log.case_ids),
            delta=None
        )
    
    with col2:
        st.metric(
            label="Mean Duration",
            value=f"{process_kpis.mean_case_duration:.2f} hours",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Completion Rate",
            value=f"{process_kpis.completion_rate:.1%}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Process Variants",
            value=process_kpis.process_variants,
            delta=None
        )
    
    # Process flow visualization
    st.subheader("🔄 Process Flow")
    
    # Create process flow chart
    transitions = []
    for case_id in event_log.case_ids:
        trace = event_log.get_case_trace(case_id)
        for i in range(len(trace) - 1):
            transitions.append((trace[i], trace[i + 1]))
    
    if transitions:
        transition_counts = pd.Series(transitions).value_counts().reset_index()
        transition_counts.columns = ['Transition', 'Count']
        
        # Create Sankey diagram
        sources = []
        targets = []
        values = []
        
        for _, row in transition_counts.iterrows():
            source, target = row['Transition']
            sources.append(source)
            targets.append(target)
            values.append(row['Count'])
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=list(set(sources + targets))
            ),
            link=dict(
                source=[list(set(sources + targets)).index(s) for s in sources],
                target=[list(set(sources + targets)).index(t) for t in targets],
                value=values
            )
        )])
        
        fig.update_layout(title_text="Process Flow Diagram", font_size=10)
        st.plotly_chart(fig, use_container_width=True)
    
    # Activity frequency
    st.subheader("📈 Activity Frequency")
    
    activity_counts = pd.Series([event.activity for event in event_log.events]).value_counts()
    
    fig = px.bar(
        x=activity_counts.index,
        y=activity_counts.values,
        title="Activity Frequency Distribution",
        labels={'x': 'Activity', 'y': 'Frequency'}
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def show_process_models(models, conformance_results):
    """Show discovered process models."""
    
    st.subheader("🔍 Discovered Process Models")
    
    for i, model in enumerate(models):
        with st.expander(f"Model {i+1}: {model.name} ({model.algorithm})"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Information:**")
                st.write(f"- Algorithm: {model.algorithm}")
                st.write(f"- Nodes: {model.graph.number_of_nodes()}")
                st.write(f"- Edges: {model.graph.number_of_edges()}")
                
                if model.parameters:
                    st.write("**Parameters:**")
                    for key, value in model.parameters.items():
                        if key not in ['initial_marking', 'final_marking', 'process_tree']:
                            st.write(f"- {key}: {value}")
            
            with col2:
                if model.name in conformance_results:
                    result = conformance_results[model.name]
                    st.write("**Conformance Metrics:**")
                    st.write(f"- Fitness: {result.fitness:.3f}")
                    st.write(f"- Precision: {result.precision:.3f}")
                    st.write(f"- Generalization: {result.generalization:.3f}")
                    st.write(f"- Simplicity: {result.simplicity:.3f}")
            
            # Model visualization
            st.write("**Process Model:**")
            
            # Create a simple text representation
            edges = list(model.graph.edges())
            if edges:
                st.write("**Process Flow:**")
                for source, target in edges[:10]:  # Show first 10 edges
                    source_label = model.graph.nodes[source].get('label', source)
                    target_label = model.graph.nodes[target].get('label', target)
                    st.write(f"  {source_label} → {target_label}")
                
                if len(edges) > 10:
                    st.write(f"  ... and {len(edges) - 10} more transitions")
            else:
                st.write("No process flow detected.")

def show_performance_metrics(process_kpis, event_log):
    """Show performance metrics and KPIs."""
    
    st.subheader("📈 Performance Metrics")
    
    # Throughput metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏱️ Throughput")
        
        throughput_data = {
            'Metric': ['Mean Duration', 'Median Duration', 'Cases/Day', 'Cases/Week'],
            'Value': [
                f"{process_kpis.mean_case_duration:.2f} hours",
                f"{process_kpis.median_case_duration:.2f} hours",
                f"{process_kpis.cases_per_day:.2f}",
                f"{process_kpis.cases_per_week:.2f}"
            ]
        }
        
        st.table(pd.DataFrame(throughput_data))
    
    with col2:
        st.subheader("✅ Quality")
        
        quality_data = {
            'Metric': ['Completion Rate', 'Error Rate', 'Rework Rate'],
            'Value': [
                f"{process_kpis.completion_rate:.1%}",
                f"{process_kpis.error_rate:.1%}",
                f"{process_kpis.rework_rate:.1%}"
            ]
        }
        
        st.table(pd.DataFrame(quality_data))
    
    # Cost analysis
    st.subheader("💰 Cost Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Cost per Case",
            value=f"${process_kpis.cost_per_case:.2f}",
            delta=None
        )
    
    with col2:
        total_cost = sum(event.cost for event in event_log.events if event.cost)
        st.metric(
            label="Total Cost",
            value=f"${total_cost:.2f}",
            delta=None
        )
    
    # Process complexity
    st.subheader("🔧 Process Complexity")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Process Variants", process_kpis.process_variants)
    
    with col2:
        st.metric("Unique Activities", len(event_log.activities))
    
    with col3:
        st.metric("Resources Used", len(event_log.resources))

def show_bottleneck_analysis(bottleneck_analysis):
    """Show bottleneck analysis results."""
    
    st.subheader("🚧 Bottleneck Analysis")
    
    if bottleneck_analysis['activity_analysis']:
        # Top bottlenecks
        st.subheader("🔴 Top Bottlenecks")
        
        top_bottlenecks = bottleneck_analysis['top_bottlenecks'][:5]
        
        for i, activity in enumerate(top_bottlenecks, 1):
            analysis = bottleneck_analysis['activity_analysis'][activity]
            
            with st.expander(f"{i}. {activity}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Bottleneck Score", f"{analysis['bottleneck_score']:.2f}")
                
                with col2:
                    st.metric("Mean Duration", f"{analysis['mean_duration']:.2f} hours")
                
                with col3:
                    st.metric("Frequency", analysis['count'])
                
                st.write(f"**Details:**")
                st.write(f"- Median Duration: {analysis['median_duration']:.2f} hours")
                st.write(f"- Standard Deviation: {analysis['std_duration']:.2f} hours")
                st.write(f"- Total Duration: {analysis['total_duration']:.2f} hours")
                st.write(f"- Cases Affected: {analysis['case_count']}")
        
        # Bottleneck visualization
        st.subheader("📊 Bottleneck Visualization")
        
        activities = list(bottleneck_analysis['activity_analysis'].keys())
        scores = [bottleneck_analysis['activity_analysis'][a]['bottleneck_score'] for a in activities]
        
        fig = px.bar(
            x=activities,
            y=scores,
            title="Bottleneck Scores by Activity",
            labels={'x': 'Activity', 'y': 'Bottleneck Score'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Duration analysis
        durations = [bottleneck_analysis['activity_analysis'][a]['mean_duration'] for a in activities]
        
        fig = px.bar(
            x=activities,
            y=durations,
            title="Mean Duration by Activity",
            labels={'x': 'Activity', 'y': 'Mean Duration (hours)'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No bottleneck analysis data available.")

def show_resource_analysis(resource_efficiency):
    """Show resource efficiency analysis."""
    
    st.subheader("👥 Resource Efficiency Analysis")
    
    if resource_efficiency['resource_analysis']:
        # Top resources
        st.subheader("⭐ Top Resources")
        
        top_resources = resource_efficiency['top_resources'][:5]
        
        for i, resource in enumerate(top_resources, 1):
            analysis = resource_efficiency['resource_analysis'][resource]
            
            with st.expander(f"{i}. {resource}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Utilization", f"{analysis['utilization']:.1%}")
                
                with col2:
                    st.metric("Total Work", f"{analysis['total_work']:.2f} hours")
                
                with col3:
                    st.metric("Activity Diversity", analysis['activity_diversity'])
                
                st.write(f"**Details:**")
                st.write(f"- Cost Efficiency: {analysis['cost_efficiency']:.2f}")
                st.write(f"- Case Efficiency: {analysis['case_efficiency']:.2f}")
                st.write(f"- Activity Count: {analysis['activity_count']}")
                st.write(f"- Case Count: {analysis['case_count']}")
                st.write(f"- Total Cost: ${analysis['total_cost']:.2f}")
        
        # Resource utilization visualization
        st.subheader("📊 Resource Utilization")
        
        resources = list(resource_efficiency['resource_analysis'].keys())
        utilizations = [resource_efficiency['resource_analysis'][r]['utilization'] for r in resources]
        
        fig = px.bar(
            x=resources,
            y=utilizations,
            title="Resource Utilization",
            labels={'x': 'Resource', 'y': 'Utilization'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Work hours distribution
        work_hours = [resource_efficiency['resource_analysis'][r]['total_work'] for r in resources]
        
        fig = px.pie(
            values=work_hours,
            names=resources,
            title="Work Hours Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Resources",
                resource_efficiency['total_resources']
            )
        
        with col2:
            st.metric(
                "Average Utilization",
                f"{resource_efficiency['avg_utilization']:.1%}"
            )
        
        with col3:
            st.metric(
                "Total Work Hours",
                f"{resource_efficiency['total_work']:.2f}"
            )
    
    else:
        st.info("No resource efficiency data available.")

def show_leaderboard(leaderboard):
    """Show model comparison leaderboard."""
    
    st.subheader("🏆 Model Comparison Leaderboard")
    
    if not leaderboard.empty:
        # Display leaderboard
        st.dataframe(leaderboard, use_container_width=True)
        
        # Best model
        best_model = leaderboard.iloc[0]
        st.success(f"🏆 Best Model: {best_model['Model']} with composite score {best_model['Composite_Score']:.3f}")
        
        # Model comparison chart
        st.subheader("📊 Model Comparison")
        
        metrics = ['Fitness', 'Precision', 'Generalization', 'Simplicity', 'Composite_Score']
        
        fig = go.Figure()
        
        for _, row in leaderboard.iterrows():
            fig.add_trace(go.Scatter(
                x=metrics,
                y=[row[metric] for metric in metrics],
                mode='lines+markers',
                name=row['Model'],
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Metrics",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model complexity analysis
        st.subheader("🔧 Model Complexity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                leaderboard,
                x='Nodes',
                y='Edges',
                size='Composite_Score',
                color='Model',
                title="Model Size vs Performance",
                labels={'Nodes': 'Number of Nodes', 'Edges': 'Number of Edges'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                leaderboard,
                x='Model',
                y='Complexity',
                title="Model Complexity",
                labels={'Complexity': 'Edge-to-Node Ratio'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No leaderboard data available.")

if __name__ == "__main__":
    main()
