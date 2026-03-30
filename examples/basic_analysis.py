#!/usr/bin/env python3
"""Basic process mining analysis example."""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from business_process_mining import (
    EventLogGenerator, ProcessDiscovery, ConformanceChecker, 
    ProcessMetrics, ProcessVisualizer
)


def main():
    """Run basic process mining analysis."""
    print("🔍 Business Process Mining - Basic Analysis Example")
    print("=" * 60)
    
    # Initialize components
    print("📊 Initializing components...")
    generator = EventLogGenerator(seed=42)
    discovery = ProcessDiscovery(seed=42)
    conformance = ConformanceChecker(seed=42)
    metrics = ProcessMetrics(seed=42)
    visualizer = ProcessVisualizer()
    
    # Generate event log
    print("📈 Generating event log...")
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    event_log = generator.generate_simple_approval_process(
        num_cases=50,
        start_date=start_date,
        end_date=end_date
    )
    
    print(f"✅ Generated {len(event_log.case_ids)} cases with {len(event_log.events)} events")
    print(f"📋 Activities: {', '.join(event_log.activities)}")
    print(f"👥 Resources: {', '.join(event_log.resources)}")
    
    # Discover process models
    print("\n🔍 Discovering process models...")
    
    # Simple graph
    simple_model = discovery.discover_simple_graph(event_log)
    print(f"✅ Simple Graph: {simple_model.graph.number_of_nodes()} nodes, {simple_model.graph.number_of_edges()} edges")
    
    # Alpha miner (may fail, that's okay)
    try:
        alpha_model = discovery.discover_alpha_miner(event_log)
        print(f"✅ Alpha Miner: {alpha_model.graph.number_of_nodes()} nodes, {alpha_model.graph.number_of_edges()} edges")
    except Exception as e:
        print(f"⚠️ Alpha Miner failed: {e}")
        alpha_model = None
    
    # Heuristics miner (may fail, that's okay)
    try:
        heuristics_model = discovery.discover_heuristics_miner(event_log)
        print(f"✅ Heuristics Miner: {heuristics_model.graph.number_of_nodes()} nodes, {heuristics_model.graph.number_of_edges()} edges")
    except Exception as e:
        print(f"⚠️ Heuristics Miner failed: {e}")
        heuristics_model = None
    
    # Collect successful models
    models = [simple_model]
    if alpha_model:
        models.append(alpha_model)
    if heuristics_model:
        models.append(heuristics_model)
    
    # Perform conformance checking
    print("\n📏 Performing conformance checking...")
    conformance_results = {}
    
    for model in models:
        result = conformance.check_conformance(event_log, model)
        conformance_results[model.name] = result
        print(f"✅ {model.name}:")
        print(f"   Fitness: {result.fitness:.3f}")
        print(f"   Precision: {result.precision:.3f}")
        print(f"   Generalization: {result.generalization:.3f}")
        print(f"   Simplicity: {result.simplicity:.3f}")
    
    # Calculate process KPIs
    print("\n📊 Calculating process KPIs...")
    process_kpis = metrics.calculate_process_kpis(event_log)
    
    print("📈 Process Performance:")
    print(f"   Mean case duration: {process_kpis.mean_case_duration:.2f} hours")
    print(f"   Median case duration: {process_kpis.median_case_duration:.2f} hours")
    print(f"   Cases per day: {process_kpis.cases_per_day:.2f}")
    print(f"   Completion rate: {process_kpis.completion_rate:.1%}")
    print(f"   Error rate: {process_kpis.error_rate:.1%}")
    print(f"   Process variants: {process_kpis.process_variants}")
    print(f"   Cost per case: ${process_kpis.cost_per_case:.2f}")
    
    # Analyze bottlenecks
    print("\n🚧 Analyzing bottlenecks...")
    bottleneck_analysis = metrics.calculate_bottleneck_analysis(event_log)
    
    if bottleneck_analysis['top_bottlenecks']:
        print("🔴 Top bottlenecks:")
        for i, bottleneck in enumerate(bottleneck_analysis['top_bottlenecks'][:3], 1):
            analysis = bottleneck_analysis['activity_analysis'][bottleneck]
            print(f"   {i}. {bottleneck}: {analysis['mean_duration']:.2f}h mean duration, {analysis['count']} occurrences")
    else:
        print("✅ No significant bottlenecks identified")
    
    # Analyze resource efficiency
    print("\n👥 Analyzing resource efficiency...")
    resource_efficiency = metrics.calculate_resource_efficiency(event_log)
    
    if resource_efficiency['resource_analysis']:
        print("⭐ Top resources by utilization:")
        for i, resource in enumerate(resource_efficiency['top_resources'][:3], 1):
            analysis = resource_efficiency['resource_analysis'][resource]
            print(f"   {i}. {resource}: {analysis['utilization']:.1%} utilization, {analysis['total_work']:.2f}h total work")
    else:
        print("✅ No resource data available")
    
    # Create leaderboard
    print("\n🏆 Model comparison leaderboard...")
    leaderboard = metrics.create_leaderboard(models, event_log)
    
    print("\n📊 Model Performance Summary:")
    for _, row in leaderboard.iterrows():
        print(f"   {row['Model']}: Composite Score = {row['Composite_Score']:.3f}")
        print(f"     Fitness: {row['Fitness']:.3f}, Precision: {row['Precision']:.3f}")
        print(f"     Generalization: {row['Generalization']:.3f}, Simplicity: {row['Simplicity']:.3f}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    
    # Create output directory
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Process model visualization
        fig = visualizer.plot_process_model(
            simple_model, 
            title="Simple Process Model",
            save_path=str(output_dir / "simple_process_model.png")
        )
        print("✅ Process model visualization saved")
        
        # Case duration distribution
        fig = visualizer.plot_case_duration_distribution(
            event_log,
            save_path=str(output_dir / "case_duration_distribution.png")
        )
        print("✅ Case duration distribution saved")
        
        # Activity frequency
        fig = visualizer.plot_activity_frequency(
            event_log,
            save_path=str(output_dir / "activity_frequency.png")
        )
        print("✅ Activity frequency plot saved")
        
        # Resource utilization
        fig = visualizer.plot_resource_utilization(
            event_log,
            save_path=str(output_dir / "resource_utilization.png")
        )
        print("✅ Resource utilization plot saved")
        
    except Exception as e:
        print(f"⚠️ Visualization generation failed: {e}")
    
    # Save results to CSV
    print("\n💾 Saving results...")
    
    try:
        # Save event log
        event_log.dataframe.to_csv(output_dir / "event_log.csv", index=False)
        print("✅ Event log saved to CSV")
        
        # Save leaderboard
        leaderboard.to_csv(output_dir / "model_leaderboard.csv", index=False)
        print("✅ Model leaderboard saved to CSV")
        
        # Save conformance results
        conformance_summary = []
        for model_name, result in conformance_results.items():
            conformance_summary.append({
                'Model': model_name,
                'Fitness': result.fitness,
                'Precision': result.precision,
                'Generalization': result.generalization,
                'Simplicity': result.simplicity
            })
        
        import pandas as pd
        conformance_df = pd.DataFrame(conformance_summary)
        conformance_df.to_csv(output_dir / "conformance_results.csv", index=False)
        print("✅ Conformance results saved to CSV")
        
    except Exception as e:
        print(f"⚠️ CSV export failed: {e}")
    
    print("\n🎉 Analysis completed successfully!")
    print(f"📁 Results saved to: {output_dir.absolute()}")
    print("\n⚠️ Remember: This is for research and educational purposes only.")
    print("   Always validate results with domain experts before making business decisions.")


if __name__ == "__main__":
    main()
