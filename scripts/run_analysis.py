#!/usr/bin/env python3
"""Command-line script to run process mining analysis."""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from business_process_mining import (
    EventLogGenerator, ProcessDiscovery, ConformanceChecker, 
    ProcessMetrics, ProcessVisualizer
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_analysis(config: dict, output_dir: str) -> None:
    """Run the complete process mining analysis.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory for results
    """
    print("🔍 Starting Business Process Mining Analysis...")
    
    # Initialize components
    generator = EventLogGenerator(seed=config['data']['seed'])
    discovery = ProcessDiscovery(seed=config['data']['seed'])
    conformance = ConformanceChecker(seed=config['data']['seed'])
    metrics = ProcessMetrics(seed=config['data']['seed'])
    visualizer = ProcessVisualizer(
        figsize=tuple(config['visualization']['figsize']),
        style=config['visualization']['style']
    )
    
    # Generate event log
    print("📊 Generating event log...")
    start_date = datetime.strptime(config['data']['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(config['data']['end_date'], '%Y-%m-%d')
    
    # For demo purposes, generate a simple approval process
    event_log = generator.generate_simple_approval_process(
        num_cases=config['data']['num_cases'],
        start_date=start_date,
        end_date=end_date
    )
    
    print(f"✅ Generated {len(event_log.case_ids)} cases with {len(event_log.events)} events")
    
    # Discover process models
    print("🔍 Discovering process models...")
    models = []
    
    for algorithm in config['discovery']['algorithms']:
        try:
            if algorithm == "simple_graph":
                model = discovery.discover_simple_graph(event_log)
            elif algorithm == "alpha_miner":
                model = discovery.discover_alpha_miner(event_log)
            elif algorithm == "heuristics_miner":
                heuristics_config = config['discovery']['heuristics_miner']
                model = discovery.discover_heuristics_miner(
                    event_log,
                    dependency_threshold=heuristics_config['dependency_threshold'],
                    and_threshold=heuristics_config['and_threshold'],
                    loop_two_threshold=heuristics_config['loop_two_threshold']
                )
            elif algorithm == "inductive_miner":
                model = discovery.discover_inductive_miner(event_log)
            else:
                print(f"⚠️ Unknown algorithm: {algorithm}")
                continue
            
            models.append(model)
            print(f"✅ Discovered model: {model.name}")
            
        except Exception as e:
            print(f"❌ Failed to discover model with {algorithm}: {e}")
    
    if not models:
        print("❌ No models could be discovered!")
        return
    
    # Perform conformance checking
    print("📏 Performing conformance checking...")
    conformance_results = {}
    
    for model in models:
        try:
            result = conformance.check_conformance(event_log, model)
            conformance_results[model.name] = result
            print(f"✅ Conformance checked for {model.name}")
        except Exception as e:
            print(f"❌ Conformance checking failed for {model.name}: {e}")
    
    # Calculate metrics
    print("📈 Calculating metrics...")
    
    # Process KPIs
    if config['evaluation']['calculate_kpis']:
        process_kpis = metrics.calculate_process_kpis(event_log)
        print("✅ Process KPIs calculated")
    
    # Bottleneck analysis
    if config['evaluation']['calculate_bottlenecks']:
        bottleneck_analysis = metrics.calculate_bottleneck_analysis(event_log)
        print("✅ Bottleneck analysis completed")
    
    # Resource efficiency
    if config['evaluation']['calculate_resource_efficiency']:
        resource_efficiency = metrics.calculate_resource_efficiency(event_log)
        print("✅ Resource efficiency analysis completed")
    
    # Create leaderboard
    if config['evaluation']['create_leaderboard']:
        leaderboard = metrics.create_leaderboard(models, event_log)
        print("✅ Model leaderboard created")
    
    # Generate visualizations
    print("📊 Generating visualizations...")
    
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Process model visualizations
    for model in models:
        try:
            fig = visualizer.plot_process_model(
                model, 
                title=f"Process Model: {model.name}",
                save_path=str(plots_dir / f"model_{model.name.lower().replace(' ', '_')}.png")
            )
            print(f"✅ Generated visualization for {model.name}")
        except Exception as e:
            print(f"❌ Visualization failed for {model.name}: {e}")
    
    # Case duration distribution
    try:
        fig = visualizer.plot_case_duration_distribution(
            event_log,
            save_path=str(plots_dir / "case_duration_distribution.png")
        )
        print("✅ Generated case duration distribution")
    except Exception as e:
        print(f"❌ Case duration visualization failed: {e}")
    
    # Activity frequency
    try:
        fig = visualizer.plot_activity_frequency(
            event_log,
            save_path=str(plots_dir / "activity_frequency.png")
        )
        print("✅ Generated activity frequency plot")
    except Exception as e:
        print(f"❌ Activity frequency visualization failed: {e}")
    
    # Resource utilization
    try:
        fig = visualizer.plot_resource_utilization(
            event_log,
            save_path=str(plots_dir / "resource_utilization.png")
        )
        print("✅ Generated resource utilization plot")
    except Exception as e:
        print(f"❌ Resource utilization visualization failed: {e}")
    
    # Save results
    if config['output']['save_results']:
        print("💾 Saving results...")
        
        results_dir = Path(output_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save event log
        event_log_df = event_log.dataframe
        event_log_df.to_csv(results_dir / "event_log.csv", index=False)
        print("✅ Event log saved")
        
        # Save leaderboard
        if 'leaderboard' in locals():
            leaderboard.to_csv(results_dir / "model_leaderboard.csv", index=False)
            print("✅ Model leaderboard saved")
        
        # Save conformance results
        if conformance_results:
            conformance_summary = []
            for model_name, result in conformance_results.items():
                conformance_summary.append({
                    'Model': model_name,
                    'Fitness': result.fitness,
                    'Precision': result.precision,
                    'Generalization': result.generalization,
                    'Simplicity': result.simplicity,
                    'Alignment_Cost': result.alignment_cost,
                    'Token_Replay_Fitness': result.token_replay_fitness
                })
            
            conformance_df = pd.DataFrame(conformance_summary)
            conformance_df.to_csv(results_dir / "conformance_results.csv", index=False)
            print("✅ Conformance results saved")
        
        # Save process KPIs
        if 'process_kpis' in locals():
            kpis_data = {
                'Metric': [
                    'Mean Case Duration', 'Median Case Duration', 'Case Duration Std',
                    'Cases Per Day', 'Cases Per Week', 'Completion Rate', 'Error Rate',
                    'Rework Rate', 'Cost Per Case', 'Process Variants'
                ],
                'Value': [
                    process_kpis.mean_case_duration, process_kpis.median_case_duration,
                    process_kpis.case_duration_std, process_kpis.cases_per_day,
                    process_kpis.cases_per_week, process_kpis.completion_rate,
                    process_kpis.error_rate, process_kpis.rework_rate,
                    process_kpis.cost_per_case, process_kpis.process_variants
                ]
            }
            
            kpis_df = pd.DataFrame(kpis_data)
            kpis_df.to_csv(results_dir / "process_kpis.csv", index=False)
            print("✅ Process KPIs saved")
    
    print("🎉 Analysis completed successfully!")
    print(f"📁 Results saved to: {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Business Process Mining Analysis")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"✅ Loaded configuration from {args.config}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    try:
        run_analysis(config, str(output_dir))
        return 0
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
