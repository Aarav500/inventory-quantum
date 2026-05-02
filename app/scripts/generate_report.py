"""Script to generate backtest report."""

import sys
import os
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from app.services.validation import DataValidator
from app.forecasting.backtest import run_backtest
from app.forecasting.naive import SeasonalNaiveForecaster
from app.forecasting.arima import ARIMAForecaster
from app.forecasting.lightgbm_model import LightGBMForecaster
from app.decision.reorder_point import ReorderPointPolicy
from app.decision.eoq import EOQPolicy
from app.decision.qubo import QUBOOptimizer
from app.monitoring.drift import DriftDetector


def load_fixture_data():
    """Load the fixture dataset."""
    fixture_path = Path(__file__).parent.parent.parent / 'tests' / 'fixtures' / 'tiny_dataset.csv'
    df = pd.read_csv(fixture_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def generate_report(data: pd.DataFrame, output_path: Path):
    """Generate the backtest report."""
    
    report_lines = []
    report_lines.append("# Inventory Backtest Report")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Dataset summary
    report_lines.append("## Dataset Summary")
    report_lines.append("")
    report_lines.append(f"- **Total Records**: {len(data)}")
    report_lines.append(f"- **SKUs**: {data['sku'].nunique()}")
    report_lines.append(f"- **Date Range**: {data['date'].min().date()} to {data['date'].max().date()}")
    report_lines.append(f"- **Total Days**: {(data['date'].max() - data['date'].min()).days + 1}")
    report_lines.append("")
    
    # Forecasting results
    report_lines.append("## Forecasting Model Comparison")
    report_lines.append("")
    
    models = ['naive', 'arima', 'lightgbm']
    backtest_results = run_backtest(data, models, n_windows=3, window_size=14)
    
    report_lines.append("| Model | MAE | RMSE | MAPE (%) | Coverage 95% |")
    report_lines.append("|-------|-----|------|----------|--------------|")
    
    for result in backtest_results:
        coverage = f"{result.coverage_95:.1f}%" if result.coverage_95 else "N/A"
        report_lines.append(
            f"| {result.model} | {result.mae:.2f} | {result.rmse:.2f} | "
            f"{result.mape:.1f} | {coverage} |"
        )
    
    report_lines.append("")
    
    # Decision optimization results
    report_lines.append("## Decision Policy Comparison")
    report_lines.append("")
    
    policies = {
        'Reorder Point': ReorderPointPolicy(),
        'EOQ': EOQPolicy(),
        'QUBO': QUBOOptimizer(n_bits=6),
    }
    
    for sku in data['sku'].unique():
        sku_data = data[data['sku'] == sku]
        
        report_lines.append(f"### SKU: {sku}")
        report_lines.append("")
        report_lines.append("| Policy | Reorder Point | Order Qty | Expected Cost | Service Level |")
        report_lines.append("|--------|--------------|-----------|---------------|---------------|")
        
        for policy_name, policy in policies.items():
            result = policy.optimize(sku_data)
            report_lines.append(
                f"| {policy_name} | {result.reorder_point:.0f} | "
                f"{result.reorder_quantity:.0f} | ${result.expected_cost:.2f} | "
                f"{result.expected_service_level:.1%} |"
            )
        
        report_lines.append("")
    
    # QUBO ablation
    report_lines.append("## QUBO Solver Ablation")
    report_lines.append("")
    
    for sku in data['sku'].unique():
        sku_data = data[data['sku'] == sku]
        optimizer = QUBOOptimizer(n_bits=6)
        
        solver_comparison = optimizer.compare_solvers(sku_data)
        
        report_lines.append(f"### SKU: {sku}")
        report_lines.append("")
        report_lines.append("| Solver | Cost |")
        report_lines.append("|--------|------|")
        
        for solver, cost in solver_comparison.items():
            report_lines.append(f"| {solver.replace('_', ' ').title()} | ${cost:.2f} |")
        
        best_solver = min(solver_comparison, key=solver_comparison.get)
        report_lines.append(f"\n**Best Solver**: {best_solver.replace('_', ' ').title()}")
        report_lines.append("")
    
    # Drift detection
    report_lines.append("## Distributional Drift Analysis")
    report_lines.append("")
    
    detector = DriftDetector()
    drift_results = detector.detect_drift(data, reference_days=30, test_days=14)
    
    if drift_results:
        report_lines.append("| SKU | PSI | KS Statistic | Drifted |")
        report_lines.append("|-----|-----|--------------|---------|")
        
        for result in drift_results:
            drifted = "⚠️ Yes" if result.is_drifted else "✓ No"
            report_lines.append(
                f"| {result.sku} | {result.psi:.4f} | "
                f"{result.ks_statistic:.4f} | {drifted} |"
            )
    else:
        report_lines.append("*Insufficient data for drift detection*")
    
    report_lines.append("")
    
    # Conclusions
    report_lines.append("## Key Findings")
    report_lines.append("")
    
    # Find best forecaster
    if backtest_results:
        best_forecaster = min(backtest_results, key=lambda x: x.rmse)
        report_lines.append(f"1. **Best Forecasting Model**: {best_forecaster.model} "
                          f"(RMSE: {best_forecaster.rmse:.2f})")
    
    report_lines.append("2. **QUBO Optimization** provides rigorous solution via binary encoding "
                       "with simulated annealing solver")
    report_lines.append("3. **Drift Detection** uses PSI and KS-test for monitoring without ground truth")
    report_lines.append("")
    
    # Write report
    report_content = "\n".join(report_lines)
    output_path.write_text(report_content)
    
    print(f"Report generated: {output_path}")
    return report_content


def main():
    """Main entry point."""
    # Ensure reports directory exists
    reports_dir = Path(__file__).parent.parent.parent / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    # Load data
    data = load_fixture_data()
    print(f"Loaded {len(data)} records")
    
    # Generate report
    output_path = reports_dir / 'inventory_backtest.md'
    generate_report(data, output_path)


if __name__ == '__main__':
    main()
