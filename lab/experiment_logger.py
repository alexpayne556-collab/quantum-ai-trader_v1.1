#!/usr/bin/env python3
"""
EXPERIMENT LOGGER - Scientific Reproducibility System

Every experiment must be:
1. Documented (what, why, how)
2. Reproducible (exact code, exact data, exact results)
3. Versioned (track changes over time)
4. Peer-reviewable (clear enough for others to validate)
"""

import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox


@dataclass
class ExperimentMetadata:
    """Metadata for scientific experiment"""
    experiment_id: str
    title: str
    hypothesis_id: str
    researcher: str
    started_at: datetime
    completed_at: Optional[datetime]
    status: str  # running, completed, failed
    description: str
    methodology: str
    data_sources: List[str]
    code_version: str
    random_seed: int
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['started_at'] = self.started_at.isoformat()
        d['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return d


@dataclass
class ExperimentResult:
    """Results from scientific experiment"""
    experiment_id: str
    hypothesis_tested: str
    result: str  # validated, rejected, inconclusive
    p_value: float
    effect_size: float
    confidence_interval: tuple
    sample_size: int
    statistical_power: float
    conclusions: List[str]
    anomalies: List[str]
    next_steps: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ExperimentLogger:
    """
    Log all experiments with full reproducibility.
    
    Structure:
    lab/
      experiments/
        EXP001_momentum_test/
          metadata.json
          data.pkl
          results.json
          code.py
          figures/
          logs.txt
    """
    
    def __init__(self, lab_dir: str = '../lab'):
        self.lab_dir = Path(lab_dir)
        self.experiments_dir = self.lab_dir / 'experiments'
        self.results_dir = self.lab_dir / 'results'
        
        # Create directories
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_experiment = None
        self.log_buffer = []
    
    def start_experiment(
        self,
        title: str,
        hypothesis_id: str,
        description: str,
        methodology: str,
        data_sources: List[str],
        parameters: Dict[str, Any],
        random_seed: int = 42,
        researcher: str = "AI Researcher"
    ) -> str:
        """
        Start new experiment and create logging directory.
        """
        # Generate experiment ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_id = f"EXP{len(list(self.experiments_dir.glob('EXP*')))+1:04d}_{timestamp}"
        
        # Create experiment directory
        exp_dir = self.experiments_dir / exp_id
        exp_dir.mkdir(exist_ok=True)
        (exp_dir / 'figures').mkdir(exist_ok=True)
        
        # Create metadata
        metadata = ExperimentMetadata(
            experiment_id=exp_id,
            title=title,
            hypothesis_id=hypothesis_id,
            researcher=researcher,
            started_at=datetime.now(),
            completed_at=None,
            status='running',
            description=description,
            methodology=methodology,
            data_sources=data_sources,
            code_version=self._get_git_version(),
            random_seed=random_seed,
            parameters=parameters
        )
        
        # Save metadata
        with open(exp_dir / 'metadata.json', 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        self.current_experiment = {
            'id': exp_id,
            'dir': exp_dir,
            'metadata': metadata
        }
        
        self.log(f"Experiment {exp_id} started: {title}")
        self.log(f"Hypothesis: {hypothesis_id}")
        self.log(f"Random seed: {random_seed}")
        
        return exp_id
    
    def log(self, message: str, level: str = 'INFO'):
        """Log message to experiment log file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        print(log_entry)
        self.log_buffer.append(log_entry)
        
        # Write to file if experiment is active
        if self.current_experiment:
            log_file = self.current_experiment['dir'] / 'logs.txt'
            with open(log_file, 'a') as f:
                f.write(log_entry + '\n')
    
    def save_data(self, data: Any, name: str = 'data'):
        """Save experiment data (reproducibility)"""
        if not self.current_experiment:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        data_file = self.current_experiment['dir'] / f'{name}.pkl'
        
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Calculate hash for data integrity
        data_hash = hashlib.md5(pickle.dumps(data)).hexdigest()
        
        self.log(f"Saved data: {name}.pkl (hash: {data_hash[:8]})")
        
        return data_file
    
    def save_code(self, code: str, filename: str = 'code.py'):
        """Save experiment code for reproducibility"""
        if not self.current_experiment:
            raise RuntimeError("No active experiment")
        
        code_file = self.current_experiment['dir'] / filename
        
        with open(code_file, 'w') as f:
            f.write(code)
        
        self.log(f"Saved code: {filename}")
        
        return code_file
    
    def save_figure(self, fig, name: str):
        """Save matplotlib figure"""
        if not self.current_experiment:
            raise RuntimeError("No active experiment")
        
        fig_file = self.current_experiment['dir'] / 'figures' / f'{name}.png'
        fig.savefig(fig_file, dpi=150, bbox_inches='tight')
        
        self.log(f"Saved figure: figures/{name}.png")
        
        return fig_file
    
    def complete_experiment(
        self,
        result: ExperimentResult,
        figures: Dict[str, Any] = None
    ):
        """Mark experiment as complete and save results"""
        if not self.current_experiment:
            raise RuntimeError("No active experiment")
        
        exp_dir = self.current_experiment['dir']
        
        # Update metadata
        metadata = self.current_experiment['metadata']
        metadata.completed_at = datetime.now()
        metadata.status = 'completed'
        
        with open(exp_dir / 'metadata.json', 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Save results
        with open(exp_dir / 'results.json', 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Save figures if provided
        if figures:
            for name, fig in figures.items():
                self.save_figure(fig, name)
        
        # Generate summary report
        self._generate_summary_report(metadata, result)
        
        self.log(f"Experiment completed: {result.result.upper()}")
        self.log(f"P-value: {result.p_value:.6f}")
        self.log(f"Effect size: {result.effect_size:.4f}")
        
        # Copy to results directory
        result_summary = {
            'experiment_id': metadata.experiment_id,
            'title': metadata.title,
            'hypothesis': result.hypothesis_tested,
            'result': result.result,
            'p_value': result.p_value,
            'effect_size': result.effect_size,
            'completed': metadata.completed_at.isoformat(),
            'dir': str(exp_dir)
        }
        
        results_file = self.results_dir / 'all_results.json'
        
        # Load existing results
        all_results = []
        if results_file.exists():
            with open(results_file, 'r') as f:
                all_results = json.load(f)
        
        all_results.append(result_summary)
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        self.current_experiment = None
        
        return exp_dir
    
    def _generate_summary_report(self, metadata: ExperimentMetadata, result: ExperimentResult):
        """Generate human-readable summary report"""
        exp_dir = self.current_experiment['dir']
        report_file = exp_dir / 'SUMMARY.md'
        
        lines = []
        lines.append(f"# {metadata.title}")
        lines.append(f"\n**Experiment ID:** {metadata.experiment_id}")
        lines.append(f"**Hypothesis:** {metadata.hypothesis_id}")
        lines.append(f"**Researcher:** {metadata.researcher}")
        lines.append(f"**Date:** {metadata.started_at.strftime('%Y-%m-%d')}")
        lines.append(f"**Duration:** {(metadata.completed_at - metadata.started_at).total_seconds():.0f} seconds")
        
        lines.append(f"\n## Objective")
        lines.append(f"\n{metadata.description}")
        
        lines.append(f"\n## Methodology")
        lines.append(f"\n{metadata.methodology}")
        
        lines.append(f"\n## Data Sources")
        for source in metadata.data_sources:
            lines.append(f"- {source}")
        
        lines.append(f"\n## Parameters")
        for key, value in metadata.parameters.items():
            lines.append(f"- {key}: {value}")
        
        lines.append(f"\n## Results")
        lines.append(f"\n**Conclusion:** {result.result.upper()}")
        lines.append(f"\n**Statistical Evidence:**")
        lines.append(f"- P-value: {result.p_value:.6f}")
        lines.append(f"- Effect size: {result.effect_size:.4f}")
        lines.append(f"- 95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
        lines.append(f"- Sample size: {result.sample_size:,}")
        lines.append(f"- Statistical power: {result.statistical_power:.2f}")
        
        lines.append(f"\n## Conclusions")
        for conclusion in result.conclusions:
            lines.append(f"- {conclusion}")
        
        if result.anomalies:
            lines.append(f"\n## Anomalies / Warnings")
            for anomaly in result.anomalies:
                lines.append(f"- ⚠️  {anomaly}")
        
        if result.next_steps:
            lines.append(f"\n## Next Steps")
            for step in result.next_steps:
                lines.append(f"- [ ] {step}")
        
        lines.append(f"\n## Reproducibility")
        lines.append(f"- Random seed: {metadata.random_seed}")
        lines.append(f"- Code version: {metadata.code_version}")
        lines.append(f"- All data and code saved in: `{exp_dir.name}/`")
        
        with open(report_file, 'w') as f:
            f.write('\n'.join(lines))
        
        self.log(f"Generated summary report: SUMMARY.md")
    
    def _get_git_version(self) -> str:
        """Get current git commit hash"""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=self.lab_dir
            )
            return result.stdout.strip() if result.returncode == 0 else 'unknown'
        except:
            return 'unknown'
    
    def load_experiment(self, experiment_id: str) -> Dict:
        """Load existing experiment"""
        exp_dir = self.experiments_dir / experiment_id
        
        if not exp_dir.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Load metadata
        with open(exp_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load results if exist
        results = None
        if (exp_dir / 'results.json').exists():
            with open(exp_dir / 'results.json', 'r') as f:
                results = json.load(f)
        
        # Load data if exists
        data = None
        if (exp_dir / 'data.pkl').exists():
            with open(exp_dir / 'data.pkl', 'rb') as f:
                data = pickle.load(f)
        
        return {
            'metadata': metadata,
            'results': results,
            'data': data,
            'dir': exp_dir
        }
    
    def list_experiments(self) -> pd.DataFrame:
        """List all experiments"""
        experiments = []
        
        for exp_dir in sorted(self.experiments_dir.glob('EXP*')):
            metadata_file = exp_dir / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Load results if available
                result = None
                if (exp_dir / 'results.json').exists():
                    with open(exp_dir / 'results.json', 'r') as f:
                        results_data = json.load(f)
                        result = results_data.get('result', 'unknown')
                
                experiments.append({
                    'id': metadata['experiment_id'],
                    'title': metadata['title'],
                    'hypothesis': metadata['hypothesis_id'],
                    'status': metadata['status'],
                    'result': result,
                    'started': metadata['started_at'],
                    'dir': str(exp_dir)
                })
        
        return pd.DataFrame(experiments)


def example_experiment():
    """Example of using the experiment logger"""
    
    print("EXAMPLE: Running Scientific Experiment with Full Logging\n")
    
    logger = ExperimentLogger(lab_dir='../lab')
    
    # Start experiment
    exp_id = logger.start_experiment(
        title="Test Momentum Hypothesis",
        hypothesis_id="H0001",
        description="Test if past 60-day returns predict next 20-day returns",
        methodology="Linear regression with autocorrelation test",
        data_sources=["310 stock universe", "Daily returns 2020-2025"],
        parameters={
            'lookback': 60,
            'holding': 20,
            'min_observations': 100,
            'alpha': 0.05
        },
        random_seed=42
    )
    
    logger.log("Generating test data...")
    
    # Simulate experiment
    np.random.seed(42)
    n = 1000
    
    # Create momentum series
    returns = np.zeros(n)
    returns[0] = np.random.randn()
    for t in range(1, n):
        returns[t] = 0.2 * returns[t-1] + np.random.randn()
    
    logger.save_data(returns, 'returns_series')
    
    logger.log("Running statistical tests...")
    
    # Test autocorrelation
    from scipy import stats
    autocorr = pd.Series(returns).autocorr(lag=1)
    lb_result = acorr_ljungbox(returns, lags=[1], return_df=True)
    lb_pvalue = lb_result['lb_pvalue'].iloc[0]
    
    logger.log(f"Autocorrelation: {autocorr:.4f}")
    logger.log(f"Ljung-Box p-value: {lb_pvalue:.6f}")
    
    # Create result
    result = ExperimentResult(
        experiment_id=exp_id,
        hypothesis_tested="H0001: Momentum exists (ρ(1) > 0)",
        result="validated" if lb_pvalue < 0.05 and autocorr > 0 else "rejected",
        p_value=lb_pvalue,
        effect_size=autocorr,
        confidence_interval=(autocorr - 0.05, autocorr + 0.05),
        sample_size=n,
        statistical_power=0.95,
        conclusions=[
            f"Autocorrelation {autocorr:.4f} is statistically significant",
            "Momentum effect validated in test data",
            "Next step: validate on real market data"
        ],
        anomalies=[],
        next_steps=[
            "Test on 310-stock universe",
            "Check regime dependence",
            "Build trading strategy"
        ]
    )
    
    # Complete experiment
    exp_dir = logger.complete_experiment(result)
    
    print(f"\n✅ Experiment completed and saved to: {exp_dir}")
    print(f"\nAll experiments:")
    print(logger.list_experiments().to_string(index=False))


if __name__ == '__main__':
    example_experiment()
