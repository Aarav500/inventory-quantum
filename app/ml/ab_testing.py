"""
A/B Testing Framework with Statistical Significance.

Properly evaluates experiments with:
- Power analysis for sample size
- Sequential testing to stop early
- Multiple comparison correction
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class ExperimentResult:
    """Result of an A/B test."""
    control_mean: float
    treatment_mean: float
    lift: float
    lift_pct: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    power: float
    sample_size_control: int
    sample_size_treatment: int


class PowerAnalysis:
    """
    Sample size calculation for A/B tests.
    """
    
    @staticmethod
    def required_sample_size(
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.8,
    ) -> int:
        """
        Calculate required sample size per group.
        
        Uses formula for difference in means.
        """
        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        # Variance (simplified for proportions)
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        pooled_p = (p1 + p2) / 2
        
        variance = 2 * pooled_p * (1 - pooled_p)
        effect = (p2 - p1) ** 2
        
        n = ((z_alpha + z_beta) ** 2 * variance) / effect
        return int(np.ceil(n))


class ABTest:
    """
    A/B Testing with proper statistical analysis.
    """
    
    def __init__(
        self,
        name: str,
        alpha: float = 0.05,
        min_samples_per_group: int = 100
    ):
        self.name = name
        self.alpha = alpha
        self.min_samples = min_samples_per_group
        self.control_data: List[float] = []
        self.treatment_data: List[float] = []
    
    def add_observation(self, value: float, is_treatment: bool):
        """Add single observation."""
        if is_treatment:
            self.treatment_data.append(value)
        else:
            self.control_data.append(value)
    
    def add_batch(self, control: List[float], treatment: List[float]):
        """Add batch of observations."""
        self.control_data.extend(control)
        self.treatment_data.extend(treatment)
    
    def analyze(self) -> ExperimentResult:
        """Run statistical analysis."""
        control = np.array(self.control_data)
        treatment = np.array(self.treatment_data)
        
        # Means
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        
        # Lift
        lift = treatment_mean - control_mean
        lift_pct = lift / (control_mean + 1e-8) * 100
        
        # T-test
        t_stat, p_value = stats.ttest_ind(treatment, control)
        
        # Confidence interval for difference
        pooled_se = np.sqrt(
            np.var(control) / len(control) + 
            np.var(treatment) / len(treatment)
        )
        margin = stats.t.ppf(1 - self.alpha / 2, len(control) + len(treatment) - 2) * pooled_se
        ci = (lift - margin, lift + margin)
        
        # Power
        effect_size = lift / (np.std(np.concatenate([control, treatment])) + 1e-8)
        power = self._calculate_power(len(control), len(treatment), effect_size)
        
        return ExperimentResult(
            control_mean=float(control_mean),
            treatment_mean=float(treatment_mean),
            lift=float(lift),
            lift_pct=float(lift_pct),
            p_value=float(p_value),
            confidence_interval=ci,
            is_significant=p_value < self.alpha,
            power=float(power),
            sample_size_control=len(control),
            sample_size_treatment=len(treatment)
        )
    
    def _calculate_power(self, n1: int, n2: int, effect_size: float) -> float:
        """Calculate achieved power."""
        df = n1 + n2 - 2
        nc = effect_size * np.sqrt(n1 * n2 / (n1 + n2))
        
        critical_t = stats.t.ppf(1 - self.alpha / 2, df)
        power = 1 - stats.nct.cdf(critical_t, df, nc) + stats.nct.cdf(-critical_t, df, nc)
        
        return max(0, min(1, power))
    
    def should_stop_early(self, method: str = 'obrien_fleming') -> Tuple[bool, str]:
        """
        Sequential testing: check if we can stop early.
        
        Uses spending functions to control Type I error.
        """
        if len(self.control_data) < self.min_samples:
            return False, "Need more samples"
        
        result = self.analyze()
        
        if method == 'obrien_fleming':
            # O'Brien-Fleming spending function
            info_fraction = len(self.control_data) / (self.min_samples * 3)
            adjusted_alpha = self.alpha * np.exp(-0.5 * stats.norm.ppf(1 - self.alpha/2)**2 / info_fraction)
            
            if result.p_value < adjusted_alpha:
                return True, f"Significant at adjusted alpha={adjusted_alpha:.4f}"
        
        # Futility check
        if result.power < 0.2 and len(self.control_data) > self.min_samples * 2:
            return True, "Futility: unlikely to reach significance"
        
        return False, "Continue testing"


class MultipleTestingCorrection:
    """
    Correct for multiple comparisons.
    """
    
    @staticmethod
    def bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
        """Bonferroni correction (conservative)."""
        adjusted_alpha = alpha / len(p_values)
        return [p < adjusted_alpha for p in p_values]
    
    @staticmethod
    def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> List[bool]:
        """Benjamini-Hochberg FDR control."""
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_idx]
        
        # BH procedure
        thresholds = np.arange(1, n + 1) / n * alpha
        significant = sorted_p <= thresholds
        
        # Find largest k where p_k <= k/n * alpha
        if not np.any(significant):
            return [False] * n
        
        k = np.max(np.where(significant)[0])
        
        result = [False] * n
        for i in range(k + 1):
            result[sorted_idx[i]] = True
        
        return result


def run_experiment(
    control_data: List[float],
    treatment_data: List[float],
    name: str = "Experiment"
) -> Dict:
    """
    Run a complete A/B test analysis.
    """
    test = ABTest(name)
    test.add_batch(control_data, treatment_data)
    result = test.analyze()
    
    can_stop, reason = test.should_stop_early()
    
    return {
        'name': name,
        'control_mean': result.control_mean,
        'treatment_mean': result.treatment_mean,
        'lift': result.lift,
        'lift_pct': result.lift_pct,
        'p_value': result.p_value,
        'is_significant': result.is_significant,
        'confidence_interval': {
            'lower': result.confidence_interval[0],
            'upper': result.confidence_interval[1]
        },
        'power': result.power,
        'sample_sizes': {
            'control': result.sample_size_control,
            'treatment': result.sample_size_treatment
        },
        'early_stopping': {
            'can_stop': can_stop,
            'reason': reason
        },
        'interpretation': _interpret_result(result)
    }


def _interpret_result(result: ExperimentResult) -> str:
    """Generate human-readable interpretation."""
    if result.is_significant:
        direction = "increases" if result.lift > 0 else "decreases"
        return (
            f"Treatment {direction} the metric by {abs(result.lift_pct):.1f}% "
            f"(95% CI: [{result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f}]). "
            f"This result is statistically significant (p={result.p_value:.4f})."
        )
    else:
        return (
            f"No significant difference detected (p={result.p_value:.4f}). "
            f"Current power: {result.power:.1%}. "
            f"Consider collecting more data or accepting null hypothesis."
        )
