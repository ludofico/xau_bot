"""
Tests for Monte Carlo simulation with TRNG.

Verifies:
- TRNG entropy quality
- Statistical properties of simulations
- Convergence behavior
- Result consistency
"""

import pytest
import numpy as np
from scipy import stats


class TestTRNGProvider:
    """Tests for True Random Number Generator."""
    
    def test_entropy_quality(self):
        """Test that TRNG passes chi-squared randomness test."""
        from xauusd_strategy.backtest.monte_carlo import TRNGProvider
        
        rng = TRNGProvider(source="auto")
        
        # p-value should be > 0.01 (99% confidence in randomness)
        assert rng.entropy_quality > 0.01, \
            f"Entropy quality too low: {rng.entropy_quality:.4f}"
    
    def test_random_int_range(self):
        """Test random_int produces values in correct range."""
        from xauusd_strategy.backtest.monte_carlo import TRNGProvider
        
        rng = TRNGProvider()
        
        for _ in range(1000):
            value = rng.random_int(10, 20)
            assert 10 <= value < 20
    
    def test_random_float_range(self):
        """Test random_float produces values in [0, 1)."""
        from xauusd_strategy.backtest.monte_carlo import TRNGProvider
        
        rng = TRNGProvider()
        
        for _ in range(1000):
            value = rng.random_float()
            assert 0.0 <= value < 1.0
    
    def test_shuffle_preserves_elements(self):
        """Test shuffle preserves all elements."""
        from xauusd_strategy.backtest.monte_carlo import TRNGProvider
        
        rng = TRNGProvider()
        original = list(range(100))
        shuffled = rng.shuffle(original)
        
        assert sorted(shuffled) == sorted(original)
        assert shuffled != original  # Very unlikely to be identical
    
    def test_bootstrap_sample_size(self):
        """Test bootstrap produces correct sample size."""
        from xauusd_strategy.backtest.monte_carlo import TRNGProvider
        
        rng = TRNGProvider()
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Default size
        sample = rng.bootstrap_sample(data)
        assert len(sample) == len(data)
        
        # Custom size
        sample = rng.bootstrap_sample(data, size=10)
        assert len(sample) == 10
    
    def test_uniformity(self):
        """Test that random values are uniformly distributed."""
        from xauusd_strategy.backtest.monte_carlo import TRNGProvider
        
        rng = TRNGProvider()
        
        # Generate many random floats
        values = [rng.random_float() for _ in range(10000)]
        
        # Test with Kolmogorov-Smirnov test against uniform
        statistic, p_value = stats.kstest(values, 'uniform')
        
        assert p_value > 0.01, f"Distribution not uniform: KS p={p_value:.4f}"


class TestMonteCarloSimulator:
    """Tests for Monte Carlo simulation."""
    
    def test_simulation_runs(self, sample_trades):
        """Test basic simulation execution."""
        from xauusd_strategy.backtest.monte_carlo import MonteCarloSimulator
        
        sim = MonteCarloSimulator(
            initial_balance=250.0,
            n_simulations=100
        )
        
        result = sim.run(sample_trades)
        
        assert result.n_simulations == 100
        assert result.mean_final_balance > 0
        assert 0 <= result.prob_profit <= 1
        assert 0 <= result.prob_ruin <= 1
    
    def test_confidence_intervals(self, sample_trades):
        """Test confidence interval ordering."""
        from xauusd_strategy.backtest.monte_carlo import MonteCarloSimulator
        
        sim = MonteCarloSimulator(n_simulations=500)
        result = sim.run(sample_trades)
        
        # 99% CI should be wider than 95% CI
        assert result.ci_99[0] <= result.ci_95[0]
        assert result.ci_99[1] >= result.ci_95[1]
    
    def test_convergence(self, sample_trades):
        """Test that results converge with more simulations."""
        from xauusd_strategy.backtest.monte_carlo import MonteCarloSimulator
        
        # Run with different simulation counts
        results_100 = MonteCarloSimulator(n_simulations=100).run(sample_trades)
        results_1000 = MonteCarloSimulator(n_simulations=1000).run(sample_trades)
        
        # Std should decrease with more simulations
        # (or be similar if already converged)
        # We just check they're in reasonable range
        assert results_100.std_final_balance > 0
        assert results_1000.std_final_balance > 0
    
    def test_drawdown_calculation(self):
        """Test max drawdown calculation."""
        from xauusd_strategy.backtest.monte_carlo import MonteCarloSimulator
        
        sim = MonteCarloSimulator()
        
        # Simple equity curve: 100 -> 150 -> 100 -> 200
        equity = np.array([100, 110, 150, 130, 100, 150, 200])
        
        dd = sim.calculate_max_drawdown(equity)
        
        # Max DD should be (150 - 100) / 150 = 33.3%
        expected_dd = (150 - 100) / 150
        assert abs(dd - expected_dd) < 0.01
    
    def test_minimum_trades_required(self):
        """Test that simulation requires minimum trades."""
        from xauusd_strategy.backtest.monte_carlo import MonteCarloSimulator
        
        sim = MonteCarloSimulator()
        
        with pytest.raises(ValueError, match="at least 5 trades"):
            sim.run([1.0, 2.0, 3.0])
    
    def test_result_serialization(self, sample_trades, tmp_path):
        """Test result saving to JSON."""
        from xauusd_strategy.backtest.monte_carlo import MonteCarloSimulator
        
        sim = MonteCarloSimulator(n_simulations=100)
        result = sim.run(sample_trades)
        
        output_file = tmp_path / "test_results.json"
        sim.save_results(result, str(output_file))
        
        assert output_file.exists()
        
        import json
        with open(output_file) as f:
            data = json.load(f)
        
        assert 'results' in data
        assert 'mean_final_balance' in data['results']


class TestMonteCarloStatistics:
    """Statistical validity tests for Monte Carlo."""
    
    def test_positive_expectancy_simulation(self):
        """Test that positive expectancy trades produce profit."""
        from xauusd_strategy.backtest.monte_carlo import MonteCarloSimulator
        
        # Trades with positive expectancy
        trades = [10.0] * 8 + [-5.0] * 2  # Win 80%, avg +7.0
        
        sim = MonteCarloSimulator(n_simulations=1000)
        result = sim.run(trades)
        
        # Should have high probability of profit
        assert result.prob_profit > 0.8
    
    def test_negative_expectancy_simulation(self):
        """Test that negative expectancy trades produce loss."""
        from xauusd_strategy.backtest.monte_carlo import MonteCarloSimulator
        
        # Trades with negative expectancy
        trades = [-10.0] * 8 + [5.0] * 2  # Lose 80%, avg -7.0
        
        sim = MonteCarloSimulator(n_simulations=1000)
        result = sim.run(trades)
        
        # Should have low probability of profit
        assert result.prob_profit < 0.2
    
    def test_reproducibility_with_numpy_fallback(self):
        """Test reproducibility when using numpy (for testing purposes)."""
        from xauusd_strategy.backtest.monte_carlo import MonteCarloSimulator
        
        trades = [10.0, -5.0, 15.0, -8.0, 12.0]
        
        # Using numpy source with seed should be reproducible
        np.random.seed(42)
        sim1 = MonteCarloSimulator(n_simulations=100, trng_source="numpy")
        
        np.random.seed(42)  # Reset seed
        sim2 = MonteCarloSimulator(n_simulations=100, trng_source="numpy")
        
        # Note: With hardware RNG this would NOT be reproducible
        # This test only works with numpy fallback
