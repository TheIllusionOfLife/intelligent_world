# Experiment Campaign Report

## Executive Summary

- **Total runs**: 60 (0 failed)
- **Best configuration**: Config A (Single-agent, no semantic mutations (baseline))
- **Best overall success rate**: 66.7%

## Recommendation

**GO** — Config A achieves 66.7% success rate across all tasks. The system demonstrates viable evolutionary optimization. Recommended next steps: scale to harder tasks, tune hyperparameters for underperforming configurations.

## Success Rates

| Config | Description | Overall | run_length_encode | slugify | two_sum_sorted |
|--------|-------------|---------|------|------|------|
| A | Single-agent, no semantic mutations (baseline) | 66.7% | 100% (5/5) | 0% (0/5) | 100% (5/5) |
| B | Single-agent, with semantic mutations | 66.7% | 100% (5/5) | 0% (0/5) | 100% (5/5) |
| C | Population, no semantic mutations | 66.7% | 100% (5/5) | 0% (0/5) | 100% (5/5) |
| D | Population, with semantic mutations | 66.7% | 100% (5/5) | 0% (0/5) | 100% (5/5) |

## Fitness Distribution

| Config | Mean | Median | Min | Max | Stdev |
|--------|------|--------|-----|-----|-------|
| A | 0.840 | 0.900 | 0.720 | 0.900 | 0.088 |
| B | 0.840 | 0.900 | 0.720 | 0.900 | 0.088 |
| C | 0.840 | 0.900 | 0.720 | 0.900 | 0.088 |
| D | 0.840 | 0.900 | 0.720 | 0.900 | 0.088 |

## Population Diversity Analysis

### Config C

- **Mean Shannon entropy**: 0.924
- No premature convergence observed

### Config D

- **Mean Shannon entropy**: 0.938
- No premature convergence observed

## Key Findings

1. **Config A** (Single-agent, no semantic mutations (baseline)): 66.7% success rate, mean fitness 0.840
2. **Config B** (Single-agent, with semantic mutations): 66.7% success rate, mean fitness 0.840
3. **Config C** (Population, no semantic mutations): 66.7% success rate, mean fitness 0.840
4. **Config D** (Population, with semantic mutations): 66.7% success rate, mean fitness 0.840

- **Semantic mutation effect (single-agent)**: 66.7% → 66.7% (+0.0%)
- **Semantic mutation effect (population)**: 66.7% → 66.7% (+0.0%)
- **Population vs single-agent (no semantic)**: 66.7% → 66.7% (+0.0%)
- **Population vs single-agent (with semantic)**: 66.7% → 66.7% (+0.0%)

