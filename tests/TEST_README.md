# TLR Regression Test

This directory contains pytest-based regression tests for the TLR (Time-Lagged Recurrence) repository.

## Test Overview

The regression test uses the `../data/datasets/lorenz_small.txt` dataset to verify that:
1. Data loading works correctly
2. TLR computation functions work as expected
3. Results are saved in the correct format (JSON)
4. Basic sanity checks on the computed values

## Installation

First, install the required dependencies:

```bash
pip install -r tests/requirements.txt
```

## Running the Test

### Option 1: Using pytest from TLR root directory
```bash
pytest tests/test_regression.py -v
```

### Option 2: From test directory
```bash
cd test
pytest test_regression.py -v
```

### Option 3: Using the shell script
```bash
./test/run_test.sh
```

### Option 4: Run specific test
```bash
pytest tests/test_regression.py::test_data_loading -v
```

## Test Details

The test performs the following operations:
1. **Data Loading**: Loads `lorenz_small.txt` and verifies the data format
2. **TLR Computation**: Runs the complete TLR pipeline:
   - Computes exceedances
   - Computes local dimension (d1)
   - Computes persistence (theta)
   - Computes alphat for multiple time lags
3. **Result Verification**: Checks that:
   - All output files are created correctly
   - JSON files contain valid data
   - Alpha values are in the expected range [0,1]
   - Basic statistics are reasonable

## Test Parameters

- **Dataset**: `../data/datasets/lorenz_small.txt` (10,000 data points)
- **Quantile**: 0.99
- **Time lags**: [5, 10, 20] (smaller values for the small dataset)
- **Theiler window**: 0
- **Distance metric**: l=0 (original alpha)

## Expected Output

When the test passes, you should see:
```
==================================================
✓ ALL TESTS PASSED!
==================================================
```

## Troubleshooting

If the test fails, check:
1. That `../data/datasets/lorenz_small.txt` exists
2. That all required Python packages are installed
3. That the TLR scripts are accessible from the test script

## Test Cleanup

The test automatically cleans up temporary files after completion.
