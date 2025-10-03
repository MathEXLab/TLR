#!/bin/bash

# Script to run TLR regression test using pytest
echo "Running TLR regression test with pytest..."

# Change to test directory
cd "$(dirname "$0")"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "pytest not found. Installing pytest..."
    pip install pytest
fi

# Run the regression test with pytest
pytest test_regression.py -v

echo "Test completed."
