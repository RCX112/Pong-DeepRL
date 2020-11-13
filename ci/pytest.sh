#!/bin/bash

# Create Log file
touch pytest_output.txt

# Counter for errors
containsErrors=
pytestFailed=0

# Run Pytest
for directory in $( cat directories.txt ); do
    echo Checking $directory
    pytest $directory --disable-warnings >> pytest_output.txt
    
    containsErrors=$?
    echo "Pytest in $directory exited with code $containsErrors"
    echo $( cat pytest_output.txt | tail -1)
    
    
    if [[ $containsErrors != 0 && $containsErrors != 5 ]]; then
        pytestFailed=1
    fi
done

if [[ $pytestFailed == 0 ]]; then
    echo "Pytest exited with 0 errors"

    # Create Badge
    anybadge -l Unit\ Testing -v Success -c green -f pytest.svg
    
    exit 0
else
    echo "Pytest found errors in the code. You might want to check your code again"
    echo "Maybe take a look at pytest_output.txt"
    
    # Create Badge
    anybadge -l Unit\ Testing -v Failed -c red -f pytest.svg
    
    exit 1
fi
