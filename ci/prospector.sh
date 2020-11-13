#!/bin/bash

# Create Log file
touch prospector_output.txt

# Counter for errors
containsErrors=
prospectorFailed=0

# Create Badge with unknown result
anybadge -l Code\ Analysis -v Unknown -c gray -f prospector.svg

# Run Prospector
for directory in $( cat directories.txt ); do
    echo Checking $directory
    #prospector $directory --strictness medium --doc-warnings --test-warnings >> prospector_output.txt
    prospector $directory >> prospector_output.txt
    
    containsErrors=$?
    echo "Prospector in $directory exited with code $containsErrors"
    cat prospector_output.txt | tail -n 10 
    
    if [[ $containsErrors != 0 ]]; then
        prospectorFailed=1
    fi
done

if [[ $prospectorFailed == 0 ]]; then
    echo "Prospector exited with 0 errors"
    
    # Create Badge
    anybadge -l Code\ Analysis -v Success -c green -f prospector.svg
    
    exit 0
else
    echo "Prospector found errors in the code. You might want to check your code again"
    echo "Maybe take a look at prospector_output.txt"

    # Create Badge
    anybadge -l Code\ Analysis -v Failed -c red -f prospector.svg

    exit 1
fi
