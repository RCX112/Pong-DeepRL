#!/bin/bash

# What percentage of the code has to be annotated
THRESHOLD=0.75

# Create Log file
touch mypy_output.txt

# Counter for errors and passes
nErrors=0
nPasses=0

# Run mypy to find all functions that are not annotated or wrong
for directory in $( cat directories.txt ); do
    echo Checking $directory
    mypy $directory --disallow-untyped-defs --disallow-incomplete-defs --disallow-untyped-calls --pretty --linecount-report . >> mypy_output.txt

    err=$( cat mypy_output.txt | tail -1 | awk '{print $2}' )
    pas=$( cat linecount.txt | head -1 | awk '{print $1}' )
    
    nErrors=$(( nErrors + err ))
    nPasses=$(( nPasses + pas ))
done

# If nothing has been checked
if [[ $nPasses == 0 && $nErrors == 0 ]]; then
    echo "mypy hasn't found any code to check"
    exit 0
else
    # Calculate the ratio of errors
    percentage=$( python3 -c "print($nPasses/($nErrors+$nPasses))" )
fi

rm linecount.txt

if [[ $percentage > $THRESHOLD ]]; then
    echo "The number of type-errors is below the threshold. Check passed"
    
    # Create Badge
    anybadge -l Type\ Check -v Success -c green -f mypy.svg
    
    exit 0
else
    echo "mypy found enough errors in the code. You might want to check your code again"
    echo "Maybe take a look at mypy_output.txt"
    
    # Create Badge
    anybadge -l Type\ Check -v Failed -c red -f mypy.svg
    
    exit 1
fi
