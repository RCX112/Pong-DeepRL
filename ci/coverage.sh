#!/bin/bash

# Create Log file
touch coverage_output.txt

# Counter for errors
errorValues=""

# Run Coverage command with pytest
for directory in $( cat directories.txt ); do
    echo Checking $directory
    coverage run -m pytest $directory --disable-warnings -q
    coverage report >> coverage_output.txt
    echo $( cat coverage_output.txt | tail -1 )
    
    errorValues=$errors" "$( cat coverage_output.txt | tail -1 | awk '{ print $NF }' )
    errorValues=${errorValues::-1}  # Remove Pecent Sign
done

# Calculate Average Coverage
arr=($errorValues)
length=${#arr[@]}

averageCoverage=0
for i in "${arr[@]}"; do
  sum=$(( sum + i))
done

averageCoverage=$(( sum / length ))

#Apply Percentage to Badge
if [[ $averageCoverage > 70 ]]; then
    anybadge -l Coverage -v $averageCoverage -c green -f coverage.svg
elif [[ $averageCoverage > 50 ]]; then
    anybadge -l Coverage -v $averageCoverage -c yellow -f coverage.svg
elif [[ $averageCoverage > 30 ]]; then
    anybadge -l Coverage -v $averageCoverage -c orange -f coverage.svg
else
    anybadge -l Coverage -v $averageCoverage -c red -f coverage.svg
fi

exit 0
