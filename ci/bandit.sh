#!/bin/bash

# Create Log file
touch bandit_output.txt

# Counter for errors
containsErrors=""

# Run Bandit
# Ignore the following:
#   - Assert (Contradiction with Pytest)
#   - Try and Except
for directory in $( cat directories.txt ); do
    echo Checking $directory
    bandit -r $directory -q --skip B101,B110,B112 >> bandit_output.txt
    
    echo $( cat bandit_output.txt)
    containsErrors=$countainsErrors" "$( cat bandit_output.txt )
done

if [[ "$containsErrors" == " " ]]; then
    echo "Bandit exited with 0 errors"
    
    # Create Badge
    anybadge -l Security -v Success -c green -f bandit.svg
    
    exit 0
else
    echo "Bandit found errors in the code. You might want to check your code again"
    echo "Maybe take a look at bandit_output.txt"
    
    # Create Badge
    anybadge -l Security -v Failed -c red -f bandit.svg
    
    exit 1
fi
