#!/bin/bash

# Set the command you want to execute on the files
command_to_execute="pytest"

# Target directory
cd pythonfiles/tests/

# Loop through the files that match the pattern
for file in test_*.py; do
    if [ -f "$file" ]; then
        # Execute the command on each matching file
        $command_to_execute "$file"
    fi
done