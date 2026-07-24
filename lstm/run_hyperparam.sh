#!/bin/bash


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"



# Backup current configuration
if [ -f "config.yml" ]; then
    cp config.yml config.yml.backup
    echo "Backed up config.yml to config.yml.backup"
fi

# Run grid search
python hyperparameter_search.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then

else
    cp config.yml.backup config.yml
    exit $EXIT_CODE
fi
