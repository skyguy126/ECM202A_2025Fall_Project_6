#!/bin/bash
# Get the name of the current directory
current_dir=$(basename "$PWD")

SCENARIO=$1

if [ "$current_dir" != "data_parsing" ]; then
    echo "Please navigate to the data_parsing folder and try again."
    exit 1
fi

echo "Parsing scenario $SCENARIO..."
python parse_edge_events.py ../../demos/$SCENARIO/edge_data
python parse_inner_events.py ../../demos/$SCENARIO/inner_data
python get_final_events.py ../../demos/$SCENARIO

# Find all *_events.json files in the scenario folder
EVENT_FILES=$(find "../../demos/${SCENARIO}" -maxdepth 1 -name "*_events.json")

echo "Final events files generated in ../../demos/${SCENARIO}:"
for file in $EVENT_FILES; do
    echo "  $file"
done
