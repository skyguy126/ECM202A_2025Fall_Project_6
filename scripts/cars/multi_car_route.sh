#!/bin/bash

COLORS=("red" "green" "blue" "yellow" "cyan" "magenta" "white" "black")
SHUFFLED_COLORS=($(printf "%s\n" "${COLORS[@]}" | shuf))


# -------------------------------------------
# check traffic lights flag
# -------------------------------------------
USE_T=false
for arg in "$@"; do
    if [ "$arg" = "-t" ]; then
        USE_T=true
        break
    fi
done

# Array to store PIDs
pids=()
# tm port base and index
BASE_TM_PORT=8000
index=0

# -------------------------------------------
# ctrl+c cleanup function
# -------------------------------------------
cleanup() {
    echo "Stopping all cars..."
    for pid in "${pids[@]}"; do
        kill "$pid"
    done
    wait
    exit
}
# Trap Ctrl+C
trap cleanup SIGINT

# -------------------------------------------
# run one_car_route.py for each car id
# -------------------------------------------
# Loop over all arguments, skipping -t
for route_id in "$@"; do
    if [ "$route_id" = "-t" ]; then
        continue
    fi

    # Compute TM port for this car
    TM_PORT=$((BASE_TM_PORT + index))

    # Pick a unique color
    CAR_COLOR="${SHUFFLED_COLORS[$index]}"

     # set car name
    CAR_NAME="Route_${route_id}_${CAR_COLOR}"
    

    if [ "$USE_T" = true ]; then
        python ./one_car_route.py --read --id "$route_id" -t --tm-port "$TM_PORT" --name "$CAR_NAME" --color "$CAR_COLOR" &
    else
        python ./one_car_route.py --read --id "$route_id" --tm-port "$TM_PORT" --name "$CAR_NAME" --color "$CAR_COLOR" &
    fi

    # Save PID
    pids+=($!)

    # Increment index for next car
    index=$((index + 1))

done

# Wait for all background processes to finish
wait
