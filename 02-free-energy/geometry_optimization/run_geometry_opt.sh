#!/bin/bash

# Start i-PI in the background and redirect output to a log file
i-pi input.xml &> log.i-pi &

# Give i-PI a moment to start up
sleep 2

# Launch the driver with appropriate options
i-pi-py_driver -u -a geop -m MorseHarmonic -o 0.2,0.75,1.11,1.0 &> log.driver &
