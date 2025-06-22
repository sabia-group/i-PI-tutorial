#!/bin/bash


lambdas=(0.0 0.2 0.4 0.6 0.8 1.0)
socket_ids=(0 2 4 6 8 10)

# Clean up sockets
for x in "${socket_ids[@]}"; do
  rm -f /tmp/ipi_f${x}
done

# Launch all i-PI instances in subdirectories
for dir in ${lambdas[@]}; do
  (
    cd "$dir"
    i-pi input.xml > log.i-pi &
  )
done

sleep 5

# Launch drivers corresponding to each lambda directory
# Assumes you use lambda values like 0.0, 0.2, ..., 1.0 (i.e., f0, f2, ..., f10 sockets)
for i in "${!lambdas[@]}"; do
  x=${socket_ids[$i]}
  dir=${lambdas[$i]}
  i-pi-py_driver -u -a f${x} -m MorseHarmonic -o 0.2,0.75,1.11,1.0 &> ${dir}/log.driver &
done

