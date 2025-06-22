#!/bin/bash

# Launch all i-PI instances in subdirectories
for dir in */; do
  (
    cd "$dir"
    i-pi input.xml > log.i-pi &
  )
done

sleep 10

# Launch drivers corresponding to each lambda directory
# Assumes you use lambda values like 0.0, 0.2, ..., 1.0 (i.e., f0, f2, ..., f10 sockets)
for x in {0..10..1}; do
  echo f${x}
  i-pi-py_driver -u -a f${x} -m MorseHarmonic -o 0.2,3.042,1.111,17.5 &
done

wait 
