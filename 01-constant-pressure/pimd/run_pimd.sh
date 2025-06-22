#!/bin/bash 
python -u $(which i-pi) npt-pimd.xml &> log.pimd &
sleep 5
for i in `seq 1 4`; do
   i-pi-driver -u -a ph2-driver-pi -m sg -o 15 &> log.driver.$i &
done
