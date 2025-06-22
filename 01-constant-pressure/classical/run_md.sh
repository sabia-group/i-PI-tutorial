#!/bin/bash 
python -u $(which i-pi) npt-md.xml &> log.md &
sleep 1
i-pi-driver -u -a ph2-driver -m sg -o 15 &> log.driver &
