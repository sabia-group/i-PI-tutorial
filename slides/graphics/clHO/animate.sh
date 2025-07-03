#!/bin/bash

convert -delay 5 -loop 0 frame_*.png $(ls frame_*.png | sort -r | tail -n +2) -layers Optimize temp.gif
ffmpeg -i temp.gif -movflags faststart -pix_fmt yuv420p clHO.mp4
mv temp.gif ../clHO.gif
cp clHO.mp4 ..

