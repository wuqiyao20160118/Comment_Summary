#! /bin/bash

for ((i=0; i<15; i++));
	do
		python final_summary.py $i
	done
