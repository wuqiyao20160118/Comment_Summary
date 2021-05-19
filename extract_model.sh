#! /bin/bash

for ((i=0; i<15; i++));
	do
		python extract_model.py $i
	done
