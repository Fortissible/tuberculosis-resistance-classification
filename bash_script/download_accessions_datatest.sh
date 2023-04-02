#!/bin/bash
File="accession_eth_pz.txt"
Lines=$(cat $File)
for Line in $Lines
do 
	python3 enaDataGet.py -f fastq -d ./datatest $Line
done
