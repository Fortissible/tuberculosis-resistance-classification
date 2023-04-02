#!/bin/bash
Dir="/home/fortissible/enaBrowserTools-1.1.0/python3/datatest/"
cd $Dir
for Folder in *
do 	
	if [ -f "${Dir}${Folder}/${Folder}_2.fastq" ]; then
		gzip -v ${Dir}${Folder}/${Folder}_1.fastq ${Dir}${Folder}/${Folder}_2.fastq
		echo "${Folder} selesai di kompress."
	else 
		echo "${Folder}_2.fastq.gz sudah di compress atau terdapat masalah."
	fi
done 
