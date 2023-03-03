#!/bin/bash
Dir="/home/fortissible/enaBrowserTools-1.1.0/python3/datatest/"
cd $Dir
for Folder in *
do 	
	if [ -f "${Dir}${Folder}/${Folder}_2.fastq.gz" ]; then
		gzip -d ${Dir}${Folder}/${Folder}_1.fastq.gz ${Dir}${Folder}/${Folder}_2.fastq.gz
		ariba run /home/fortissible/CARDdb/prefref_out ${Dir}${Folder}/${Folder}_1.fastq ${Dir}${Folder}/${Folder}_2.fastq /home/fortissible/CARDdb/ariba_run_out/ --force --verbose --noclean --threads 8
		continue
	else 
	    	echo "${Folder} fastq file does not exist or not completed yet."
	fi
done 

