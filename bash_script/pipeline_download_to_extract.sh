#!/bin/bash
Dir="/mnt/c/DataSkripsi/1501-2000/"
ReportDir="${Dir}ariba_run_report/"
OutputDir="${Dir}ariba_run_output/"
cd $Dir
for Folder in *
do
	if [ -f "${OutputDir}${Folder}/report.tsv" ]; then
		if [ -d "${OutputDir}${Folder}/Logs" ]; then
			rm -r ${OutputDir}${Folder}/Logs ${OutputDir}${Folder}/clusters ${OutputDir}${Folder}/.fails
			echo "Removed Folder for Accession number ${Folder}"
		fi
		if [ ! -f "${ReportDir}${Folder}_report.tsv" ]; then
			cp -u ${OutputDir}${Folder}/report.tsv ${ReportDir}${Folder}_report.tsv
			echo "Copying report result for accession ${Folder}..."
		fi
		continue
	elif [ -f "${Dir}${Folder}/${Folder}_2.fastq.gz" ] && [ -f "${Dir}${Folder}/${Folder}_1.fastq.gz" ]; then
		echo "Unzipping ${Folder}.fastq.gz..."
		gzip -dk ${Dir}${Folder}/${Folder}_1.fastq.gz ${Dir}${Folder}/${Folder}_2.fastq.gz
		ariba run /home/fortissible/skripsi/ariba_prefref/prepareref.out.card ${Dir}${Folder}/${Folder}_1.fastq ${Dir}${Folder}/${Folder}_2.fastq ${Dir}ariba_run_output/${Folder}/ --force --noclean --threads 8
		rm -r ${Dir}${Folder}/${Folder}_1.fastq ${Dir}${Folder}/${Folder}_2.fastq
		rm -r ${OutputDir}${Folder}/Logs ${OutputDir}${Folder}/clusters ${OutputDir}${Folder}/.fails
		cp -u ${OutputDir}${Folder}/report.tsv ${ReportDir}${Folder}_report.tsv
		echo "${Folder} extraction complete, now copying report.tsv and delete Ariba run tmp folder..."
		continue
	else
		echo "${Folder} fastq.gz file does not exist or download not completed yet."
	fi
done

