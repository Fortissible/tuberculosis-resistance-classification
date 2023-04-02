DirDatatest="/home/fortissible/enaBrowserTools-1.1.0/python3/datatest/"
DirAribaRunOut="/home/fortissible/CARDdb/"
cd $DirDatatest
for Read in *
do	
	if [ -d ${DirAribaRunOut}ariba_run_out_${Read} ]; then
		if [ -f ${DirAribaRunOut}ariba_run_out_${Read}/assembled_genes.fa.gz ]; then
			continue
		else 
			echo "$Read AMR extraction is incomplete!"
		fi
	else 
	    	echo "ariba_run_out_${Read} folder does not exist!"
	    	continue
	fi
done
