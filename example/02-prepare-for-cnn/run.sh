#!/bin/bash

rm -rf image/temp_48_small_partial_charge_radius/*
rm -rf processed
rm run.log
for name in `cat ../01-input/pdb_list.txt`
do
    echo $name
    mkdir -p processed
    inpdb=../01-input/${name}.pdb
    outpdb=./processed/${name}.pdb
    outpdbqt=./processed/${name}.pdbqt
    outrnapdb=./processed/${name}_RNA.pdb
    echo "--->copy inpdb"
    echo "cp ${inpdb} ${outpdb}"
    cp ${inpdb} ${outpdb}
    echo "---> get RNA part"
    echo "vmd -dispdev text -e 0-get_RNA_part.tcl -args ${inpdb} ${outrnapdb}"
    vmd -dispdev text -e 0-get_RNA_part.tcl -args ${inpdb} ${outrnapdb}
    echo "---> remove altloc"
    echo "python 1-remove_altloc.py ${outpdb} ${outpdb}"
    python 1-remove_altloc.py ${outpdb} ${outpdb}
    echo "python 1-remove_altloc.py ${outrnapdb} ${outrnapdb}"
    python 1-remove_altloc.py ${outrnapdb} ${outrnapdb}
    echo "---> generate pdbqt"
    echo "python 2-generate_pdbqt.py ${outpdb} ${outpdbqt} rna true true"
    python 2-generate_pdbqt.py ${outpdb} ${outpdbqt} rna true true
    echo "---> voxelization"
    python 3-voxelization.py ${outrnapdb} ${outpdbqt} ./image/temp_48_small_partial_charge_radius/
    echo "---> finish ${name}"
    echo "##########################################"
done

