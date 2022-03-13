### Script to select and save RNA part only
set inDir original
set outDir processed

file mkdir ${outDir}

### first get the list of all pdb names
set PDB_files [glob ./original/*.pdb]
set PDBNames {}
puts [llength $PDB_files]
foreach f $PDB_files {
    puts $f
    set PDBid [string range [lindex [split $f /] end] 0 3]
    puts $PDBid
    lappend PDBNames $PDBid
}
puts $PDBNames
puts [llength $PDBNames]

foreach id $PDBNames {
    file mkdir ${outDir}/${id}
    file copy ./${inDir}/${id}.pdb ${outDir}/${id}/${id}_original.pdb
    ### read the reference model
    set ref [mol load pdb ./${inDir}/${id}.pdb]
#    set MG [atomselect $ref "resname MG"]
    set RNA [atomselect $ref "nucleic and (altloc '' or altloc A)"]
#    set PROTEIN [atomselect $ref "protein"]

#    $MG writepdb ${outDir}/${id}_MG.pdb
    $RNA writepdb ${outDir}/${id}/${id}_RNA.pdb
#    $PROTEIN writepdb ${outDir}/${id}_PROTEIN.pdb

#    $MG delete
    $RNA delete
#    $PROTEIN delete
    mol delete $ref
}

exit
