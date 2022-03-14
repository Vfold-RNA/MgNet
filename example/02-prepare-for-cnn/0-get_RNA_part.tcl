### Script to select and save RNA part only

if { $argc != 3 } {
    puts "usage: vmd -dispdev text -e get-RNA-part.tcl in_pdb out_rna_pdb"
    exit
}

set inpdb [ lindex $argv 0 ]
set outrnapdb [ lindex $argv 1 ]

### read the reference model
set ref [mol load pdb ${inpdb}]
#    set MG [atomselect $ref "resname MG"]
set RNA [atomselect $ref "nucleic and (altloc '' or altloc A)"]
#    set PROTEIN [atomselect $ref "protein"]
#    $MG writepdb ${outDir}/${id}_MG
$RNA writepdb ${outrnapdb}
#    $PROTEIN writepdb ${outDir}/${id}_PROTEIN
#    $MG delete
$RNA delete
#    $PROTEIN delete
mol delete $ref

exit
