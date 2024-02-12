import sys
import os

# print("usage: python *.py inpdb outpdbqt check_nphs_lps_waters(false or true) check_hydrogen(false or true)")
inpdb = sys.argv[1]
outpdbqt = sys.argv[2]
mol_type = sys.argv[3]
check_nphs_lps_waters = sys.argv[4]
check_hydrogen = sys.argv[5]

cmd = ""
if mol_type == 'ligand':
    cmd = "/src/mgltools_x86_64Linux2_1.5.6/bin/pythonsh /src/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py -l "+inpdb+" -o "+outpdbqt
elif mol_type == 'rna':
    cmd = "/src/mgltools_x86_64Linux2_1.5.6/bin/pythonsh /src/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r "+inpdb+" -o "+outpdbqt
else:
    print(f'unknown mol_type: {mol_type}')
    exit()
if check_nphs_lps_waters=="true":
    cmd += " -U nphs_lps_waters"
if check_hydrogen=="true":
    cmd += " -A checkhydrogens"

os.system(cmd)
