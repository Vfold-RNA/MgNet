#import numpy as np
#import math
import sys
import os
import os.path
import subprocess

print("usage: python *.py input_folder file_specifier(e.g, _protein.pdb) check_nphs_lps_waters(false or true) check_hydrogen(false or true)")
input_folder = sys.argv[1]
file_specifier = sys.argv[2]
check_nphs_lps_waters = sys.argv[3]
check_hydrogen = sys.argv[4]


def is_image_file(filename):
    return ((filename.find(file_specifier)!=-1) and filename.find('.pdbqt')==-1)

def dataset_list(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir,followlinks=True)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images



assert os.path.isdir(input_folder), 'Error: no '+input_folder+' directory found!'

for img_path in dataset_list(input_folder):
    path = img_path
    file_name = path[path.rfind('/')+1:]
    input_format = file_name[file_name.rfind('.')+1:]
    output_path = path[:path.rfind('.')]+'.pdbqt'
    cmd = ""
    if file_name.find("_ligand.")!=-1:
        cmd = "~/.local/mgltools_x86_64Linux2_1.5.6/bin/pythonsh ~/.local/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py -l "+path+" -o "+output_path
    else:
        cmd = "~/.local/mgltools_x86_64Linux2_1.5.6/bin/pythonsh ~/.local/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r "+path+" -o "+output_path
    if check_nphs_lps_waters=="true":
        cmd += " -U nphs_lps_waters"
    if check_hydrogen=="true":
        cmd += " -A checkhydrogens"

    error_outfile = open("pdbqt_error.log",'a')
    normal_outfile = open("pdbqt_normal.log",'a')
    error_outfile.write(file_name+'\n')
    normal_outfile.write(file_name+'\n')
    subprocess.run([cmd], shell=True, stdout=normal_outfile, stderr=error_outfile)
    normal_outfile.write("=====================================================================\n")
    error_outfile.write("=====================================================================\n")
normal_outfile.write("#########################################################################\n")
error_outfile.write("#########################################################################\n")
normal_outfile.close()
error_outfile.close()

