#!/opt/conda/bin/python
import os
import sys
import shutil
import argparse
import datetime

parser = argparse.ArgumentParser(description='Process arguments passed to mgnet.py')
parser.add_argument('-i',  '--in_rna', type=str, required=True, help='input path for the target RNA in pdb format')
parser.add_argument('-o', '--out_dir', type=str, default='./', help='folder where the predicted magnesium ion site in pdb format will be saved, default is current folder')
args = parser.parse_args()

in_rna, out_dir = args.in_rna, args.out_dir

assert os.path.exists(in_rna), f'input RNA -> {in_rna} does not exist'
if not in_rna.endswith('.pdb'):
    print('input RNA should be in pdb format', flush=True)
os.makedirs(out_dir, exist_ok=True)

# prepare
tmp_dir = '/tmp/mgnet/'
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
os.makedirs(tmp_dir, exist_ok=True)
work_name = os.path.basename(in_rna).split('.')[0]
tmp_work_dir = f'/tmp/mgnet/{work_name}/'
os.makedirs(tmp_work_dir, exist_ok=True)

print(f'preparing {in_rna} ...', flush=True)

script_dir = '/src/MgNet/script/'
tmp_pdb       = f'{tmp_work_dir}/{work_name}.pdb'
tmp_rna_pdb   = f'{tmp_work_dir}/{work_name}_rna.pdb'
tmp_pdbqt     = f'{tmp_work_dir}/{work_name}.pdbqt'
shutil.copy(in_rna, tmp_pdb)
print("######## get RNA part ########", flush=True)
os.system(f'vmd -dispdev text -e {script_dir}/0-get_RNA_part.tcl -args {tmp_pdb} {tmp_rna_pdb}')
print("", flush=True)
print("######## remove altloc ########", flush=True)
os.system(f'python {script_dir}/1-remove_altloc.py {tmp_pdb} {tmp_pdb}')
os.system(f'python {script_dir}/1-remove_altloc.py {tmp_rna_pdb} {tmp_rna_pdb}')
print("######## generate pdbqt ########", flush=True)
os.system(f'python {script_dir}/2-generate_pdbqt.py {tmp_pdb} {tmp_pdbqt} rna true true')
print("######## voxelization ########", flush=True)
tmp_image_dir = f'{tmp_work_dir}/image/'
os.system(f'python {script_dir}/3-voxelization.py {tmp_rna_pdb} {tmp_pdbqt} {tmp_image_dir}')
print("######## predict, density, cluster ########", flush=True)
for cv in range(1, 6):
    tmp_result_dir = f'{tmp_work_dir}/result/cv{cv}/'
    os.system(f'python {script_dir}/4-predict.py {cv} {tmp_image_dir} {tmp_result_dir}/raw/')
    cubic_step = 0.5
    os.system(f'{script_dir}/density/density {tmp_result_dir}/raw/ {cubic_step}')
    DB_res_factor = 50
    kmean_cluster_size = 380
    os.system(f'python {script_dir}/5-cluster.py {DB_res_factor} {kmean_cluster_size} {tmp_rna_pdb} {tmp_result_dir}/density/ {out_dir}/{work_name}_model_{cv}_prediction.pdb')
    # os.system(f'python average_cluster.py {DB_res_factor} {kmean_cluster_size}')
print('######## MgNet completed ########', flush=True)