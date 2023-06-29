import sys
import os
import os.path
from pathlib import Path
from Bio.PDB import *

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
#from sklearn import metrics
#from sklearn.datasets.samples_generator import make_blobs
#from sklearn.preprocessing import StandardScaler

IMG_EXTENSIONS = ['.xyz']

parser = PDBParser()

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataset_list(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

print('usage: python cluster.py xx yy')

res_factor = int(sys.argv[1])
cluster_size = int(sys.argv[2])

output_folder = './average_result/'

file_list = dataset_list(output_folder)
#clusterPDBs = [f.strip().split("/")[-1][0:4] for f in file_list if f.find("_prediction_")!=-1]
for preFile in file_list:
    pdbname = preFile.strip().split("/")[-1]
    pdbname = pdbname[:pdbname.find('_RNA_prediction_40.xyz')]
    print(pdbname)

    clean_expt_structure = parser.get_structure(pdbname, '../../02-prepare/processed/'+pdbname+'_RNA.pdb')
    num_res = 0
    clean_expt_models = []
    for m in clean_expt_structure:
        clean_expt_models.append(m)
    for model in [clean_expt_models[0]]:
        for chain in model:
            for residue in chain:
                num_res += 1
    
    expt_structure = parser.get_structure(pdbname, '../../02-prepare/processed/'+pdbname+'.pdb')
    old_wrong_num_res = 0
    expt_models = []
    expt_MGs = []
    for m in expt_structure:
        expt_models.append(m)
    for model in [expt_models[0]]:
        chain_count = 0
        for chain in model:
            chain_count += 1
            residue_count = 0
            for residue in chain:
                old_wrong_num_res += 1
                residue_count += 1
                residue_name = residue.get_resname().strip()
                for atom in residue:
                    if atom.get_name()=="MG":
                        expt_MGs.append(atom)
    num_expt_MG = len(expt_MGs)
    print(pdbname, 'num_expt_MG',num_expt_MG)
    print(pdbname, 'num_res', num_res)
    print(pdbname, 'old_wrong_num_res', old_wrong_num_res)

    density_distribution = np.loadtxt(preFile)
    density_distribution = density_distribution[0:num_res*res_factor,:]
    atoms_coords = density_distribution[:,0:3:1]
    scores = density_distribution[:,-1::1]
    # print(atom_coords.shape)
    # print(atom_coords[0])
    # Compute DBSCAN
    db = DBSCAN(eps=1.0, min_samples=15).fit(atoms_coords)
    #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    #core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    print(atoms_coords.shape[0]==len(labels))
    
    #print(labels)
    #print(core_samples_mask)
    #print(db.core_sample_indices_)
    #print(db.components_)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    
    
    dbclusters = {}
    for i in range(n_clusters_):
        dbclusters[i] = {'coords':[],'scores':[]}
    dbclusters[-1] = {'coords':[],'scores':[]}
    for i in range(len(labels)):
        dbclusters[labels[i]]['coords'].append(atoms_coords[i])
        dbclusters[labels[i]]['scores'].append(scores[i][0])

    clusters = {}
    for i in range(n_clusters_):
        clusters[i] = {'coords':[],'scores':[]}
    for k,v in dbclusters.items():
        print(k,len(v['coords']))
        if k!=-1:
            num = int(len(v['coords'])/cluster_size)
            if num == 0:
                num_cluster = 1
            else:
                num_cluster = num
            kmeans = KMeans(n_clusters=num_cluster,random_state=0).fit(np.array(v['coords']))
            clusters[k]['coords'] = kmeans.cluster_centers_
            clusters[k]['scores'] = [0]*clusters[k]['coords'].shape[0]
            kmeans_labels = kmeans.labels_
            for i in range(len(kmeans_labels)):
                clusters[k]['scores'][kmeans_labels[i]] += v['scores'][i]
    hits = []
    for k,v in clusters.items():
        for i in range(v['coords'].shape[0]):
            hits.append({'coord':v['coords'][i],'score':v['scores'][i]})
    hits = sorted(hits, key = lambda i: -i['score'])
    if len(hits) >= 10:
        hits = hits[0:int(len(hits)*3/4)]
    with open(output_folder+pdbname+'_cluster.pdb','w') as f:
        serial = 0
        for h in hits:
            serial+=1
            f.write("HETATM%5d MG   MG  Z%4d    %8.3f%8.3f%8.3f  1.00%6.2f          Mg  \n" % (serial,serial,h['coord'][0],h['coord'][1],h['coord'][2],h['score']/hits[0]['score']))
    #print(clusters)
