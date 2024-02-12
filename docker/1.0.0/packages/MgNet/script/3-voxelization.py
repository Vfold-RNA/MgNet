import numpy as np
#import math
from htmd.molecule.util import boundingBox
from htmd.molecule import vdw
import ctypes
import htmd.home
from numba import cuda, jit
import numba
from math import sqrt, exp
import datetime

import sys
import os
import os.path
import subprocess
from htmd.ui import *
from htmd.molecule import voxeldescriptors
from Bio.PDB import *

#libdir = htmd.home.home(libDir=True)
#print(libdir)
#occupancylib = ctypes.cdll.LoadLibrary(os.path.join(libdir, "occupancy_ext.so"))
occupancylib = np.ctypeslib.load_library("libdescriptor_ext", "/src/MgNet/script/descriptor/")

thread_number = 8
image_size = 48
half_image_length = image_size/2.0
image_resolution = 0.5


parser = PDBParser()


def getVoxelDescriptors(mol, usercenters=None, voxelsize=1, buffer=0, channels_sigmas=None, channels_values=None, method='C'):
    N = None
    if usercenters is None:
        # Calculate the bbox and the number of voxels
        [bbm, bbM] = boundingBox(mol)
        bbm -= buffer
        bbM += buffer
        N = np.ceil((bbM - bbm) / voxelsize).astype(int) + 1

        # Calculate grid centers
        centers = voxeldescriptors._getGridCenters(bbm, N, voxelsize)
        centers = centers.reshape(np.prod(N), 3)
    else:
        centers = usercenters

    # Calculate features
    if method.upper() == 'C':
        features = _getOccupancyC(mol.coords[:, :, mol.frame].copy(), centers, channels_sigmas, channels_values)
    elif method.upper() == 'CUDA':
        # print(channels_sigmas)
        features = voxeldescriptors._getOccupancyCUDA(mol.coords[:, :, mol.frame].copy(), centers, channels_sigmas, channels_values)
    elif method.upper() == 'NUMBA':
        features = voxeldescriptors._getOccupancyNUMBA(mol.coords[:, :, mol.frame].copy(), centers, channels_sigmas, channels_values, 5)


    if N is None:
        return features, centers
    else:
        return features, centers, N

def _getOccupancyC(coords, centers, channels_sigmas, channels_values):
    """ Calls the C code to calculate the voxels values for each property."""
    centers = centers.astype(np.float64)
    channels_sigmas = channels_sigmas.astype(np.float64)
    channels_values = channels_values.astype(np.float64)
    coords = coords.astype(np.float32)
    nchannels = channels_sigmas.shape[1]
    occus = np.zeros((centers.shape[0], nchannels))
    #print(channels_sigmas)
    # print("Time:", datetime.datetime.now())
    # double *centers, float *coords, double *channels_sigmas, double *occus, int number_centers, int number_atoms, int number_channels, double resolution, int thread_number
    occupancylib.descriptor_ext(centers.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       coords.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       channels_sigmas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       channels_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       occus.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       ctypes.c_int(occus.shape[0]),  # n of centers
                       ctypes.c_int(coords.shape[0]),  # n of atoms
                       ctypes.c_int(nchannels), # n of channels
                    #    ctypes.c_double(image_resolution), # resolution i.e. cubic_step
                       ctypes.c_int(thread_number)) # thread_number
    # print("Time:", datetime.datetime.now())
    return occus

def PointDescriptors(mol, point, size, resolution=1.0, channels_sigmas=None, channels_values=None):
    # print(channels.dtype)
    size = np.array(size)
    bbm = point - (size * resolution) / 2 + resolution / 2  # Position centers + half res.
    inbox = voxeldescriptors._getGridCenters(bbm, size, resolution)
    inbox = inbox.reshape(np.prod(size), 3)
    # print(channels)
    features, _ = getVoxelDescriptors(mol, usercenters=inbox, channels_sigmas=channels_sigmas, channels_values=channels_values, method='C')
    features = features.reshape((size[0], size[1], size[2], features.shape[1]))
    return features








#ATOMorder = ("OP1", "OP2", "O2'", "O3'", "O4'", "O5'", "A_N1", "A_N3", "A_N6", "A_N7", "A_N9", "G_O6", "G_N1", "G_N2", "G_N3", "G_N7", "G_N9", "C_O2", "C_N1", "C_N3", "C_N4", "U_O2", "U_O4", "U_N1", "U_N3")
#def getAtomChannel(mol):
#    """ define our own atom types to specific properties
#    Parameters
#    """
#    from collections import OrderedDict
#    props = OrderedDict()
#    # print(mol.element)
#    # atomnames = np.array([el.upper() for el in mol.name])
#    atomnames = mol.name
#    resnames = mol.resname
#    # print(atomnames)
#
#    props['OP1'] = atomnames == 'OP1'
#    props['OP2'] = atomnames == 'OP2'
#
#    props["O2'"] = atomnames == "O2'"
#    props["O3'"] = atomnames == "O3'"
#    props["O4'"] = atomnames == "O4'"
#    props["O5'"] = atomnames == "O5'"
#
#    props["A_N1"] = (resnames == 'A') & (atomnames == 'N1')
#    props["A_N3"] = (resnames == 'A') & (atomnames == 'N3')
#    props["A_N6"] = (resnames == 'A') & (atomnames == 'N6')
#    props["A_N7"] = (resnames == 'A') & (atomnames == 'N7')
#    props["A_N9"] = (resnames == 'A') & (atomnames == 'N9')
#
#    props["G_O6"] = (resnames == 'G') & (atomnames == 'O6')
#    props["G_N1"] = (resnames == 'G') & (atomnames == 'N1')
#    props["G_N2"] = (resnames == 'G') & (atomnames == 'N2')
#    props["G_N3"] = (resnames == 'G') & (atomnames == 'N3')
#    props["G_N7"] = (resnames == 'G') & (atomnames == 'N7')
#    props["G_N9"] = (resnames == 'G') & (atomnames == 'N9')
#
#    props["C_O2"] = (resnames == 'C') & (atomnames == 'O2')
#    props["C_N1"] = (resnames == 'C') & (atomnames == 'N1')
#    props["C_N3"] = (resnames == 'C') & (atomnames == 'N3')
#    props["C_N4"] = (resnames == 'C') & (atomnames == 'N4')
#
#    props["U_O2"] = (resnames == 'U') & (atomnames == 'O2')
#    props["U_O4"] = (resnames == 'U') & (atomnames == 'O4')
#    props["U_N1"] = (resnames == 'U') & (atomnames == 'N1')
#    props["U_N3"] = (resnames == 'U') & (atomnames == 'N3')
#
#    channels = np.zeros((len(atomnames), len(props)), dtype=bool)
#    for i, p in enumerate(ATOMorder):
#        channels[:, i] = props[p]
#    return channels


#KDEEPorder = ('hydrophobic', 'aromatic', 'hbond_acceptor', 'hbond_donor', 'positive_ionizable',
#'negative_ionizable', 'metal', 'occupancies')
#def getKDEEPChannel(mol):
#    """ Matches PDBQT atom types to specific properties
#    Parameters
#    """
#    from collections import OrderedDict
#    props = OrderedDict()
#    # print(mol.element)
#    # elements = np.array([el.upper() for el in mol.element])
#    elements = np.array([el for el in mol.element])
#    # print(elements)
#
#    props['hydrophobic'] = (elements == 'C') | (elements == 'A')
#    props['aromatic'] = elements == 'A'
#    props['hbond_acceptor'] = (elements == 'NA') | (elements == 'NS') | (elements == 'OA') | (elements == 'OS') | (elements == 'SA')
#    props['hbond_donor'] = voxeldescriptors._findDonors(mol, mol._getBonds())
#    props['positive_ionizable'] = (mol.charge > 0) & (elements != 'MG') & (elements != 'Mg')
#    props['negative_ionizable'] = (mol.charge < 0) & (elements != 'MG') & (elements != 'Mg')
#    props['metal'] = (elements == 'ZN') | (elements == 'MN') | (elements == 'CA') | (elements == 'FE') | (elements == 'Zn') | (elements == 'Mn') | (elements == 'Ca') | (elements == 'Fe')
#    props['occupancies'] = (elements != 'H') & (elements != 'HS') & (elements != 'HD') & (elements != 'MG') & (elements != 'Mg')
#
#    channels = np.zeros((len(elements), len(props)), dtype=bool)
#    for i, p in enumerate(KDEEPorder):
#        channels[:, i] = props[p]
#
#    if channels.dtype == bool:
#        # Calculate for each channel the atom sigmas
#        sigmas = voxeldescriptors._getRadii(mol)
#        channels_sigmas = sigmas[:, np.newaxis] * channels.cpoy().astype(float)
#        channels_values = 1.0 * channels.cpoy().astype(float)
#
#    return channels_sigmas,channels_values

MYorder = ('occupancies', 'partial_charges')
def getMYChannel(mol):
    from collections import OrderedDict
    props = OrderedDict()
    # print(mol.element)
    # elements = np.array([el.upper() for el in mol.element])
    elements = np.array([el for el in mol.element])
    # print(elements)

    #props['positive_ionizable'] = (mol.charge > 0) & (elements != 'MG') & (elements != 'Mg') & (elements != 'H') & (elements != 'HS') & (elements != 'HD')
    #props['negative_ionizable'] = (mol.charge < 0) & (elements != 'MG') & (elements != 'Mg') & (elements != 'H') & (elements != 'HS') & (elements != 'HD')
    props['occupancies'] = (elements != 'MG') & (elements != 'Mg')
    props['partial_charges']  = (elements != 'MG') & (elements != 'Mg')

    channels = np.zeros((len(elements), len(props)), dtype=bool)
    for i, p in enumerate(MYorder):
        channels[:, i] = props[p]

    # Calculate for each channel the atom sigmas
    sigmas = voxeldescriptors._getRadii(mol)
    channels_sigmas = sigmas[:, np.newaxis] * channels.copy().astype(float)

    channels_values = 1.0 * channels.copy().astype(float)
    #print(channels_values[:,-1].shape, mol.charge.shape)
    channels_values[:,-1] = mol.charge
    # 0.7 is set to all partial charge atom radius
    channels_sigmas[:,-1] = 0.65 * channels.copy().astype(float)[:,-1]

    #print(channels_sigmas[:,-1], channels_values[:,-1])


    return channels_sigmas,channels_values



def getMGChannel(mol,MG_radius):
    from collections import OrderedDict
    props = OrderedDict()
    # elements = np.array([el.upper() for el in mol.element])
    elements = np.array([el for el in mol.element])
    props['MG'] = (elements == 'MG') | (elements == 'Mg')
    channels = np.zeros((len(elements), len(props)), dtype=bool)
    channels[:, 0] = props['MG']
    channels_sigmas = MG_radius * channels.copy().astype(float)
    channels_values = 1.0 * channels.copy().astype(float)
    # print(channels)
    return channels_sigmas,channels_values


def getATOMMAPChannel(mol):
    from collections import OrderedDict
    props = OrderedDict()
    # elements = np.array([el.upper() for el in mol.element])
    elements = np.array([el for el in mol.element])
    props['MG'] = (elements == 'MG') | (elements == 'Mg')
    channels = np.zeros((len(elements), len(props)), dtype=bool)
    channels[:, 0] = props['MG']
    channels_sigmas = MG_radius * channels.copy().astype(float)
    channels_values = 1.0 * channels.copy().astype(float)
    # print(channels)
    return channels_sigmas,channels_values



# def getRNAEdgeChannel(mol,edge_radius):
#     from collections import OrderedDict
#     props = OrderedDict()
#     # elements = np.array([el.upper() for el in mol.element])
#     elements = np.array([el for el in mol.element])
#     props['RNA_edge'] = (elements != 'H') & (elements != 'HS') & (elements != 'HD') & (elements != 'MG') & (elements != 'Mg')
#     channels = np.zeros((len(elements), len(props)), dtype=bool)
#     channels[:, 0] = props['RNA_edge']
#     if channels.dtype == bool:
#         sigmas = voxeldescriptors._getRadii(mol)
#         # Calculate for each channel the atom sigmas
#         sigmas = sigmas + edge_radius
#         # print(sigmas)
#     channels = sigmas[:, np.newaxis] * channels.astype(float)
#     return channels


def save_image(mol,origin,inverse_transformation_matrix,save_path,PDBname,chainid,resid,resname):
    # make the transform here
#    image_channels = getKDEEPChannel(mol)
    image_channels_sigmas,image_channels_values = getMYChannel(mol)
    target_channels_sigmas,target_channels_values = getMGChannel(mol,2.5)

    image_features = PointDescriptors(mol, point=(0,0,0), size=[image_size,image_size,image_size], resolution=image_resolution, channels_sigmas=image_channels_sigmas, channels_values=image_channels_values)
    target_features = PointDescriptors(mol, point=(0,0,0), size=[image_size,image_size,image_size], resolution=image_resolution, channels_sigmas=target_channels_sigmas, channels_values=target_channels_values)

    #print(image_features.shape)
    ##print(image_features)
    #print(np.count_nonzero(image_features))

    image=np.zeros((image_features.shape[-1],image_size,image_size,image_size), dtype=float)
    target=np.zeros((target_features.shape[-1],image_size,image_size,image_size), dtype=float)

    #print(image.shape)
    #print(np.count_nonzero(image))
    for i in range(image_features.shape[-1]):
        image[i,:,:,:]=image_features[:,:,:,i]
#    for i in range(target1_features.shape[-1]):
#        target1[i,:,:,:]=target1_features[:,:,:,i]
#    for i in range(target2_features.shape[-1]):
#        target2[i,:,:,:]=target2_features[:,:,:,i]
    for i in range(target_features.shape[-1]):
        target[i,:,:,:]=target_features[:,:,:,i]
#    for i in range(target4_features.shape[-1]):
#        target4[i,:,:,:]=target4_features[:,:,:,i]
#    for i in range(target5_features.shape[-1]):
#        target5[i,:,:,:]=target5_features[:,:,:,i]

    # print(PDBname, chainid, resid, resname, np.count_nonzero(image), np.count_nonzero(image[-1]))

    # print(image.shape,target1.shape)
    # edge_non_zero_dict = {"0.0":np.count_nonzero(edge1), "0.5":np.count_nonzero(edge2), "1.0":np.count_nonzero(edge3)}
    image_non_zero = np.count_nonzero(image)
    mg_non_zero = np.count_nonzero(target)
    print(PDBname, chainid, resid, resname, 'image non zero', image_non_zero, 'Mg non zero', mg_non_zero)
    channel_non_zero = {}
    for i, p in enumerate(MYorder):
        channel_non_zero[p] = np.count_nonzero(image[i])
        print(p, np.count_nonzero(image[i]), end=" ")
    print("")
#    pic = {'image':image, 'occupied':image[-1], 'label':label_dict, 'PDB':PDBname, 'chainid':chainid, 'resid':resid, 'resname':resname, 'origin':origin, 'inverse_transformation_matrix':inverse_transformation_matrix, 'image_non_zero_count':np.count_nonzero(image), 'occupied_non_zero_count':np.count_nonzero(image[-1]), 'label_non_zero_count':label_non_zero_dict}

    pic = {'image':image, 'label':target, 'PDB':PDBname, 'chainid':chainid, 'resid':resid, 'resname':resname, 'origin':origin, 'inverse_transformation_matrix':inverse_transformation_matrix, 'image_non_zero_count':image_non_zero, 'label_non_zero_count': mg_non_zero, 'channel_non_zero_dict':channel_non_zero}
    np.save(save_path,pic)



















#########################################################################################
print('usage: python 3-voxelization.py inrnapdb inpdbqt save_folder')

inrnapdb = sys.argv[1]
inpdbqt = sys.argv[2]
save_folder = sys.argv[3]

PDBname = inrnapdb[inrnapdb.rfind('/')+1:-4]

os.makedirs(save_folder+'/'+PDBname, exist_ok=True)

N1baseList = ["C","U"]
N9baseList = ["A","G"]



rna_structure = parser.get_structure(PDBname, inrnapdb)
whole_structure_path = inpdbqt

original = Molecule(whole_structure_path)

rna_models = []
for m in rna_structure:
    rna_models.append(m)
for model in [rna_models[0]]:
    chain_count = 0
    for chain in model:
        chain_count += 1
        residue_count = 0
        for residue in chain:
            residue_count += 1
            residue_name = residue.get_resname().strip()
            save_path = save_folder+'/'+PDBname+'/'+str(chain_count)+'_'+str(residue_count)+'_'+residue_name+'.npy'
            C1Prime=None
            O4Prime=None
            NitrogenBase=None
            for atom in residue:
                if atom.get_name()=="C1'":
                    C1Prime=atom
                if atom.get_name() == "O4'":
                    O4Prime=atom
                if residue_name in N9baseList and atom.get_name() == "N9":
                    NitrogenBase=atom
                if residue_name in N1baseList and atom.get_name() == "N1":
                    NitrogenBase=atom
            if C1Prime!=None and O4Prime!=None and NitrogenBase!=None:
                # print(C1Prime.shape,O4Prime.shape,NitrogenBase.shape)
                axis_x = NitrogenBase.get_coord()-C1Prime.get_coord()
                axis_y = O4Prime.get_coord()-C1Prime.get_coord()
                axis_z = np.cross(axis_x, axis_y)
                axis_y = np.cross(axis_z, axis_x)
                origin = (NitrogenBase.get_coord()+C1Prime.get_coord())/2.0
                # print('origin',origin)
                # print('x*y',np.dot(axis_x,axis_y))
                # print('z*y',np.dot(axis_z,axis_y))
                # print('z*x',np.dot(axis_z,axis_y))
                axis_x = axis_x / np.linalg.norm(axis_x)
                axis_y = axis_y / np.linalg.norm(axis_y)
                axis_z = axis_z / np.linalg.norm(axis_z)
                original_copy = original.copy()
                # print('axis',axis_x,axis_y,axis_z)
                transformation_matrix = np.array([axis_x,axis_y,axis_z]).transpose()
                inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

                #print(np.dot(transformation_matrix,inverse_transformation_matrix))

                # rotated_original = original_copy.rotateBy(M=rotationMatrix([0, 1, 0], 1.57),center=origin)

                coords = original_copy.get('coords', sel='all')
                # print(coords[0],origin,(coords - origin)[0])
                newcoords = coords - origin
                # print(np.dot(newcoords[0],axis_x),np.dot(newcoords[0],axis_y),np.dot(newcoords[0],axis_z))
                newcoords = np.dot(newcoords, transformation_matrix)
                original_copy.set('coords', newcoords, sel='all')
                transformed_coords = original_copy.get('coords', sel='all')
                # print(transformed_coords[0])
                #print(transformation_matrix)
                #print(upper_corner_site,origin,lower_corner_site)
                #print('after',upper_corner_site,origin,lower_corner_site)
                os.makedirs(save_folder+PDBname, exist_ok=True)
                # if not (PDBname=='3t1y' and chain_count <= 1 and residue_count <= 793):
                save_image(mol=original_copy,origin=origin,inverse_transformation_matrix=inverse_transformation_matrix,save_path=save_path,PDBname=PDBname,chainid=str(chain_count),resid=str(residue_count),resname=residue_name)
                # print(axis_x,axis_y,axis_z)
            else:
                print(PDBname,'chaincount',chain_count,'rescount',residue_count,'resname',residue_name,C1Prime,O4Prime,NitrogenBase)
print('--------------------------------------------------------------------------')


