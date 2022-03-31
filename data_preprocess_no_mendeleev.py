# %%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 09:10:12 2020

@author: Harry, Helena, and Linh
"""

import torch

from data_helpers import DataPeriodicNeighbors

from pymatgen.ext.matproj import MPRester
import pymatgen.analysis.magnetism.analyzer as pg
import numpy as np
import pickle

import time


# %% Process Materials Project Data
order_list_mp = []
structures_list_mp = []
formula_list_mp = []
sites_list = []
id_list_mp = []
y_values_mp = []
order_encode = {"NM": 0, "AFM": 1, "FM": 2, "FiM": 2}

magnetic_atoms = ['Ga', 'Tm', 'Y', 'Dy', 'Nb', 'Pu', 'Th', 'Er', 'U',
                  'Cr', 'Sc', 'Pr', 'Re', 'Ni', 'Np', 'Nd', 'Yb', 'Ce',
                  'Ti', 'Mo', 'Cu', 'Fe', 'Sm', 'Gd', 'V', 'Co', 'Eu',
                  'Ho', 'Mn', 'Os', 'Tb', 'Ir', 'Pt', 'Rh', 'Ru']

m = MPRester(endpoint=None, include_user_agent=True)
structures = m.query(criteria={"elements": {"$in": magnetic_atoms}, 'blessed_tasks.GGA+U Static': {
                     '$exists': True}}, properties=["material_id", "pretty_formula", "structure", "blessed_tasks", "nsites"])

structures_copy = structures.copy()
for struc in structures_copy:
    if len(struc["structure"]) > 250:
        structures.remove(struc)
        print("MP Structure Deleted")

# %%
order_list = []
for i in range(len(structures)):
    order = pg.CollinearMagneticStructureAnalyzer(structures[i]["structure"])
    order_list.append(order.ordering.name)
id_NM = []
id_FM = []
id_AFM = []
for i in range(len(structures)):
    if order_list[i] == 'NM':
        id_NM.append(i)
    if order_list[i] == 'AFM':
        id_AFM.append(i)
    if order_list[i] == 'FM' or order_list[i] == 'FiM':
        id_FM.append(i)
np.random.shuffle(id_FM)
np.random.shuffle(id_NM)
np.random.shuffle(id_AFM)
id_AFM, id_AFM_to_delete = np.split(id_AFM, [int(len(id_AFM))])
id_NM, id_NM_to_delete = np.split(id_NM, [int(1.2*len(id_AFM))])
id_FM, id_FM_to_delete = np.split(id_FM, [int(1.2*len(id_AFM))])

structures_mp = [structures[i] for i in id_NM] + [structures[j]
                                                  for j in id_FM] + [structures[k] for k in id_AFM]
np.random.shuffle(structures_mp)


for structure in structures_mp:
    analyzed_structure = pg.CollinearMagneticStructureAnalyzer(
        structure["structure"])
    order_list_mp.append(analyzed_structure.ordering)
    structures_list_mp.append(structure["structure"])
    formula_list_mp.append(structure["pretty_formula"])
    id_list_mp.append(structure["material_id"])
    sites_list.append(structure["nsites"])

for order in order_list_mp:
    y_values_mp.append(order_encode[order.name])

# structures, y_values, formula_list_mp, sites_list, id_list = pickle.load(open('structure_info.p', 'rb')
# )
structures = structures_list_mp
y_values = y_values_mp
id_list = id_list_mp

elements = pickle.load(open('element_info.p', 'rb'))

torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {'len_embed_feat': 64,
          'num_channel_irrep': 32,
          'num_e3nn_layer': 2,
          'max_radius': 5,
          'num_basis': 10,
          'adamw_lr': 0.005,
          'adamw_wd': 0.03,
          'radial_layers': 3
          }

# Used for debugging
identification_tag = "1:1:1.1 Relu wd:0.03 4 Linear"
cost_multiplier = 1.0

print('Length of embedding feature vector: {:3d} \n'.format(params.get('len_embed_feat')) +
      'Number of channels per irreducible representation: {:3d} \n'.format(params.get('num_channel_irrep')) +
      'Number of tensor field convolution layers: {:3d} \n'.format(params.get('num_e3nn_layer')) +
      'Maximum radius: {:3.1f} \n'.format(params.get('max_radius')) +
      'Number of basis: {:3d} \n'.format(params.get('num_basis')) +
      'AdamW optimizer learning rate: {:.4f} \n'.format(params.get('adamw_lr')) +
      'AdamW optimizer weight decay coefficient: {:.4f}'.format(
          params.get('adamw_wd'))
      )


run_name = (time.strftime("%y%m%d-%H%M", time.localtime()))


# structures = structures_list_mp
# y_values = y_values_mp
# id_list = id_list_mp


species = set()
count = 0
for struct in structures[:]:
    try:
        species = species.union(list(set(map(str, struct.species))))
        count += 1
    except:
        print(count)
        count += 1
        continue
species = sorted(list(species))
print("Distinct atomic species ", len(species))

len_element = 118
atom_types_dim = 3*len_element
embedding_dim = params['len_embed_feat']
lmax = 1
# Roughly the average number (over entire dataset) of nearest neighbors for a given atom
n_norm = 35


data = []
count = 0
indices_to_delete = []
for i, struct in enumerate(structures):
    try:
        print(
            f"Encoding sample {i+1:5d}/{len(structures):5d}", end="\r", flush=True)
        input = torch.zeros(len(struct), 3*len_element)
        for j, site in enumerate(struct):
            input[j, int(elements[str(site.specie)]['atomic_number'])] = elements[str(site.specie)]['atomic_radius']
            input[j, len_element + int(elements[str(site.specie)]['atomic_number']) + 1] = elements[str(site.specie)]['en_pauling']
            input[j, 2*len_element + int(elements[str(site.specie)]['atomic_number']) + 1] = elements[str(site.specie)]['dipole_polarizability']
        data.append(DataPeriodicNeighbors(
            x=input, Rs_in=None,
            pos=torch.tensor(struct.cart_coords.copy()), lattice=torch.tensor(struct.lattice.matrix.copy()),
            r_max=params['max_radius'],
            y=(torch.tensor([y_values[i]])).to(torch.long),
            n_norm=n_norm,
            order = struct
        ))

        count += 1
    except Exception as e:
        indices_to_delete.append(i)
        print(f"Error: {count} {e}", end="\n")
        count += 1
        continue


struc_dictionary = dict()
for i in range(len(structures)):
    struc_dictionary[i] = structures[i]

id_dictionary = dict()
for i in range(len(id_list)):
    id_dictionary[i] = id_list[i]

for i in indices_to_delete:
    del struc_dictionary[i]
    del id_dictionary[i]

structures2 = []
for i in range(len(structures)):
    if i in struc_dictionary.keys():
        structures2.append(struc_dictionary[i])
structures = structures2

id2 = []
for i in range(len(id_list)):
    if i in id_dictionary.keys():
        id2.append(id_dictionary[i])
id_list = id2

compound_list = []
for i, struc in enumerate(structures):
    str_struc = (str(struc))
    count = 0
    while str_struc[count] != ":":
        count += 1
    str_struc = str_struc[count+2:]
    count = 0
    while str_struc[count:count+3] != "abc":
        count += 1
    str_struc = str_struc[:count]
    compound_list.append(str_struc)

torch.save(data, run_name+'_data.pt')
pickle.dump((formula_list_mp, sites_list, id_list), open(run_name+'_formula_and_sites.p', 'wb'))
