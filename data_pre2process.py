import torch
import torch_scatter

from data_helpers import DataPeriodicNeighbors
# are there different sorts of convolutions for different datatypes?
from e3nn.nn.models.gate_points_2101 import Convolution, Network
from e3nn.o3 import Irreps

from pymatgen.ext.matproj import MPRester
import pymatgen.analysis.magnetism.analyzer as pg
import numpy as np
import pickle
from mendeleev import element

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

# m = MPRester(api_key='PqU1TATsbzHEOkSX', endpoint=None, notify_db_version=True, include_user_agent=True)
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


structures = structures_list_mp
y_values = y_values_mp
id_list = id_list_mp

pickle.dump((structures, y_values, formula_list_mp, sites_list, id_list), open('structure_info.p', 'wb'))

