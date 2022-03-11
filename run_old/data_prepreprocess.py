# %%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 09:10:12 2020

@author: Harry, Helena, and Linh
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import torch_scatter

import e3nn
from e3nn import rs, o3
from e3nn.point.data_helpers import DataPeriodicNeighbors
from e3nn.networks import GatedConvParityNetwork
from e3nn.kernel_mod import Kernel
from e3nn.point.message_passing import Convolution

import pymatgen as mg
import pymatgen.io
from pymatgen.core.structure import Structure
from pymatgen.ext.matproj import MPRester
import pymatgen.analysis.magnetism.analyzer as pg
import numpy as np
import pickle
from mendeleev import element
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import io
import random
import math
import sys
import time
import os
import datetime

def load_obj(filename):
    return pickle.load(open(filename, "rb"))

def save_obj(obj, filename):
    pickle.dump(obj, open(filename, "wb"))

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

# structures_copy = structures.copy()
# for struc in structures_copy:
#     if len(struc["structure"])>250:
#         structures.remove(struc)
#         print("MP Structure Deleted")

# #%%
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
    id_list_mp.append(structure["material_id"])

torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {'len_embed_feat': 64,
          'num_channel_irrep': 32,
          'num_e3nn_layer': 2,
          'max_radius': 5,
          'num_basis': 10,
          'adamw_lr': 0.005,
          'adamw_wd': 0.03
          }

# #Used for debugging
identification_tag = "1:1:1.1 Relu wd:0.03 4 Linear"
cost_multiplier = 1.0


run_name = (time.strftime("%y%m%d-%H%M", time.localtime()))


id_list = id_list_mp


save_obj((structures, structures_mp, id_list, formula_list_mp, sites_list), 'structure_lists.p')
