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
import numpy as np
import pickle

import time

ORDER_ENCODE = {"NM": 0, "AFM": 1, "FM": 2, "FiM": 2}

MAGNETIC_ATOMS = ['Ga', 'Tm', 'Y', 'Dy', 'Nb', 'Pu', 'Th', 'Er', 'U',
                  'Cr', 'Sc', 'Pr', 'Re', 'Ni', 'Np', 'Nd', 'Yb', 'Ce',
                  'Ti', 'Mo', 'Cu', 'Fe', 'Sm', 'Gd', 'V', 'Co', 'Eu',
                  'Ho', 'Mn', 'Os', 'Tb', 'Ir', 'Pt', 'Rh', 'Ru']

ELEMENTS = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

torch.set_default_dtype(torch.float64)

def get_element_info():
    from mendeleev import element

    element_info = {}
    for e in ELEMENTS:
        element_info[e] = {
            'atomic_number': element(e).atomic_number,
            'atomic_radius': element(e).atomic_radius,
            'en_pauling': element(e).en_pauling,
            'dipole_polarizability': element(e).dipole_polarizability,
        }

    pickle.dump(element_info, open('elements.p', 'wb'))
    return element_info

def get_dataset(run_name, save_query=False, local_data=False, local_elements=True):
    ### Get raw data
    if local_data:
        structures = pickle.load(structures, open(f'mpquery_{run_name}.p', 'rb'))
    else:
        m = MPRester(endpoint=None, include_user_agent=True)
        structures = m.query(criteria={"elements": {"$in": MAGNETIC_ATOMS}, 'blessed_tasks.GGA+U Static': {'$exists': True}}, 
                            properties=["material_id", "pretty_formula", "structure", "nsites", "magnetism"])
        structures = list(filter(lambda struc: len(struc["structure"]) <= 250, structures))
        if save_query: pickle.dump(structures, open(f'mpquery_{run_name}.p', 'wb'))

    ### Split data + shuffle
    id_NM = []
    id_FM = []
    id_AFM = []
    for i in range(len(structures)):
        if structures[i]["magnetism"]["ordering"] == 'NM':
            id_NM.append(i)
        elif structures[i]["magnetism"]["ordering"] == 'AFM':
            id_AFM.append(i)
        elif structures[i]["magnetism"]["ordering"] in ['FM', 'FiM']:
            id_FM.append(i)
    np.random.shuffle(id_FM)
    np.random.shuffle(id_NM)
    np.random.shuffle(id_AFM)
    id_AFM, _ = np.split(id_AFM, [int(len(id_AFM))])
    id_NM, _ = np.split(id_NM, [int(1.2*len(id_AFM))])
    id_FM, _ = np.split(id_FM, [int(1.2*len(id_AFM))])

    structures_mp = [structures[i] for i in id_NM] + [structures[j] for j in id_FM] + [structures[k] for k in id_AFM]
    np.random.shuffle(structures_mp)

    ### Gather more specific information about structures
    order_list_mp = []
    formula_list_mp = []
    sites_list = []
    structures_list = []
    y_values = []
    id_list = []

    for structure in structures_mp:
        order_list_mp.append(structure["magnetism"]["ordering"])
        structures_list.append(structure["structure"])
        formula_list_mp.append(structure["pretty_formula"])
        id_list.append(structure["material_id"])
        sites_list.append(structure["nsites"])

    y_values = [ORDER_ENCODE[order] for order in order_list_mp]

    ### Get element information
    if local_elements:
        elements = pickle.load(open('element_info.p', 'rb'))
    else:
        elements = get_element_info()

    species = set()
    for i, struct in enumerate(structures_list):
        try:
            species = species.union(set(map(str, struct.species)))
        except:
            print(i)
            continue
    print("Distinct atomic species ", len(species))

    ### Create final dataset
    len_element = len(ELEMENTS)
    # Roughly the average number (over entire dataset) of nearest neighbors for a given atom
    n_norm = 35
    max_radius = 5
    data = []
    indices_to_delete = set()
    for i, struct in enumerate(structures_list):
        try:
            print(f"Encoding sample {i+1:5d}/{len(structures_list):5d}", end="\r", flush=True)
            input_data = torch.zeros(len(struct), 3*len_element)
            for j, site in enumerate(struct):
                input_data[j, int(elements[str(site.specie)]['atomic_number'])] = elements[str(site.specie)]['atomic_radius']
                input_data[j, len_element + int(elements[str(site.specie)]['atomic_number']) + 1] = elements[str(site.specie)]['en_pauling']
                input_data[j, 2*len_element + int(elements[str(site.specie)]['atomic_number']) + 1] = elements[str(site.specie)]['dipole_polarizability']
            data.append(DataPeriodicNeighbors(
                x=input_data,
                Rs_in=None,
                pos=torch.tensor(struct.cart_coords.copy()), lattice=torch.tensor(struct.lattice.matrix.copy()),
                r_max=max_radius,
                y=(torch.tensor([y_values[i]])).to(torch.long),
                n_norm=n_norm,
                order=struct
            ))
        except Exception as e:
            indices_to_delete.add(i)
            print(f"Error: {i} {e}", end="\n")
            continue

    id_list_f = [id_list[i] for i in range(len(id_list)) if i not in indices_to_delete]

    torch.save(data, run_name+'_data.pt')
    pickle.dump((formula_list_mp, sites_list, id_list_f), open(run_name+'_formula_and_sites.p', 'wb'))


if __name__ == "__main__":
    get_dataset(time.strftime("%y%m%d-%H%M", time.localtime()))
