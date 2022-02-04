#import itertools 
#import pymatgen as mg
#import pymatgen.io
#from pymatgen.core.structure import Structure
#from pymatgen.ext.matproj import MPRester
#import pymatgen.analysis.magnetism.analyzer as pg
import matplotlib.pyplot as plt
#import glob
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np
#m = MPRester(api_key='tF3g5QMe4wiUqymj', endpoint=None, include_user_agent=True)
#run_name='211026-1221'
#steplist=['0', '1', '4', '9', '14', '19', '24', '29', '34', '39', '44']
#stepnumlist=[0, 1, 4, 9, 14, 19, 24, 29, 34, 39, 44]
accuracylist=[]
#confusionmatrix=[[0]*3]*3
#print(confusionmatrix)
confusionmatrix=[[0,0,0],[0,0,0],[0,0,0]]
crm=[[0,0,0],[0,0,0],[0,0,0]]
for i in range(1,21): 
    file1 = open(str(i)+'testing_results.txt', 'r')
    Lines = file1.readlines()
 
    count_line = 0
    count_true=0
    count_total=0
    mpid_list=[]
    pred_list=[]
    actual_list=[]
    #space_group_list=[]
    # Strips the newline character
    for line in Lines:
        count_line += 1
        if line[0]=='m':
            splitstring=line.split(" ")
            mpid=splitstring[0]
            pred=splitstring[3]
            actual=splitstring[5][8]
            confusionmatrix[int(actual)][int(pred)]+=1
            #structure=m.get_structure_by_material_id(mpid)
            #group=structure.get_space_group_info()
            #space_group_list.append(group[1])
            mpid_list.append(mpid)
            pred_list.append(splitstring[3])
            actual_list.append(splitstring[5][8])
            if splitstring[3]==splitstring[5][8]:
                count_true+=1
            count_total+=1
    print('classification report: {i}')
    a=classification_report(actual_list,pred_list,output_dict=True)
    crm[0][0]+=a['0']['precision']
    crm[0][1]+=a['0']['recall']
    crm[0][2]+=a['0']['f1-score']
    crm[1][0]+=a['1']['precision']
    crm[1][1]+=a['1']['recall']
    crm[1][2]+=a['1']['f1-score']
    crm[2][0]+=a['2']['precision']
    crm[2][1]+=a['2']['recall']
    crm[2][2]+=a['2']['f1-score']

    #print(f'f1 score: {f1_score(actual_list,pred_list)}')
    accuracylist.append(count_true/count_total)

#valid=accuracylist
print(crm)
crm=[[x/20 for x in y] for y in crm]
print(crm)
#print(accuracylist)
#print(confusionmatrix)
confusionmatrix_average=[[x/20 for x in y] for y in confusionmatrix]
#print(confusionmatrix_average)
#lattice_name=['Triclinic','Monoclinic','Orthorhombic','Tetragonal','Trigonal','Hexagonal','Cubic']
#lattice_range=[[1,2],[3,15],[16,74],[75,142],[143,167],[168,194],[195,230]]
#total_lattice=[0,0,0,0,0,0,0]
#correct_lattice=[0,0,0,0,0,0,0]
#print(f'in total {count_total} samples, {count_true} of which are predicted correctly, accuracy is {count_true/count_total:.2f}')

#
#[126, 233, 115, 41, 52, 9, 25]
#[94, 179, 88, 32, 32, 4, 18]
    
    #print("Line{}: {}".format(count, line.strip()))
#print(mpid_list[3])
#print(pred_list[3])
#print(actual_list[3])
#print(space_group_list[:100])

#m = MPRester(api_key='tF3g5QMe4wiUqymj', endpoint=None, include_user_agent=True)
#structures = m.query(criteria={"elements": {"$in":['Ga', 'Tm', 'Y', 'Dy', 'Nb', 'Pu', 'Th', 'Er', 'U', 'Cr', 'Sc', 'Pr', 'Re', 'Ni', 'Np', 'Nd', 'Yb', 'Ce', 'Ti', 'Mo', 'Cu', 'Fe', 'Sm', 'Gd', 'V', 'Co', 'Eu', 'Ho', 'Mn', 'Os', 'Tb', 'Ir', 'Pt', 'Rh', 'Ru']}, 'blessed_tasks.GGA+U Static': {'$exists': True}}, properties=["material_id","pretty_formula","structure","blessed_tasks", "nsites"])


#[[0.9097959556599884, 0.9192676635455745, 0.9143364708802754], [0.7031620432127331, 0.6751828695946998, 0.6872388979896138], [0.6839156346210122, 0.6984392618563626, 0.6902003374241954]] classification report