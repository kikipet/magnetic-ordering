import numpy as np
from pymatgen.ext.matproj import MPRester
m = MPRester(api_key='tF3g5QMe4wiUqymj', endpoint=None, include_user_agent=True)
vector_list=[]
magnetization_list=[]
atomnumber_list=[]
no_data_count=0
mpid_list=[]
for i in range(1,21):
    print(f"reading {i}\n")
    file1 = open(f'test_summary_stop/{i}testing_results.txt', 'r')
    Lines = file1.readlines()
    count = 0
    for line in Lines:
        if line[0]=='O':   #read output vector
            line=" ".join(line.split("  "))
            splitstring=line.split(" ")
            if len(splitstring)==9:
                vec=[float(splitstring[4][9:-1]),float(splitstring[5][0:-1]),float(splitstring[6][0:-3])]
            elif len(splitstring)==10:
                vec=[float(splitstring[5][0:-1]),float(splitstring[6][0:-1]),float(splitstring[7][0:-3])]
            else:
                print(splitstring)
            vec=np.array(vec)
            vec=vec/np.sum(abs(vec))
        if line[0]=='m':
            splitstring=line.split(" ")
            mpid=splitstring[0]   #read mpid
            try:   #grab magnetization and atom number with mpid
                structure=m.get_entry_by_material_id(mpid,property_data=['total_magnetization'])
                magnetization=structure.data['total_magnetization']
                magnetization_list.append(magnetization)
                structure=m.get_structure_by_material_id(mpid)
                atomnumber_list.append(len(structure))
                vector_list.append(vec)
                mpid_list.append(mpid)
            except:
                no_data_count+=1  #count the number of mpid that fail to grabbing magnetization and atom number
                

