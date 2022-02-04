import matplotlib.pyplot as plt
import ternary
from matplotlib import gridspec
import numpy as np
import matplotlib as mpl
magnetization=np.load('magnetization2.npy') #total magnetization
vector=np.load('vector2.npy')  #output vector in ternary form
atomnumber=np.load('atomnumber.npy') #atom number
magnetization=magnetization/atomnumber  #magnetization per atom
fig=plt.figure(constrained_layout=True,figsize=(8,4))
fontsize = 10
offset = 0.14
gs = gridspec.GridSpec(2, 2,wspace=0.2,hspace=0.1,height_ratios= [7,1],figure=fig)
ax = fig.add_subplot(gs[0, 0])

figure, tax = ternary.figure(ax=ax,scale=1.0)
#figure.set_size_inches(10, 10)
tax.boundary(linewidth=1)
tax.gridlines(multiple=0.2, color="black")
tax.right_corner_label("NM", fontsize=fontsize)
tax.top_corner_label("AFM", fontsize=fontsize)
tax.left_corner_label("FM/FiM", fontsize=fontsize)
tax.get_axes().axis('off')
cutoff=max(magnetization)
magnetization_cutoff=[x if x<cutoff else cutoff for x in magnetization]
sc=tax.scatter(vector, s=2, edgecolors='none',c=magnetization_cutoff, cmap=mpl.cm.cool,vmin=0,vmax=cutoff)
tax.legend(loc='upper left',fontsize=fontsize)
ax2 = fig.add_subplot(gs[1,0])
cmap = mpl.cm.cool
norm = mpl.colors.Normalize(vmin=0, vmax=cutoff)

cb1 = mpl.colorbar.ColorbarBase(ax=ax2, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
cb1.set_label('magnetization per atom')
ax3 = fig.add_subplot(gs[:,1])
ax3.set_xlabel('magnetization per atom')
plt.hist(magnetization_cutoff,100)
#plt.savefig('magnetization_unit_cf0_002.pdf',format='pdf',dpi=300,bbox_inches='tight')