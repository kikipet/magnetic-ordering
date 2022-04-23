# magnetic-ordering

Classify magnetic orderings using the current version of e3nn. This is an updated version of the code used by Merker et al.

The most recent run was at 71.7% accuracy overall; more information in statistics.txt.

## Organization

Data preprocessing and model training/testing are in two separate files. Both files can be run separately, or both at once using `run_magnetic_ordering.py`.

## Dependencies

Non-extensive list of notable dependencies:

* e3nn >= 0.4.4
* torch==1.10.2
* torch-geometric (installed with cuda version 11.3)
* numpy==1.22.1
* pymatgen==2022.3.29
* ase==3.22.1
