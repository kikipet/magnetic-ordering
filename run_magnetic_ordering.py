from data_preprocess import *
from magnetic_ordering import *

run_name = time.strftime("%y%m%d-%H%M", time.localtime())
get_dataset(run_name)
run_model(run_name, f'{run_name}_data.pt', f'{run_name}_formula_and_sites.p')
