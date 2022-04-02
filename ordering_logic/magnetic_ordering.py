# %%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from util import *

### Setup
run_name = (time.strftime("%y%m%d-%H%M", time.localtime()))

# if we already have a .pt file to pull data from
data = torch.load('220320-1356_data.pt')
formula_list_mp, sites_list, id_list = pickle.load(open('220320-1356_formula_and_sites.p', 'rb'))

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Set params + initialize model
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


len_element = 118
atom_types_dim = 3*len_element
embedding_dim = params['len_embed_feat']
lmax = 1
# Roughly the average number (over entire dataset) of nearest neighbors for a given atom
n_norm = 35

# num_atom_types scalars (L=0) with even parity
irreps_in = Irreps([(45, (0, 1))])
irreps_hidden = Irreps([(64, (0, 1)), (64, (1, 1))])  # not sure
irreps_out = Irreps([(3, (0, 1))])  # len_dos scalars (L=0) with even parity

model_kwargs = {
    "irreps_in": irreps_in,
    "irreps_hidden": irreps_hidden,
    "irreps_out": irreps_out,
    "irreps_node_attr": '3x0e',
    "irreps_edge_attr": '0e+1o',  # relative distance vector 
    # "irreps_edge_attr": '1o',  # relative distance vector 
    "layers": params['num_e3nn_layer'],
    "max_radius": params['max_radius'],
    "number_of_basis": params['num_basis'],
    "radial_layers": params['radial_layers'],
    "radial_neurons": 35, # not really sure
    "num_neighbors": 35,
    "num_nodes": 35 # not really sure
}
print(model_kwargs)


model = AtomEmbeddingAndSumLastLayer(atom_types_dim, embedding_dim, Network(**model_kwargs))
opt = torch.optim.AdamW(model.parameters(), lr=params['adamw_lr'], weight_decay=params['adamw_wd'])


### prepare data
index_tr, index_va, index_te, dataloader, dataloader_valid = create_dataloaders(data, batch_size=1)

with open('loss.txt', 'a') as f:
    f.write(f"Iteration: {identification_tag}")

loss_fn = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.78)


### train
for results in train(model, opt, loss_fn, dataloader, dataloader_valid, scheduler, device=device, max_iter=45):
    with open(run_name+'_trial_run_full_data.torch', 'wb') as f:
        results['model_kwargs'] = model_kwargs
        torch.save(results, f)

### create training accuracy plots
plots(run_name)

### run model train/valid/test data
run_write_data('training', index_tr, data, model, device, formula_list_mp, id_list)
run_write_data('validation', index_va, data, model, device, formula_list_mp, id_list)
y_test, y_pred = run_write_data('testing', index_te, data, model, device, formula_list_mp, id_list)

accuracy_score = accuracy_score(y_test, y_pred)

with open('y_pred.txt', 'a') as f:
    f.write("Predicted Values \n")
    f.write(str(y_pred))

with open('y_test.txt', 'a') as f:
    f.write("Actual Values \n")
    f.write(str(y_test))

with open('statistics.txt', 'a') as f:
    f.write("\n")
    f.write("Network Analytics: \n")
    f.write(f"Identification tag: {identification_tag}\n")
    f.write(f"Accuracy score: {accuracy_score}\n")
    f.write("Classification Report: \n")
    f.write(classification_report(y_test, y_pred, target_names=["NM", "AFM", "FM"]))
    f.write("\n")
