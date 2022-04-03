import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric as tg

import e3nn
from e3nn import o3
from data_helpers import DataPeriodicNeighbors
from e3nn.nn.models.gate_points_2101 import Convolution, Network
from e3nn.o3 import Irreps

from pymatgen.core.structure import Structure
from pymatgen.ext.matproj import MPRester
import pymatgen.analysis.magnetism.analyzer as pg
import numpy as np
import pickle
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


LETTERS = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}


#################
# Model-related #
#################

class AtomEmbeddingAndSumLastLayer(torch.nn.Module):
    def __init__(self, atom_type_in, atom_type_out, model):
        super().__init__()
        self.linear = torch.nn.Linear(atom_type_in, 128)
        self.model = model
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128, 96)
        self.linear3 = torch.nn.Linear(96, 64)
        self.linear4 = torch.nn.Linear(64, 45)
        #self.linear5 = torch.nn.Linear(45, 32)
        #self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, *args, batch=None, **kwargs):
        output = self.linear(x)
        output = self.relu(output)
        print(f"Input: {x}")
        output = self.linear2(output)
        output = self.relu(output)
        output = self.linear3(output)
        output = self.relu(output)
        output = self.linear4(output)
        output = self.relu(output)
        output = self.model({'x': output, 'batch': batch, **kwargs})
        if batch is None:
            N = output.shape[0]
            batch = output.new_ones(N)
        print(f"Output: {output}")
        return output


def create_dataloaders(data, batch_size=1):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    index_tr, index_va, index_te = np.split(
        indices, [int(.8 * len(indices)), int(.9 * len(indices))])

    assert set(index_tr).isdisjoint(set(index_te))
    assert set(index_tr).isdisjoint(set(index_va))
    assert set(index_te).isdisjoint(set(index_va))

    pickle.dump((index_tr, index_va, index_te), open("data_splits.p", "wb"))

    dataloader = tg.loader.DataLoader([data[i] for i in index_tr], batch_size=batch_size, shuffle=True)
    dataloader_valid = tg.loader.DataLoader([data[i] for i in index_va], batch_size=batch_size)

    return index_tr, index_va, index_te, dataloader, dataloader_valid


def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step * (1 - math.exp(-t * rate / step)))


def evaluate(model, loss_fn, dataloader, device, cost_multiplier=1.0):
    model.eval()
    loss_cumulative = 0.
    with torch.no_grad():
        for _, d in enumerate(dataloader):
            d.to(device)
            output = model(x=d.x, batch=d.batch, pos=d.pos, z=d.pos.new_ones((d.pos.shape[0], 3)))
            if d.y.item() == 2:
                loss = cost_multiplier*loss_fn(output, d.y).cpu()
                print("Multiplied Loss Index \n")
            elif d.y.item() == 0 or d.y.item() == 1:
                loss = loss_fn(output, d.y).cpu()
                print("Standard Loss Index \n")
            else:
                print("Lost datapoint \n")
            loss_cumulative = loss_cumulative + loss.detach().item()
    return loss_cumulative / len(dataloader)


def train(model, optimizer, loss_fn, dataloader, dataloader_valid, scheduler, max_iter=101, device="cpu"):
    model.to(device)

    checkpoint_generator = loglinspace(3.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()
    dynamics = []

    for step in range(max_iter):
        model.train()
        loss_cumulative = 0.
        for j, d in enumerate(dataloader):
            d.to(device)
            output = model(x=d.x, batch=d.batch, pos=d.pos, z=d.pos.new_ones((d.pos.shape[0], 3)))
            loss = loss_fn(output, d.y).cpu()
            print(f"Iteration {step+1:4d}    batch {j+1:5d} / {len(dataloader):5d}   " + f"batch loss = {loss.data}", end="\r", flush=True)
            loss_cumulative = loss_cumulative + loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        wall = end_time - start_time

        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step

            valid_avg_loss = evaluate(model, loss_fn, dataloader_valid, device)
            train_avg_loss = evaluate(model, loss_fn, dataloader, device)

            dynamics.append({
                'step': step,
                'wall': wall,
                'batch': {'loss': loss.item(),},
                'valid': {'loss': valid_avg_loss,},
                'train': {'loss': train_avg_loss,},
            })

            yield {
                'dynamics': dynamics,
                'state': model.state_dict()
            }

            print(f"Iteration {step+1:4d}    batch {j+1:5d} / {len(dataloader):5d}   " +
                  f"train loss = {train_avg_loss:8.3f}   " +
                  f"valid loss = {valid_avg_loss:8.3f}   " +
                  f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")
            with open('loss.txt', 'a') as f:
                f.write(f"train average loss: {str(train_avg_loss)} \n")
                f.write(f" validation average loss: {str(valid_avg_loss)} \n")
        scheduler.step()


def plots(run_name):
    saved = torch.load(run_name+'_trial_run_full_data.torch')
    steps = [d['step'] + 1 for d in saved['dynamics']]
    valid = [d['valid']['loss'] for d in saved['dynamics']]
    train = [d['train']['loss'] for d in saved['dynamics']]

    plt.plot(steps, train, 'o-', label="train")
    plt.plot(steps, valid, 'o-', label="valid")
    plt.legend()
    plt.savefig(run_name+'_hist.png', dpi=300)


def run_write_data(stage, indices, data, model, device, formula_list_mp, id_list):
    composition_dict = {}
    sites_dict = {}
    y_pred = []
    # only used if stage=='testing'
    y_test = []
    y_score = []

    for _, index in enumerate(indices):
        d = tg.data.Batch.from_data_list([data[index]])
        d.to(device)
        # run the model on the current batch
        #   pos: position of the nodes (atoms)
        #   z: attributes of nodes, initialized as blank
        output = model(x=d.x, batch=d.batch, pos=d.pos, z=d.pos.new_ones((d.pos.shape[0], 3)))

        # if this is the test set, we should also prepare y_test and y_score to return
        if stage == 'testing':
            y_test.append(d.y.item())
            y_score.append(output)

        with open(f'{stage}_results.txt', 'a') as f:
            f.write(f"Output for below sample: {torch.exp(output)} \n")

        # find the output encoding
        if max(output[0][0], output[0][1], output[0][2]) == output[0][0]:
            output = 0
        elif max(output[0][0], output[0][1], output[0][2]) == output[0][1]:
            output = 1
        else:
            output = 2
        y_pred.append(output)
        with open(f'{stage}_results.txt', 'a') as f:
            f.write(f"{id_list[index]} {formula_list_mp[index]} Prediction: {output} Actual: {d.y} \n")
        
        correct_flag = d.y.item() == output

        # Accuracy per element calculation
        current_element = ""
        for char_index in range(len(formula_list_mp[index])):
            print("Entered Loop")
            formula = formula_list_mp[index]

            if formula[char_index] in LETTERS:
                current_element += formula[char_index]
                print(f"Using char: {formula[char_index]}")
                if char_index + 1 == len(formula) or formula[char_index + 1].isupper() or formula[char_index + 1] not in LETTERS:
                    print(f"printing to dict {current_element}")
                    if correct_flag:
                        current_entry = composition_dict.get(current_element, [0, 0])
                        current_entry = [current_entry[0] + 1, current_entry[1] + 1]
                    else:
                        current_entry = composition_dict.get(current_element, [0, 0])
                        current_entry = [current_entry[0], current_entry[1] + 1]
                    composition_dict[current_element] = current_entry
                    current_element = ""

    # Accuracy per element depiction
    with open(f'{stage}_composition_info.txt', 'a') as f:
        f.write(f"{stage.capitalize()} Composition Ratios: \n")
        for key, value in composition_dict.items():
            f.write(f"Element: {key} Ratio: {value[0]}/{value[1]} Fraction: {value[0]/value[1]}\n")

    # Accuracy per nsites depiction
    with open(f'{stage}_nsites_info.txt', 'a') as f:
        f.write(f"{stage.capitalize()} Nsites Info: \n")
        for key, value in sites_dict.items():
            f.write(f"nsites: {key} Ratio: {value[0]}/{value[1]} Fraction: {value[0]/value[1]}\n")

    if stage == 'testing':
        return y_test, y_pred
