### for viewing purposes only

import math
from typing import Dict, Union

import torch
from torch_geometric.data import Data
from torch_cluster import radius_graph
from torch_scatter import scatter

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode


class Convolution(torch.nn.Module):
    r"""equivariant convolution

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input node features

    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the node attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes

    irreps_out : `e3nn.o3.Irreps` or None
        representation of the output node features

    number_of_basis : int
        number of basis on which the edge length are projected

    radial_layers : int
        number of hidden layers in the radial fully connected network

    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network

    num_neighbors : float
        typical number of nodes convolved over
    """

    def __init__(
        self,
        irreps_in,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_out,
        number_of_basis,
        radial_layers,
        radial_neurons,
        num_neighbors
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        self.sc = FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_node_attr, self.irreps_out)

        self.lin1 = FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_node_attr, self.irreps_in)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            self.irreps_in,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet([number_of_basis] + radial_layers *
                                    [radial_neurons] + [tp.weight_numel], torch.nn.functional.silu)
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(
            irreps_mid, self.irreps_node_attr, self.irreps_out)

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedded) -> torch.Tensor:
        weight = self.fc(edge_length_embedded)

        x = node_input

        s = self.sc(x, node_attr)
        x = self.lin1(x, node_attr)

        edge_features = self.tp(x[edge_src], edge_attr, weight)
        x = scatter(edge_features, edge_dst, dim=0,
                    dim_size=x.shape[0]).div(self.num_neighbors**0.5)

        x = self.lin2(x, node_attr)

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        m = self.sc.output_mask
        c_x = (1 - m) + c_x * m
        return c_s * s + c_x * x


class Network(torch.nn.Module):
    r"""equivariant neural network

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps` or None
        representation of the input features
        can be set to ``None`` if nodes don't have input features

    irreps_hidden : `e3nn.o3.Irreps`
        representation of the hidden features

    irreps_out : `e3nn.o3.Irreps`
        representation of the output features

    irreps_node_attr : `e3nn.o3.Irreps` or None
        representation of the nodes attributes
        can be set to ``None`` if nodes don't have attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes
        the edge attributes are :math:`h(r) Y(\vec r / r)`
        where :math:`h` is a smooth function that goes to zero at ``max_radius``
        and :math:`Y` are the spherical harmonics polynomials

    layers : int
        number of gates (non linearities)

    max_radius : float
        maximum radius for the convolution

    number_of_basis : int
        number of basis on which the edge length are projected

    radial_layers : int
        number of hidden layers in the radial fully connected network

    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network

    num_neighbors : float
        typical number of nodes at a distance ``max_radius``

    num_nodes : float
        typical number of nodes in a graph
    """

    def __init__(
        self,
        irreps_in,
        irreps_hidden,
        irreps_out,
        irreps_node_attr,
        irreps_edge_attr,
        layers,
        max_radius,
        number_of_basis,
        radial_layers,
        radial_neurons,
        num_neighbors,
        num_nodes,
        reduce_output=True,
    ) -> None:
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output

        self.irreps_in = o3.Irreps(
            irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(
            irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        self.input_has_node_in = (irreps_in is not None)
        self.input_has_node_attr = (irreps_node_attr is not None)

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps(
            "0e")

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.layers = torch.nn.ModuleList()

        for _ in range(layers):
            irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(
                irreps, self.irreps_edge_attr, ir)])
            irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(
                irreps, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(
                irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars, [act[ir.p]
                                 for _, ir in irreps_scalars],  # scalar
                irreps_gates, [act_gates[ir.p]
                               for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors
            )
            irreps = gate.irreps_out
            self.layers.append(Compose(conv, gate))

        self.layers.append(
            Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors
            )
        )


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False