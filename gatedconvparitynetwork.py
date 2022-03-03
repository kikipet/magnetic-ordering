from functools import partial

import torch

from e3nn import o3, rs
from kernels import Kernel
from e3nn.non_linearities import GatedBlockParity
from e3nn.non_linearities.rescaled_act import sigmoid, swish, tanh
# from e3nn.point.operations import Convolution
from e3nn.nn.models.gate_points_2101 import Convolution
from e3nn.radial import GaussianRadialModel

class GatedConvParityNetwork(torch.nn.Module):
    """A basic network architecture that alternates between Convolutions and GatedBlockParity nonlinearities.
    This network architecture should be used if you want to be fully equivariant to rotations, translations, and inversion (including mirrors).
    Args:
        Rs_in (rs.TY_RS_STRICT): Representation list of input data. Must have parity specified (1 for even, -1 for odd).
        mul (int > 0): multiplicity of irreps of intermediate data.
        Rs_out (rs.TY_RS_STRICT): Representation list of output data. Must have parity specified (1 for even, -1 for odd).
        lmax (int > 0): Maximum L used for spherical harmonic in kernel and sets max L for intermediate data.
        layers (int, optional): Number of convolution + GatedBlock layers. Defaults to 3.
        max_radius (float, optional): Maximum radius of radial basis functions. Used to initialize RadialFunction for kernel. Defaults to 1.0.
        number_of_basis (int, optional): Number of spaced Gaussian radial basis functions used for RadialFunction. Defaults to 3.
        radial_layers (int, optional): Number of dense layers applied to radial basis functions to create RadialFunction. Defaults to 3.
        kernel (kernel class, optional): Equivariant kernel class. Defaults to e3nn.kernel.Kernel.
        convolution (convolution class, optional): Equivariant convolution operation. Defaults to e3nn.point.operations.Convolution. For torch_geometric.data.Data input use e3nn.point.message_passing.Convolution.
        min_radius (float, optional): Minimum radius of radial basis functions. Used to initialize RadialFunction for kernel. Defaults to 0.0.
    Example::
        ### For use on torch_geometric.data.Data objects. Otherwise, do not replace convolution kwarg.
        N_atom_types = 3  # For example hydrogen, carbon, oxygen
        Rs_in = [(N_atom_types, 0, 0)]  # GatedBlock acts on irreps of SO(3) so parity (last number) should be omitted or 0.
        Rs_out = [(1, 1, 0)]  # Predict vectors on each atom.
        model_kwargs = {
            'convolution': e3nn.point.message_passing.Convolution,  # Use this convolution with data from e3nn.data.DataNeighbors or e3nn.data.DataPeriodicNeighbors
            'Rs_in': Rs_in,
            'Rs_out': Rs_out,
            'mul': 4,  # An example intermediate Rs might be [(4, 0, 1), (4, 0, -1), (4, 1, 1), (4, 1, -1), (4, 2, 1), (4, 2, -1)]
            'lmax': 2,  # Includes spherical harmonics 0, 1, and 2 in kernel and intermediate tensors go up to L=2.
            'layers': 3,  # 3 * (conv + nonlin) + conv layers
            'max_radius': r_max,  # Should match whatever is used to generate edges with e3nn.data.DataNeighbors or e3nn.data.DataPeriodicNeighbors
            'number_of_basis': 10,
        }
        model = GatedConvParityNetwork(**model_kwargs)
    """

    def __init__(self, Rs_in, mul, Rs_out, lmax, layers=3,
                 max_radius=1.0, number_of_basis=3, radial_layers=3,
                 kernel=Kernel, convolution=Convolution,
                 min_radius=0.0):
        super().__init__()

        R = partial(GaussianRadialModel, max_radius=max_radius,
                    number_of_basis=number_of_basis, h=100,
                    L=radial_layers, act=swish, min_radius=min_radius)
        K = partial(kernel, RadialModel=R, selection_rule=partial(
            o3.selection_rule_in_out_sh, lmax=lmax))

        modules = []

        Rs = rs.convention(Rs_in)
        for _ in range(layers):
            scalars = [(mul, l, p) for mul, l, p in [(mul, 0, +1),
                                                     (mul, 0, -1)] if rs.haslinearpath(Rs, l, p)]
            act_scalars = [(mul, swish if p == 1 else tanh)
                           for mul, l, p in scalars]

            nonscalars = [(mul, l, p) for l in range(1, lmax + 1)
                          for p in [+1, -1] if rs.haslinearpath(Rs, l, p)]
            if rs.haslinearpath(Rs, 0, +1):
                gates = [(rs.mul_dim(nonscalars), 0, +1)]
                act_gates = [(-1, sigmoid)]
            else:
                gates = [(rs.mul_dim(nonscalars), 0, -1)]
                act_gates = [(-1, tanh)]

            act = GatedBlockParity(scalars, act_scalars,
                                   gates, act_gates, nonscalars)
            conv = convolution(K(Rs, act.Rs_in))

            Rs = act.Rs_out

            block = torch.nn.ModuleList([conv, act])
            modules.append(block)

        self.layers = torch.nn.ModuleList(modules)

        K = partial(K, allow_unused_inputs=True)
        self.layers.append(convolution(K(Rs, Rs_out)))

    def forward(self, input, *args, **kwargs):
        """Consult the convolution operation used to initalize this class for specifics on what input should be given.
        Args:
            input (torch.tensor with dtype=torch.float64): input tensor with shape [batch, N, C] if default or [N, C] if e3nn.point.message_passing.Convolution
        Examples::
            ## Example for e3nn.point.message_passing.Convolution
            n_norm = 6  # Normalization of convolution to keep variance of output close to 1. Typically, this is equal to slightly larger than the dataset average for number of nearest neighbors.
            data = dh.DataNeighbors(features, Rs_in, pos, r_max, y=target)
            # data.x is input with shape [num_nodes, channels], data.edge_index has shape [2, num_edges], data.edge_attr stores relative distance vectors and has shape [num_edges, 3 (xyz)]
            output = model(data.x, data.edge_index, data.edge_attr, n_norm=n_norm)
            loss = ((output - target) ** 2).mean()
        """
        output = input
        N = args[0].shape[-2]
        if 'n_norm' not in kwargs:
            kwargs['n_norm'] = N

        for conv, act in self.layers[:-1]:
            output = conv(output, *args, **kwargs)
            output = act(output)

        layer = self.layers[-1]
        output = layer(output, *args, **kwargs)
        return output
