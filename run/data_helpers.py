# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long, not-callable

import torch
import torch_geometric as tg
import numpy as np
import ase.neighborlist


class DataNeighbors(tg.data.Data):
    """Builds a graph from points with features for message passing convolutions.
    Wraps ``_neighbor_list_and_relative_vec``; see it for edge conventions.
    Args:
        x (torch.Tensor shape [N, M]): per-node M-dimensional features.
        pos (torch.Tensor shape [N, 3]): node positions.
        r_max (float): neighbor cutoff radius.
        cell (ase.Cell/ndarray [3,3], optional): cell (box) for the points. Defaults to ``None``.
        pbc (bool or 3-tuple of bool, optional): whether to apply periodic boundary conditions to all or each of the three cell vector directions. Defaults to ``False``.
        self_interaction (bool, optional): whether to include self edges for points. Defaults to ``True``.
        **kwargs (optional): other attributes to pass to the ``torch_geometric.data.Data`` constructor.
    """

    def __init__(self, x, pos, r_max, cell=None, pbc=False, self_interaction=True, **kwargs):
        # BEGIN EDIT
        if pos is None:
            pos = []
        # END EDIT
        edge_index, edge_attr = _neighbor_list_and_relative_vec(
            pos,
            r_max,
            self_interaction=self_interaction,
            cell=cell,
            pbc=pbc
        )
        if cell is not None:
            # For compatability: the old DataPeriodicNeighbors put the cell
            # in the Data object as `lattice`.
            kwargs['lattice'] = cell
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, **kwargs)

    @classmethod
    def from_ase(cls, atoms, r_max, features=None, **kwargs):
        """Build a ``DataNeighbors`` from an ``ase.Atoms`` object.
        Respects ``atoms``'s ``pbc`` and ``cell``.
        Args:
            atoms (ase.Atoms): the input.
            r_max (float): neighbor cutoff radius.
            features (torch.Tensor shape [N, M], optional): per-atom M-dimensional feature vectors. If ``None`` (the default), uses a one-hot encoding of the species present in ``atoms``.
            **kwargs (optional): other arguments for the ``DataNeighbors`` constructor.
        Returns:
            A ``DataNeighbors``.
        """
        if features is None:
            _, species_ids = np.unique(
                atoms.get_atomic_numbers(), return_inverse=True)
            features = torch.nn.functional.one_hot(torch.as_tensor(
                species_ids)).to(dtype=torch.get_default_dtype())
        return cls(
            x=features,
            pos=torch.as_tensor(atoms.positions),
            r_max=r_max,
            cell=atoms.get_cell(),
            pbc=atoms.pbc,
            **kwargs
        )


class DataPeriodicNeighbors(DataNeighbors):
    """Compatability wrapper for ``DataNeighbors``.
    Arguments are the same as ``DataNeighbors``, but ``lattice`` is accepted
    as an alias for ``cell`` and ``pbc`` is always set to ``True``.
    """

    def __init__(self, x, pos, lattice, r_max, self_interaction=True, **kwargs):
        super().__init__(
            x=x, pos=pos, cell=lattice, pbc=True, r_max=r_max,
            self_interaction=self_interaction, **kwargs
        )


def _neighbor_list_and_relative_vec(pos, r_max, self_interaction=True, cell=None, pbc=False):
    """Create neighbor list and neighbor vectors based on radial cutoff.
    Create neighbor list (``edge_index``) and relative vectors
    (``edge_attr``) based on radial cutoff.
    Edges are given by the following convention:
    - ``edge_index[0]`` is the *source* (convolution center).
    - ``edge_index[1]`` is the *target* (neighbor).
    Thus, ``edge_index`` has the same convention as the relative vectors:
    :math:`\\vec{r}_{source, target}`
    Args:
        pos (shape [N, 3]): Positional coordinate; Tensor or numpy array. If Tensor, must be detached & on CPU.
        r_max (float): Radial cutoff distance for neighbor finding.
        cell (numpy shape [3, 3]): Cell for periodic boundary conditions. Ignored if ``pbc == False``.
        pbc (bool or 3-tuple of bool): Whether the system is periodic in each of the three cell dimensions.
        self_interaction (bool): Whether or not to include self-edges in the neighbor list.
    Returns:
        edge_index (torch.tensor shape [2, num_edges]): List of edges.
        edge_attr (torch.tensor shape [num_edges, 3]): Relative vectors corresponding to each edge.
    """
    if isinstance(pbc, bool):
        pbc = (pbc,)*3
    if cell is None:
        # ASE will "complete" this correctly.
        cell = np.zeros((3, 3))
    cell = ase.geometry.complete_cell(cell)
    first_idex, second_idex, displacements = ase.neighborlist.primitive_neighbor_list(
        'ijD',
        pbc,
        np.asarray(cell),
        np.asarray(pos),
        cutoff=r_max,
        self_interaction=self_interaction,
        use_scaled_positions=False
    )
    edge_index = torch.vstack((
        torch.LongTensor(first_idex),
        torch.LongTensor(second_idex)
    ))
    edge_attr = torch.as_tensor(displacements)
    return edge_index, edge_attr
