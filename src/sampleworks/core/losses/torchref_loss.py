"""TorchRef refinement targets as a guidance loss: X-ray likelihood plus geometry restraints.

Each ensemble member is a ``ModelFT`` subclass built straight from the atom table, whose
``xyz`` module returns that member's sampled coordinates. The members are combined as
TorchRef combines states, in a ``MixedModel`` with fixed 1/N fractions (``F = Σ w_i F_i``),
which the scaler and the X-ray target (sigma_A ML by default) see; the bulk-solvent mask
is the first member's, and each member carries its own geometry restraints. Targets and
weights sit in a ``LossState`` laid out as TorchRef's multi-model refinement lays it out.
TorchRef's forward caching is switched off (process-wide), since the coordinates change on
every call.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torchref.config
from biotite.structure import AtomArray
from jaxtyping import Float, Int
from torchref.io import ReflectionData
from torchref.model.context import ModelContext
from torchref.model.mixed_model import MixedModel
from torchref.model.model_ft import ModelFT
from torchref.model.parameter_wrappers import MixedTensor
from torchref.refinement.base_refinement import DEFAULT_GROUP_WEIGHTS
from torchref.refinement.loss_state import LossState
from torchref.refinement.targets.combined import TotalGeometryTarget
from torchref.refinement.targets.xray import create_xray_target
from torchref.scaling.scaler import Scaler

from sampleworks.core.losses.options import TorchRefOptions
from sampleworks.core.losses.registry import LossBuildContext


if TYPE_CHECKING:
    from sampleworks.core.losses.protocol import LossInputs


class SampledXYZ(MixedTensor):
    """Coordinate wrapper returning externally set coordinates, graph attached.

    Parameters
    ----------
    reference
        Coordinates ``[n_atoms, 3]`` returned until ``sampled`` is set.
    """

    def __init__(self, reference: torch.Tensor):
        super().__init__(reference, requires_grad=False, name="xyz", device=reference.device)
        self.sampled: torch.Tensor | None = None

    def forward(self) -> torch.Tensor:
        """The sampled coordinates ``[n_atoms, 3]``, or the reference before any are set."""
        return super().forward() if self.sampled is None else self.sampled


class SampledModel(ModelFT):
    """A ``ModelFT`` built from an atom table, whose coordinates come from the sampler.

    Parameters
    ----------
    table
        TorchRef atom table (see :func:`atom_table`); its coordinates are the reference.
    reflections
        Observed data supplying the cell and space group.
    max_res
        High-resolution limit of the structure-factor grid, in Å.
    device
        Torch device.
    """

    def __init__(self, table: pd.DataFrame, reflections: ReflectionData, max_res: float, device):
        super().__init__(max_res=max_res, device=device)
        self.ctx, values = ModelContext.from_atoms(
            table,
            reflections.cell,
            reflections.spacegroup,
            dtype=self.dtype_float,
            device=self.device,
        )
        reference = torch.tensor(values.xyz, dtype=self.dtype_float, device=self.device)
        self._install_parameters(values, xyz=SampledXYZ(reference))

    def set_coordinates(self, coordinates: Float[torch.Tensor, "n_atoms 3"]) -> None:
        """Use ``coordinates`` (table order) for every following evaluation."""
        # TorchRef's CUDA restraint kernels misread strided xyz (HatPdotS/TorchRef#113).
        self.xyz.sampled = coordinates.to(self.dtype_float).contiguous()


def atom_table(atom_array: AtomArray, coords: np.ndarray, b_factors: np.ndarray) -> pd.DataFrame:
    """TorchRef atom table of one conformer, rows in ``atom_array`` order, occupancy 1.

    Parameters
    ----------
    atom_array
        Atom identities, in coordinate order.
    coords, b_factors
        Coordinates ``[n_atoms, 3]`` and B-factors ``[n_atoms]``.

    Returns
    -------
    pandas.DataFrame
        Identity and value columns as TorchRef's readers produce them.
    """
    return pd.DataFrame(
        {
            "name": atom_array.atom_name,
            "element": atom_array.element,
            "altloc": "",
            "chainid": atom_array.chain_id,
            "resseq": atom_array.res_id,
            "icode": atom_array.ins_code,
            "resname": atom_array.res_name,
            "ATOM": np.where(atom_array.hetero, "HETATM", "ATOM"),
            "charge": 0,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "tempfactor": b_factors,
            "occupancy": 1.0,
        }
    )


class TorchRefLoss:
    """TorchRef's X-ray and geometry targets on the sampled ensemble, in one ``LossState``.

    After every call, ``r_work`` and ``r_free`` hold TorchRef's R-factors.

    Parameters
    ----------
    mtzfile
        MTZ with the observed data and an R-free flag column.
    expcolumns
        ``[amplitude, sigma]`` column names, or ``None`` for TorchRef's detection.
    resolution
        High-resolution limit in Å, or ``None`` for the data's own.
    xray_mode
        TorchRef X-ray target; ``"ml"`` (TorchRef's default) is sigma_A ML.
    geometry_weight
        Weight of the ``geometry`` group relative to ``xray`` (TorchRef's default 0.2).
    """

    def __init__(
        self,
        mtzfile: str | Path,
        *,
        expcolumns: list[str] | None = None,
        resolution: float | None = None,
        xray_mode: str = "ml",
        geometry_weight: float = DEFAULT_GROUP_WEIGHTS["geometry"],
    ):
        self.mtzfile, self.resolution, self.xray_mode = str(mtzfile), resolution, xray_mode
        self.columns = {"F": expcolumns[0], "SIGF": expcolumns[1]} if expcolumns else None
        self.weights = {**DEFAULT_GROUP_WEIGHTS, "geometry": geometry_weight}
        self.state: LossState | None = None
        self.r_work = self.r_free = None

    def prepare(self, loss_inputs: LossInputs, *, device: torch.device | str = "cpu") -> None:
        """Build one TorchRef model per member, the mixture, scaler and targets on ``device``.

        Parameters
        ----------
        loss_inputs
            Inputs for the sampled coordinates; their batch size is the ensemble size.
        device
            Torch device everything is built on.

        Raises
        ------
        ValueError
            If TorchRef's model does not hold exactly the given atoms, in their order.
        """
        torchref.config.caching.value = False
        device = torch.device(device)
        self.reflections = self._load_reflections(device)
        self.members = self._build_members(loss_inputs, device)
        self.mixture = MixedModel(self.members, frozen_fractions=True)
        # One bulk-solvent mask, from the first member; the scale is fitted to the mixture.
        self.scaler = Scaler(self.members[0], self.reflections, device=device)
        self.scaler.initialize(self.reflections.structure_factors(self.mixture))
        self.xray = create_xray_target(
            data=self.reflections, model=self.mixture, scaler=self.scaler, mode=self.xray_mode
        )
        self.state = self._build_loss_state(device)

    def _load_reflections(self, device: torch.device) -> ReflectionData:
        """Observed data from the MTZ, cut at ``resolution`` when one is set.

        Parameters
        ----------
        device
            Torch device for the reflection tensors.

        Returns
        -------
        ReflectionData
            The data the X-ray target and scaler fit against.
        """
        reflections = ReflectionData(device=device).load_mtz(
            self.mtzfile, column_names=self.columns
        )
        if self.resolution is not None:
            reflections = reflections.filter_by_resolution(d_min=self.resolution)
        return reflections

    def _build_members(self, loss_inputs: LossInputs, device: torch.device) -> list[SampledModel]:
        """One TorchRef model per ensemble member, over the atoms of ``loss_inputs``.

        Parameters
        ----------
        loss_inputs
            Inputs for the sampled coordinates; their batch size is the ensemble size.
        device
            Torch device.

        Returns
        -------
        list of SampledModel
            Models whose rows follow the atom order of ``loss_inputs.atom_array``.

        Raises
        ------
        ValueError
            If TorchRef's model does not hold exactly the given atoms.
        """
        atom_array = loss_inputs.to_atom_array()
        n_atoms = atom_array.array_length()
        reference_coords = loss_inputs.input_coords.reshape(-1, n_atoms, 3).detach().cpu().numpy()
        reference_b_factors = loss_inputs.b_factors.reshape(-1, n_atoms).detach().cpu().numpy()
        max_res = self.resolution or self.reflections.d_min
        members = [
            SampledModel(
                atom_table(atom_array, member_coords, member_b_factors),
                self.reflections,
                max_res,
                device,
            )
            for member_coords, member_b_factors in zip(
                reference_coords, reference_b_factors, strict=True
            )
        ]
        if members[0].n_atoms != n_atoms:
            raise ValueError(
                f"TorchRef's model holds {members[0].n_atoms} atoms, not the {n_atoms} given."
            )
        return members

    def _build_loss_state(self, device: torch.device) -> LossState:
        """The X-ray target and each member's geometry, weighted as TorchRef weights them.

        Parameters
        ----------
        device
            Torch device.

        Returns
        -------
        LossState
            Targets keyed ``xray`` and ``geometry/model_<i>/...``.
        """
        state = LossState(device=device)
        state.register_target("xray", self.xray)
        for i, member in enumerate(self.members):
            state.register_target(f"geometry/model_{i}", TotalGeometryTarget(member))
        # TorchRef turns Ramachandran off under "geometry/"; the per-member keys need it too.
        ramachandran = {key: 0.0 for key in state.targets if key.endswith("ramachandran")}
        state.set_weights({**self.weights, **ramachandran})
        return state

    def __call__(
        self,
        coordinates: Float[torch.Tensor, "batch n_atoms 3"],
        elements: Int[torch.Tensor, "batch n_atoms"] | None = None,
        b_factors: Float[torch.Tensor, "batch n_atoms"] | None = None,
        occupancies: Float[torch.Tensor, "batch n_atoms"] | None = None,
        unique_combinations: torch.Tensor | None = None,
        inverse_indices: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, ""]:
        """Weighted TorchRef loss (work-set X-ray plus geometry) of the ensemble.

        Refits the scaler and resets the sigma_A estimate first, so both follow the
        coordinates (side effect). ``elements``, ``b_factors`` and ``occupancies`` are
        fixed at :meth:`prepare` and ignored here.

        Parameters
        ----------
        coordinates
            Ensemble coordinates ``[n_members, n_atoms, 3]`` in the prepared atom order.

        Returns
        -------
        torch.Tensor
            Scalar ``LossState`` aggregate.
        """
        if self.state is None:
            raise RuntimeError(
                "TorchRefLoss.prepare() must be called before the loss is evaluated."
            )
        if coordinates.shape[0] != len(self.members):
            raise ValueError(
                f"Prepared for {len(self.members)} members, got {coordinates.shape[0]}."
            )
        for member, member_coords in zip(self.members, coordinates, strict=True):
            member.set_coordinates(member_coords)
        self.scaler.update_solvent()
        self.scaler.refine_lbfgs(self.reflections.structure_factors(self.mixture))
        self.xray.maintenance()  # sigma_A is re-estimated at these coordinates
        loss = self.state.aggregate()
        self.r_work, self.r_free = self.xray.get_rfactor()
        return loss


def build_torchref_loss(options: TorchRefOptions, context: LossBuildContext) -> TorchRefLoss:
    """Build the TorchRef loss from its options; it binds to the atoms in ``prepare``.

    Parameters
    ----------
    options
        See :class:`~sampleworks.core.losses.options.TorchRefOptions`.
    context
        Unused; the topology and device arrive in :meth:`TorchRefLoss.prepare`.

    Returns
    -------
    TorchRefLoss
        The unprepared loss.
    """
    del context
    if options.mtzfile is None:
        raise ValueError("The torchref loss needs a target MTZ (--mtzfile).")
    return TorchRefLoss(
        options.mtzfile,
        expcolumns=options.expcolumns,
        resolution=options.resolution,
        xray_mode=options.xray_mode,
        geometry_weight=options.geometry_weight,
    )
