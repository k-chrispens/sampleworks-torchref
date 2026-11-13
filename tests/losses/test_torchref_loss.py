"""Tests for the TorchRef refinement-target loss.

Scored against the synthetic 1vme MTZ the SFC loss tests use (``mtz_path_1vme``, its
``Fprotein`` set), with the first altloc of the model it was computed from: close to, but
not exactly, the structure behind the data.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray
from sampleworks.core.losses.options import TorchRefOptions
from sampleworks.core.losses.protocol import LossInputs
from sampleworks.core.losses.registry import LossBuildContext
from sampleworks.core.losses.torchref_loss import build_torchref_loss, TorchRefLoss


@pytest.fixture(scope="module")
def protein(structure_1vme_sf: AtomArray) -> AtomArray:
    """Single-conformer 1vme chain A: blank and ``A`` altlocs only."""
    atoms = structure_1vme_sf[np.isin(structure_1vme_sf.altloc_id, ["", ".", "A"])].copy()
    atoms.occupancy[:] = 1.0
    return atoms


def _prepared(mtz: Path, atoms: AtomArray, device, n_members: int = 1, **kwargs) -> TorchRefLoss:
    loss = TorchRefLoss(mtz, expcolumns=["Fprotein", "SIGFprotein"], **kwargs)
    loss.prepare(
        LossInputs.from_atom_array(atoms, ensemble_size=n_members, device=device), device=device
    )
    return loss


@pytest.fixture(scope="module")
def loss(mtz_path_1vme, protein, device) -> TorchRefLoss:
    return _prepared(mtz_path_1vme, protein, device)


@pytest.fixture(scope="module")
def coords(protein, device) -> torch.Tensor:
    return torch.from_numpy(protein.coord).to(device=device, dtype=torch.float32)[None]


def test_true_structure_scores_with_finite_gradient(loss, coords):
    x = coords.clone().requires_grad_(True)
    value = loss(x)
    value.backward()
    assert torch.isfinite(value)
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert loss.r_work < 0.2


def test_ranks_perturbed_structure_worse(loss, coords):
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(coords.shape, generator=generator).to(coords)
    assert loss(coords).item() < loss(coords + 0.3 * noise).item()


def test_strided_coordinates_score_as_contiguous(loss, coords):
    """A member slice of a non-contiguous ensemble tensor scores like a dense copy."""
    strided = torch.stack([coords[0], coords[0]], dim=1)[:, 0][None]  # row stride 6, not 3
    assert not strided[0].is_contiguous()
    torch.testing.assert_close(loss(strided), loss(coords.contiguous()), rtol=1e-5, atol=1e-3)


def test_identical_members_score_as_one(mtz_path_1vme, protein, device, coords):
    """Two identical members at fraction 1/2 give the X-ray loss of one alone."""
    single = _prepared(mtz_path_1vme, protein, device, geometry_weight=0.0)
    pair = _prepared(mtz_path_1vme, protein, device, n_members=2, geometry_weight=0.0)
    torch.testing.assert_close(pair(coords.expand(2, -1, -1)), single(coords), rtol=1e-4, atol=1e-3)


def test_rejects_other_ensemble_size(loss, coords):
    with pytest.raises(ValueError, match="members"):
        loss(coords.expand(2, -1, -1))


def test_call_before_prepare_raises(mtz_path_1vme, coords):
    with pytest.raises(RuntimeError, match="prepare"):
        TorchRefLoss(mtz_path_1vme)(coords)


def test_builder_requires_mtz():
    with pytest.raises(ValueError, match="MTZ"):
        build_torchref_loss(TorchRefOptions(), LossBuildContext(structure={}))
