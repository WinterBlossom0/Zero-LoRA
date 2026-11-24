"""Save, load and apply deletion masks: one bit per weight of each masked matrix (1 keeps the weight, 0 deletes it).

File format: masks.npz holds, for each matrix, "<name>.bits" (numpy.packbits of the flat keep mask) and
"<name>.shape"; masks.json holds the deleted and total weight counts.
"""

import json
import os

import numpy as np
import torch
from torch import nn


def save_masks(masks: dict[str, torch.Tensor], directory: str) -> None:
    """Save {parameter name: bool keep mask} to <directory>/masks.npz and the counts to <directory>/masks.json."""
    os.makedirs(directory, exist_ok=True)
    arrays = {}
    for name, keep in masks.items():
        keep_array = keep.to(torch.bool).cpu().numpy()
        arrays[f"{name}.bits"] = np.packbits(keep_array.ravel())
        arrays[f"{name}.shape"] = np.array(keep_array.shape, dtype=np.int64)
    np.savez_compressed(os.path.join(directory, "masks.npz"), **arrays)
    deleted = sum(int((~keep).sum()) for keep in masks.values())
    total = sum(keep.numel() for keep in masks.values())
    with open(os.path.join(directory, "masks.json"), "w") as stats_file:
        json.dump({"deleted": deleted, "total": total, "deleted_fraction": deleted / total}, stats_file, indent=2)


def load_masks(directory: str) -> dict[str, torch.Tensor]:
    """Return {parameter name: bool keep mask} from <directory>/masks.npz."""
    masks = {}
    with np.load(os.path.join(directory, "masks.npz")) as arrays:
        for key in arrays.files:
            if not key.endswith(".shape"):
                continue
            name = key[: -len(".shape")]
            shape = tuple(arrays[key].tolist())
            bits = np.unpackbits(arrays[f"{name}.bits"])[: int(np.prod(shape))]
            masks[name] = torch.from_numpy(bits.astype(bool).reshape(shape))
    return masks


@torch.no_grad()
def apply_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> int:
    """Set the deleted weights of ``model`` to exactly 0 in place; return the number of deleted weights."""
    parameters = dict(model.named_parameters())
    missing = sorted(set(masks) - set(parameters))
    assert not missing, f"masks name parameters that the model does not have: {missing[:3]}"
    for name, keep in masks.items():
        weight = parameters[name]
        assert keep.shape == weight.shape, f"{name}: mask {tuple(keep.shape)}, weight {tuple(weight.shape)}"
        weight.mul_(keep.to(device=weight.device, dtype=weight.dtype))
    return sum(int((~keep).sum()) for keep in masks.values())
