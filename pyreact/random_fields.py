import os
import numpy as np
import gstools as gs
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from pathlib import Path


def generate_one_sample(it, model, grid_pos, min_ls, max_ls, base_var, min_p, max_p, folder=os.getcwd(), trunc=3):
    """
    Generate a single random field with a random length scale between min_ls and max_ls and variance.
    A porosity field is the generated. The random field is truncated at trunc*sigma, and mapped to the range [min_p, max_p]
    Then, the porosity field is transformed to a realtive diffusivity field using the formula D/D0 = porosity^exponent
    """
    # Generate the GRF
    idx, rng = it
    current_len_scale = rng.uniform(min_ls, max_ls)
    srf = gs.SRF(model)
    srf.set_pos(grid_pos, "structured")
    srf.model.len_scale = current_len_scale
    srf.model.var = base_var
    srf(seed=rng.integers(0, 1000000))
    field = srf.structured(grid_pos)

    # Map to porosity
    sigma = np.sqrt(base_var)
    porosity = np.clip(field, -trunc * sigma, trunc * sigma)
    porosity = ((porosity + trunc * sigma) / (2 * trunc * sigma)) * (max_p - min_p) + min_p

    # save fields
    np.save(os.path.join(folder, "GRF", f"GRF_{idx:04d}.npy"), field)
    np.save(os.path.join(folder, "PORO", f"porosity_{idx:04d}.npy"), porosity)


def generate_fields(
    n_samples,
    grid_pos,
    porosity_avg,
    porosity_std,
    min_ls,
    max_ls,
    model=gs.Gaussian(dim=2),
    base_var=1,
    folder=os.getcwd(),
    trunc=3,
):
    """
    Generate random fields with a random length scale between min_ls and max_ls and variance.
    A porosity field is the generated. The random field is truncated at trunc*sigma, and mapped to the range [min_p, max_p]
    Then, the porosity field is transformed to a realtive diffusivity field using the formula D/D0 = porosity^exponent
    inputs:
    n_samples: number of samples to generate
    grid_pos: grid position of the field [x, y] with x, y 1D numpy arrays
    porosity_avg: average porosity
    porosity_std: standard deviation of porosity
    min_ls: minimum length scale
    max_ls: maximum length scale
    base_var: base variance of the random field
    folder: folder to save the fields
    trunc: truncation factor for the random field (usually +/-3*sigma)
    exponent: exponent for the porosity to relative diffusivity transformation
    """
    # Create folder if it does not exist
    Path(os.path.join(folder, "GRF")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(folder, "PORO")).mkdir(parents=True, exist_ok=True)

    min_p = porosity_avg - trunc * porosity_std
    max_p = porosity_avg + trunc * porosity_std

    assert min_p > 0, "minimum porosity must be > 0, change the truncation factor"
    print(f"Creating {n_samples} fields with min_p: {min_p:4.2f}, max_p: {max_p:4.2f}")

    # Generate a list of random fields
    it = [(i, np.random.default_rng(i)) for i in range(n_samples)]
    func = partial(
        generate_one_sample,
        model=model,
        grid_pos=grid_pos,
        min_ls=min_ls,
        max_ls=max_ls,
        base_var=base_var,
        min_p=min_p,
        max_p=max_p,
        folder=folder,
        trunc=trunc,
    )
    t0 = time.time()
    # Use multiprocessing to generate the fields
    with Pool(cpu_count() - 1) as pool:
        pool.map(func, it)
    t1 = time.time()
    print(f"Generated {n_samples} fields in {t1-t0:.2f} seconds, {n_samples/(t1-t0):.2f} fields/s")
