import typing
import numpy as np
import os
import torch
from phiml.backend import set_global_precision
import phi.flow as flow
from tqdm import trange


def diffusivity_model(porosity, d0: float = 1, exponent: float = 1.5):
    """
    Map porosity values to diffusion coefficients using a power-law relationship.

    Args:
        porosity (numpy.ndarray): Porosity field.
        exponent (float): Exponent for the power-law relationship.

    Returns:
        numpy.ndarray: Diffusion coefficient field.
    """
    return d0 * (porosity**exponent)


def step(c, D, dt, solver=flow.Solve("CG"), implicit=False):
    """
    Perform a single time step of the diffusion simulation using the explicit method.

    Args:
        c (CenteredGrid): Concentration field.
        D (CenteredGrid): Diffusion coefficient field.
        dt (float): Time step size.
        implicit (bool): If True, use implicit diffusion scheme.

    Returns:
        CenteredGrid: Updated concentration field after one time step.
    """
    # Apply the diffusion operator
    if implicit:
        c = flow.diffuse.implicit(c, diffusivity=D, dt=dt, solve=solver)
    else:
        c = flow.diffuse.explicit(c, diffusivity=D, dt=dt)
    return c


class Diffusionsolver:

    def __init__(self, bc, h, solver=flow.Solve("CG"), implicit=True, device=None, usefloat64=True):
        """
        Initialize the diffusion solver with the given parameters.

        Args:
            c0 (CenteredGrid): Initial concentration field.
            dfield (CenteredGrid): Diffusion coefficient field.
            boundaries (extrapolation): Boundary conditions.
            h (float): Grid spacing.
            dt (float): Time step size.
            it (int): Number of iterations to run.
        """

        self.bc = bc
        self.h = h
        self.solver = solver
        self.implicit = implicit
        self.usefloat64 = usefloat64
        self.results = []

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if usefloat64:
            set_global_precision(64)

    def run(self, dataloader, dt, iterations, batch_to_process=None, folder=None, keep_results=False, implicit=None):

        self.results = []
        if folder is not None:
            os.makedirs(folder, exist_ok=True)

        if implicit is not None:
            self.implicit = implicit

        idx = 0

        for i, inputs in enumerate(dataloader):
            if i == batch_to_process:
                break
            if isinstance(inputs, (list, tuple)):
                dfield = inputs[0]
                c0 = inputs[1]
            else:
                dfield = inputs
                c0 = torch.zeros_like(dfield)

            batch_size = flow.batch(id=dfield.shape[0])
            nx, ny = dfield.shape[1:]
            domain = flow.Box(x=nx * self.h, y=ny * self.h)
            dfield = flow.wrap(dfield.to(self.device), batch_size, flow.spatial("x,y"))
            diffusivity_grid = flow.CenteredGrid(dfield, bounds=domain, extrapolation=flow.extrapolation.PERIODIC)
            c0 = flow.wrap(c0.to(self.device), batch_size, flow.spatial("x,y"))
            c0_grid = flow.CenteredGrid(c0, bounds=domain, extrapolation=self.bc)

            c_evol = flow.iterate(
                step,
                flow.batch(time=iterations),
                c0_grid,
                f_kwargs={"D": diffusivity_grid, "dt": dt, "implicit": self.implicit, "solver": self.solver},
                range=trange,
            )

            if folder is not None:
                for b in range(c_evol.shape.get_size("id")):
                    np.save(os.path.join(folder, f"c_evol_{idx:04d}.npy"), c_evol.id[b])
                    idx += 1

            if keep_results:
                self.results.append(c_evol)

        if keep_results:
            self.results = flow.concat(self.results, flow.batch("id"))
