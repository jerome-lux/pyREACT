from typing import Union, Sequence, Callable, Iterable
import numpy as np
from functools import partial
import os
import torch
from phiml.backend import set_global_precision
from phiml.math import jit_compile_linear
import phi.flow as flow
from tqdm import trange
from .models import diffusivity_model


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

    def step(self, c, D, dt, solver=flow.Solve("CG"), implicit=False):
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
                self.step,
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


class simpleDRsolver:

    def __init__(
        self,
        bc,
        h,
        shape,
        dinf,
        solid_density,
        mineral_molar_mass,
        Keq,
        rate_cst,
        stoech_coeffs,
        d_model=diffusivity_model,
        solver=flow.Solve("CG"),
        max_substeps=10,
        substeps_tol=1e-6,
        max_porosity_delta=1e-3,
        device=None,
        usefloat64=True,
    ):
        """
        Initializes the Diffusion-reaction solver with the specified parameters.

        This solver models a single reaction of the form nA + mB <=> AnBm, where AB is a solid product.
        It supports both implicit and explicit time integration, and can run on CPU or GPU.

        Parameters:
            bc (list): Phiflow Boundary conditions for each species (e.g., extrapolation, usually a dict)
            h (float): Grid spacing in meters.
            shape (tuple): Shape of the simulation domain as (nx, ny), where nx and ny are the number of grid cells in each dimension.
            dinf (list): List of diffusivity coefficients (one scalar per species).
            solid_density (float): Density of the solid phase in kg/m³. For 2D simulations, each pixel is assumed to be a cube of size h³.
            mineral_molar_mass (float): Molar mass of the mineral (g/mol).
            Keq (float): Equilibrium constant for the reaction.
            rate_cst (float): Reaction rate constant.
            stoech_coeffs (list): Stoechiometric coefficients for the reaction for each species.
            d_model (callable): Function or model to compute diffusivity.
            solver (callable, optional): Linear solver to use for the system. Defaults to flow.Solve("CG").
            max_substeps (int, optional): Maximum number of substeps for the solver. Defaults to 5.
            c_tol (float, optional): Tolerance for concentration convergence. Defaults to 1e-6.
            device (str or None, optional): Device to run the computation on ("cpu" or "cuda:0"). If None, automatically selects GPU if available.
            usefloat64 (bool, optional): Whether to use 64-bit floating point precision. Defaults to True.
        """

        self.bc = bc
        self.h = h
        self.mineral_molar_mass = mineral_molar_mass  # g/mol
        self.solid_density = solid_density  # kg/m3
        self.mineral_molar_volume = 1e3 * solid_density / mineral_molar_mass  # rho/M in mol/m3
        self.nx, self.ny = shape
        self.dinf = dinf
        self.solver = solver
        self.usefloat64 = usefloat64
        self.Keq = Keq
        self.rate_cst = rate_cst
        self.d_model = d_model
        self.stoech_coeffs = stoech_coeffs
        self.max_substeps = max_substeps
        self.substeps_tol = substeps_tol
        self.max_porosity_delta = (max_porosity_delta,)
        self.c_results = []
        self.p_results = []
        self.history = []

        # Domain
        self.domain = flow.Box(x=self.nx * self.h, y=self.ny * self.h)

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if usefloat64:
            set_global_precision(64)

    def reaction_rate(self, c_field):
        """
        Compute the rate of dissolution/precipitation (mol/(m³·s)) using a simplified expression.
        The rate is positive for precipitation and negative for dissolution.

        The rate equation is:
            $$
            \text{rate} = k_m \prod_i a_i^{n_i} \cdot A \left( \frac{Q_m}{K_{eq}} - 1 \right) = \text{rate\_cst} \left( \frac{Q_m}{K_{eq}} - 1 \right)
            $$
        Note that the specific area is generally given as the area of the interface over the mineral volume
        So in order to take into account the evolution o fthe total surface in a grid cel, the rate should be multiply by (1 - porosity) in the conservation equation

        Inputs:
            c_field (List): List of concentration fields (CenteredGrid Fields).
        Class attributes
            Keq (float): The equilibrium constant, e.g., $[c_i]^{eq} [c_j]^{eq}$.
            rate_cst (float): Replaces the term $A \cdot k_m \prod \text{act}_i$.
            stoech_coeffs (list): stoechiometric coefficients of the reaction

        Returns:
            A Field with the same shape as the concentration fields (mol/(m³·s)).
            The reaction rate is the same for both reacting species.
        """
        return self.rate_cst * (
            (c_field[0].values ** self.stoech_coeffs[0] * c_field[1].values ** self.stoech_coeffs[1]) / self.Keq - 1
        )

    def explicit_step(self, c_field, porosity, dt=None):
        # the time step is computed dynamically. We take the min value over the batch
        R = self.reaction_rate(c_field)
        diffusivity = self.update_diffusivity(porosity)

        max_rate = np.max([flow.math.max((1 - porosity) * s * flow.math.abs(R)).numpy() for s in self.stoech_coeffs])
        dt = 0.2 * self.h**2 / np.max([flow.math.max(d) for d in diffusivity])
        if max_rate > 0:
            dt = min(dt, self.max_porosity_delta / max_rate)
        c_new = [c for c in c_field]

        for i, c in enumerate(c_field):
            d_grid = flow.CenteredGrid(diffusivity[i], bounds=self.domain, boundary=flow.extrapolation.REFLECT)
            c_new[i] = (
                c_field[i].values * porosity
                + flow.diffuse.differential(c_field[i], d_grid * dt).values
                - dt * (1 - porosity) * self.stoech_coeffs[i] * R
            ) / porosity
            c_new[i] = flow.CenteredGrid(c_new[i], bounds=self.domain, boundary=self.bc[i])

        # R = self.reaction_rate(c_new)
        porosity = porosity - (1 - porosity) * R * dt / self.mineral_molar_volume
        self.history[-1]["dt"] = dt

        return (c_new, porosity)

    def implicit_step(self, c_field, porosity, dt, substeps=10, method="euler"):
        """
        Advances the concentration and porosity fields by one time step using operator splitting for reaction-diffusion equations.
        Args:
            c_field (list of CenteredGrid Fields): List of concentration fields for each chemical species.
            porosity (CenteredGrid Field): Porosity field of the domain.
            dt (float): Time step size.
            substeps (int, optional): Maximum number of sub-iterations for convergence within the time step. Default is 10.
        Returns:
            tuple: Updated (c_field, porosity) after one time step.
        Notes:
            - The method iteratively solves the coupled reaction-diffusion equations using a fixed-point approach.
            - Diffusivity is updated at each substep based on the current porosity.
            - The reaction rate and porosity are updated after each substep.
            - Convergence is checked based on the relative change in concentrations.
        """
        # TODO: do an explicit step for the first time step of the simulation !

        c_new = [c for c in c_field]
        c_old = [c for c in c_field]
        R_old = self.reaction_rate(c_field)
        diffusivity = self.update_diffusivity(porosity)
        diffusivity_prev = diffusivity
        porosity_new = porosity
        porosity_old = porosity

        for it in range(substeps):
            # Update each species concentration field

            for i, c in enumerate(c_old):
                # Solving the transfer equation

                if method == "euler":
                    rhs = porosity * c_field[i] - dt * (1 - porosity_old) * self.stoech_coeffs[i] * R_old

                    def lhs(x, i):
                        d_grid = flow.CenteredGrid(
                            diffusivity[i], bounds=self.domain, boundary=flow.extrapolation.REFLECT
                        )
                        x_grid = flow.CenteredGrid(x, bounds=self.domain, boundary=self.bc[i])
                        return x * porosity_old + flow.diffuse.differential(x_grid, -d_grid * dt).values

                else:
                    d_grid = flow.CenteredGrid(
                        diffusivity_prev, bounds=self.domain, boundary=flow.extrapolation.REFLECT
                    )
                    rhs = (
                        porosity * c_field[i]
                        + 0.5 * flow.diffuse.differential(c_field[i], -d_grid * dt).values
                        - dt * (1 - porosity_old) * self.stoech_coeffs[i] * R_old
                    )

                    def lhs(x, i):
                        d_grid = flow.CenteredGrid(
                            diffusivity[i], bounds=self.domain, boundary=flow.extrapolation.REFLECT
                        )
                        x_grid = flow.CenteredGrid(x, bounds=self.domain, boundary=self.bc[i])
                        return x * porosity_old + flow.diffuse.differential(x_grid, -0.5 * d_grid * dt).values

                self.solver.x0 = c.values
                c_new[i] = flow.solve_linear(jit_compile_linear(partial(lhs, i=i)), y=rhs, solve=self.solver)
                c_new[i] = flow.CenteredGrid(c_new[i], bounds=self.domain, boundary=self.bc[i])

            # Update the reaction rate, porosity and diffusivity [We update from variables at previous global iteration]
            R_new = self.reaction_rate(c_new)
            porosity_new = porosity - (1.0 - porosity_old) * R_new * dt / self.mineral_molar_volume
            # Ensure that there is still a small amount of solid in a cell
            porosity_new = flow.math.where(porosity >= (1 - 1e-6), 1 - 1e-6, porosity_new)
            diffusivity = self.update_diffusivity(porosity_new)

            # Compute the delta C and delta P between two sub-iterations.
            delta_C = np.array(
                [flow.math.max(flow.math.abs(c_new[i].values - c_old[i].values)).numpy() for i, _ in enumerate(c_old)]
            )
            avg_c = flow.math.mean(c_new[i].values).numpy()
            delta_p = flow.math.max(flow.math.abs(porosity_new - porosity_old))
            # print(f"Substep {it+1}: max(C_new - C_old) = {[d for d in delta_C]}")
            # print(f"AVG C_new = {avg_c}")
            # print(f"Delta p = {delta_p.numpy()}")
            # ravg = flow.math.mean((1 - porosity_old) * R_new * dt / self.mineral_molar_volume).numpy()
            # print(f"(1 - phi) * Rm * dt / Vm = {ravg}")

            c_old = [c for c in c_new]
            R_old = R_new
            porosity_old = porosity_new

            if it == 0 and np.all(delta_p.numpy() < self.substeps_tol):
                break
            elif it > 0 and np.all(delta_C < self.substeps_tol) and np.all(delta_p.numpy() < self.substeps_tol):
                break
            elif it >= substeps - 1:
                print("No convergence ")
                break

        self.history[-1]["dt"] = dt
        self.history[-1]["substeps"] = it + 1
        return (c_new, porosity_new)

    def update_diffusivity(self, porosity):
        # Porosity should be a CenteredGrid
        return [self.d_model(d, porosity) for d in self.dinf]

    def run(
        self,
        dataloader,
        dt,
        iterations,
        c0: Sequence,
        explicit=False,
        method="euler",
        batch_to_process=None,
        folder=None,
        keep_results=False,
        save_step=1,
    ):
        """
        Runs the simulation over batches of porosity fields, evolving concentration and porosity over time.

        Args:
            dataloader: An iterable that yields batches of porosity fields, typically a DataLoader or similar object.
            dt (float): Time step size for each iteration.
            iterations (int): Number of time steps to run the simulation for each batch.
            c0 (Sequence): Initial concentration values for each species.
            batch_to_process (int, optional): If specified, stops processing after this batch index. Defaults to None (processes all batches).
            folder (str, optional): Directory path to save the evolution of concentration fields. If None, results are not saved to disk.
            keep_results (bool, optional): If True, stores the evolution of concentration and porosity fields in the instance for later access.
            BEWARE: if keep_results, the saved arrays can eat a lot of memory depending on the number of time steps!

        Side Effects:
            - May create directories and save .npy files with concentration field evolution if `folder` is specified.
            - Updates `self.c_results` and `self.p_results` if `keep_results` is True.

        Returns:
            None
        """
        # clear results
        self.c_results = []
        self.p_results = []
        self.history = []

        if folder is not None:
            os.makedirs(folder, exist_ok=True)

        idx = 0

        for n, porosity in enumerate(dataloader):

            if n == batch_to_process:
                break

            batch_size = porosity.shape[0]
            flow_batch_size = flow.batch(id=porosity.shape[0])
            nx, ny = porosity.shape[1:]

            porosity = flow.wrap(porosity.to(self.device), flow_batch_size, flow.spatial("x,y"))
            # porosity = flow.CenteredGrid(porosity, bounds=self.domain, extrapolation=flow.extrapolation.ONE)

            # create concentration fields. To use different bcs for each c field, we must create one Field per species
            c_field = [torch.ones((batch_size, nx, ny)).to(self.device) * c for c in c0]
            c_field = [flow.wrap(c, flow_batch_size, flow.spatial("x,y")) for c in c_field]
            c_field = [
                flow.CenteredGrid(c, bounds=self.domain, extrapolation=self.bc[i]) for i, c in enumerate(c_field)
            ]
            avg_c = np.array([flow.math.mean(c.values).numpy() for c in c_field])

            # TO Store the evolution of concentration field & porosity over time
            c_evol = [flow.stack([c.values for c in c_field], dim=flow.channel("species"))]
            p_evol = [porosity]
            self.history.append({"iteration": 0})
            if not explicit:
                self.history[-1]["substeps"] = 0
            self.history[-1]["dt"] = 0

            progress_bar = trange(iterations)
            t = 0
            step = (
                partial(self.explicit_step)
                if explicit
                else partial(self.implicit_step, substeps=self.max_substeps, method=method)
            )

            for it in progress_bar:
                self.history.append({"iteration": it + 1})
                c_field, porosity = step(c_field, porosity, dt)
                if it % save_step == 0:
                    c_evol.append(flow.stack([c.values for c in c_field], dim=flow.channel("species")))
                    p_evol.append(porosity)
                dt = self.history[-1]["dt"]
                t += dt
                if explicit:
                    progress_bar.set_postfix(dt=dt, time=t)
                else:
                    progress_bar.set_postfix(dt=dt, time=t, substeps=self.history[-1]["substeps"])

            c_evol = flow.stack(c_evol, dim=flow.channel("time"))
            p_evol = flow.stack(p_evol, dim=flow.channel("time"))

            if keep_results:
                self.c_results.append(c_evol)
                self.p_results.append(p_evol)

            if folder is not None:
                os.makedirs(folder, exist_ok=True)
                os.makedirs(os.path.join(folder, "c_fields"), exist_ok=True)
                os.makedirs(os.path.join(folder, "p_fields"), exist_ok=True)
                for b in range(c_evol.shape.get_size("id")):
                    np.save(
                        os.path.join(folder, "c_fields", f"c_evol_{idx:04d}.npy"),
                        c_evol.id[b].numpy(("time", "x", "y", "species")),
                    )
                    np.save(
                        os.path.join(folder, "p_fields", f"p_evol_{idx:04d}.npy"),
                        p_evol.id[b].numpy(("time", "x", "y")),
                    )
                    idx += 1

        if keep_results:
            self.c_results = flow.concat(self.c_results, flow.batch("id"))
            self.p_results = flow.concat(self.p_results, flow.batch("id"))
