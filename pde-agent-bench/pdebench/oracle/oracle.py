"""Unified oracle entry point."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

from .common import OracleResult
from .linear_elasticity import LinearElasticitySolver
from .convection_diffusion import ConvectionDiffusionSolver
from .biharmonic import BiharmonicSolver
from .heat import HeatSolver
from .helmholtz import HelmholtzSolver
from .navier_stokes import NavierStokesSolver
from .poisson import PoissonSolver
from .stokes import StokesSolver
from .darcy import DarcySolver
from .reaction_diffusion import ReactionDiffusionSolver


class OracleSolver:
    """Dispatch to PDE-specific ground-truth solvers."""

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        pde_type = case_spec["pde"]["type"]

        if pde_type == "poisson":
            return PoissonSolver().solve(case_spec)
        if pde_type == "heat":
            return HeatSolver().solve(case_spec)
        if pde_type == "convection_diffusion":
            return ConvectionDiffusionSolver().solve(case_spec)
        if pde_type == "stokes":
            return StokesSolver().solve(case_spec)
        if pde_type == "navier_stokes":
            return NavierStokesSolver().solve(case_spec)
        if pde_type == "helmholtz":
            return HelmholtzSolver().solve(case_spec)
        if pde_type == "biharmonic":
            return BiharmonicSolver().solve(case_spec)
        if pde_type == "linear_elasticity":
            return LinearElasticitySolver().solve(case_spec)
        if pde_type == "darcy":
            return DarcySolver().solve(case_spec)
        if pde_type == "reaction_diffusion":
            return ReactionDiffusionSolver().solve(case_spec)

        raise ValueError(f"Unsupported PDE type: {pde_type}")


def generate(oracle_config: Dict[str, Any], output_dir: Union[str, Path]) -> OracleResult:
    """
    Generate oracle reference solution and save to output directory.
    
    Args:
        oracle_config: Oracle configuration dictionary containing PDE specification
        output_dir: Directory to save the reference solution
    
    Returns:
        OracleResult with the reference solution
    """
    import numpy as np
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    solver = OracleSolver()
    result = solver.solve(oracle_config)
    
    # Generate x, y grid arrays from oracle_config
    grid_cfg = oracle_config.get("output", {}).get("grid", {})
    bbox = grid_cfg.get("bbox", [0, 1, 0, 1])
    nx = grid_cfg.get("nx", 50)
    ny = grid_cfg.get("ny", 50)
    x = np.linspace(bbox[0], bbox[1], nx)
    y = np.linspace(bbox[2], bbox[3], ny)
    
    # Save reference solution to npz file
    # Save with keys expected by validator: x, y, u_star
    np.savez(
        output_dir / 'reference.npz',
        x=x,
        y=y,
        u_star=result.reference,
        reference=result.reference,
        baseline_error=result.baseline_error,
        baseline_time=result.baseline_time,
        num_dofs=result.num_dofs,
        solver_info=result.solver_info
    )
    
    # Also save u.npy for specialized metrics
    np.save(output_dir / 'u.npy', result.reference)
    
    return result
