import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc
import json
import sys
import time
import argparse


def solve(case_spec: dict) -> dict:
    """
    Solve steady incompressible Navier-Stokes (double lid-driven cavity).
    """
    comm = MPI.COMM_WORLD
    
    # Extract parameters from case_spec
    oracle_config = case_spec.get("oracle_config", case_spec)
    pde = oracle_config.get("pde", {})
    nu_val = pde.get("pde_params", {}).get("nu", 0.18)
    
    # Get boundary conditions
    bc_config = oracle_config.get("bc", {})
    bc_dirichlet = bc_config.get("dirichlet", [])
    
    # Get output grid
    output_config = oracle_config.get("output", {})
    grid_config = output_config.get("grid", {})
    nx_out = grid_config.get("nx", 50)
    ny_out = grid_config.get("ny", 50)
    bbox = grid_config.get("bbox", [0, 1, 0, 1])
    
    # Use resolution 48 (fast, should be accurate enough for 8.66e-02 threshold)
    N = 48
    result = _solve_ns(comm, N, nu_val, bc_dirichlet, nx_out, ny_out, bbox)
    return result


def _solve_ns(comm, N, nu_val, bc_dirichlet, nx_out, ny_out, bbox):
    """Core Navier-Stokes solver matching oracle formulation."""
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Create mixed Taylor-Hood elements (P2/P1)
    deg_u = 2
    deg_p = 1
    vel_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), deg_u, shape=(gdim,))
    pres_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), deg_p)
    mel = basix.ufl.mixed_element([vel_el, pres_el])
    
    W = fem.functionspace(domain, mel)
    
    # Get sub-spaces for BCs
    V_sub, _ = W.sub(0).collapse()
    Q_sub, _ = W.sub(1).collapse()
    
    # Boundary conditions (using geometrical like oracle)
    bcs = _setup_bcs(domain, W, V_sub, bc_dirichlet, gdim)
    
    # Add pressure point BC (pin p=0 at origin, like oracle)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q_sub),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    if len(p_dofs[0]) > 0:
        p0 = fem.Function(Q_sub)
        p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))
    
    # Define solution and test functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Viscosity and source
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))
    
    # === Step 1: Stokes initialization (like oracle) ===
    (u_s, p_s) = ufl.TrialFunctions(W)
    (v_s, q_s) = ufl.TestFunctions(W)
    a_stokes = (
        nu * ufl.inner(ufl.grad(u_s), ufl.grad(v_s)) * ufl.dx
        - ufl.div(v_s) * p_s * ufl.dx
        - q_s * ufl.div(u_s) * ufl.dx
    )
    L_stokes = ufl.inner(f, v_s) * ufl.dx
    
    stokes_problem = petsc.LinearProblem(
        a_stokes,
        L_stokes,
        bcs=bcs,
        petsc_options={
            "ksp_type": "minres",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
        },
        petsc_options_prefix="stokes_init_",
    )
    w0 = stokes_problem.solve()
    w.x.array[:] = w0.x.array
    
    # === Step 2: Newton solve for full NS (matching oracle formulation) ===
    # Oracle form: dot(grad(u), u) + nu*grad(u):grad(v) - p*div(v) - q*div(u) - f·v = 0
    F_ns = (
        ufl.inner(ufl.dot(ufl.grad(u), u), v) * ufl.dx
        + nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 50,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-12,
    }
    
    problem = petsc.NonlinearProblem(
        F_ns, w,
        bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options,
    )
    
    w_sol = problem.solve()
    
    snes = problem.solver
    n_newton = snes.getIterationNumber()
    
    # Extract velocity
    u_sol = w_sol.sub(0).collapse()
    
    # Evaluate on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    vel_values = _evaluate_function(domain, u_sol, points_3d, gdim)
    
    vel_mag = np.sqrt(vel_values[:, 0]**2 + vel_values[:, 1]**2)
    u_grid = vel_mag.reshape(nx_out, ny_out)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": deg_u,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "rtol": 1e-8,
        "nonlinear_iterations": [int(n_newton)],
    }
    
    return {"u": u_grid, "solver_info": solver_info}


def _setup_bcs(domain, W, V_sub, bc_dirichlet, gdim):
    """Set up boundary conditions using geometrical DOF location (like oracle)."""
    bcs = []
    
    if bc_dirichlet:
        for bc_item in bc_dirichlet:
            location = bc_item.get("on", "")
            value = bc_item.get("value", ["0.0", "0.0"])
            
            marker = _get_boundary_marker(location)
            
            val_floats = [float(v) for v in value]
            
            u_bc = fem.Function(V_sub)
            u_bc.interpolate(lambda x, v=val_floats: np.array([[v[i]] * x.shape[1] for i in range(len(v))]))
            
            dofs = fem.locate_dofs_geometrical((W.sub(0), V_sub), marker)
            bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))
    else:
        # Default double lid-driven cavity BCs
        bc_data = [
            ("y1", [1.0, 0.0]),
            ("x1", [0.0, -0.6]),
            ("x0", [0.0, 0.0]),
            ("y0", [0.0, 0.0]),
        ]
        for loc, val in bc_data:
            marker = _get_boundary_marker(loc)
            u_bc = fem.Function(V_sub)
            u_bc.interpolate(lambda x, v=val: np.array([[v[i]] * x.shape[1] for i in range(len(v))]))
            dofs = fem.locate_dofs_geometrical((W.sub(0), V_sub), marker)
            bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))
    
    return bcs


def _get_boundary_marker(location):
    """Convert location string to boundary marker function."""
    loc = location.lower().strip()
    if loc in ("y1", "top", "y=1", "y=1.0", "ymax"):
        return lambda x: np.isclose(x[1], 1.0)
    elif loc in ("y0", "bottom", "y=0", "y=0.0", "ymin"):
        return lambda x: np.isclose(x[1], 0.0)
    elif loc in ("x0", "left", "x=0", "x=0.0", "xmin"):
        return lambda x: np.isclose(x[0], 0.0)
    elif loc in ("x1", "right", "x=1", "x=1.0", "xmax"):
        return lambda x: np.isclose(x[0], 1.0)
    elif loc in ("all", "boundary", "*"):
        return lambda x: np.ones(x.shape[1], dtype=bool)
    else:
        return lambda x: np.ones(x.shape[1], dtype=bool)


def _evaluate_function(domain, u_func, points_3d, gdim):
    """Evaluate a dolfinx Function at given 3D points."""
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    n_points = points_3d.shape[0]
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    result = np.zeros((n_points, gdim))
    
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc)
        cells = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts, cells)
        if vals.ndim == 1:
            vals = vals.reshape(-1, gdim)
        for idx, global_idx in enumerate(eval_map):
            result[global_idx] = vals[idx, :gdim]
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()
    
    outdir = args.outdir
    
    # Default case spec for double lid cavity
    case_spec = {
        "oracle_config": {
            "pde": {
                "type": "navier_stokes",
                "pde_params": {"nu": 0.18},
                "source_term": ["0.0", "0.0"],
            },
            "domain": {"type": "unit_square"},
            "bc": {
                "dirichlet": [
                    {"on": "y1", "value": ["1.0", "0.0"]},
                    {"on": "x1", "value": ["0.0", "-0.6"]},
                    {"on": "x0", "value": ["0.0", "0.0"]},
                    {"on": "y0", "value": ["0.0", "0.0"]},
                ]
            },
            "output": {
                "format": "npz",
                "field": "velocity_magnitude",
                "grid": {"bbox": [0, 1, 0, 1], "nx": 50, "ny": 50},
            },
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    solver_info = result["solver_info"]
    
    # Save solution
    grid = case_spec["oracle_config"]["output"]["grid"]
    nx, ny = grid["nx"], grid["ny"]
    x = np.linspace(grid["bbox"][0], grid["bbox"][1], nx)
    y = np.linspace(grid["bbox"][2], grid["bbox"][3], ny)
    
    np.savez(f"{outdir}/solution.npz", x=x, y=y, u=u_grid)
    np.save(f"{outdir}/u.npy", u_grid)
    
    meta = {
        "wall_time_sec": elapsed,
        "solver_info": solver_info,
    }
    with open(f"{outdir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Solution shape: {u_grid.shape}")
    print(f"Max velocity magnitude: {np.max(u_grid):.6f}")
    print(f"Min velocity magnitude: {np.min(u_grid):.6f}")
    print(f"Mean velocity magnitude: {np.mean(u_grid):.6f}")
    print(f"L2 norm: {np.linalg.norm(u_grid):.6f}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Solver info: {solver_info}")
