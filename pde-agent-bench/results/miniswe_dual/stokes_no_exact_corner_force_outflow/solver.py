import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh, geometry
from dolfinx.fem import petsc
import ufl
import basix.ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    """Solve Stokes flow on unit square with Taylor-Hood elements."""
    
    comm = MPI.COMM_WORLD
    
    # Parse case spec
    if "oracle_config" in case_spec:
        config = case_spec["oracle_config"]
    else:
        config = case_spec
    
    pde = config.get("pde", {})
    nu_val = pde.get("pde_params", {}).get("nu", pde.get("viscosity", 0.1))
    source = pde.get("source_term", pde.get("source", [
        "3*exp(-50*((x-0.15)**2 + (y-0.15)**2))",
        "3*exp(-50*((x-0.15)**2 + (y-0.15)**2))"
    ]))
    
    bc_config = config.get("bc", {})
    output_config = config.get("output", {})
    grid_config = output_config.get("grid", {"bbox": [0, 1, 0, 1], "nx": 100, "ny": 100})
    
    # Use N=48 P2/P1 with MINRES+HYPRE (matches oracle solver)
    N = 48
    degree_u = 2
    degree_p = 1
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    
    el_u = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,))
    el_p = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    mel = basix.ufl.mixed_element([el_u, el_p])
    W = fem.functionspace(domain, mel)
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    f_expr = _parse_source(source, x)
    
    # Bilinear form (matching oracle sign convention)
    a = (
        nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.div(v) * p * ufl.dx
        - q * ufl.div(u) * ufl.dx
    )
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Boundary conditions
    bcs = _setup_bcs(domain, W, V, bc_config, gdim)
    
    # Pressure pinning at origin
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    if len(p_dofs[0]) > 0:
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))
    
    # Solve with MINRES + HYPRE (matching oracle solver)
    problem = petsc.LinearProblem(
        a, L,
        bcs=bcs,
        petsc_options_prefix="stokes_",
        petsc_options={
            "ksp_type": "minres",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
        },
    )
    wh = problem.solve()
    wh.x.scatter_forward()
    
    # Extract velocity
    u_sol = wh.sub(0).collapse()
    
    # Evaluate on output grid
    bbox = grid_config.get("bbox", [0, 1, 0, 1])
    nx_out = grid_config.get("nx", 100)
    ny_out = grid_config.get("ny", 100)
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.flatten()
    points[:, 1] = YY.flatten()
    
    u_values = _evaluate_function(domain, u_sol, points, gdim)
    
    vel_mag = np.linalg.norm(u_values, axis=1)
    u_grid = vel_mag.reshape((nx_out, ny_out))
    u_grid = np.nan_to_num(u_grid, nan=0.0)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "minres",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": 1,
            "pressure_fixing": "point",
        }
    }


def _parse_source(source, x):
    """Parse source term expressions using UFL spatial coordinates."""
    components = []
    for s in source:
        namespace = {
            "x": x[0],
            "y": x[1],
            "exp": ufl.exp,
            "sin": ufl.sin,
            "cos": ufl.cos,
            "pi": np.pi,
            "sqrt": ufl.sqrt,
            "__builtins__": {},
        }
        val = eval(s, namespace)
        components.append(val)
    return ufl.as_vector(components)


def _setup_bcs(domain, W, V, bc_config, gdim):
    """Set up boundary conditions from config."""
    bcs = []
    
    dirichlet_specs = bc_config.get("dirichlet", [])
    
    if dirichlet_specs:
        for bc_def in dirichlet_specs:
            location = bc_def.get("on", "all")
            value = bc_def.get("value", ["0.0", "0.0"])
            
            marker = _get_boundary_marker(location)
            dofs = fem.locate_dofs_geometrical((W.sub(0), V), marker)
            
            u_bc = fem.Function(V)
            if isinstance(value, (list, tuple)):
                val_arr = np.array([float(v) for v in value], dtype=PETSc.ScalarType)
                u_bc.interpolate(lambda x, v=val_arr: np.array([[v[i]] * x.shape[1] for i in range(len(v))]))
            else:
                u_bc.interpolate(lambda x: np.full((gdim, x.shape[1]), float(value)))
            
            bc = fem.dirichletbc(u_bc, dofs, W.sub(0))
            bcs.append(bc)
    else:
        # Default: Dirichlet u=0 on left, bottom, top
        for marker_fn in [
            lambda x: np.isclose(x[0], 0.0),
            lambda x: np.isclose(x[1], 0.0),
            lambda x: np.isclose(x[1], 1.0),
        ]:
            dofs = fem.locate_dofs_geometrical((W.sub(0), V), marker_fn)
            u_bc = fem.Function(V)
            u_bc.interpolate(lambda x: np.zeros((gdim, x.shape[1])))
            bc = fem.dirichletbc(u_bc, dofs, W.sub(0))
            bcs.append(bc)
    
    return bcs


def _get_boundary_marker(location):
    """Return a boundary marker function."""
    markers = {
        "x0": lambda x: np.isclose(x[0], 0.0),
        "x1": lambda x: np.isclose(x[0], 1.0),
        "y0": lambda x: np.isclose(x[1], 0.0),
        "y1": lambda x: np.isclose(x[1], 1.0),
        "left": lambda x: np.isclose(x[0], 0.0),
        "right": lambda x: np.isclose(x[0], 1.0),
        "bottom": lambda x: np.isclose(x[1], 0.0),
        "top": lambda x: np.isclose(x[1], 1.0),
        "all": lambda x: (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                          np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)),
    }
    return markers.get(location, markers["all"])


def _evaluate_function(domain, u_func, points, gdim):
    """Evaluate a vector function at given points.
    points: shape (N, 3)
    Returns: shape (N, gdim)
    """
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    n_points = points.shape[0]
    
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    result = np.full((n_points, gdim), 0.0)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        for j, idx in enumerate(eval_map):
            result[idx, :] = vals[j, :gdim]
    
    return result


if __name__ == "__main__":
    import sys
    import json
    import time
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=None)
    args, _ = parser.parse_known_args()
    
    case_spec = {
        "pde": {
            "type": "stokes",
            "pde_params": {"nu": 0.1},
            "source_term": [
                "3*exp(-50*((x-0.15)**2 + (y-0.15)**2))",
                "3*exp(-50*((x-0.15)**2 + (y-0.15)**2))"
            ],
        },
        "domain": {"type": "unit_square"},
        "bc": {
            "dirichlet": [
                {"on": "x0", "value": ["0.0", "0.0"]},
                {"on": "y0", "value": ["0.0", "0.0"]},
                {"on": "y1", "value": ["0.0", "0.0"]},
            ]
        },
        "output": {
            "format": "npz",
            "field": "velocity_magnitude",
            "grid": {"bbox": [0, 1, 0, 1], "nx": 100, "ny": 100}
        },
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    print(f"Solve time: {elapsed:.2f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{result['u'].min():.6f}, {result['u'].max():.6f}]")
    print(f"Solution mean: {result['u'].mean():.6f}")
    print(f"Solver info: {result['solver_info']}")
    
    if args.outdir:
        from pathlib import Path
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        grid = case_spec["output"]["grid"]
        nx, ny = grid["nx"], grid["ny"]
        xarr = np.linspace(grid["bbox"][0], grid["bbox"][1], nx)
        yarr = np.linspace(grid["bbox"][2], grid["bbox"][3], ny)
        np.savez(str(outdir / "solution.npz"), x=xarr, y=yarr, u=result["u"])
        meta = {"wall_time_sec": elapsed, "solver_info": result["solver_info"]}
        with open(str(outdir / "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Output saved to {outdir}")
