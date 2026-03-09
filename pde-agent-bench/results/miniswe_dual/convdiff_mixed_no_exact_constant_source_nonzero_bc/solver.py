import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import sys
import json

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None) -> dict:
    """Solve convection-diffusion equation with SUPG stabilization.
    
    -eps * laplacian(u) + beta . grad(u) = f  in Omega
    u = g on dOmega
    """
    
    # Default parameters from problem description
    epsilon = 0.005
    beta_vec = [12.0, 0.0]
    f_val = 1.0
    g_val = 1.0  # nonzero BC value
    nx_out, ny_out = 50, 50
    
    # Parse case_spec
    if case_spec is not None:
        try:
            print(f"[solver] case_spec: {json.dumps(case_spec, indent=2, default=str)}", file=sys.stderr)
        except Exception:
            pass
        
        pde = case_spec.get("pde", {})
        params = pde.get("parameters", {})
        epsilon = float(params.get("epsilon", epsilon))
        beta_raw = params.get("beta", beta_vec)
        if isinstance(beta_raw, (list, tuple, np.ndarray)):
            beta_vec = [float(b) for b in beta_raw]
        
        # Source term
        source = pde.get("source", {})
        if isinstance(source, dict):
            f_val = float(source.get("value", f_val))
        elif isinstance(source, (int, float)):
            f_val = float(source)
        
        # Boundary conditions
        bc_info = pde.get("boundary_conditions", pde.get("bc", {}))
        if isinstance(bc_info, dict):
            g_val = float(bc_info.get("value", bc_info.get("g", g_val)))
        elif isinstance(bc_info, list):
            for bc_item in bc_info:
                if isinstance(bc_item, dict):
                    if bc_item.get("type", "").lower() in ("dirichlet", "all"):
                        g_val = float(bc_item.get("value", g_val))
        
        # Output grid
        output_spec = case_spec.get("output", {})
        nx_out = int(output_spec.get("nx", nx_out))
        ny_out = int(output_spec.get("ny", ny_out))
    
    comm = MPI.COMM_WORLD
    
    # Adaptive mesh refinement
    resolutions = [64, 128]
    prev_norm = None
    final_result = None
    
    for N in resolutions:
        result = _solve_at_resolution(comm, N, epsilon, beta_vec, f_val, g_val, nx_out, ny_out)
        current_norm = result["norm"]
        
        if prev_norm is not None:
            rel_change = abs(current_norm - prev_norm) / (abs(current_norm) + 1e-15)
            if rel_change < 0.01:
                return result["output"]
        
        prev_norm = current_norm
        final_result = result["output"]
    
    return final_result


def _solve_at_resolution(comm, N, epsilon, beta_vec, f_val, g_val, nx_out, ny_out):
    """Solve at a given mesh resolution with SUPG stabilization."""
    
    element_degree = 1
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    f = fem.Constant(domain, ScalarType(f_val))
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    beta = fem.Constant(domain, np.array(beta_vec, dtype=ScalarType))
    
    # Standard Galerkin: -eps*laplacian(u) + beta.grad(u) = f
    a_std = (eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
             + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx)
    L_std = f * v * ufl.dx
    
    # SUPG stabilization for convection-dominated problems
    h = ufl.CellDiameter(domain)
    beta_mag = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_mag * h / (2.0 * eps_c)
    
    # Optimal SUPG parameter
    tau = h / (2.0 * beta_mag + 1e-10) * (1.0 / ufl.tanh(Pe_cell + 1e-10) - 1.0 / (Pe_cell + 1e-10))
    
    # SUPG modification: for P1 elements, div(grad(u)) = 0 inside each element
    # So residual = beta.grad(u) - f
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = ufl.dot(beta, ufl.grad(u)) * v_supg * ufl.dx
    L_supg = f * v_supg * ufl.dx
    
    a = a_std + a_supg
    L = L_std + L_supg
    
    # Boundary conditions: u = g on entire boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def all_boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    
    facets_all = mesh.locate_entities_boundary(domain, fdim, all_boundary)
    dofs_all = fem.locate_dofs_topological(V, fdim, facets_all)
    bc_all = fem.dirichletbc(ScalarType(g_val), dofs_all, V)
    bcs = [bc_all]
    
    # Solve with iterative solver
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": str(rtol),
                "ksp_max_it": "5000",
            },
            petsc_options_prefix="cdiff_"
        )
        u_sol = problem.solve()
    except Exception:
        # Fallback to direct solver
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
            },
            petsc_options_prefix="cdiff2_"
        )
        u_sol = problem.solve()
    
    # Evaluate on output grid
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    norm_val = np.nanmean(np.abs(u_grid))
    
    output = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 0,
        }
    }
    
    return {"output": output, "norm": norm_val}


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    u = result["u"]
    print(f"Solution shape: {u.shape}")
    print(f"Solution range: [{np.nanmin(u):.6f}, {np.nanmax(u):.6f}]")
    print(f"Solution mean: {np.nanmean(u):.6f}")
    print(f"NaN count: {np.isnan(u).sum()}")
    print(f"Solver info: {result['solver_info']}")
    print(f"Wall time: {elapsed:.2f}s")
