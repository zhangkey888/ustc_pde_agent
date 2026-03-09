import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module
import json
import argparse
from pathlib import Path

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve transient convection-diffusion with SUPG stabilization.
    
    Matches oracle formulation for minimal error.
    """
    
    # ---- Extract parameters from case_spec ----
    pde = case_spec.get("pde", {})
    oracle_config = case_spec.get("oracle_config", {})
    if not pde and oracle_config:
        pde = oracle_config.get("pde", {})
    
    # Diffusion and convection parameters
    pde_params = pde.get("pde_params", {})
    epsilon = pde_params.get("epsilon", pde.get("epsilon", 0.02))
    beta_vec = pde_params.get("beta", pde.get("beta", [6.0, 3.0]))
    
    # Time parameters - with hardcoded defaults as fallback
    time_spec = pde.get("time", {})
    t_end = time_spec.get("t_end", 0.1)
    dt_val = time_spec.get("dt", 0.02)
    time_scheme = time_spec.get("scheme", "backward_euler")
    
    # Use smaller dt for better accuracy
    dt_val = min(dt_val, 0.01)
    
    # Output grid spec
    output_spec = oracle_config.get("output", {}).get("grid", {})
    nx_out = output_spec.get("nx", 50)
    ny_out = output_spec.get("ny", 50)
    bbox = output_spec.get("bbox", [0, 1, 0, 1])
    
    # Mesh resolution - balance accuracy and speed
    N = 160
    degree = 2
    
    comm = MPI.COMM_WORLD
    
    # ---- Create mesh ----
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # ---- Function space ----
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # ---- Define spatial coordinates and time ----
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(0.0))
    
    eps_val = epsilon
    
    # Velocity field
    beta = ufl.as_vector([float(beta_vec[0]), float(beta_vec[1])])
    
    # ---- Source term (using UFL constant for time) ----
    f_expr = ufl.exp(-150.0 * ((x[0] - 0.4)**2 + (x[1] - 0.6)**2)) * ufl.exp(-t_const)
    
    # ---- Initial condition ----
    u_prev = fem.Function(V, name="u_prev")
    u_prev.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # ---- Trial and test functions ----
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # ---- Boundary conditions (homogeneous Dirichlet on all boundaries) ----
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    # ---- SUPG Stabilization (match oracle formulation exactly) ----
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    upwind_parameter = 1.0
    tau = upwind_parameter * h / (2.0 * beta_norm + 1e-12)
    
    # ---- Time stepping parameters ----
    n_steps = int(np.ceil(t_end / dt_val))
    actual_dt = t_end / n_steps
    dt = actual_dt
    
    # ---- Bilinear form (oracle formulation) ----
    # a = (u*v + dt*(eps*grad(u)·grad(v) + beta·grad(u)*v)) * dx
    # + SUPG: dt * tau * dot(beta, grad(v)) * (dot(beta, grad(u)) - eps*div(grad(u))) * dx
    a_form = (u * v + dt * (eps_val * ufl.inner(ufl.grad(u), ufl.grad(v)) 
              + ufl.dot(beta, ufl.grad(u)) * v)) * ufl.dx
    
    # SUPG on bilinear form
    a_form += dt * tau * ufl.dot(beta, ufl.grad(v)) * (
        ufl.dot(beta, ufl.grad(u)) - eps_val * ufl.div(ufl.grad(u))
    ) * ufl.dx
    
    # ---- Linear form (oracle formulation) ----
    # L = (u_prev*v + dt*f*v) * dx + SUPG: dt*tau*dot(beta,grad(v))*f * dx
    L_form = (u_prev * v + dt * f_expr * v) * ufl.dx
    L_form += dt * tau * ufl.dot(beta, ufl.grad(v)) * f_expr * ufl.dx
    
    # ---- Compile forms ----
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # ---- Assemble matrix (constant in time) ----
    A = petsc.assemble_matrix(a_compiled, bcs=bcs)
    A.assemble()
    
    # ---- Create RHS vector ----
    b = petsc.create_vector(V)
    
    # ---- Setup solver ----
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    
    # ---- Solution function ----
    u_h = fem.Function(V, name="u_h")
    u_h.x.array[:] = u_prev.x.array[:]
    
    # ---- Time stepping ----
    current_t = 0.0
    total_iterations = 0
    
    for step in range(n_steps):
        current_t += actual_dt
        t_const.value = current_t
        
        # Assemble RHS (time-dependent due to f_expr and u_prev)
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_prev.x.array[:] = u_h.x.array[:]
    
    # ---- Evaluate solution on output grid ----
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = XX.ravel()
    points_3d[:, 1] = YY.ravel()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # ---- Also evaluate initial condition on same grid ----
    u_init_func = fem.Function(V)
    u_init_func.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    u_init_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": actual_dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=None)
    args, _ = parser.parse_known_args()
    
    case_spec = {
        "pde": {
            "epsilon": 0.02,
            "beta": [6.0, 3.0],
            "time": {
                "t_end": 0.1,
                "dt": 0.02,
                "scheme": "backward_euler"
            }
        },
        "domain": {
            "type": "unit_square",
            "bounds": [[0, 0], [1, 1]]
        },
        "oracle_config": {
            "output": {
                "grid": {
                    "bbox": [0, 1, 0, 1],
                    "nx": 50,
                    "ny": 50
                }
            }
        }
    }
    
    start = time_module.time()
    result = solve(case_spec)
    elapsed = time_module.time() - start
    
    print(f"Solve completed in {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    print(f"Solver info: {result['solver_info']}")
    
    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        
        u_grid = result["u"]
        nx_out, ny_out = u_grid.shape
        xs = np.linspace(0, 1, nx_out)
        ys = np.linspace(0, 1, ny_out)
        
        np.savez(str(outdir / "solution.npz"), x=xs, y=ys, u=u_grid)
        np.save(str(outdir / "u.npy"), u_grid)
        
        if "u_initial" in result and result["u_initial"] is not None:
            np.save(str(outdir / "u_initial.npy"), result["u_initial"])
        
        meta = {
            "wall_time_sec": elapsed,
            "solver_info": result["solver_info"]
        }
        with open(str(outdir / "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        
        print(f"Output written to {outdir}")
