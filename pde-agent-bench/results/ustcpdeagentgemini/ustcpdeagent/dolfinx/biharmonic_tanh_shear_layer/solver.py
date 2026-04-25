import numpy as np
import sympy as sp
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    mesh_res = 128
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res)
    
    # Define sympy exact solution
    x_sym, y_sym = sp.symbols('x y')
    u_exact_sym = sp.tanh(6*(y_sym - 0.5)) * sp.sin(sp.pi * x_sym)
    
    # Compute v = -Delta u
    u_xx = sp.diff(u_exact_sym, x_sym, 2)
    u_yy = sp.diff(u_exact_sym, y_sym, 2)
    v_exact_sym = -(u_xx + u_yy)
    
    # Compute f = Delta^2 u = -Delta v
    v_xx = sp.diff(v_exact_sym, x_sym, 2)
    v_yy = sp.diff(v_exact_sym, y_sym, 2)
    f_sym = -(v_xx + v_yy)
    
    # Convert to C code for numpy evaluation
    from sympy.utilities.lambdify import lambdify
    u_lam = lambdify((x_sym, y_sym), u_exact_sym, 'numpy')
    v_lam = lambdify((x_sym, y_sym), v_exact_sym, 'numpy')
    f_lam = lambdify((x_sym, y_sym), f_sym, 'numpy')
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Interpolate exact solutions for BCs
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(lambda x: u_lam(x[0], x[1]))
    
    v_exact_func = fem.Function(V)
    v_exact_func.interpolate(lambda x: v_lam(x[0], x[1]))
    
    # Define source term
    f_func = fem.Function(V)
    f_func.interpolate(lambda x: f_lam(x[0], x[1]))
    
    # Boundary conditions
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    bc_v = fem.dirichletbc(v_exact_func, bdofs)
    bc_u = fem.dirichletbc(u_exact_func, bdofs)
    
    # First solve: -Delta v = f
    v_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    a_v = ufl.inner(ufl.grad(v_trial), ufl.grad(v_test)) * ufl.dx
    L_v = ufl.inner(f_func, v_test) * ufl.dx
    
    problem_v = petsc.LinearProblem(
        a_v, L_v, bcs=[bc_v],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
        petsc_options_prefix="v_"
    )
    v_sol = problem_v.solve()
    
    # Second solve: -Delta u = v
    u_trial = ufl.TrialFunction(V)
    u_test = ufl.TestFunction(V)
    a_u = ufl.inner(ufl.grad(u_trial), ufl.grad(u_test)) * ufl.dx
    L_u = ufl.inner(v_sol, u_test) * ufl.dx
    
    problem_u = petsc.LinearProblem(
        a_u, L_u, bcs=[bc_u],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
        petsc_options_prefix="u_"
    )
    u_sol = problem_u.solve()
    
    # Sample onto grid
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())]
    
    tree = bb_tree(domain, domain.topology.dim)
    cell_candidates = compute_collisions_points(tree, pts)
    colliding = compute_colliding_cells(domain, cell_candidates, pts)
    
    cells = []
    points_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)
            
    u_vals = np.full(len(pts), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()
        
    u_grid = u_vals.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": 0
    }
    
    return {"u": u_grid, "solver_info": solver_info}

if __name__ == "__main__":
    case = {"output": {"grid": {"nx": 100, "ny": 100, "bbox": [0,1,0,1]}}}
    res = solve(case)
    print("Done! Output shape:", res["u"].shape)
