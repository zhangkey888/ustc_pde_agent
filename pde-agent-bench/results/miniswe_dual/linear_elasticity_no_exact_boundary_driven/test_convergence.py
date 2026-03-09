import time
import numpy as np

# Test with different mesh resolutions to check convergence
case_spec = {
    "pde": {
        "type": "linear_elasticity",
        "parameters": {"E": 1.0, "nu": 0.3},
        "source": ["0.0", "0.0"],
    },
    "domain": {
        "x_range": [0.0, 1.0],
        "y_range": [0.0, 1.0],
    },
    "boundary_conditions": [
        {"type": "dirichlet", "location": "bottom", "value": [0.0, 0.0]},
        {"type": "dirichlet", "location": "top", "value": [0.1, 0.0]},
        {"type": "dirichlet", "location": "left", "component": 1, "value": 0.0},
        {"type": "dirichlet", "location": "right", "component": 1, "value": 0.0},
    ],
    "output": {
        "nx": 50,
        "ny": 50,
        "field": "displacement_magnitude",
    },
}

# Modify solver to accept N as parameter - just import and test multiple resolutions
import sys
sys.path.insert(0, '.')

from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve_with_N(case_spec, N, deg):
    pde = case_spec.get("pde", {})
    params = pde.get("parameters", {})
    E_val = float(params.get("E", 1.0))
    nu_val = float(params.get("nu", 0.3))
    mu_val = E_val / (2.0 * (1.0 + nu_val))
    lam_val = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
    
    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])
    bcs_spec = case_spec.get("boundary_conditions", [])
    nx_out = 50
    ny_out = 50

    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(comm,
        [np.array([x_range[0], y_range[0]]), np.array([x_range[1], y_range[1]])],
        [N, N], cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", deg, (gdim,)))
    mu = fem.Constant(domain, PETSc.ScalarType(mu_val))
    lam = fem.Constant(domain, PETSc.ScalarType(lam_val))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    def epsilon(u): return ufl.sym(ufl.grad(u))
    def sigma(u): return 2.0*mu*epsilon(u) + lam*ufl.tr(epsilon(u))*ufl.Identity(gdim)
    f_expr = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    tdim = domain.topology.dim
    fdim = tdim - 1
    bcs = []
    for bc_spec in bcs_spec:
        location = bc_spec.get("location", "all")
        value = bc_spec.get("value", None)
        component = bc_spec.get("component", None)
        xr0, xr1 = x_range[0], x_range[1]
        yr0, yr1 = y_range[0], y_range[1]
        if location == "left": marker = lambda x, a=xr0: np.isclose(x[0], a)
        elif location == "right": marker = lambda x, a=xr1: np.isclose(x[0], a)
        elif location == "bottom": marker = lambda x, a=yr0: np.isclose(x[1], a)
        elif location == "top": marker = lambda x, a=yr1: np.isclose(x[1], a)
        else: marker = lambda x: np.full(x.shape[1], True)
        facets = mesh.locate_entities_boundary(domain, fdim, marker)
        if component is not None:
            comp = int(component)
            V_sub = V.sub(comp)
            V_collapsed, _ = V_sub.collapse()
            u_bc = fem.Function(V_collapsed)
            if isinstance(value, (list, tuple)):
                bc_val = float(value[comp]) if len(value) > comp else float(value[0])
            elif value is not None: bc_val = float(value)
            else: bc_val = 0.0
            u_bc.interpolate(lambda x, bv=bc_val: np.full(x.shape[1], bv))
            dofs = fem.locate_dofs_topological((V_sub, V_collapsed), fdim, facets)
            bc = fem.dirichletbc(u_bc, dofs, V_sub)
        else:
            if isinstance(value, (list, tuple)):
                val_vec = np.array([float(vv) for vv in value])
            elif value is not None: val_vec = np.array([float(value), float(value)])
            else: val_vec = np.array([0.0, 0.0])
            u_bc = fem.Function(V)
            vv0, vv1 = val_vec[0], val_vec[1]
            u_bc.interpolate(lambda x, a=vv0, b=vv1: np.vstack([np.full(x.shape[1], a), np.full(x.shape[1], b)]))
            dofs = fem.locate_dofs_topological(V, fdim, facets)
            bc = fem.dirichletbc(u_bc, dofs)
        bcs.append(bc)
    
    problem = petsc.LinearProblem(a, L, bcs=bcs,
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": "1e-12", "ksp_max_it": "5000"},
        petsc_options_prefix=f"test{N}_")
    u_sol = problem.solve()
    
    x_pts = np.linspace(x_range[0], x_range[1], nx_out)
    y_pts = np.linspace(y_range[0], y_range[1], ny_out)
    X, Y = np.meshgrid(x_pts, y_pts, indexing='ij')
    points_3d = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx_out*ny_out)])
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)
    n_points = nx_out * ny_out
    u_values = np.zeros((n_points, gdim))
    pts_list, cells_list, emap = [], [], []
    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_list.append(points_3d[:, i])
            cells_list.append(links[0])
            emap.append(i)
    if len(pts_list) > 0:
        vals = u_sol.eval(np.array(pts_list), np.array(cells_list, dtype=np.int32))
        for idx, mi in enumerate(emap):
            u_values[mi, :] = vals[idx, :gdim]
    disp_mag = np.sqrt(np.sum(u_values**2, axis=1))
    return disp_mag.reshape((nx_out, ny_out))

# Reference: N=256, deg=2
print("Computing reference (N=256, deg=2)...")
t0 = time.time()
ref = solve_with_N(case_spec, 256, 2)
print(f"  Time: {time.time()-t0:.2f}s")

for N in [32, 48, 64, 80, 128]:
    for deg in [1, 2]:
        t0 = time.time()
        sol = solve_with_N(case_spec, N, deg)
        elapsed = time.time() - t0
        diff = np.abs(sol - ref)
        ref_norm = np.sqrt(np.sum(ref**2))
        err = np.sqrt(np.sum(diff**2)) / (ref_norm + 1e-15)
        print(f"N={N:3d}, deg={deg}: rel_L2_err={err:.6e}, time={elapsed:.3f}s")
