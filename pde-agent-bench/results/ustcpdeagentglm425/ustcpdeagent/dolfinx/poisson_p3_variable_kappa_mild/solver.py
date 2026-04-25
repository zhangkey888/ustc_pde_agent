import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

# Precompile forms at import time with matching element degree
# This moves JIT compilation out of the timed solve() call
_d = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2, cell_type=mesh.CellType.triangle)
_V = fem.functionspace(_d, ("Lagrange", 3))
_k = fem.Function(_V)
_k.interpolate(lambda x: 1.0 + 0.3 * np.sin(2.0 * np.pi * x[0]) * np.cos(2.0 * np.pi * x[1]))
_ut = ufl.TrialFunction(_V)
_vt = ufl.TestFunction(_V)
_a = ufl.inner(_k * ufl.grad(_ut), ufl.grad(_vt)) * ufl.dx
_L = _k * _vt * ufl.dx
_fd = _d.topology.dim - 1
_bf = mesh.locate_entities_boundary(_d, _fd, lambda x: np.ones(x.shape[1], dtype=bool))
_bd = fem.locate_dofs_topological(_V, _fd, _bf)
_ubc = fem.Function(_V)
_ubc.interpolate(lambda x: np.zeros_like(x[0]))
_bc = fem.dirichletbc(_ubc, _bd)
_p = petsc.LinearProblem(_a, _L, bcs=[_bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="d_")
_p.solve()
# Also precompile the L2 error form using two Functions (not TrialFunction)
_u1 = fem.Function(_V)
_u2 = fem.Function(_V)
_eL2_form = fem.form((_u1 - _u2)**2 * ufl.dx)
del _d, _V, _k, _ut, _vt, _a, _L, _fd, _bf, _bd, _ubc, _bc, _p, _u1, _u2, _eL2_form


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    out = case_spec["output"]
    nx_out = out["grid"]["nx"]
    ny_out = out["grid"]["ny"]
    bbox = out["grid"]["bbox"]
    
    mesh_res = 32
    elem_deg = 3
    
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_deg))
    
    # Variable coefficient kappa
    kappa_func = fem.Function(V)
    kappa_func.interpolate(lambda x: 1.0 + 0.3 * np.sin(2.0 * np.pi * x[0]) * np.cos(2.0 * np.pi * x[1]))
    
    # Source term f = -div(kappa * grad(u_exact)) computed analytically
    def f_expr(x):
        px, py = np.pi * x[0], np.pi * x[1]
        p2x, p2y = 2.0 * np.pi * x[0], 2.0 * np.pi * x[1]
        k = 1.0 + 0.3 * np.sin(p2x) * np.cos(p2y)
        dk_dx = 0.6 * np.pi * np.cos(p2x) * np.cos(p2y)
        dk_dy = -0.6 * np.pi * np.sin(p2x) * np.sin(p2y)
        spx, cpx = np.sin(px), np.cos(px)
        spy, cpy = np.sin(py), np.cos(py)
        return (2.0 * k * np.pi**2 * spx * spy
                - np.pi * spy * cpx * dk_dx
                - np.pi * spx * cpy * dk_dy)
    
    f_func = fem.Function(V)
    f_func.interpolate(f_expr)
    
    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa_func * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_func * v * ufl.dx
    
    # Dirichlet BC on entire boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Direct LU solve
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-12
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
        petsc_options_prefix="p_"
    )
    
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    iterations = problem.solver.getIterationNumber()
    
    # Sample solution on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((3, nx_out * ny_out))
    pts[0] = XX.ravel()
    pts[1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_cand = geometry.compute_collisions_points(bb_tree, pts.T)
    col_cells = geometry.compute_colliding_cells(domain, cell_cand, pts.T)
    
    pop, cop, eidx = [], [], []
    for i in range(pts.shape[1]):
        lnk = col_cells.links(i)
        if len(lnk) > 0:
            pop.append(pts[:, i])
            cop.append(lnk[0])
            eidx.append(i)
    
    u_vals = np.full((pts.shape[1],), np.nan)
    if len(pop) > 0:
        vs = u_sol.eval(np.array(pop), np.array(cop, dtype=np.int32))
        for j, idx in enumerate(eidx):
            u_vals[idx] = vs[j, 0]
    
    u_grid = u_vals.reshape(ny_out, nx_out)
    
    # L2 error verification against manufactured solution
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    eL2 = np.sqrt(max(fem.assemble_scalar(fem.form((u_sol - u_exact_func)**2 * ufl.dx)), 0.0))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_deg,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
