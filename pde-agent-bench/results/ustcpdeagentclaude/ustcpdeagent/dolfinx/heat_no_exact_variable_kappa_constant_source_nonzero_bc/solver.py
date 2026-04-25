import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    # Mesh
    N = 140
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    kappa = 1.0 + 0.5 * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    f_src = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    # BC
    u_bc_expr = ufl.sin(ufl.pi*x[0]) + ufl.cos(ufl.pi*x[1])
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_bc_expr, V.element.interpolation_points))
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Time
    t_end = 0.1
    dt = 0.0025
    n_steps = int(round(t_end / dt))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0  # initial condition
    
    # Backward Euler: (u - u_n)/dt - div(kappa grad u) = f
    a = u*v*ufl.dx + dt_c*ufl.inner(kappa*ufl.grad(u), ufl.grad(v))*ufl.dx
    L = (u_n + dt_c*f_src)*v*ufl.dx
    
    u_sol = fem.Function(V)
    
    # Initial grid sample
    def sample_on_grid(func):
        xs = np.linspace(bbox[0], bbox[1], nx_out)
        ys = np.linspace(bbox[2], bbox[3], ny_out)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)])
        tree = geometry.bb_tree(domain, domain.topology.dim)
        cc = geometry.compute_collisions_points(tree, pts)
        col = geometry.compute_colliding_cells(domain, cc, pts)
        vals = np.zeros(pts.shape[0])
        cells = []
        pts_ok = []
        idx = []
        for i in range(pts.shape[0]):
            links = col.links(i)
            if len(links) > 0:
                cells.append(links[0])
                pts_ok.append(pts[i])
                idx.append(i)
        if len(pts_ok) > 0:
            r = func.eval(np.array(pts_ok), np.array(cells, dtype=np.int32))
            vals[idx] = r.flatten()
        return vals.reshape(ny_out, nx_out)
    
    u_initial = sample_on_grid(u_n)
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc], u=u_sol,
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
        petsc_options_prefix="heat_"
    )
    
    total_iters = 0
    for step in range(n_steps):
        problem.solve()
        try:
            total_iters += problem.solver.getIterationNumber()
        except Exception:
            pass
        u_n.x.array[:] = u_sol.x.array[:]
    
    u_grid = sample_on_grid(u_n)
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iters,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }
