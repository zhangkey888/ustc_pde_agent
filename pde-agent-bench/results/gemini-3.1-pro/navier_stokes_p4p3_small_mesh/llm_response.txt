array[:] = 0.0
    
    # Newton Solver
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-9
    solver.atol = 1e-10
    solver.max_it = 20
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    # Optional: use MUMPS if available
    # pc.setFactorSolverType("mumps")
    
    n_iters, converged = solver.solve(w)
    w.x.scatter_forward()
    
    # Evaluate on 50x50 grid
    x_coords = np.linspace(0, 1, 50)
    y_coords = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    points = np.vstack((X.flatten(), Y.flatten(), np.zeros_like(X.flatten())))
    
    bb_tree = geometry.bb_tree(domain