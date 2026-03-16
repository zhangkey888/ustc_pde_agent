# SUPG Stabilization
    h = ufl.CellDiameter(domain)
    vnorm = ufl.sqrt(ufl.dot(beta, beta))
    tau = 1.0 / ufl.sqrt((2.0 * vnorm / h)**2 + 9.0 * (4.0 * epsilon / h**2)**2)
    
    # Residual terms
    L_op_u = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = a + ufl.inner(L_op_u, v_supg) * ufl.dx
    L_supg = L + ufl.inner(f, v_supg) * ufl.dx