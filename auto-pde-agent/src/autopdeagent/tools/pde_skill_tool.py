# autopdeagent/tools/pde_skill_tool.py

"""PDE domain-specific skill knowledge base."""


class PdeSkillTool:
    """Registry of PDE-specific solving methodology guides.
    
    Each skill is a plain-text methodology guide (no framework-specific code),
    keyed by a canonical equation type name.
    """

    def __init__(self):
        self._skills: dict[str, str] = {}
        self._register_builtin_skills()

    def _register_builtin_skills(self):
        """Register all built-in PDE skills."""
        self._skills["navier_stokes"] = _NAVIER_STOKES_SKILL
        self._skills["biharmonic"] = _BIHARMONIC_SKILL
        self._skills["convection_diffusion"] = _CONVECTION_DIFFUSION_SKILL
        self._skills["darcy"] = _DARCY_SKILL
        self._skills["helmholtz"] = _HELMHOLTZ_SKILL
        self._skills["heat"] = _HEAT_SKILL
        self._skills["linear_elasticity"] = _LINEAR_ELASTICITY_SKILL
        self._skills["poisson"] = _POISSON_SKILL
        self._skills["reaction_diffusion"] = _REACTION_DIFFUSION_SKILL
        self._skills["stokes"] = _STOKES_SKILL
        # 未来扩展点：
        # self._skills["poisson"] = _POISSON_SKILL
        # self._skills["stokes"] = _STOKES_SKILL
        # self._skills["advection_diffusion"] = _ADVECTION_DIFFUSION_SKILL

    @property
    def available_skills(self) -> list[str]:
        return list(self._skills.keys())

    def run(self, equation_type: str) -> str:
        """Look up and return the skill guide for the given equation type.
        
        Args:
            equation_type: canonical name like 'navier_stokes', 'poisson', etc.
        
        Returns:
            The skill text, or a structured 'not found' message with available options.
        """
        key = equation_type.strip().lower().replace("-", "_").replace(" ", "_")

        # 尝试模糊匹配常见别名
        alias_map = {
            "ns": "navier_stokes",
            "navier_stokes": "navier_stokes",
            "navierstokes": "navier_stokes",
            "incompressible_ns": "navier_stokes",
            "fluid_flow": "navier_stokes",
            "biharmonic": "biharmonic",
            "bi_harmonic": "biharmonic",
            "fourth_order": "biharmonic",
            "plate_bending": "biharmonic",
            "delta_squared": "biharmonic",
            "bilaplacian": "biharmonic",
            "convection_diffusion": "convection_diffusion",
            "advection_diffusion": "convection_diffusion",
            "convdiff": "convection_diffusion",
            "conv_diff": "convection_diffusion",
            "transport": "convection_diffusion",
            "scalar_transport": "convection_diffusion",
            "darcy": "darcy",
            "darcy_flow": "darcy",
            "porous_media": "darcy",
            "porous_medium": "darcy",
            "groundwater": "darcy",
            "permeability": "darcy",
            "helmholtz": "helmholtz",
            "wave_equation_frequency": "helmholtz",
            "indefinite_elliptic": "helmholtz",
            "acoustic": "helmholtz",
            "heat": "heat",
            "heat_equation": "heat",
            "diffusion": "heat",
            "thermal": "heat",
            "parabolic": "heat",
            "transient_diffusion": "heat",
            "linear_elasticity": "linear_elasticity",
            "elasticity": "linear_elasticity",
            "elastic": "linear_elasticity",
            "solid_mechanics": "linear_elasticity",
            "lame": "linear_elasticity",
            "displacement": "linear_elasticity",
            "poisson": "poisson",
            "laplace": "poisson",
            "laplacian": "poisson",
            "diffusion_steady": "poisson",
            "reaction_diffusion": "reaction_diffusion",
            "allen_cahn": "reaction_diffusion",
            "allen-cahn": "reaction_diffusion",
            "fisher_kpp": "reaction_diffusion",
            "fisher-kpp": "reaction_diffusion",
            "logistic_diffusion": "reaction_diffusion",
            "stokes": "stokes",
            "stokes_flow": "stokes",
            "creeping_flow": "stokes",
            "slow_flow": "stokes",
            # 未来扩展
            # "poisson": "poisson",
            # "laplace": "poisson",
            # "diffusion": "poisson",
            # "stokes": "stokes",
            # "advection_diffusion": "advection_diffusion",
            # "convection_diffusion": "advection_diffusion",
        }

        resolved_key = alias_map.get(key, key)

        if resolved_key in self._skills:
            header = (
                f"═══ PDE SKILL LOADED: {resolved_key.upper().replace('_', ' ')} ═══\n"
                f"Follow the methodology below STRICTLY when implementing your solver.\n"
                f"{'═' * 60}\n\n"
            )
            return header + self._skills[resolved_key]
        else:
            available = ", ".join(sorted(self._skills.keys()))
            return (
                f"⚠️ No skill found for equation type: '{equation_type}'\n"
                f"Available skills: [{available}]\n\n"
                f"Proceed with general FEM best practices:\n"
                f"  - Choose appropriate element type and degree for your unknowns\n"
                f"  - Verify boundary conditions are well-posed\n"
                f"  - For nonlinear problems, use Newton with Stokes/linear init\n"
                f"  - For coupled systems, consider mixed elements with inf-sup stable pairs\n"
                f"  - Test with a manufactured solution if possible\n"
            )


# ═══════════════════════════════════════════════════════════════
#  SKILL DEFINITIONS (pure methodology, no framework-specific code)
# ═══════════════════════════════════════════════════════════════

_NAVIER_STOKES_SKILL = r"""
## 1. Mixed FE Discretization (Taylor-Hood)
- Velocity: k-th order continuous Lagrange (recommend k=2)
- Pressure: (k-1)-th order continuous Lagrange
- Combine into mixed space W = V × Q, solve coupled
- This pairing auto-satisfies inf-sup stability. Velocity degree MUST be strictly higher than pressure.

## 2. Variational Residual Form
Residual F(w)=0 where w=(u,p), test functions (v,q):
  F = [(u·∇)u]·v + ν ∇u:∇v − p(∇·v) − q(∇·u) − f·v  (integrated over Ω)
Jacobian J = dF/dw (automatic differentiation of F w.r.t. w).

## 3. Manufactured Solution Verification (MMS)
If exact solution is provided:
  1. Verify ∇·u_exact = 0 symbolically. Abort if not satisfied.
  2. Back-compute source: f_i = Σ_j u_j ∂_j u_i − ν Δu_i + ∂_i p  (use symbolic differentiation)
  3. Solve with this f, compare numerical vs analytical solution.

## 4. Dirichlet Boundary Conditions
- NS equations REQUIRE Dirichlet BCs; otherwise ill-posed.
- CRITICAL PITFALL: For "all boundary" selection, use TOPOLOGICAL method:
    locate boundary facets first → find DOFs on those facets.
  DO NOT use geometric method with `np.ones(...)` — it matches interior nodes too,
  bypassing the entire FEM solve.
- Geometric method (e.g., `np.isclose(x[0], 0.0)`) is safe for specific boundary segments.

## 5. Pressure Pinning
When ALL boundaries have velocity Dirichlet BCs, pressure is determined only up to a constant.
Fix by: set p=0 at one corner point (e.g., origin).
Not needed if natural outflow BCs exist.

## 6. Newton Solver Configuration
- Each iteration solves: J δw = −F, then w ← w + α·δw
- Use backtracking line search ("bt") to prevent divergence
- Linear sub-solver: GMRES + LU preconditioner (small/medium), GMRES + AMG (large)
- Typical tolerances: relative 1e-10, absolute 1e-12, max iterations 50

## 7. Initialization Strategy (CRITICAL for convergence)
Choose based on Reynolds number:
  | Strategy              | Method                                              | When to use        |
  |-----------------------|-----------------------------------------------------|--------------------|
  | Stokes init (DEFAULT) | Drop convection term, solve linear Stokes as initial| Low-to-moderate Re |
  | Zero init             | w₀ = 0                                              | Only very low Re   |
  | Exact init            | Use known analytical solution                       | Code verification  |
  | Viscosity continuation| Ramp ν from high→target in log-spaced steps         | Moderate-to-high Re|

## 8. Viscosity Continuation (High-Re Essential)
When Stokes init fails to converge Newton:
  1. Set ν_start = 1.0 (very low Re)
  2. Build log-spaced sequence: ν_k = ν_start · (ν_target/ν_start)^(k/N), N≈8 steps
  3. Step k=1: initialize from Stokes solution at ν_start
  4. Step k>1: initialize from converged solution at ν_{k-1}
  5. Final step yields solution at target ν
Principle: adjacent steps have small solution changes → Newton always converges.

## 9. Source Term Handling
Priority: manufactured solution (symbolic back-computation) > user-specified expression > zero.
For constant source terms, use Constant object (efficient) rather than per-node interpolation.
""".strip()







_BIHARMONIC_SKILL = r"""
## 1. Problem Decomposition (Mixed Poisson Splitting)
The biharmonic equation Δ²u = f is a 4th-order PDE. DO NOT attempt to discretize it directly with standard C⁰ Lagrange elements — they lack the required C¹ continuity.

Instead, split into two coupled 2nd-order Poisson problems by introducing an auxiliary variable w = −Δu:
  - Step 1: −Δw = f  (solve for w)
  - Step 2: −Δu = w  (solve for u, using w from Step 1)

Each sub-problem is a standard Poisson equation solvable with C⁰ Lagrange elements.

## 2. Finite Element Discretization
- Both u and w live in the SAME scalar function space V
- Standard continuous Lagrange elements (degree ≥ 1, recommend degree 2+)
- No mixed element pairing needed — the two solves are SEQUENTIAL, not coupled simultaneously
- No inf-sup stability concern (unlike Stokes/NS)

## 3. Variational Forms
Both sub-problems have identical bilinear form structure:
  a(trial, test) = ∫ ∇trial · ∇test dx
  
Step 1: a(w, v) = ∫ f·v dx
Step 2: a(u, q) = ∫ w_h·q dx  (w_h is the computed solution from Step 1)

Key: Step 2's RHS uses the COMPUTED w_h (a FEM Function), not a symbolic expression.

## 4. Boundary Conditions
Each sub-problem needs its own Dirichlet BC:
- For w: boundary value = g_w (if manufactured: g_w = −Δu_exact; otherwise typically w=0)
- For u: boundary value = g_u (if manufactured: g_u = u_exact; otherwise from problem spec, default 0)

CRITICAL: Both sub-problems require well-posed BCs independently. Forgetting the w boundary condition is a common mistake that silently produces wrong results.

## 5. Manufactured Solution Verification (MMS)
If exact u is given:
  1. Symbolically compute w_exact = −Δu_exact = −(∂²u/∂x² + ∂²u/∂y²)
  2. Symbolically compute f = −Δw_exact = Δ²u_exact
  3. Use g_w = w_exact and g_u = u_exact as Dirichlet BCs
  4. Solve both steps, compare u_h with u_exact

All symbolic differentiation should use a symbolic math engine to avoid errors.

## 6. Solver Configuration
Both sub-problems are symmetric positive-definite (SPD) linear systems.
- Optimal solver: CG (Conjugate Gradient) + AMG preconditioner (e.g., Hypre BoomerAMG)
- This is MUCH faster than direct solvers for large meshes
- Typical tolerances: relative 1e-10, absolute 1e-12
- No Newton iteration needed — both problems are LINEAR

## 7. Source Term Handling
Same priority as other equations:
  if manufactured solution → symbolic back-computation (f = Δ²u_exact)
  elif user-specified → parse expression; use Constant object for constant values
  else → f = 0 (Constant)

## 8. Key Pitfalls
- DO NOT use standard Lagrange elements to directly discretize Δ²u — requires C¹ elements (Hermite, Argyris) which are complex and often unavailable. The splitting approach avoids this entirely.
- The solve order is STRICT: w first, then u. Reversing gives nonsense.
- If w boundary conditions are wrong/missing, u will converge to the wrong solution with no obvious error signal.
- For non-manufactured cases with no explicit w BC, w=0 on ∂Ω is the standard choice (corresponding to "simply supported" plate conditions).
""".strip()





_CONVECTION_DIFFUSION_SKILL = r"""
## 1. Problem Structure
Scalar PDE with competing convection and diffusion:
  Steady:    −ε Δu + β·∇u = f
  Transient: ∂u/∂t − ε Δu + β·∇u = f

Key parameter: Peclet number Pe = |β|h/(2ε), where h is mesh size.
- Pe ≪ 1: diffusion-dominated, standard Galerkin works fine
- Pe ≫ 1: convection-dominated, standard Galerkin produces spurious oscillations → MUST stabilize

## 2. Finite Element Discretization
- Standard scalar continuous Lagrange elements (degree ≥ 1)
- Single function space V, no mixed formulation needed
- The problem is LINEAR — no Newton iteration required

## 3. Variational Form (Steady)
Trial function u, test function v:
  a(u,v) = ∫ [ε ∇u·∇v + (β·∇u)v] dx
  L(v)   = ∫ f·v dx

The bilinear form is NON-SYMMETRIC due to the convection term β·∇u.
Therefore: DO NOT use CG solver. Use GMRES + ILU preconditioner.

## 4. SUPG Stabilization (Critical for High Peclet)
When Pe > 1, add Streamline-Upwind Petrov-Galerkin (SUPG) terms to BOTH forms:

  Stabilization parameter: τ = α·h / (2|β| + small_eps)
    where α is tunable upwind parameter (default 1.0), h = cell diameter

  Added to bilinear form:
    a += ∫ τ (β·∇v)(β·∇u − ε Δu) dx

  Added to linear form:
    L += ∫ τ (β·∇v) f dx

KEY POINTS:
- SUPG modifies the test function effectively to "upwind" along streamlines
- The ε·Δu term inside SUPG involves second derivatives of the trial function.
  For linear elements (degree=1), Δu = 0 element-wise, so this term vanishes.
  For degree ≥ 2, this term is nonzero and must be included.
- The stabilization must be added to BOTH a and L consistently, otherwise the method is inconsistent.
- The small_eps (e.g., 1e-12) in the denominator prevents division by zero when β=0.

## 5. Manufactured Solution Verification (MMS)
If exact u is given:
  Steady:    f = −ε Δu_exact + β·∇u_exact
  Transient: f = ∂u_exact/∂t − ε Δu_exact + β·∇u_exact
Use symbolic differentiation for all derivatives.

## 6. Transient Case — Backward Euler Time Stepping
Semi-discrete with implicit Euler (unconditionally stable):
  (u^{n+1} − u^n)/dt − ε Δu^{n+1} + β·∇u^{n+1} = f^{n+1}

Variational form at each time step:
  a(u,v) = ∫ [u·v + dt·(ε ∇u·∇v + (β·∇u)v)] dx
  L(v)   = ∫ [u_prev·v + dt·f·v] dx

KEY POINTS:
- The bilinear form a is TIME-INDEPENDENT (constant dt, ε, β) → assemble ONCE, reuse every step
- Only L changes each step (through u_prev and possibly time-dependent f and BCs)
- SUPG stabilization in transient: add dt·τ(β·∇v)(…) to both a and L, same structure as steady
- Update u_prev ← u_new at end of each step (array copy)
- For time-dependent BCs or source: re-evaluate expressions at each time level t^{n+1}

## 7. Solver Configuration
- Linear system each step → use Krylov solver, NOT Newton
- GMRES + ILU is the standard choice (non-symmetric system)
- CG is WRONG here because the convection term breaks symmetry
- Typical tolerances: relative 1e-10, absolute 1e-12

## 8. Key Pitfalls
- Using CG solver on a convection-diffusion system → CG requires SPD matrix, convection term makes it non-symmetric → solver may silently produce garbage or diverge
- Forgetting SUPG when Pe > 1 → wild oscillations near boundary layers, solution looks "noisy"
- Applying SUPG to linear form but not bilinear form (or vice versa) → inconsistent method, wrong convergence order
- In transient problems, reassembling the bilinear form every step when it hasn't changed → massive performance waste; assemble once and only rebuild RHS
- Not updating boundary conditions at each time step when they are time-dependent → wrong solution that "lags" in time
""".strip()




_DARCY_SKILL = r"""
## 1. Problem Structure & Formulation Choice
Darcy flow models slow flow through porous media. Two equivalent formulations:

**Elliptic (pressure-only, simpler, preferred when only pressure is needed):**
  −∇·(κ ∇p) = f,  p = g on ∂Ω

**Mixed (flux-pressure, needed when accurate flux is required):**
  u + κ ∇p = 0    (constitutive law)
  ∇·u      = f    (mass conservation)

Choose elliptic unless the problem explicitly asks for flux output or uses RT elements.

## 2. Elliptic Formulation
- Standard scalar Lagrange elements, single function space
- Bilinear form: a(p,v) = ∫ κ ∇p·∇v dx,  linear form: L(v) = ∫ f·v dx
- System is SPD → use CG + AMG preconditioner (optimal)
- If flux is needed post-solve: compute u = −κ ∇p via projection into a DG vector space

## 3. Mixed Formulation
- Use H(div)×L² stable pair: RT_k × DG_{k-1} (Raviart-Thomas for flux, discontinuous Galerkin for pressure)
- RT elements are VECTOR-VALUED intrinsically — do NOT add extra shape=(dim,)
- Variational form (saddle-point):
    a = ∫ (1/κ) u·v dx − ∫ p ∇·v dx + ∫ q ∇·u dx
    L = ∫ f·q dx
- System is symmetric-indefinite → AMG is NOT safe. Use direct solver (LU) for robustness.
- Flux BC: impose u·n on boundary via topological DOF location on boundary facets.

## 4. Pressure Pinning (Mixed Only)
When flux BCs are imposed everywhere, pressure is determined up to a constant.
Fix by pinning one DG pressure DOF to zero.
PITFALL: DG DOFs are cell-based, NOT vertex-based. Geometric lookup at a point may return empty.
Fallback: pin the first available DOF from the collapsed sub-space map.

## 5. Permeability κ Handling
- κ must be strictly positive everywhere
- Can be constant (use fem.Constant, efficient) or spatially varying expression
- For manufactured solutions, κ participates in source term derivation:
    f = −∇·(κ ∇p_exact) = ∇·u_exact
  Use symbolic differentiation including κ(x) in the chain rule.

## 6. MMS (Manufactured Solution)
Given exact pressure p_exact and κ:
  1. u_exact = −κ ∇p_exact (symbolic)
  2. f = ∇·u_exact (symbolic, includes ∂κ/∂x terms if κ varies spatially)
  3. BCs from exact solutions: p = p_exact (elliptic) or u·n = u_exact·n (mixed)

## 7. Key Pitfalls
- Using AMG on the mixed saddle-point system → likely divergence or garbage; use LU
- Forgetting that RT elements are already vector-valued → wrapping with shape=(dim,) creates a tensor field, wrong space
- DG pressure DOF pinning via geometric point query fails silently → always have a fallback to pin first available DOF
- Spatially varying κ: must include ∇κ terms when deriving manufactured source (product rule)
""".strip()







_HELMHOLTZ_SKILL = r"""
## 1. Problem Structure
Scalar indefinite elliptic PDE:
  −Δu − k²u = f  in Ω,   u = g on ∂Ω

The −k²u term (reaction with NEGATIVE sign) makes the operator indefinite for moderate/large wave number k.
This is fundamentally different from Poisson (−Δu = f, SPD) and changes solver requirements completely.

## 2. Finite Element Discretization
- Standard scalar continuous Lagrange elements
- CRITICAL: mesh must resolve the wave — rule of thumb: at least 10 DOFs per wavelength λ = 2π/k
  i.e., h ≤ λ/10 = 2π/(10k). For large k, this demands fine meshes.
- Higher-order elements (degree ≥ 2) reduce pollution error and are strongly recommended for large k.

## 3. Variational Form
  a(u,v) = ∫ [∇u·∇v − k²uv] dx
  L(v)   = ∫ f·v dx

The bilinear form is SYMMETRIC but INDEFINITE (not positive-definite).
Therefore: CG solver is WRONG. CG requires SPD. Use GMRES instead.

## 4. Solver Configuration
- Default: GMRES + ILU preconditioner (handles indefiniteness)
- For large k or ill-conditioned cases: fallback to direct solver (LU)
- DO NOT use CG + AMG — the system is not SPD, CG will fail or give wrong results
- Typical tolerances: relative 1e-10, absolute 1e-12

## 5. MMS (Manufactured Solution)
If exact u is given:
  f = −Δu_exact − k²·u_exact
  = −(∂²u/∂x² + ∂²u/∂y²) − k²·u_exact
Use symbolic differentiation.

## 6. Key Pitfalls
- Using CG solver → system is indefinite, CG assumes SPD, will diverge or silently give wrong answer
- Insufficient mesh resolution for large k → "pollution effect" where numerical dispersion accumulates; increase mesh density or element degree
- Forgetting the negative sign: the operator is −Δu − k²u, both terms have specific signs in the weak form; ∇u·∇v comes from −Δu (positive after integration by parts), −k²uv stays negative
""".strip()





_HEAT_SKILL = r"""
## 1. Problem Structure
Parabolic PDE (transient diffusion):
  ∂u/∂t − ∇·(κ ∇u) = f  in Ω×(0,T],   u = g on ∂Ω,   u(0) = u₀

Scalar unknown, single function space, LINEAR at each time step.

## 2. Time Discretization — Backward Euler (Implicit)
Semi-discrete: (u^{n+1} − u^n)/dt − ∇·(κ ∇u^{n+1}) = f^{n+1}

Variational form at each step:
  a(u,v) = ∫ [u·v + dt·κ ∇u·∇v] dx
  L(v)   = ∫ [u_prev·v + dt·f·v] dx

KEY: bilinear form a is TIME-INDEPENDENT (constant dt, κ) → assemble ONCE, reuse every step.
Only L changes (through u_prev and possibly time-dependent f, BCs).

## 3. Solver Configuration
Each time step is an SPD linear system (mass + stiffness).
- Optimal: CG + AMG preconditioner (symmetric positive-definite)
- This is the ONE case where CG is correct and optimal — unlike Helmholtz or convection-diffusion.

## 4. Variable Conductivity κ
κ can be constant or spatially varying κ(x).
- For MMS with variable κ: source term must include product rule:
    f = ∂u/∂t − [∂/∂x(κ ∂u/∂x) + ∂/∂y(κ ∂u/∂y)]
  The ∂κ/∂x terms matter — do NOT simplify to f = ∂u/∂t − κΔu when κ varies.

## 5. Time-Dependent BCs and Source
- Re-evaluate f and boundary values at each time level t^{n+1}
- Update BC function's DOF values BEFORE solving each step
- Forgetting to update BCs → solution "freezes" at initial BC values

## 6. Key Pitfalls
- Reassembling bilinear form every step when it hasn't changed → massive performance waste
- Using explicit Euler without stability check: explicit requires dt ≤ Ch² (very restrictive); backward Euler is unconditionally stable, always prefer it
- Variable κ MMS: forgetting ∇κ terms in manufactured source → wrong convergence rate
""".strip()




_LINEAR_ELASTICITY_SKILL = r"""
## 1. Problem Structure
Vector-valued elliptic PDE (2D/3D solid mechanics, small strain):
  −div(σ(u)) = f  in Ω,   u = g on ∂Ω

where:
  ε(u) = sym(∇u)                          (symmetric strain tensor)
  σ(u) = 2μ ε(u) + λ tr(ε(u)) I          (Cauchy stress, isotropic)

Unknown u is a VECTOR field (displacement), not scalar.

## 2. Material Parameters (Lamé Constants)
Two equivalent parameterizations — must support both:
  - Direct: (λ, μ) given
  - Engineering: (E, ν) given → convert via plane strain formulas:
      μ = E / [2(1+ν)]
      λ = Eν / [(1+ν)(1−2ν)]

PITFALL: ν → 0.5 makes λ → ∞ (incompressible limit). For ν > 0.45, standard displacement formulation locks — consider mixed methods or reduced integration.

## 3. Finite Element Discretization
- Vector-valued continuous Lagrange elements (same degree for all displacement components)
- Single vector function space V, NOT a mixed space
- System is SPD → use CG + AMG preconditioner

## 4. Variational Form
  a(u,v) = ∫ σ(u) : ε(v) dx
  L(v)   = ∫ f · v dx

Expand: a(u,v) = ∫ [2μ ε(u):ε(v) + λ tr(ε(u)) tr(ε(v))] dx

CRITICAL: use sym(grad(u)) for strain, NOT grad(u). Using unsymmetrized gradient gives a different (wrong) operator.

## 5. MMS (Manufactured Solution)
Given exact displacement u_exact = (u₁, u₂):
  1. Compute strain: ε₁₁ = ∂u₁/∂x, ε₂₂ = ∂u₂/∂y, ε₁₂ = ½(∂u₁/∂y + ∂u₂/∂x)
  2. Compute stress: σᵢⱼ = 2μ εᵢⱼ + λ(ε₁₁+ε₂₂)δᵢⱼ
  3. Source: f₁ = −(∂σ₁₁/∂x + ∂σ₁₂/∂y), f₂ = −(∂σ₂₁/∂x + ∂σ₂₂/∂y)
All via symbolic differentiation.

## 6. Boundary Conditions
- Same topological vs geometric pitfall as NS: "all boundary" must use topological method
- For vector BCs, each component can be specified independently
- Dirichlet BCs are applied to the vector space directly (not per-component sub-spaces)

## 7. Key Pitfalls
- Using grad(u) instead of sym(grad(u)) → wrong physics, asymmetric "strain"
- Near-incompressible material (ν ≈ 0.5) with standard elements → volumetric locking, artificially stiff solution
- Forgetting that f is a VECTOR source term matching displacement dimension
- Boundary DOF selection for "all": must use topological approach, same as NS
""".strip()




_POISSON_SKILL = r"""
## 1. Problem Structure
The simplest elliptic PDE — scalar unknown, linear, steady:
  −∇·(κ ∇u) = f  in Ω,   u = g on ∂Ω

When κ=1: reduces to −Δu = f (standard Poisson).

## 2. Finite Element Discretization
- Standard scalar continuous Lagrange elements (degree ≥ 1)
- Single function space, no mixed formulation needed
- System is SPD → CG + AMG is the optimal solver

## 3. Variational Form
  a(u,v) = ∫ κ ∇u·∇v dx
  L(v)   = ∫ f·v dx

Linear, symmetric, coercive — the most well-behaved FEM problem.

## 4. Variable Coefficient κ(x)
When κ varies spatially, it enters the bilinear form as a weight on the gradient.
For MMS, source term must use product rule:
  f = −∇·(κ ∇u_exact) = −[∂/∂x(κ ∂u/∂x) + ∂/∂y(κ ∂u/∂y)]
DO NOT simplify to f = −κΔu when κ is not constant.

## 5. Solver
- CG + AMG (e.g., Hypre BoomerAMG): optimal O(N) complexity
- Typical tolerances: relative 1e-10, absolute 1e-12
- This is the ONE equation where everything "just works" — if your Poisson solve fails, something is fundamentally wrong with your setup

## 6. Key Pitfalls
- Pure Neumann BCs (no Dirichlet anywhere): solution is determined only up to a constant → must either pin one DOF or add a mean-zero constraint
- Variable κ MMS: forgetting ∇κ terms in product rule → wrong convergence rate
- Using GMRES instead of CG: works but wastes memory; system is SPD, CG is sufficient and cheaper
""".strip()



_REACTION_DIFFUSION_SKILL = r"""
## 1. Problem Structure
Scalar PDE with diffusion and pointwise nonlinear reaction:
  Steady:    −ε Δu + R(u) = f
  Transient: ∂u/∂t − ε Δu + R(u) = f

The reaction R(u) determines whether the problem is linear or nonlinear:
  | Type        | R(u)              | Linear? |
  |-------------|-------------------|---------|
  | linear      | αu                | YES     |
  | cubic       | αu + βu³          | NO      |
  | Allen-Cahn  | λ(u³ − u)         | NO      |
  | Fisher-KPP  | ρu(1 − u)         | NO      |

CRITICAL DECISION: Check the reaction type FIRST — it determines whether you need Newton or a simple linear solve.

## 2. Variational Form
Steady linear (R = αu):
  a(u,v) = ∫ [ε ∇u·∇v + αuv] dx,   L(v) = ∫ fv dx
  → SPD system, use CG + AMG

Steady nonlinear (e.g. R = αu + βu³):
  Residual form: F = ∫ [ε ∇u·∇v + R(u)v − fv] dx = 0
  Jacobian: J = dF/du (automatic differentiation)
  → Newton iteration with GMRES + ILU (or LU for robustness)

## 3. Transient — Backward Euler
  F = ∫ [(u−u_prev)/dt · v + ε ∇u·∇v + R(u)v − fv] dx = 0

Even if R(u) is nonlinear, each time step requires Newton.
Update u_prev ← u_new after each step. Re-evaluate time-dependent f and BCs at t^{n+1}.

## 4. Newton Solver for Nonlinear Reactions
- Use backtracking line search to handle cubic/higher-order nonlinearities
- Zero initialization is usually fine for moderate reaction strengths
- For strong nonlinearities (large β or λ), consider:
  - Initializing from the linear problem (set nonlinear coefficients to 0)
  - Parameter continuation on the nonlinear coefficient

## 5. MMS (Manufactured Solution)
  Steady: f = −ε Δu_exact + R(u_exact)
  Transient: f = ∂u_exact/∂t − ε Δu_exact + R(u_exact)
Substitute the SYMBOLIC u_exact into R(u) before differentiating. Use symbolic engine throughout.

## 6. Key Pitfalls
- Treating a nonlinear reaction as linear → assembling R(u) into the bilinear form with TrialFunction is WRONG for cubic terms; u³ cannot appear in a bilinear form. Must use residual + Newton.
- Linear reaction with α > 0 keeps the system SPD (reaction adds positive mass). But Allen-Cahn has −λu term which can make the linearized system indefinite → use GMRES, not CG, for Newton steps.
- Transient: reassembling the FULL nonlinear problem each step is correct but expensive. The Jacobian structure doesn't change between steps — reuse sparsity pattern where possible.
""".strip()







_STOKES_SKILL = r"""
## 1. Problem Structure
Steady incompressible Stokes — LINEAR saddle-point system (no convection):
  −ν Δu + ∇p = f    in Ω
  ∇·u        = 0    in Ω
  u = g              on ∂Ω

This is the linearized version of Navier-Stokes (drops the (u·∇)u convection term).
The system is LINEAR — no Newton iteration needed.

## 2. Mixed FE Discretization (Taylor-Hood)
Same as Navier-Stokes:
- Velocity: k-th order continuous Lagrange (recommend k=2)
- Pressure: (k-1)-th order continuous Lagrange
- Mixed space W = V × Q, solve coupled
- Satisfies inf-sup stability automatically. Velocity degree MUST be strictly higher than pressure.

## 3. Variational Form
Trial (u,p), test (v,q):
  a = ∫ [ν ∇u:∇v − p(∇·v) − q(∇·u)] dx
  L = ∫ f·v dx

The system is symmetric-indefinite (saddle-point structure).

## 4. Solver Configuration
- MINRES + AMG preconditioner is the natural choice (symmetric indefinite system)
- MINRES exploits the symmetry — more efficient than GMRES for this structure
- Alternative: direct solver (LU) for small/medium problems
- DO NOT use CG — system is indefinite (not positive-definite)
- Typical tolerance: relative 1e-10

## 5. Pressure Pinning
Same as Navier-Stokes: when all boundaries have velocity Dirichlet BCs, pressure is determined only up to a constant. Fix by pinning p=0 at one corner point.

## 6. Boundary Conditions
Same topological pitfall as NS:
- "All boundary" → MUST use topological method (locate facets first, then DOFs)
- DO NOT use geometric `np.ones(...)` for all-boundary — matches interior DOFs too

## 7. MMS (Manufactured Solution)
Given exact (u, p):
  f_i = −ν Δu_i + ∂p/∂x_i
No convection term (unlike NS). Verify ∇·u_exact = 0 for consistency.

## 8. Stokes vs Navier-Stokes — When to Use Which
- If the problem has NO convection term (u·∇)u → Stokes (linear, one solve)
- If the problem HAS convection → Navier-Stokes (nonlinear, Newton required)
- Stokes is often used as the INITIAL GUESS for NS Newton iteration
""".strip()