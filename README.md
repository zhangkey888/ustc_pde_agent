🌊 PDEAgent: Autonomous LLM-Driven PDE SolverPDEAgent is an autonomous code-generation system that bridges the gap between natural language and high-performance traditional scientific computing (FDM/FEM).Designed to overcome the efficiency bottlenecks of Physics-Informed Neural Networks (PINNs) and the steep learning curves of traditional FEM libraries, PDEAgent translates human instructions into optimal, context-aware, and physically consistent solver codes.✨ Key Features🔍 Context-Aware Inspect Tool (Zero API Hallucination)Scientific computing libraries (e.g., FEniCS, SciPy) evolve rapidly. PDEAgent features a dynamic reflection engine that actively introspects your local C++/Python bindings at runtime. By selectively retrieving class signatures and docstrings via importlib and inspect, it guarantees zero API hallucination while maintaining extreme token economy.🛡️ AST-Driven Physical Reviewer AgentCode that Returns 0 isn't necessarily physically correct. PDEAgent incorporates a Reviewer Agent that parses the Abstract Syntax Tree (AST) of the generated code before execution. It strictly enforces physical constraints (e.g., checking if dynamic CFL conditions like $\Delta t \le \frac{\Delta x}{u_{max}}$ are present in explicit schemes) and boundary condition consistency.🔄 Resilient ReAct SandboxPDEAgent executes generated codes in an isolated environment, automatically capturing deep execution errors—ranging from Python tracebacks to underlying C++ Segmentation Faults—and utilizes them as observations for self-reflection and multi-turn iterative fixing.🏆 SOTA on PDEAgentBenchmarkAchieved State-Of-The-Art performance on the comprehensive PDEAgentBenchmark, evaluated strictly on Relative L2 Error against ground-truth numerical solutions across various Out-Of-Distribution (OOD) physical scenarios.🚀 Quick StartInstallationBashgit clone https://github.com/YourName/PDEAgent.git
cd PDEAgent
pip install -r requirements.txt
UsageSimply describe your physical problem in natural language, and PDEAgent will handle the meshing, boundary conditions, and solver execution.Pythonfrom pdeagent import PDEAgent

# Initialize the agent
agent = PDEAgent(model="gpt-4o")

# Define your problem
prompt = """
Solve the 2D Poisson equation on a unit square. 
The boundary conditions are Dirichlet u=0 on all boundaries. 
Use FEniCS (dolfinx) to compute the solution and plot the result.
"""

# Agent will auto-inspect local FEniCS APIs, write code, review AST, and execute.
result = agent.solve(prompt)

print(f"Solver executed successfully. Output saved to: {result.output_dir}")
🧠 Architecture HighlightsDual-Environment Routing: Intercepts exploratory commands (inspect, search) to run instantly in the main Python process while routing high-risk solver codes to a secure sandbox.Graceful Degradation for C-Extensions: Custom handlers for pybind11 generated C++ modules that fail standard inspect.signature checks, ensuring the Agent never crashes during environmental scans.📊 Benchmark ResultsPDEAgent has been rigorously evaluated on the PDEAgentBenchmark. Instead of merely checking for syntax errors, success is strictly defined by achieving a Relative L2 Error $< 1e^{-3}$ against high-fidelity ground truth data.(Insert your benchmark comparison table/chart here)📄 CitationIf you find this project useful in your research, please consider citing our paper:代码段@article{yourname2025pdeagent,
  title={PDEAgent: An Autonomous Large Language Model System for Partial Differential Equation Solving},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
🤝 ContributingIssues and Pull Requests are greatly appreciated! Please read our Contributing Guidelines for more details.📜 LicenseThis project is licensed under the MIT License - see the LICENSE file for details.
