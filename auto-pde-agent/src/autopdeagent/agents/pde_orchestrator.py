# import yaml

# # === 🔄 修改 1：替换包名为 pdeagent ===
# from autopdeagent.agents.interactive import InteractiveAgent

# class PDEOrchestrator:
#     def __init__(self, model, env, base_config, task_prompts_path):
#         """
#         Args:
#             model: LitellmModel
#             env: LocalEnvironment
#             base_config: 从 default.yaml 读取的完整 agent 配置 (包含通用 system_template)
#             task_prompts_path: pde_task_prompts.yaml 的路径
#         """
#         self.model = model
#         self.env = env
#         self.base_config = base_config
        
#         # 加载特定的 PDE 任务 Prompt
#         with open(task_prompts_path, 'r', encoding='utf-8') as f:
#             self.task_prompts = yaml.safe_load(f)

#     def _create_agent(self, role_key, extra_cost=0.0):
#         """
#         工厂方法：创建一个混合了 Base Prompt + Task Prompt 的 Agent
#         """
#         # 1. 复制基础配置
#         config = self.base_config.copy()
        
#         # 2. 提取 Base System Template (Bash 规则)
#         base_sys = config.get("system_template", "")
        
#         # 3. 提取特定的 Task Prompt (PDE 逻辑)
#         # role_key 例如 "generator_task" 或 "verifier_task"
#         specific_task = self.task_prompts.get(role_key, "")
        
#         # 4. === 核心拼接 ===
#         # 将 Bash 规则在前，PDE 业务逻辑在后
#         combined_prompt = f"{base_sys}\n\n{specific_task}"
        
#         # 5. 更新配置
#         config["system_template"] = combined_prompt
#         if extra_cost > 0:
#             config["cost_limit"] = extra_cost
            
#         return InteractiveAgent(
#             model=self.model,
#             env=self.env,
#             **config
#         )

#     def run(self, user_query: str):
#         # ==========================================
#         # Phase 1: Generator (Agent 1)
#         # ==========================================
#         print("\n🚀 [Orchestrator] Generator Phase")
        
#         # 创建 Agent 1，混入 generator_task
#         agent1 = self._create_agent("generator_task", extra_cost=4.0)
        
#         status1, result1 = agent1.run(user_query)

#         if status1 != "Submitted":
#             return status1, f"Generator failed: {result1}"

#         # ==========================================
#         # Phase 2: Verifier (Agent 2)
#         # ==========================================
#         print("\n🧐 [Orchestrator] Verifier Phase")
        
#         # 创建 Agent 2，混入 verifier_task
#         agent2 = self._create_agent("verifier_task", extra_cost=4.0)
        
#         handoff_msg = f"""
#         **Task Handoff**
#         The Generator agent has finished writing `solver.py` for the following problem:
#         {user_query}
        
#         Please proceed with the REVIEW and FIX protocol.
#         """
        
#         status2, result2 = agent2.run(handoff_msg)
        
#         return status2, result2

import yaml
from autopdeagent.agents.interactive import InteractiveAgent

class PDEOrchestrator:
    def __init__(self, model, env, base_config, task_prompts_path):
        """
        初始化 Orchestrator，加载基础配置和 Prompt 库。
        """
        self.model = model
        self.env = env
        self.base_config = base_config
        
        with open(task_prompts_path, 'r', encoding='utf-8') as f:
            self.task_prompts = yaml.safe_load(f)

    def _create_agent(self, role_key, extra_cost=0.0):
        """
        创建单体 Agent，并将 JIT 技能指南作为参数注入，供底层拦截器使用。
        """
        config = self.base_config.copy()
        
        # 1. 拼接全局系统提示词 (Bash 规则等) 和具体角色任务 (Generator)
        base_sys = config.get("system_template", "")
        specific_task = self.task_prompts.get(role_key, "")
        config["system_template"] = f"{base_sys}\n\n{specific_task}"
        
        # 2. 动态调整预算限额
        if extra_cost > 0:
            # 如果配置中原本有 cost_limit，则追加；否则直接设置
            current_limit = config.get("cost_limit", 0.0)
            config["cost_limit"] = current_limit + extra_cost
            
        # 3. 🌟 核心：提取 Profiling 指南文本
        # 这段文本不再直接塞进 Prompt，而是作为变量传给 Agent
        guide_text = self.task_prompts.get("profiling_skill", "Error: Profiling guide not found in prompts.")
            
        # 4. 实例化 InteractiveAgent
        return InteractiveAgent(
            model=self.model,
            env=self.env,
            profiling_guide_text=guide_text,  # 传递给 interactive.py 中的拦截器
            **config
        )

    def run(self, user_query: str):
        """
        执行单智能体自循环，让 Agent 自主决策、调用工具并自我修正。
        """
        print("\n🚀 [Orchestrator] Starting Autonomous Solver Agent (JIT Skill Mode)...")
        
        # 创建单体 Agent，使用 generator_task 规则，并给予充足预算
        agent = self._create_agent("generator_task", extra_cost=8.0)
        
        # 将任务交给 Agent，Agent 内部会通过 ReAct 循环不断尝试，直到调用 submit 命令
        status, result = agent.run(user_query)

        if status != "Submitted":
            return status, f"Agent failed to converge or stopped early. Final status: {status}\nResult: {result}"

        print("\n✅ [Orchestrator] Task Completed and Verified by Autonomous Agent!")
        return status, result