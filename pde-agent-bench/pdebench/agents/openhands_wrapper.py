# import os
# import time
# import asyncio
# from pathlib import Path
# from typing import Dict, Any

# from .base_agent import BaseAgent, AgentResponse

# class OpenHandsWrapper(BaseAgent):
#     """
#     OpenHands Wrapper (Host Runtime 模式 - 免 Docker)
#     """
    
#     def _setup(self):
#         self.model_name = self.config.get('model', 'deepseek-chat')
#         self.api_key = self.config.get('api_key', '')
#         self.base_url = self.config.get('base_url', '')
        
#         # 创建一个极其安全的隔离文件夹作为它的工作区
#         self.workspace_dir = self.config.get(
#             'workspace_dir', 
#             '/data5/store1/zky/ustc_pde_agent/pde-agent-bench/workspace_openhands'
#         )
#         os.makedirs(self.workspace_dir, exist_ok=True)

#     def generate_solution(self, prompt: str, context: Dict[str, Any]) -> AgentResponse:
#         start_time = time.time()
        
#         # 1. 构造强限制的 Prompt，防止它在本地乱搞破坏
#         task_instruction = (
#             f"{prompt}\n\n"
#             f"【CRITICAL INSTRUCTION】\n"
#             f"You are running in a restricted Host environment (No Docker).\n"
#             f"1. Write the complete DOLFINx Python code to solve the PDE.\n"
#             f"2. Save the code EXACTLY to a file named 'solver.py' in the current directory.\n"
#             f"3. DO NOT execute the code.\n"
#             f"4. DO NOT install packages or modify environment variables."
#         )

#         try:
#             # 2. 异步调用 OpenHands
#             result_state = asyncio.run(self._run_openhands_task(task_instruction))
            
#             # 3. 从安全目录读取生成的代码
#             solver_path = os.path.join(self.workspace_dir, "solver.py")
#             if os.path.exists(solver_path):
#                 with open(solver_path, "r", encoding="utf-8") as f:
#                     generated_code = f.read()
#             else:
#                 generated_code = "" # 如果它没按指令生成文件

#             latency = time.time() - start_time
            
#             return AgentResponse(
#                 success=True,
#                 code=generated_code,
#                 raw_response=str(result_state),
#                 agent_name=self.agent_name,
#                 usage={'latency_sec': latency, 'total_tokens': 0}
#             )
            
#         except Exception as e:
#             latency = time.time() - start_time
#             return AgentResponse(
#                 success=False,
#                 code='',
#                 raw_response='',
#                 agent_name=self.agent_name,
#                 error=str(e),
#                 usage={'latency_sec': latency, 'total_tokens': 0}
#             )

#     async def _run_openhands_task(self, instruction: str):
#         # 局部导入，防止初始化框架时报错
#         from openhands.core.config import AppConfig, LLMConfig
#         from openhands.core.main import run_controller
        
#         # 配置 LLM
#         llm_config = LLMConfig(
#             model=self.model_name,
#             api_key=self.api_key,
#             base_url=self.base_url,
#         )
        
#         # 配置 OpenHands (核心：runtime="host")
#         app_config = AppConfig(
#             llm=llm_config,
#             workspace_base=self.workspace_dir,
#             runtime="host",  # 绕过 Docker，使用当前宿主机环境
#             run_as_openhands=False,
#         )
        
#         # 启动 Agent
#         state = await run_controller(config=app_config, task_str=instruction)
#         return state
#     def cleanup(self):
#         """
#         清理资源。
#         在 Benchmark 评测完当前 case 后，框架会自动调用这个方法。
#         """
#         # 如果你想在每次跑完后清空草稿本，可以写 shutil.rmtree(self.workspace_dir)
#         # 但为了方便我们观察它到底写了啥，这里先直接 pass 放过它
#         pass    

# import os
# import time
# import subprocess
# from typing import Dict, Any
# import sys
# from .base_agent import BaseAgent, AgentResponse

# class OpenHandsWrapper(BaseAgent):
#     def _setup(self):
#         self.model_name = self.config.get('model', 'deepseek-chat')
#         self.api_key = self.config.get('api_key', '')
#         self.base_url = self.config.get('base_url', '')
        
#         self.workspace_dir = self.config.get(
#             'workspace_dir', 
#             '/data5/store1/zky/ustc_pde_agent/pde-agent-bench/results/openhands/workspace'
#         )
#         os.makedirs(self.workspace_dir, exist_ok=True)

#     def generate_solution(self, prompt: str, context: Dict[str, Any]) -> AgentResponse:
#         start_time = time.time()
        
#         # 构造提示词，锁死它的行为
#         task_instruction = (
#             f"{prompt}\n\n"
#             f"【CRITICAL INSTRUCTION】\n"
#             f"You are running in a restricted Host environment (No Docker).\n"
#             f"1. Write the complete DOLFINx Python code to solve the PDE.\n"
#             f"2. Save the code EXACTLY to a file named 'solver.py' in the current directory.\n"
#             f"3. DO NOT execute the code.\n"
#             f"4. DO NOT install packages or modify environment variables."
#         )

#         try:
#             # 1. 暴力塞入环境变量，绕过配置类的版本差异
#             env = os.environ.copy()
#             safe_model = self.model_name
#             if "deepseek" in safe_model and not safe_model.startswith("openai/"):
#                 safe_model = f"openai/{safe_model}"
            
#             safe_base_url = self.base_url
#             if safe_base_url and not safe_base_url.endswith("v1") and not safe_base_url.endswith("v1/"):
#                 safe_base_url = safe_base_url.rstrip("/") + "/v1"

#             env["LLM_MODEL"] = safe_model
#             env["LLM_API_KEY"] = self.api_key
#             if safe_base_url:
#                 env["LLM_BASE_URL"] = safe_base_url

#             # 解决嫌疑人1：强制豁免本地通信，绝对不要走系统代理！
#             env["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,::1," + env.get("NO_PROXY", "")
#             env["no_proxy"] = env["NO_PROXY"]
#             # env["LLM_MODEL"] = self.model_name
#             # env["LLM_API_KEY"] = self.api_key
#             # if self.base_url:
#             #     env["LLM_BASE_URL"] = self.base_url
            
#             # 强行指定本地运行模式 (适配不同版本的环境变量命名)
#             env["RUNTIME"] = "local"
#             env["LOCAL_RUNTIME_MODE"] = "1"
#             env["WORKSPACE_BASE"] = self.workspace_dir

#             # 2. 将 Prompt 写入文件，防止因为命令行字符太长被 Linux 截断
#             instruction_file = os.path.join(self.workspace_dir, "instruction.txt")
#             with open(instruction_file, "w", encoding="utf-8") as f:
#                 f.write(task_instruction)

#             # 3. 像人类敲键盘一样调起 OpenHands CLI
#             cmd = [
#                 sys.executable, 
#                 "-m", "openhands.core.main", 
#                 "-t", task_instruction
#             ]
            
            
            
#             print(f"\n🚀 Starting OpenHands CLI in workspace: {self.workspace_dir} ...")
            
#             # 启动子进程并等待它运行完毕 (设置超时防止死循环)
#             result = subprocess.run(
#                 cmd,
#                 env=env,
#                 cwd=self.workspace_dir,
#                 capture_output=True,
#                 text=True,
#             )
            
#             # 打印一些日志方便我们观察它到底干了啥
#             if result.returncode != 0:
#                 print(f"⚠️ OpenHands exited with non-zero code {result.returncode}")
#                 # 只打印最后 500 个字符的报错信息
#                 print(f"📝 Error Log Snippet: {result.stderr[-500:]}")

#             # 4. 去草稿本里收卷子
#             solver_path = os.path.join(self.workspace_dir, "solver.py")
#             if os.path.exists(solver_path):
#                 with open(solver_path, "r", encoding="utf-8") as f:
#                     generated_code = f.read()
#             else:
#                 generated_code = ""
#                 print("⚠️ OpenHands did not generate solver.py")

#             latency = time.time() - start_time
            
#             return AgentResponse(
#                 success=True,
#                 code=generated_code,
#                 # 把最后一部分输出塞进 raw_response，供外层框架记录
#                 raw_response=result.stdout[-2000:] if result.stdout else str(result.stderr),
#                 agent_name=self.agent_name,
#                 usage={'latency_sec': latency, 'total_tokens': 0}
#             )
            
#         except Exception as e:
#             return AgentResponse(
#                 success=False,
#                 code='',
#                 raw_response='',
#                 agent_name=self.agent_name,
#                 error=str(e),
#                 usage={'latency_sec': time.time() - start_time, 'total_tokens': 0}
#             )

#     def cleanup(self):
#         pass
import os
import time
from typing import Dict, Any
from .base_agent import BaseAgent, AgentResponse

class OpenHandsWrapper(BaseAgent):
    def _setup(self):
        self.model_name = self.config.get('model', 'deepseek-chat')
        self.api_key = self.config.get('api_key', '')
        self.base_url = self.config.get('base_url', '')
        
        self.workspace_dir = self.config.get(
            'workspace_dir', 
            '/data5/store1/zky/ustc_pde_agent/pde-agent-bench/results/openhands/workspace'
        )
        os.makedirs(self.workspace_dir, exist_ok=True)

    def generate_solution(self, prompt: str, context: Dict[str, Any]) -> AgentResponse:
        start_time = time.time()
        
        # 构造安全指令
        task_instruction = (
            f"{prompt}\n\n"
            f"【CRITICAL INSTRUCTION】\n"
            f"1. Write the complete DOLFINx Python code to solve the PDE.\n"
            f"2. Save the code EXACTLY to a file named 'solver.py' in the current directory.\n"
            f"3. DO NOT execute the code, test it, or install packages.\n"
        )

        try:
            # 导入全新的 SDK 模块
            from openhands.sdk import LLM, Agent, Conversation, Tool
            from openhands.tools.file_editor import FileEditorTool
            from openhands.tools.terminal import TerminalTool
            
            # 1. 规范化 LLM 配置 (防 LiteLLM 报错)
            safe_model = self.model_name
            if "deepseek" in safe_model and not safe_model.startswith("openai/"):
                safe_model = f"openai/{safe_model}"
            
            safe_base_url = self.base_url
            if safe_base_url and not safe_base_url.endswith("v1") and not safe_base_url.endswith("v1/"):
                safe_base_url = safe_base_url.rstrip("/") + "/v1"

            # 2. 初始化大模型
            llm = LLM(
                model=safe_model,
                api_key=self.api_key,
                base_url=safe_base_url,
            )
            
            # 3. 初始化 Agent (只给它终端和文件编辑权限，物理阉割浏览器)
            agent = Agent(
                llm=llm,
                tools=[
                    Tool(name=TerminalTool.name),
                    Tool(name=FileEditorTool.name),
                ],
            )
            
            # 4. 创建对话 (原生支持本地 Workspace)
            conversation = Conversation(
                agent=agent, 
                workspace=self.workspace_dir
            )
            
            print(f"\n🚀 Starting OpenHands SDK Agent in {self.workspace_dir} ...")
            
            # 5. 运行！(新版 SDK 直接封装了同步运行，极其清爽)
            conversation.send_message(task_instruction)
            conversation.run()
            
            # 6. 收卷子
            solver_path = os.path.join(self.workspace_dir, "solver.py")
            if os.path.exists(solver_path):
                with open(solver_path, "r", encoding="utf-8") as f:
                    generated_code = f.read()
            else:
                generated_code = ""
                print("⚠️ Agent finished but did not create solver.py")

            latency = time.time() - start_time
            return AgentResponse(
                success=True,
                code=generated_code,
                raw_response="SDK execution completed successfully.",
                agent_name=self.agent_name,
                usage={'latency_sec': latency, 'total_tokens': 0}
            )
            
        except ImportError as ie:
            return AgentResponse(
                success=False, code='', raw_response='', agent_name=self.agent_name,
                error=f"SDK Import Error: {ie}. Make sure OpenHands >= 1.4.0 is installed.",
                usage={'latency_sec': time.time() - start_time, 'total_tokens': 0}
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return AgentResponse(
                success=False, code='', raw_response='', agent_name=self.agent_name,
                error=str(e), usage={'latency_sec': time.time() - start_time, 'total_tokens': 0}
            )

    def cleanup(self):
        pass