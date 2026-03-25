
"""A small generalization of the default agent that puts the user in the loop.

There are three modes:
- human: commands issued by the user are executed immediately
- confirm: commands issued by the LM but not whitelisted are confirmed by the user
- yolo: commands issued by the LM are executed immediately without confirmation
"""

import re
from typing import Literal

from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console
from rich.rule import Rule

from autopdeagent import global_config_dir
from autopdeagent.agents.default import AgentConfig, DefaultAgent, LimitsExceeded, NonTerminatingException, Submitted
from autopdeagent.tools.web_search import SearchTool
from autopdeagent.tools.inspect_tool import InspectTool

console = Console(highlight=False)
prompt_session = PromptSession(history=FileHistory(global_config_dir / "interactive_history.txt"))


class InteractiveAgentConfig(AgentConfig):
    mode: Literal["human", "confirm", "yolo"] = "confirm"
    """Whether to confirm actions."""
    whitelist_actions: list[str] = []
    """Never confirm actions that match these regular expressions."""
    confirm_exit: bool = True
    """If the agent wants to finish, do we ask for confirmation from user?"""


class InteractiveAgent(DefaultAgent):
    _MODE_COMMANDS_MAPPING = {"/u": "human", "/c": "confirm", "/y": "yolo"}

    def __init__(self, *args, config_class=InteractiveAgentConfig, **kwargs):
        # 🌟 1. 提取传入的指南文本（如果没传，给个默认报错）
        self.profiling_guide_text = kwargs.pop("profiling_guide_text", "Error: Profiling guide not configured.")
        
        super().__init__(*args, config_class=config_class, **kwargs)
        self.cost_last_confirmed = 0.0
        self.search_tool = SearchTool()
        self.inspect_tool = InspectTool()
    def add_message(self, role: str, content: str, **kwargs):
        # Extend supermethod to print messages
        super().add_message(role, content, **kwargs)
        if role == "assistant":
            # === 🔄 修改 2：修改终端打印时的 Agent 名称 ===
            console.print(
                f"\n[red][bold]PDEAgent[/bold] (step [bold]{self.model.n_calls}[/bold], [bold]${self.model.cost:.2f}[/bold]):[/red]\n",
                end="",
                highlight=False,
            )
        else:
            console.print(f"\n[bold green]{role.capitalize()}[/bold green]:\n", end="", highlight=False)
        console.print(content, highlight=False, markup=False)

    def query(self) -> dict:
        # Extend supermethod to handle human mode
        if self.config.mode == "human":
            match command := self._prompt_and_handle_special("[bold yellow]>[/bold yellow] "):
                case "/y" | "/c":  # Just go to the super query, which queries the LM for the next action
                    pass
                case _:
                    msg = {"content": f"\n```bash\n{command}\n```"}
                    self.add_message("assistant", msg["content"])
                    return msg
        try:
            with console.status("Waiting for the LM to respond..."):
                return super().query()
        except LimitsExceeded:
            console.print(
                f"Limits exceeded. Limits: {self.config.step_limit} steps, ${self.config.cost_limit}.\n"
                f"Current spend: {self.model.n_calls} steps, ${self.model.cost:.2f}."
            )
            self.config.step_limit = int(input("New step limit: "))
            self.config.cost_limit = float(input("New cost limit: "))
            return super().query()

    def step(self) -> dict:
        # Override the step method to handle user interruption
        try:
            console.print(Rule())
            return super().step()
        except KeyboardInterrupt:
            # We always add a message about the interrupt and then just proceed to the next step
            interruption_message = self._prompt_and_handle_special(
                "\n\n[bold yellow]Interrupted.[/bold yellow] "
                "[green]Type a comment/command[/green] (/h for available commands)"
                "\n[bold yellow]>[/bold yellow] "
            ).strip()
            if not interruption_message or interruption_message in self._MODE_COMMANDS_MAPPING:
                interruption_message = "Temporary interruption caught."
            raise NonTerminatingException(f"Interrupted by user: {interruption_message}")
      
    # def execute_action(self, action: dict) -> dict:
    #     """
    #     Intercept 'search' commands to run locally via Python tool.
    #     Other commands are passed to the superclass (DefaultAgent) to run in the shell environment.
    #     """
    #     command = action.get("action", "").strip()

    #     # 🆕 3. 拦截 search 命令
    #     if command.startswith("search "):
    #         # 依然遵循用户的确认模式 (Confirm Mode)
    #         if self.should_ask_confirmation(command):
    #             self.ask_confirmation()
            
    #         # 提取查询词
    #         query = command[7:].strip() # 去掉 "search " 前缀
            
    #         # 运行搜索 (在当前 Python 进程，不走 Docker/Shell)
    #         console.print(f"[bold blue]🔍 Searching web for: {query}...[/bold blue]")
    #         result_str = self.search_tool.run(query)
            
    #         # 构造返回结果 (模拟 DefaultAgent 的返回格式)
    #         # 这样 Agent 就会认为它成功执行了一个命令并看到了输出
    #         return {
    #             "output": result_str,
    #             "returncode": 0  # 假装这是 exit code 0 (成功)
    #         }
    #     if command.startswith("inspect "):
    #         if self.should_ask_confirmation(command):
    #             self.ask_confirmation()
            
    #         target = command[8:].strip() # 去掉 "inspect "
    #         console.print(f"[bold magenta]🔍 Inspecting python object: {target}...[/bold magenta]")
            
    #         result_str = self.inspect_tool.run(target)
            
    #         return {
    #             "output": result_str,
    #             "returncode": 0 # 必须带上这个！
    #         }
    #     # --- 原有逻辑 (处理普通 Shell 命令) ---
    #     if command == "get_profiling_guide":
    #         if self.should_ask_confirmation(command):
    #             self.ask_confirmation()
            
    #         console.print("[bold cyan]📖 Retrieving Profiling Guide for the Agent...[/bold cyan]")
            
    #         # 直接将存好的文本包装成 output 返回给 Agent
    #         return {
    #             "output": self.profiling_guide_text,
    #             "returncode": 0
    #         }

    #     # --- 原有逻辑 (处理普通 Shell 命令) ---
    #     if self.should_ask_confirmation(action["action"]):
    #         self.ask_confirmation()

    #     return super().execute_action(action)
        
    def execute_action(self, action: dict) -> dict:
        """
        Intercept tool commands to run locally via Python tools.
        Other commands are passed to the superclass (DefaultAgent) to run in the shell.
        """
        command = action.get("action", "").strip()

        # ── 拦截 search 命令 ──
        if command.startswith("search "):
            if self.should_ask_confirmation(command):
                self.ask_confirmation()
            query = command[7:].strip()
            console.print(f"[bold blue]🔍 Searching web for: {query}...[/bold blue]")
            result_str = self.search_tool.run(query)
            return {"output": result_str, "returncode": 0}

        # ── 拦截 inspect 命令 ──
        if command.startswith("inspect "):
            if self.should_ask_confirmation(command):
                self.ask_confirmation()
            target = command[8:].strip()
            console.print(f"[bold magenta]🔍 Inspecting python object: {target}...[/bold magenta]")
            result_str = self.inspect_tool.run(target)
            return {"output": result_str, "returncode": 0}

        # ── 拦截 get_profiling_guide 命令 ──
        if command == "get_profiling_guide":
            if self.should_ask_confirmation(command):
                self.ask_confirmation()
            console.print("[bold cyan]📖 Retrieving Profiling Guide...[/bold cyan]")
            return {"output": self.profiling_guide_text, "returncode": 0}

        # ── 🆕 拦截 get_pde_skill 命令 ──
        if command.startswith("get_pde_skill"):
            if self.should_ask_confirmation(command):
                self.ask_confirmation()
            
            # 解析方程类型参数
            parts = command.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                # agent 调用了 get_pde_skill 但没给参数
                available = ", ".join(self.pde_skill_tool.available_skills)
                result_str = (
                    f"⚠️ Usage: get_pde_skill <equation_type>\n"
                    f"Available skills: [{available}]\n"
                    f"Example: get_pde_skill navier_stokes"
                )
            else:
                equation_type = parts[1].strip()
                console.print(
                    f"[bold yellow]📚 Loading PDE skill: {equation_type}...[/bold yellow]"
                )
                result_str = self.pde_skill_tool.run(equation_type)
            
            return {"output": result_str, "returncode": 0}

        # ── 兜底：普通 Shell 命令走原逻辑 ──
        if self.should_ask_confirmation(action["action"]):
            self.ask_confirmation()

        return super().execute_action(action)

    def should_ask_confirmation(self, action: str) -> bool:
        return self.config.mode == "confirm" and not any(re.match(r, action) for r in self.config.whitelist_actions)

    def ask_confirmation(self) -> None:
        prompt = (
            "[bold yellow]Execute?[/bold yellow] [green][bold]Enter[/bold] to confirm[/green], "
            "or [green]Type a comment/command[/green] (/h for available commands)\n"
            "[bold yellow]>[/bold yellow] "
        )
        match user_input := self._prompt_and_handle_special(prompt).strip():
            case "" | "/y":
                pass  # confirmed, do nothing
            case "/u":  # Skip execution action and get back to query
                raise NonTerminatingException("Command not executed. Switching to human mode")
            case _:
                raise NonTerminatingException(
                    f"Command not executed. The user rejected your command with the following message: {user_input}"
                )

    def _prompt_and_handle_special(self, prompt: str) -> str:
        """Prompts the user, takes care of /h (followed by requery) and sets the mode. Returns the user input."""
        console.print(prompt, end="")
        user_input = prompt_session.prompt("")
        if user_input == "/h":
            console.print(
                f"Current mode: [bold green]{self.config.mode}[/bold green]\n"
                f"[bold green]/y[/bold green] to switch to [bold yellow]yolo[/bold yellow] mode (execute LM commands without confirmation)\n"
                f"[bold green]/c[/bold green] to switch to [bold yellow]confirmation[/bold yellow] mode (ask for confirmation before executing LM commands)\n"
                f"[bold green]/u[/bold green] to switch to [bold yellow]human[/bold yellow] mode (execute commands issued by the user)\n"
            )
            return self._prompt_and_handle_special(prompt)
        if user_input in self._MODE_COMMANDS_MAPPING:
            if self.config.mode == self._MODE_COMMANDS_MAPPING[user_input]:
                return self._prompt_and_handle_special(
                    f"[bold red]Already in {self.config.mode} mode.[/bold red]\n{prompt}"
                )
            self.config.mode = self._MODE_COMMANDS_MAPPING[user_input]
            console.print(f"Switched to [bold green]{self.config.mode}[/bold green] mode.")
            return user_input
        return user_input

    def has_finished(self, output: dict[str, str]):
        try:
            return super().has_finished(output)
        except Submitted as e:
            if self.config.confirm_exit:
                console.print(
                    "[bold green]Agent wants to finish.[/bold green] "
                    "[green]Type a comment to give it a new task or press enter to quit.\n"
                    "[bold yellow]>[/bold yellow] ",
                    end="",
                )
                if new_task := self._prompt_and_handle_special("").strip():
                    raise NonTerminatingException(f"The user added a new task: {new_task}")
            raise e