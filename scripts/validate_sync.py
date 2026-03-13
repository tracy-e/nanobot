#!/usr/bin/env python3
"""Post-sync validation script.

Run after rebasing/merging upstream changes to catch interface breakage
before starting the gateway.

Usage:
    python3 scripts/validate_sync.py
"""

import importlib
import subprocess
import sys
from pathlib import Path

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

errors: list[str] = []
warnings: list[str] = []


def check(label: str, fn):
    """Run a check function, collect errors."""
    try:
        fn()
        print(f"  {GREEN}✓{RESET} {label}")
    except Exception as e:
        errors.append(f"{label}: {e}")
        print(f"  {RED}✗{RESET} {label}: {e}")


def warn(label: str, msg: str):
    warnings.append(f"{label}: {msg}")
    print(f"  {YELLOW}!{RESET} {label}: {msg}")


# ── 1. Core module imports ──────────────────────────────────────────

def check_imports():
    """Verify all critical modules can be imported without error."""
    print("\n[1/5] Core imports")

    modules = [
        ("providers", "nanobot.providers"),
        ("ClaudeOAuthProvider", "nanobot.providers.claude_oauth_provider"),
        ("LiteLLMProvider", "nanobot.providers.litellm_provider"),
        ("AgentLoop", "nanobot.agent.loop"),
        ("SubagentManager", "nanobot.agent.subagent"),
        ("Config schema", "nanobot.config.schema"),
        ("Config loader", "nanobot.config.loader"),
        ("Config paths", "nanobot.config.paths"),
        ("CronService", "nanobot.cron.service"),
        ("HeartbeatService", "nanobot.heartbeat.service"),
        ("ChannelManager", "nanobot.channels.manager"),
        ("FeishuChannel", "nanobot.channels.feishu"),
        ("TelegramChannel", "nanobot.channels.telegram"),
        ("DiscordChannel", "nanobot.channels.discord"),
        ("SlackChannel", "nanobot.channels.slack"),
        ("MessageTool", "nanobot.agent.tools.message"),
        ("Filesystem tools", "nanobot.agent.tools.filesystem"),
        ("Memory", "nanobot.agent.memory"),
        ("CLI commands", "nanobot.cli.commands"),
    ]
    for label, mod in modules:
        check(label, lambda m=mod: importlib.import_module(m))


# ── 2. Interface contracts ──────────────────────────────────────────

def check_interfaces():
    """Verify that cross-module interfaces match expectations."""
    print("\n[2/5] Interface contracts")

    # MessageTool must have start_turn and _sent_in_turn
    from nanobot.agent.tools.message import MessageTool
    mt = MessageTool()
    check("MessageTool.start_turn()", lambda: mt.start_turn())
    check("MessageTool._sent_in_turn", lambda: getattr(mt, "_sent_in_turn"))

    # LLMProvider base must have sanitize methods
    from nanobot.providers.base import LLMProvider
    check("LLMProvider._sanitize_empty_content",
          lambda: getattr(LLMProvider, "_sanitize_empty_content"))
    check("LLMProvider._sanitize_request_messages",
          lambda: getattr(LLMProvider, "_sanitize_request_messages"))

    # split_message must exist in helpers
    from nanobot.utils.helpers import split_message
    check("split_message()", lambda: split_message("test", 10))

    # get_data_dir must be importable from config.paths
    from nanobot.config.paths import get_data_dir
    check("get_data_dir()", lambda: get_data_dir())

    # Filesystem tools must accept allowed_dir (not workspace)
    from nanobot.agent.tools.filesystem import ReadFileTool
    import inspect
    sig = inspect.signature(ReadFileTool.__init__)
    params = list(sig.parameters.keys())
    check("ReadFileTool(allowed_dir=...)",
          lambda: None if "allowed_dir" in params else (_ for _ in ()).throw(
              AssertionError(f"params: {params}")))
    if "workspace" in params:
        warn("ReadFileTool", "'workspace' param exists — check AgentLoop._register_default_tools")

    # DiscordConfig must have group_policy
    from nanobot.config.schema import DiscordConfig
    check("DiscordConfig.group_policy",
          lambda: DiscordConfig().group_policy)

    # HeartbeatService constructor
    sig = inspect.signature(
        importlib.import_module("nanobot.heartbeat.service").HeartbeatService.__init__
    )
    hs_params = list(sig.parameters.keys())
    check("HeartbeatService(provider=, model=, on_execute=)",
          lambda: None if all(p in hs_params for p in ("provider", "model", "on_execute"))
          else (_ for _ in ()).throw(AssertionError(f"params: {hs_params}")))
    if "on_heartbeat" in hs_params:
        warn("HeartbeatService", "'on_heartbeat' param — check commands.py")

    # Subagent bootstrap files
    from nanobot.agent.subagent import SubagentManager
    check("SubagentManager.SUBAGENT_BOOTSTRAP_FILES",
          lambda: SubagentManager.SUBAGENT_BOOTSTRAP_FILES)


# ── 3. Config loading ───────────────────────────────────────────────

def check_config():
    """Verify config.json loads without errors."""
    print("\n[3/5] Config loading")
    try:
        import json
        from nanobot.config.schema import Config
        config_path = Path.home() / ".nanobot" / "config.json"
        if not config_path.exists():
            warn("config.json", "not found, skipping")
            return
        with open(config_path) as f:
            cfg = Config(**json.load(f))
        print(f"  {GREEN}✓{RESET} config.json loaded successfully")
        print(f"    Gateway port: {cfg.gateway.port}")
    except Exception as e:
        errors.append(f"Config loading: {e}")
        print(f"  {RED}✗{RESET} Config loading: {e}")


# ── 4. Gateway dry-run (import chain only) ──────────────────────────

def check_gateway_entrypoint():
    """Simulate the gateway() import chain without actually starting."""
    print("\n[4/5] Gateway entry point")
    try:
        from nanobot.agent.loop import AgentLoop
        from nanobot.bus.queue import MessageBus
        from nanobot.channels.manager import ChannelManager
        from nanobot.config.loader import load_config
        from nanobot.config.paths import get_data_dir
        from nanobot.cron.service import CronService
        from nanobot.heartbeat.service import HeartbeatService
        from nanobot.session.manager import SessionManager
        print(f"  {GREEN}✓{RESET} All gateway imports resolved")
    except Exception as e:
        errors.append(f"Gateway imports: {e}")
        print(f"  {RED}✗{RESET} Gateway imports: {e}")


# ── 5. Tests ────────────────────────────────────────────────────────

def check_tests():
    """Run pytest collection (no execution) to catch import-time failures."""
    print("\n[5/5] Test collection")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q",
         "--ignore=tests/test_matrix_channel.py"],
        capture_output=True, text=True, timeout=30,
        cwd=Path(__file__).resolve().parent.parent,
    )
    if result.returncode == 0:
        lines = result.stdout.strip().split("\n")
        count = lines[-1] if lines else "?"
        print(f"  {GREEN}✓{RESET} {count}")
    else:
        err_lines = [l for l in result.stdout.split("\n")
                     if "ERROR" in l or "ImportError" in l]
        for line in err_lines[:5]:
            print(f"  {RED}✗{RESET} {line.strip()}")
        errors.append(f"Test collection failed: {len(err_lines)} errors")


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Nanobot post-sync validation")
    print("=" * 60)

    check_imports()
    check_interfaces()
    check_config()
    check_gateway_entrypoint()
    check_tests()

    print("\n" + "=" * 60)
    if errors:
        print(f"  {RED}FAILED: {len(errors)} error(s){RESET}")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    elif warnings:
        print(f"  {YELLOW}PASSED with {len(warnings)} warning(s){RESET}")
        for w in warnings:
            print(f"    - {w}")
    else:
        print(f"  {GREEN}ALL CHECKS PASSED{RESET}")
    print("=" * 60)


if __name__ == "__main__":
    main()
