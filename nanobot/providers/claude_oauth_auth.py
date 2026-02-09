"""Read Claude Code OAuth token from macOS Keychain."""

import json
import logging
import subprocess

logger = logging.getLogger(__name__)


def get_claude_oauth_token() -> str | None:
    """Read OAuth access token from macOS Keychain (Claude Code credentials).

    Claude Code stores its OAuth credentials in the macOS Keychain under
    the service name "Claude Code-credentials". This function retrieves
    the access token from that entry.

    Returns:
        The OAuth access token string, or None if not found.
    """
    try:
        username = subprocess.run(
            ["whoami"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.SubprocessError:
        logger.warning("Failed to get current username")
        return None

    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                "Claude Code-credentials",
                "-a",
                username,
                "-w",
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        logger.warning("macOS 'security' command not found (non-macOS platform?)")
        return None

    if result.returncode != 0:
        logger.warning("No Claude Code credentials found in Keychain")
        return None

    try:
        data = json.loads(result.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        logger.warning("Failed to parse Keychain credential data as JSON")
        return None

    token = data.get("claudeAiOauth", {}).get("accessToken")
    if not token:
        logger.warning("No accessToken found in Claude Code credentials")
    return token
