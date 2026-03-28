"""Read Claude Code OAuth token from macOS Keychain."""

import json
import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)

KEYCHAIN_SERVICE = "Claude Code-credentials"


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


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_username() -> str | None:
    try:
        return subprocess.run(
            ["whoami"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.SubprocessError:
        logger.warning("Failed to get current username")
        return None


def _read_keychain_data(username: str) -> dict | None:
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
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
        return json.loads(result.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        logger.warning("Failed to parse Keychain credential data as JSON")
        return None


# ------------------------------------------------------------------
# Extended credential functions
# ------------------------------------------------------------------


def get_claude_oauth_credentials() -> dict | None:
    """Read complete OAuth credentials from macOS Keychain.

    Returns:
        Dict with keys ``accessToken``, ``refreshToken``, ``expiresAt``
        (millisecond timestamp), or None if unavailable.
    """
    username = _get_username()
    if not username:
        return None

    data = _read_keychain_data(username)
    if not data:
        return None

    oauth = data.get("claudeAiOauth", {})
    access_token = oauth.get("accessToken")
    if not access_token:
        logger.warning("No accessToken found in Claude Code credentials")
        return None

    return {
        "accessToken": access_token,
        "refreshToken": oauth.get("refreshToken"),
        "expiresAt": oauth.get("expiresAt"),
    }


def trigger_claude_token_refresh() -> bool:
    """Trigger Claude Code to refresh its OAuth token.

    Runs ``claude -p "hi"`` which causes Claude Code to check and refresh
    its token if expired, then write the new token back to Keychain.
    This keeps token ownership with Claude Code, avoiding conflicts.

    Returns:
        True if the command succeeded, False otherwise.
    """
    claude_path = shutil.which("claude")
    if not claude_path:
        logger.warning("'claude' CLI not found in PATH")
        return False

    try:
        result = subprocess.run(
            [claude_path, "-p", "hi"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        logger.warning("claude -p timed out after 30s")
        return False
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        logger.warning("Failed to run claude CLI: %s", exc)
        return False

    if result.returncode != 0:
        logger.warning(
            "claude -p failed (rc=%d): %s", result.returncode, result.stderr[:200]
        )
        return False

    logger.info("Triggered Claude Code token refresh via CLI")
    return True
