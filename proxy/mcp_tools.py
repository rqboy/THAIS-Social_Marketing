"""MCP tools mounted inside the existing THAIS FastAPI service."""

import os
from typing import Any, Optional

import httpx
from mcp.server.fastmcp import FastMCP

THAIS_BASE_URL = os.getenv(
    "THAIS_BASE_URL", "https://thais-social-marketing.onrender.com"
).rstrip("/")

mcp = FastMCP("THAIS Social Marketing", stateless_http=True)


async def _request(
    method: str,
    path: str,
    *,
    json: Optional[dict[str, Any]] = None,
    timeout: float = 120.0,
) -> Any:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.request(f"{THAIS_BASE_URL}{path}", json=json)
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def thais_health() -> dict[str, Any]:
    """Check THAIS availability and whether its GitHub token is loaded."""
    return await _request("GET", "/health", timeout=20)


@mcp.tool()
async def list_thais_assets() -> list[dict[str, Any]]:
    """List THAIS image assets with filenames, sizes, and URLs."""
    return await _request("GET", "/api/assets", timeout=60)


@mcp.tool()
async def get_thais_history() -> list[dict[str, Any]]:
    """Read validated THAIS posts."""
    return await _request("GET", "/api/history", timeout=60)


@mcp.tool()
async def get_thais_sources(document: str = "sources.md") -> dict[str, Any]:
    """Read THAIS editorial sources or calendar."""
    if document not in {"sources.md", "calendrier.md"}:
        raise ValueError("document must be 'sources.md' or 'calendrier.md'")
    return await _request("GET", f"/api/text/{document}", timeout=30)


@mcp.tool()
async def save_validated_post(
    cluster: str,
    title: str,
    angle: str = "",
    asset: str = "",
    copy: str = "",
) -> dict[str, Any]:
    """Save a human-approved post to THAIS history."""
    return await _request(
        "POST",
        "/api/history",
        json={
            "cluster": cluster,
            "titre": title,
            "angle": angle,
            "asset": asset,
            "copy": copy,
        },
        timeout=60,
    )


@mcp.tool()
async def draft_with_thais_claude(
    messages: list[dict[str, str]],
    system: str = "",
    max_tokens: int = 3000,
    use_web_search: bool = False,
    allowed_domains: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Use the Claude API already configured inside THAIS to draft content."""
    if not messages:
        raise ValueError("messages must contain at least one message")
    return await _request(
        "POST",
        "/api/claude",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "use_web_search": use_web_search,
            "allowed_domains": allowed_domains,
        },
        timeout=180,
    )


@mcp.tool()
async def generate_thais_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    sample_count: int = 1,
) -> dict[str, Any]:
    """Generate an image through THAIS Imagen/Gemini fallback."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    return await _request(
        "POST",
        "/api/imagen",
        json={
            "prompt": prompt,
            "model": "imagen-3.0-generate-002",
            "aspect_ratio": aspect_ratio,
            "sample_count": sample_count,
        },
        timeout=180,
    )
