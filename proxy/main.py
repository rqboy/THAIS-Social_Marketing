"""THAÏS Studio LinkedIn — un seul service : interface + API.

Variables d'environnement (Render) :
  ANTHROPIC_API_KEY  clé Claude
  GOOGLE_API_KEY     clé Google (Imagen / Gemini image)
  GITHUB_TOKEN       token fine-grained, repo THAIS-Social_Marketing, Contents read/write
                     (lecture de la banque d'images + écriture de l'historique)
  GITHUB_REPO        défaut : rqboy/THAIS-Social_Marketing
  GITHUB_BRANCH      défaut : main
  ASSET_FOLDER       défaut : "Nouveau dossier"
"""
import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

REPO = os.environ.get("GITHUB_REPO", "rqboy/THAIS-Social_Marketing")
BRANCH = os.environ.get("GITHUB_BRANCH", "main")
ASSET_FOLDER = os.environ.get("ASSET_FOLDER", "Nouveau dossier")
HISTORY_PATH = "data/historique.json"  # hors du dossier proxy/ : n'entraîne pas de redéploiement
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="THAÏS Studio LinkedIn")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])


# ---------------------------------------------------------------- interface
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    return {"status": "ok", "service": "THAÏS Studio LinkedIn", "github_token": bool(os.environ.get("GITHUB_TOKEN"))}


# ---------------------------------------------------------------- GitHub
def _gh_headers():
    h = {"Accept": "application/vnd.github+json"}
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/" + "/".join(quote(p) for p in path.split("/"))


@app.get("/api/assets")
async def list_assets():
    url = f"https://api.github.com/repos/{REPO}/contents/{quote(ASSET_FOLDER)}?ref={BRANCH}"
    exts = (".jpg", ".jpeg", ".png", ".webp")
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(url, headers=_gh_headers())
        if r.status_code == 200:
            files = [f for f in r.json() if f.get("type") == "file" and f["name"].lower().endswith(exts)]
            return [{"file": f["name"], "size": f.get("size", 0), "url": _raw_url(f"{ASSET_FOLDER}/{f['name']}")} for f in files]
        # Secours sans token : index public de jsDelivr (pas de limite par IP partagée)
        j = await c.get(f"https://data.jsdelivr.com/v1/packages/gh/{REPO}@{BRANCH}?structure=flat")
        if j.status_code == 200:
            prefix = f"/{ASSET_FOLDER}/"
            out = []
            for f in j.json().get("files", []):
                name = f.get("name", "")
                if name.startswith(prefix) and "/" not in name[len(prefix):] and name.lower().endswith(exts):
                    fn = name[len(prefix):]
                    out.append({"file": fn, "size": f.get("size", 0), "url": _raw_url(f"{ASSET_FOLDER}/{fn}")})
            if out:
                return out
    raise HTTPException(r.status_code, f"Lecture de la banque d'images impossible : {r.text[:200]}")


@app.get("/api/text/{name}")
async def repo_text(name: str):
    if name not in ("sources.md", "calendrier.md"):
        raise HTTPException(404, "Fichier non autorisé")
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(_raw_url(name))
    return {"found": r.status_code == 200, "content": r.text if r.status_code == 200 else ""}


async def _read_history(c: httpx.AsyncClient):
    r = await c.get(f"https://api.github.com/repos/{REPO}/contents/{HISTORY_PATH}?ref={BRANCH}", headers=_gh_headers())
    if r.status_code == 404:
        return [], None
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"Lecture de l'historique impossible : {r.text[:200]}")
    d = r.json()
    return json.loads(base64.b64decode(d["content"]).decode("utf-8") or "[]"), d["sha"]


@app.get("/api/history")
async def get_history():
    async with httpx.AsyncClient(timeout=30) as c:
        hist, _ = await _read_history(c)
    return hist


class HistoryEntry(BaseModel):
    cluster: str
    titre: str
    angle: str = ""
    asset: str = ""
    copy: str = ""


@app.post("/api/history")
async def add_history(e: HistoryEntry):
    if not os.environ.get("GITHUB_TOKEN"):
        raise HTTPException(500, "GITHUB_TOKEN absent sur Render : l'historique ne peut pas être enregistré")
    entry = e.model_dump()
    entry["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    async with httpx.AsyncClient(timeout=30) as c:
        for _ in range(3):  # réessaie si conflit d'écriture simultanée
            hist, sha = await _read_history(c)
            hist.append(entry)
            body = {
                "message": f"historique : {entry['cluster']} — {entry['titre'][:60]}",
                "content": base64.b64encode(json.dumps(hist, ensure_ascii=False, indent=1).encode("utf-8")).decode(),
                "branch": BRANCH,
            }
            if sha:
                body["sha"] = sha
            r = await c.put(f"https://api.github.com/repos/{REPO}/contents/{HISTORY_PATH}", headers=_gh_headers(), json=body)
            if r.status_code in (200, 201):
                return {"ok": True, "count": len(hist)}
            if r.status_code != 409:
                raise HTTPException(r.status_code, f"Écriture de l'historique impossible : {r.text[:200]}")
    raise HTTPException(409, "Conflit d'écriture répété sur l'historique")


# ---------------------------------------------------------------- Claude
class Message(BaseModel):
    role: str
    content: str


class ClaudeRequest(BaseModel):
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 3000
    system: str = ""
    messages: List[Message]
    use_web_search: bool = False
    allowed_domains: Optional[List[str]] = None


@app.post("/api/claude")
async def proxy_claude(req: ClaudeRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(500, "ANTHROPIC_API_KEY not set")
    payload = {"model": req.model, "max_tokens": req.max_tokens, "messages": [m.model_dump() for m in req.messages]}
    if req.system:
        payload["system"] = req.system
    if req.use_web_search:
        tool = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
        if req.allowed_domains:
            tool["allowed_domains"] = req.allowed_domains
        payload["tools"] = [tool]
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0)) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json=payload,
            )
    except httpx.HTTPError as e:
        raise HTTPException(504, f"Claude n'a pas répondu à temps ({type(e).__name__}) : réessaie")
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, resp.text)
    return resp.json()


# ---------------------------------------------------------------- Images
class ImagenRequest(BaseModel):
    prompt: str
    model: str = "imagen-3.0-generate-002"
    aspect_ratio: str = "1:1"
    sample_count: int = 1


@app.post("/api/imagen")
async def proxy_imagen(req: ImagenRequest):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(500, "GOOGLE_API_KEY not set")
    try:
        return await _imagen(req, api_key)
    except httpx.HTTPError as e:
        raise HTTPException(504, f"Le générateur d'images n'a pas répondu à temps ({type(e).__name__}) : réessaie")


async def _imagen(req: ImagenRequest, api_key: str):
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
        resp = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{req.model}:predict?key={api_key}",
            json={"instances": [{"prompt": req.prompt}], "parameters": {"sampleCount": req.sample_count, "aspectRatio": req.aspect_ratio}},
        )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (403, 404):  # repli sur le modèle d'image Gemini
            gem = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent?key={api_key}",
                json={"contents": [{"parts": [{"text": req.prompt}]}], "generationConfig": {"responseModalities": ["IMAGE"]}},
            )
            if gem.status_code == 200:
                try:
                    parts = gem.json()["candidates"][0]["content"]["parts"]
                    b64 = next(p["inlineData"]["data"] for p in parts if "inlineData" in p)
                    return {"predictions": [{"bytesBase64Encoded": b64}], "fallback": "gemini-2.5-flash-image"}
                except (KeyError, IndexError, StopIteration):
                    raise HTTPException(502, "Réponse Gemini sans image exploitable")
            raise HTTPException(gem.status_code, f"Imagen refusé ({resp.status_code}) et repli Gemini échoué : {gem.text[:300]}")
    raise HTTPException(resp.status_code, resp.text)
