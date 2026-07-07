import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="THAÏS Social Marketing Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restreindre à ton domaine en prod si besoin
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ClaudeRequest(BaseModel):
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 3000
    system: str = ""
    messages: List[Message]
    use_web_search: bool = False

class ImagenRequest(BaseModel):
    prompt: str
    model: str = "imagen-3.0-generate-002"
    aspect_ratio: str = "1:1"
    sample_count: int = 1

@app.get("/")
def health():
    return {"status": "ok", "service": "THAÏS Social Marketing Proxy"}

@app.post("/api/claude")
async def proxy_claude(req: ClaudeRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    payload = {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "messages": [m.dict() for m in req.messages],
    }
    if req.system:
        payload["system"] = req.system
    if req.use_web_search:
        payload["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return resp.json()

@app.post("/api/imagen")
async def proxy_imagen(req: ImagenRequest):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set")

    payload = {
        "instances": [{"prompt": req.prompt}],
        "parameters": {"sampleCount": req.sample_count, "aspectRatio": req.aspect_ratio},
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{req.model}:predict?key={api_key}",
            headers={"content-type": "application/json"},
            json=payload,
        )

        if resp.status_code == 200:
            return resp.json()

        # Imagen refusé (403 accès/région, 404 modèle indisponible) → repli sur le
        # modèle d'image Gemini, accessible plus largement, réponse remise au même format.
        if resp.status_code in (403, 404):
            gem_payload = {
                "contents": [{"parts": [{"text": req.prompt}]}],
                "generationConfig": {"responseModalities": ["IMAGE"]},
            }
            gem = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent?key={api_key}",
                headers={"content-type": "application/json"},
                json=gem_payload,
            )
            if gem.status_code == 200:
                data = gem.json()
                try:
                    parts = data["candidates"][0]["content"]["parts"]
                    b64 = next(p["inlineData"]["data"] for p in parts if "inlineData" in p)
                    return {"predictions": [{"bytesBase64Encoded": b64}], "fallback": "gemini-2.5-flash-image"}
                except (KeyError, IndexError, StopIteration):
                    raise HTTPException(status_code=502, detail="Réponse Gemini sans image exploitable")
            raise HTTPException(
                status_code=gem.status_code,
                detail=f"Imagen refusé ({resp.status_code}) et repli Gemini échoué : {gem.text[:300]}",
            )

    raise HTTPException(status_code=resp.status_code, detail=resp.text)

