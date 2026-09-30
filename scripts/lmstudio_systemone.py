"""A System One endpoint backed by a chat model in LM Studio (Gemma 4 by default), for demos without Kev weights.

Each question is its own chat request (so questions are isolated by construction): the state, the instructions and the
options as letters, answered with one letter. The option probabilities are the model's top-logprob mass on those letters,
renormalized over the offered options. This is zero-shot prompting of a chat model, not a trained Kev checkpoint: no
LoRA, no pointer head, no calibration. It exists so the playground and the TypeSafe SDK can be pointed at LM Studio.

    uv run --no-project --with fastapi --with uvicorn --with httpx python scripts/lmstudio_systemone.py \\
        --lmstudio http://192.168.2.55:1234 --model gemma-4-e4b-it-mlx --port 8009

Requests and responses use kev.api (pydantic only, no torch), so the answers have exactly kev.serve's shape.
"""
import argparse, asyncio, math, os, random, string, sys, time, uuid

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kev.api import SystemOneRequest, to_answers, to_record   # noqa: E402

LETTERS = string.ascii_uppercase
TOP_LOGPROBS = 10   # LM Studio rejects more; letters outside the top 10 get no mass
CFG = {"lmstudio": "http://127.0.0.1:1234", "model": "gemma-4-e4b-it-mlx"}


def prompt(state, q):
    opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(q["options"]))
    return (f"Read the document and answer the question by choosing exactly one option.\n\n"
            f"Document:\n{state}\n\nQuestion: {q['instr']}\n\nOptions:\n{opts}\n\n"
            f"Answer with the option letter only.")


def letter_probs(choice, k):
    """Probability per option letter from the first content token that is an option letter: its top logprobs, summed per
    letter (' B', 'B', 'b' all count), renormalized over the k offered letters. Uniform when no letter appears."""
    valid = LETTERS[:k]
    for t in (choice.get("logprobs") or {}).get("content") or []:
        if t["token"].strip().upper() not in valid:
            continue
        mass = [0.0] * k
        for alt in t.get("top_logprobs") or [{"token": t["token"], "logprob": t["logprob"]}]:
            s = alt["token"].strip().upper()
            if s in valid: mass[valid.index(s)] += math.exp(alt["logprob"])
        total = sum(mass)
        if total > 0: return [m / total for m in mass]
    return [1 / k] * k


async def ask(client, state, q):
    """One question -> (probabilities in option order, prompt tokens)."""
    k = len(q["options"])
    if k > len(LETTERS): raise HTTPException(422, f"at most {len(LETTERS)} options per question with the LM Studio backend (got {k})")
    body = {"model": CFG["model"], "messages": [{"role": "user", "content": prompt(state, q)}], "max_tokens": 16,
            "temperature": 0, "logprobs": True, "top_logprobs": TOP_LOGPROBS}
    r = await client.post(f"{CFG['lmstudio']}/v1/chat/completions", json=body, timeout=180)
    if r.status_code != 200: raise HTTPException(502, f"LM Studio: {r.status_code} {r.text[:200]}")
    d = r.json()
    return letter_probs(d["choices"][0], k), d.get("usage", {}).get("prompt_tokens", 0)


async def answer(req, sequential=False):
    rec, meta = to_record(req)
    t = time.perf_counter()
    async with httpx.AsyncClient() as client:
        if sequential:
            res = [await ask(client, rec["state"], q) for q in rec["questions"]]
        else:
            res = await asyncio.gather(*(ask(client, rec["state"], q) for q in rec["questions"]))
    answers = to_answers([p for p, _ in res], meta)
    return {"model": req.model, "answers": answers,
            "usage": {"input_tokens": sum(n for _, n in res), "output_tokens": len(res)},
            "latency_ms": round((time.perf_counter() - t) * 1000, 1)}


app = FastAPI(title="kev-lmstudio")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["x-typesafe-request-id"])


@app.middleware("http")
async def request_id(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["x-typesafe-request-id"] = request.headers.get("x-typesafe-request-id") or uuid.uuid4().hex
    return resp


@app.post("/v1/systemone")
async def systemone(req: SystemOneRequest):
    return await answer(req)


@app.post("/v1/systemone/separate")
async def systemone_separate(req: SystemOneRequest):
    """Each question is already its own request here; this runs them one after another, for the playground's comparison."""
    return await answer(req, sequential=True)


class PermuteSystemOne(BaseModel):
    request: SystemOneRequest
    question: str
    n_perm: int = Field(default=6, ge=1, le=64)
    seed: int = 0


@app.post("/v1/systemone/permute")
async def systemone_permute(r: PermuteSystemOne):
    q = r.request.questions.get(r.question)
    if q is None or q.type != "choice": raise HTTPException(422, "question must be an existing choice question")
    rng = random.Random(r.seed); keys = list(q.criteria); runs = []
    for i in range(r.n_perm):
        order = list(keys)
        if i > 0: rng.shuffle(order)
        one = r.request.model_copy(update={"questions": {r.question: q.model_copy(update={"criteria": {k: q.criteria[k] for k in order}})}})
        resp = await answer(one); a = resp["answers"][r.question]
        runs.append({"order": order, "probabilities": a["probabilities"], "choice": a["choice"], "latency_ms": resp["latency_ms"]})
    spread = {k: max(x["probabilities"][k] for x in runs) - min(x["probabilities"][k] for x in runs) for k in keys}
    return {"runs": runs, "argmax_stable": len({x["choice"] for x in runs}) == 1, "spread": spread}


@app.get("/v1/models")
def models():
    card = {"description": f"{CFG['model']} in LM Studio at {CFG['lmstudio']}, prompted zero-shot (not a trained Kev checkpoint)",
            "release_date": time.strftime("%Y-%m-%d"), "run": f"lmstudio:{CFG['model']}", "base": CFG["model"], "backend": "lmstudio"}
    return {"models": [{"name": name, **card} for name in ("kev-latest", "jev-latest")]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmstudio", default=CFG["lmstudio"], help="LM Studio server URL")
    ap.add_argument("--model", default=CFG["model"], help="model id as LM Studio lists it under /v1/models")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8009)
    a = ap.parse_args()
    CFG.update(lmstudio=a.lmstudio.rstrip("/"), model=a.model)
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
