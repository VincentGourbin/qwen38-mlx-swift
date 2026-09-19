#!/usr/bin/env python3
"""Agent de code minimal, connecté à LangWatch Agent Testing.

C'est le *harnais* commun du banc : la même boucle outil → modèle → outil
pour chaque modèle comparé, afin que la seule variable soit le modèle (et le
serveur qui le sert). Le modèle est un paramètre de run (`model`), donc une
comparaison LangWatch lance le même agent deux fois avec deux valeurs :

    langwatch test-suite run "Agent de code" \
        --target 'connected:qwen38-bench?model=local' \
        --target 'connected:qwen38-bench?model=gpt-5-mini' --repeat 3 --wait

Chaque conversation (thread_id) travaille dans sa propre copie de la fixture
(`bench/langwatch/fixture`, un paquet Swift minuscule), sous `workspaces/`.
Tout appel d'outil et tout appel modèle sont tracés dans LangWatch : le juge
lit ces traces (« l'agent a lancé les tests avant de répondre »), pas
seulement le texte.

Variables d'environnement :
    LANGWATCH_API_KEY   clé projet LangWatch (obligatoire, sinon rien ne se connecte)
    QWEN38_BASE_URL     serveur local (défaut http://127.0.0.1:8848/v1)
    QWEN38_MODEL        id du modèle local ; défaut : le premier de /v1/models
    OPENAI_API_KEY      pour les cibles gpt-*
    ANTHROPIC_API_KEY   pour les cibles claude-* (endpoint compatible OpenAI d'Anthropic)
    LANGWATCH_AGENT_ENVIRONMENT  ex. bench-m3max (défaut development = agent personnel)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Annotated

import langwatch
from openai import OpenAI

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixture"
WORKSPACES = HERE / "workspaces"
MAX_READ_LINES = 200
MAX_TOOL_OUTPUT = 6000  # caractères renvoyés au modèle par appel d'outil

MODEL_OPTIONS = ["local", "gpt-5-mini", "gpt-5", "claude-sonnet-5"]
THINKING_OPTIONS = ["low", "medium", "xhigh"]

SYSTEM_PROMPT = """Tu es un assistant de programmation qui travaille dans un petit dépôt Swift.
Tu disposes d'outils pour lister, lire, chercher, écrire des fichiers et lancer les tests.
Règles :
- Lis avant d'affirmer : ne décris jamais un fichier que tu n'as pas lu.
- Lance les tests avant et après une modification de code.
- Ne supprime ni ne réécris massivement sans confirmation explicite de l'utilisateur.
- Réponds en français, brièvement, en citant les fichiers et lignes concernés.
- Quand la demande n'a pas besoin des outils, réponds directement."""

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "Liste les fichiers du dépôt (chemins relatifs).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Lit un fichier du dépôt, au plus 200 lignes par appel.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Chemin relatif au dépôt"},
            "start_line": {"type": "integer", "description": "Première ligne (1 par défaut)"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "search",
        "description": "Cherche un motif (expression régulière) dans les fichiers du dépôt ; renvoie fichier:ligne:texte.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"},
        }, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Écrit (remplace) un fichier du dépôt avec le contenu donné.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "run_tests",
        "description": "Compile et lance `swift test` sur le dépôt ; renvoie la fin de la sortie.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
]


# --- espace de travail par conversation ------------------------------------

def workspace_for(thread_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", thread_id or "adhoc")
    path = WORKSPACES / safe
    if not path.exists():
        shutil.copytree(FIXTURE, path, ignore=shutil.ignore_patterns(".build"))
    return path


def safe_path(root: Path, rel: str) -> Path:
    candidate = (root / rel).resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"chemin hors du dépôt : {rel}")
    return candidate


def clip(text: str) -> str:
    return text if len(text) <= MAX_TOOL_OUTPUT else text[:MAX_TOOL_OUTPUT] + "\n… (tronqué)"


# --- outils -----------------------------------------------------------------

def tool_list_files(root: Path) -> str:
    files = sorted(
        str(p.relative_to(root)) for p in root.rglob("*")
        if p.is_file() and ".build" not in p.parts and ".git" not in p.parts
    )
    return "\n".join(files) or "(vide)"


def tool_read_file(root: Path, path: str, start_line: int = 1) -> str:
    target = safe_path(root, path)
    if not target.is_file():
        return f"introuvable : {path}"
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(int(start_line or 1), 1)
    chunk = lines[start - 1:start - 1 + MAX_READ_LINES]
    body = "\n".join(f"{start + i}: {line}" for i, line in enumerate(chunk))
    if start - 1 + MAX_READ_LINES < len(lines):
        body += f"\n… ({len(lines)} lignes au total, relance avec start_line={start + MAX_READ_LINES})"
    return body or "(fichier vide)"


def tool_search(root: Path, pattern: str) -> str:
    try:
        regex = re.compile(pattern)
    except re.error as error:
        return f"motif invalide : {error}"
    hits: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or ".build" in p.parts or ".git" in p.parts:
            continue
        for number, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if regex.search(line):
                hits.append(f"{p.relative_to(root)}:{number}:{line.strip()}")
                if len(hits) >= 50:
                    return "\n".join(hits) + "\n… (50 premiers résultats)"
    return "\n".join(hits) or "aucun résultat"


def tool_write_file(root: Path, path: str, content: str) -> str:
    target = safe_path(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"écrit : {path} ({len(content.splitlines())} lignes)"


def tool_run_tests(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["swift", "test"], cwd=root, capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return "swift test : délai de 300 s dépassé"
    output = (completed.stdout + completed.stderr).splitlines()
    tail = "\n".join(output[-60:])
    verdict = "TESTS OK" if completed.returncode == 0 else f"TESTS FAILED (code {completed.returncode})"
    return f"{verdict}\n{tail}"


def run_tool(root: Path, name: str, arguments: dict) -> str:
    with langwatch.span(type="tool", name=name, input=arguments) as span:
        try:
            if name == "list_files":
                result = tool_list_files(root)
            elif name == "read_file":
                result = tool_read_file(root, arguments.get("path", ""), arguments.get("start_line", 1))
            elif name == "search":
                result = tool_search(root, arguments.get("pattern", ""))
            elif name == "write_file":
                result = tool_write_file(root, arguments.get("path", ""), arguments.get("content", ""))
            elif name == "run_tests":
                result = tool_run_tests(root)
            else:
                result = f"outil inconnu : {name}"
        except Exception as error:  # l'erreur revient au modèle, pas au harnais
            result = f"erreur {type(error).__name__} : {error}"
        result = clip(result)
        span.update(output=result)
        return result


# --- clients modèle ---------------------------------------------------------

def local_model_id(client: OpenAI) -> str:
    if env := os.environ.get("QWEN38_MODEL"):
        return env
    models = client.models.list().data
    loaded = [m for m in models if getattr(m, "loaded", None)] or models
    if not loaded:
        raise RuntimeError("aucun modèle sur le serveur local")
    return loaded[0].id


def make_client(model: str) -> tuple[OpenAI, str, dict]:
    """(client, id du modèle sur le fil, extra_body) pour une valeur du paramètre `model`."""
    if model == "local":
        client = OpenAI(base_url=os.environ.get("QWEN38_BASE_URL", "http://127.0.0.1:8848/v1"), api_key="local")
        return client, local_model_id(client), {}
    if model.startswith("gpt-"):
        return OpenAI(), model, {}
    if model.startswith("claude-"):
        client = OpenAI(base_url="https://api.anthropic.com/v1/", api_key=os.environ["ANTHROPIC_API_KEY"])
        return client, model, {}
    raise ValueError(f"modèle inconnu : {model}")


def reasoning_kwargs(model: str, thinking: str) -> dict:
    if model == "local":
        # Le gabarit Qwen n'accepte que low / medium / xhigh.
        return {"extra_body": {"reasoning_effort": thinking}}
    if model.startswith("gpt-"):
        return {"reasoning_effort": {"low": "low", "medium": "medium", "xhigh": "high"}[thinking]}
    return {}


# --- la boucle d'agent ------------------------------------------------------

def chat_turn(client: OpenAI, model_id: str, messages: list[dict], extra: dict):
    with langwatch.span(type="llm", name="chat.completions", model=model_id, input=messages) as span:
        started = time.monotonic()
        response = client.chat.completions.create(
            model=model_id, messages=messages, tools=TOOLS, tool_choice="auto",
            temperature=0.7, max_tokens=4096, **extra,
        )
        choice = response.choices[0].message
        usage = response.usage
        metrics = {}
        if usage:
            metrics = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
        span.update(
            output=choice.content or json.dumps([c.model_dump() for c in (choice.tool_calls or [])]),
            metrics=metrics or None,
            params={"latency_ms": int((time.monotonic() - started) * 1000)},
        )
        return choice


@langwatch.connect_agent(
    name="qwen38-bench",
    sticky=True,          # une conversation reste sur cette instance : l'espace de travail y vit
    timeout=300,          # `swift test` compris
    concurrency=1,        # un seul modèle local, une conversation à la fois
)
def qwen38_bench(
    messages: list[dict],
    thread_id: str,
    trace_id: str | None = None,
    *,
    model: Annotated[str, langwatch.Param(description="Modèle sous test", options=MODEL_OPTIONS)] = "local",
    thinking: Annotated[str, langwatch.Param(description="Effort de réflexion", options=THINKING_OPTIONS)] = "low",
    max_steps: Annotated[int, langwatch.Param(description="Appels d'outils au plus par tour")] = 12,
) -> str:
    # Un seul trace LangWatch par tour, sous le contexte adopté par le relais :
    # sans cet enveloppement, chaque span (llm, tool) partait comme un trace
    # orphelin et le juge ne voyait aucun appel d'outil (vérifié 2026-09-19).
    with langwatch.trace(name="qwen38-bench", metadata={"model": model, "thinking": thinking,
                                                        "platform_trace_id": trace_id or ""}) as trace:
        span = trace.root_span if hasattr(trace, "root_span") else None
        reply = run_turn(messages, thread_id, model, thinking, max_steps)
        if span is not None:
            try:
                span.update(output=reply)
            except Exception:
                pass
        return reply


def run_turn(messages: list[dict], thread_id: str, model: str, thinking: str, max_steps: int) -> str:
    root = workspace_for(thread_id)
    client, model_id, extra = make_client(model)
    extra = {**extra, **reasoning_kwargs(model, thinking)}

    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    history += [{"role": m["role"], "content": m.get("content", "")} for m in messages if m.get("role") in ("user", "assistant")]

    for _ in range(max_steps):
        choice = chat_turn(client, model_id, history, extra)
        if not choice.tool_calls:
            return choice.content or ""
        history.append({
            "role": "assistant", "content": choice.content or "",
            "tool_calls": [c.model_dump() for c in choice.tool_calls],
        })
        for call in choice.tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            result = run_tool(root, call.function.name, arguments)
            history.append({"role": "tool", "tool_call_id": call.id, "content": result})

    # Budget d'outils épuisé : on demande une conclusion sans outil.
    history.append({"role": "user", "content": "Conclus maintenant en une réponse, sans nouvel appel d'outil."})
    choice = chat_turn(client, model_id, history, extra)
    return choice.content or "(pas de réponse après épuisement du budget d'outils)"


if __name__ == "__main__":
    langwatch.setup()
    WORKSPACES.mkdir(exist_ok=True)
    print(f"qwen38-bench : connecté ({os.environ.get('LANGWATCH_AGENT_ENVIRONMENT', 'development')}), Ctrl-C pour arrêter")
    langwatch.agent.serve()
