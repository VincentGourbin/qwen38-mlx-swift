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
import threading
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
THINKING_OPTIONS = ["off", "low", "medium", "xhigh"]

# La plateforme plafonne un tour d'agent à 300 s. Le tour est chronométré dès
# son entrée (attente du verrou comprise) : chaque appel modèle reçoit un
# `max_tokens` proportionnel au temps restant, et passé TURN_BUDGET_S l'agent
# conclut sans outil au lieu de laisser la plateforme couper (ERROR). Une
# conclusion forcée peut être jugée FAILED : c'est un résultat, pas une erreur
# du banc (ASK L-4, 2026-09-19 : une génération de 3 739 jetons ≈ 250 s).
TURN_BUDGET_S = 150.0      # 4 tours × 150 s + simulateur + juge < plafond du scénario (~900 s)
TOKENS_PER_S = 12.0        # débit local prudent (Bonsai 2 ≈ 15-20, Flash-Next ≈ 13)
CONCLUDE_RESERVE_S = 50.0  # réservé à la conclusion : les outils s'arrêtent avant
MAX_CALL_TOKENS = 3072
BLANK_REPLY = "Je n'ai pas pu formuler de réponse dans le temps imparti ; reformule ou précise la demande."
UNFINISHED_REPLY = "Je n'ai pas terminé : il me reste des vérifications à faire. Dis-moi si je continue."
TOOL_CALL_XML = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)

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
            ["swift", "test"], cwd=root, capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "swift test : délai de 120 s dépassé"
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
    # max_retries=0 : le client OpenAI réessaie deux fois par défaut après un
    # délai dépassé, soit trois générations complètes pour un seul tour.
    if model == "local":
        client = OpenAI(base_url=os.environ.get("QWEN38_BASE_URL", "http://127.0.0.1:8848/v1"),
                        api_key="local", max_retries=0)
        return client, local_model_id(client), {}
    if model.startswith("gpt-"):
        return OpenAI(max_retries=0), model, {}
    if model.startswith("claude-"):
        client = OpenAI(base_url="https://api.anthropic.com/v1/", api_key=os.environ["ANTHROPIC_API_KEY"],
                        max_retries=0)
        return client, model, {}
    raise ValueError(f"modèle inconnu : {model}")


def reasoning_kwargs(model: str, thinking: str) -> dict:
    if model == "local":
        # Le gabarit Qwen n'accepte que low / medium / xhigh ; "off" coupe la
        # réflexion (`enable_thinking: false`).
        if thinking == "off":
            return {"extra_body": {"enable_thinking": False}}
        return {"extra_body": {"reasoning_effort": thinking}}
    if model.startswith("gpt-"):
        return {"reasoning_effort": {"off": "low", "low": "low", "medium": "medium", "xhigh": "high"}[thinking]}
    return {}


def clamp(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def no_thinking(model: str, extra: dict) -> dict:
    """Variante de `extra` sans réflexion, pour une conclusion courte : avec la
    réflexion active, un petit `max_tokens` est entièrement consommé par le
    bloc <think> et le contenu revient vide."""
    if model == "local":
        return {"extra_body": {"enable_thinking": False}}
    if model.startswith("gpt-"):
        return {"reasoning_effort": "low"}
    return extra


class Turn:
    """Résultat d'un appel modèle, accumulé depuis le flux."""

    def __init__(self):
        self.content = ""
        self.tool_calls: dict[int, dict] = {}
        self.finish_reason = None
        self.usage = None
        self.aborted = False

    @property
    def calls(self) -> list[dict]:
        return [self.tool_calls[i] for i in sorted(self.tool_calls)]


# --- la boucle d'agent ------------------------------------------------------

def chat_turn(client: OpenAI, model_id: str, messages: list[dict], extra: dict,
              *, max_tokens: int, deadline: float, with_tools: bool = True) -> Turn:
    """Un appel modèle en flux. À l'échéance, le flux est fermé : le serveur
    local annule alors la génération (chemin diffusé), au lieu de la finir
    dans le vide et de bloquer la file pour le tour suivant."""
    with langwatch.span(type="llm", name="chat.completions", model=model_id, input=messages) as span:
        started = time.monotonic()
        tool_kwargs = {"tools": TOOLS, "tool_choice": "auto"} if with_tools else {}
        turn = Turn()
        stream = client.chat.completions.create(
            model=model_id, messages=messages, temperature=0.7, max_tokens=max_tokens,
            stream=True, timeout=90, **tool_kwargs, **extra,
        )
        try:
            for chunk in stream:
                if time.monotonic() > deadline:
                    turn.aborted = True
                    break
                if getattr(chunk, "usage", None):
                    turn.usage = chunk.usage
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta and delta.content:
                    turn.content += delta.content
                for tc in (delta.tool_calls or []) if delta else []:
                    slot = turn.tool_calls.setdefault(tc.index or 0, {"id": None, "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
                if choice.finish_reason:
                    turn.finish_reason = choice.finish_reason
        finally:
            try:
                stream.close()
            except Exception:
                pass
        metrics = {}
        if turn.usage:
            metrics = {"prompt_tokens": turn.usage.prompt_tokens, "completion_tokens": turn.usage.completion_tokens}
        span.update(
            output=turn.content or json.dumps(turn.calls, ensure_ascii=False),
            metrics=metrics or None,
            params={"latency_ms": int((time.monotonic() - started) * 1000), "max_tokens": max_tokens,
                    "aborted": turn.aborted},
        )
        return turn


# Un seul modèle local à la fois : le serveur sert une conversation après
# l'autre, et un appel refusé (`agent_busy`) fait échouer le run côté
# plateforme après ~90 s de réessais (observé 2026-09-19 : 7 runs sur 8 en
# ERROR). D'où : accepter plusieurs appels (une cible du marché peut tourner
# en parallèle) mais sérialiser les tours du modèle local derrière ce verrou,
# et lancer les scénarios un par un (voir run.sh).
LOCAL_TURN = threading.Lock()


@langwatch.connect_agent(
    name="qwen38-bench",
    sticky=True,          # une conversation reste sur cette instance : l'espace de travail y vit
    timeout=300,          # `swift test` compris — plafond de la plateforme
    concurrency=4,        # les appels en attente du verrou comptent ici
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
    deadline = time.monotonic() + TURN_BUDGET_S   # l'attente du verrou compte
    with langwatch.trace(name="qwen38-bench", metadata={"model": model, "thinking": thinking,
                                                        "platform_trace_id": trace_id or ""}) as trace:
        span = trace.root_span if hasattr(trace, "root_span") else None
        if model == "local":
            with LOCAL_TURN:
                reply = run_turn(messages, thread_id, model, thinking, max_steps, deadline)
        else:
            reply = run_turn(messages, thread_id, model, thinking, max_steps, deadline)
        if span is not None:
            try:
                span.update(output=reply)
            except Exception:
                pass
        return reply


def run_turn(messages: list[dict], thread_id: str, model: str, thinking: str, max_steps: int,
             deadline: float) -> str:
    root = workspace_for(thread_id)
    client, model_id, extra = make_client(model)
    extra = {**extra, **reasoning_kwargs(model, thinking)}

    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    history += [{"role": m["role"], "content": m.get("content", "")} for m in messages if m.get("role") in ("user", "assistant")]
    partial = ""   # dernier texte visible, rendu si le budget s'épuise

    def remaining() -> float:
        return deadline - time.monotonic()

    def spoken(text: str, wanted_tools: bool = False) -> str:
        """Jamais de tour blanc (un blanc relance le simulateur en boucle
        jusqu'au plafond du scénario, « Prudence », 2026-09-20), et jamais de
        XML <tool_call> brut dans ce que voit l'utilisateur simulé
        (« Correction », 2026-09-21 : la conclusion sans outils déclarés
        laissait passer le XML que le modèle émet par habitude)."""
        text = TOOL_CALL_XML.sub("", text or "").strip()
        if text:
            return text
        return UNFINISHED_REPLY if wanted_tools else BLANK_REPLY

    # Les outils ne dépassent jamais `deadline - CONCLUDE_RESERVE_S` : la
    # conclusion a toujours sa place, quel que soit le temps pris par les
    # appels (préremplissage compris, qui n'est pas compté par TOKENS_PER_S).
    tools_deadline = deadline - CONCLUDE_RESERVE_S
    for _ in range(max_steps):
        if time.monotonic() > tools_deadline - 20:
            break
        try:
            turn = chat_turn(
                client, model_id, history, extra,
                max_tokens=clamp((tools_deadline - time.monotonic()) * TOKENS_PER_S - 100, 256, MAX_CALL_TOKENS),
                deadline=tools_deadline)
        except Exception as error:   # délai réseau, serveur : on conclut avec ce qu'on a
            partial = partial or f"(appel interrompu : {type(error).__name__})"
            break
        if turn.content.strip():
            partial = turn.content
        if turn.aborted:
            break
        if not turn.calls:
            if turn.content.strip():
                return turn.content
            # Réponse vide (le modèle a émis sa fin de tour sans texte, observé
            # sur Bonsai 2 après des appels d'outils sur une demande vague) :
            # une relance, la même pour toutes les cibles, sans réflexion.
            break
        history.append({
            "role": "assistant", "content": turn.content or "",
            "tool_calls": [{"id": c["id"] or f"call_{i}", "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                           for i, c in enumerate(turn.calls)],
        })
        for i, call in enumerate(turn.calls):
            try:
                arguments = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            result = run_tool(root, call["name"], arguments)
            history.append({"role": "tool", "tool_call_id": call["id"] or f"call_{i}", "content": result})

    # Budget d'outils épuisé, tour vide ou appel coupé : une conclusion courte,
    # sans réflexion, dans la réserve. Les outils restent *déclarés* (sinon le
    # serveur ne reconnaît pas le XML <tool_call> que le modèle émet quand même
    # et il fuit en texte) ; un appel d'outil demandé ici est ignoré.
    if remaining() < 15:
        return spoken(partial)
    history.append({"role": "user", "content": "Réponds maintenant à l'utilisateur en une réponse brève et concrète, sans nouvel appel d'outil."})
    try:
        turn = chat_turn(
            client, model_id, history, no_thinking(model, extra),
            max_tokens=clamp(remaining() * TOKENS_PER_S - 100, 128, 400),
            deadline=deadline - 3)
        if turn.content.strip():
            return spoken(turn.content)
        return spoken(partial, wanted_tools=bool(turn.calls))
    except Exception:
        return spoken(partial)


if __name__ == "__main__":
    langwatch.setup()
    WORKSPACES.mkdir(exist_ok=True)
    print(f"qwen38-bench : connecté ({os.environ.get('LANGWATCH_AGENT_ENVIRONMENT', 'development')}), Ctrl-C pour arrêter")
    langwatch.agent.serve()
