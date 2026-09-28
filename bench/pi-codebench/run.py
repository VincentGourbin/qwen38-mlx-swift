#!/usr/bin/env python3
"""Banc de création de code piloté par pi.dev.

Un run = une copie neuve de la fixture AgentKit, un découpage du plan en
fiches (mono, split2, split4…), une session pi par fiche (ou une seule
session poursuivie), puis la notation par les tests d'acceptation cachés
(`hidden/`). Tout est mesuré : durée, tours, appels d'outils, jetons (bloc
`usage` du serveur, repris par pi dans sa session), compactions, et l'état
réel du paquet après chaque fiche.

    ./run.py --plan split4                       # modèle chargé par qwen38 serve
    ./run.py --plan mono --reps 2
    ./run.py --provider ollama --model glm-5.3-flash:cloud --plan mono
    ./run.py --dry-check                         # la notation seule, sur la solution de référence

pi tourne avec un dossier de configuration isolé (PI_CODING_AGENT_DIR) :
la configuration de l'utilisateur (~/.pi/agent) n'est jamais modifiée.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixture"
TASKS = HERE / "tasks"
HIDDEN = HERE / "hidden"
RUNS = HERE / "runs"
RESULTS = HERE / "results.jsonl"
TASK_IDS = ["T1", "T2", "T3", "T4"]

PLANS = {
    "mono": [["T1", "T2", "T3", "T4"]],
    "split2": [["T1", "T2"], ["T3", "T4"]],
    "split4": [["T1"], ["T2"], ["T3"], ["T4"]],
    # Mêmes quatre fiches, mais une seule session pi poursuivie (--continue) :
    # isole l'effet « session neuve par fiche » de l'effet « fiche courte ».
    "split4-continue": [["T1"], ["T2"], ["T3"], ["T4"]],
}

AGENTS_MD = """# AgentKit

Paquet Swift sans dépendance (Foundation seul) : la boucle d'un petit agent
qui explore un dossier avec des outils (`list_files`, `read_file`, `grep`,
`final_answer`). Sources dans `Sources/AgentKit/`, tests (swift-testing) dans
`Tests/AgentKitTests/`.

Commandes (quelques secondes chacune) : `swift build`, `swift test`.

Règles :
- Travaille uniquement dans ce dossier. Ne modifie pas `Package.swift`.
- Une fiche est terminée quand `swift build` **et** `swift test` passent :
  lance-les toi-même avant de t'arrêter.
- Ne supprime ni ne désactive de test existant. Si un test existant contredit
  le nouveau comportement demandé par la fiche, mets-le à jour.
- Ne crée pas de commit git.
- Termine par un compte rendu court : fichiers modifiés, résultat de `swift test`.
"""

PROMPT = "Exécute la fiche {fiche} en suivant AGENTS.md. Quand elle est terminée, arrête-toi."


# ---------------------------------------------------------------- utilitaires

def sh(cmd: list[str], cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def now() -> float:
    return time.monotonic()


def health(base: str) -> dict:
    with urllib.request.urlopen(f"{base}/healthz", timeout=5) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------- énergie

def system_load_watts() -> float | None:
    """Puissance consommée par tout le Mac (`SystemLoad` de la télémétrie de
    la batterie, en mW, lisible sans sudo), ou `None` si indisponible."""
    out = subprocess.run(["ioreg", "-rn", "AppleSmartBattery"], capture_output=True, text=True).stdout
    m = re.search(r'"SystemLoad"=(\d+)', out)
    return int(m.group(1)) / 1000 if m else None


class PowerSampler:
    """Échantillonne la puissance du Mac toutes les `period` secondes dans un
    fil à part ; `stop()` rend l'énergie (Wh) et la puissance moyenne (W)."""

    def __init__(self, period: float = 2.0):
        import threading
        self.period, self.samples = period, []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            w = system_load_watts()
            if w is not None:
                self.samples.append((now(), w))
            self._stop.wait(self.period)

    def start(self) -> "PowerSampler":
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        if len(self.samples) < 2:
            return {}
        energy = sum((t1 - t0) * w0 for (t0, w0), (t1, _) in zip(self.samples, self.samples[1:])) / 3600
        span = self.samples[-1][0] - self.samples[0][0]
        return {"energy_wh": round(energy, 3), "mean_w": round(energy * 3600 / span, 1) if span else 0}


# ---------------------------------------------------------------- pi

def write_pi_config(agent_dir: Path, provider: str, model: str, base_url: str,
                    context_window: int, max_tokens: int) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    if provider == "qwen38-local":
        prov = {
            "api": "openai-completions",
            "apiKey": "local",
            "baseUrl": base_url + "/v1",
            "compat": {"maxTokensField": "max_tokens", "supportsDeveloperRole": False,
                       "supportsReasoningEffort": True, "thinkingFormat": "qwen"},
            "models": [{
                "id": model, "name": model, "input": ["text"], "reasoning": True,
                "contextWindow": context_window, "maxTokens": max_tokens,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                # Le gabarit Qwen n'accepte que low / medium / xhigh.
                "thinkingLevelMap": {"minimal": "low", "low": "low", "medium": "medium",
                                     "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"},
            }],
        }
    else:  # ollama : modèles cloud via le démon local, sans clé
        prov = {
            "api": "openai-completions", "apiKey": "ollama",
            "baseUrl": "http://127.0.0.1:11434/v1",
            "models": [{"id": model, "input": ["text"], "contextWindow": context_window,
                        "maxTokens": max_tokens}],
        }
    (agent_dir / "models.json").write_text(json.dumps({"providers": {provider: prov}}, indent=2))
    # Les clés de compaction doivent être imbriquées sous `compaction` (issue #1).
    (agent_dir / "settings.json").write_text(json.dumps({
        "defaultProvider": provider, "defaultModel": model,
        "compaction": {"reserveTokens": 16384, "keepRecentTokens": 12000},
        "quietStartup": True,
    }, indent=2))


def run_pi(ws: Path, agent_dir: Path, session_dir: Path, provider: str, model: str,
           thinking: str, fiche: str, cont: bool, timeout: int, log: Path) -> dict:
    cmd = ["pi", "-p", "--mode", "json", "--approve", "--no-extensions", "--no-skills",
           "--no-prompt-templates", "--offline", "--thinking", thinking,
           "--model", f"{provider}/{model}", "--session-dir", str(session_dir)]
    if cont:
        cmd.append("--continue")
    cmd.append(PROMPT.format(fiche=fiche))
    env = dict(os.environ, PI_CODING_AGENT_DIR=str(agent_dir))
    start = now()
    timed_out = False
    with open(log, "w") as out, open(log.with_suffix(".stderr"), "w") as err:
        proc = subprocess.Popen(cmd, cwd=ws, env=env, stdout=out, stderr=err,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGINT)
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                rc = proc.wait()
    return {"rc": rc, "timed_out": timed_out, "wall_s": round(now() - start, 1)}


def session_metrics(session_file: Path, since_entries: int = 0) -> tuple[dict, int]:
    """Métriques d'une session pi, à partir de l'entrée `since_entries`
    (pour une session poursuivie, ne compter que la fiche courante)."""
    entries = [json.loads(l) for l in session_file.read_text().splitlines() if l.strip()]
    m = {"turns": 0, "tool_calls": 0, "tool_errors": 0, "tools": {}, "test_runs": 0,
         "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "max_context": 0,
         "compactions": 0, "stop_reason": None, "error": None}
    for e in entries[since_entries:]:
        if e.get("type") == "compaction":
            m["compactions"] += 1
        if e.get("type") != "message":
            continue
        msg = e["message"]
        role = msg.get("role")
        if role == "assistant":
            m["turns"] += 1
            u = msg.get("usage") or {}
            m["input_tokens"] += u.get("input", 0)
            m["cached_tokens"] += u.get("cacheRead", 0)
            m["output_tokens"] += u.get("output", 0)
            m["max_context"] = max(m["max_context"], u.get("input", 0) + u.get("cacheRead", 0))
            m["stop_reason"] = msg.get("stopReason")
            if msg.get("errorMessage"):
                m["error"] = msg["errorMessage"][:300]
            for part in msg.get("content") or []:
                if part.get("type") == "toolCall":
                    m["tool_calls"] += 1
                    name = part.get("name", "?")
                    m["tools"][name] = m["tools"].get(name, 0) + 1
                    if name == "bash" and "swift test" in json.dumps(part.get("arguments", {})):
                        m["test_runs"] += 1
        elif role == "toolResult" and msg.get("isError"):
            m["tool_errors"] += 1
    return m, len(entries)


# ---------------------------------------------------------------- notation

TEST_LINE = re.compile(r'Test "(T\d) caché[^"]*" (passed|failed)')


def grade(ws: Path, task_ids: list[str]) -> dict:
    """État réel du paquet : build, suite de l'agent, puis chaque test caché
    ajouté seul (un test caché qui ne compile pas = tâche ratée, sans
    empêcher la notation des autres)."""
    g: dict = {}
    b = sh(["swift", "build"], ws)
    g["build"] = b.returncode == 0
    t = sh(["swift", "test"], ws)
    g["tests"] = t.returncode == 0
    m = re.search(r"Test run with (\d+) tests?", t.stdout)
    g["test_count"] = int(m.group(1)) if m else 0
    dest = ws / "Tests" / "AgentKitTests"
    for tid in task_ids:
        f = dest / f"Hidden{tid}.swift"
        shutil.copy(HIDDEN / f"Hidden{tid}.swift", f)
        try:
            r = sh(["swift", "test", "--filter", f"hidden{tid}"], ws)
            ok = [s for (k, s) in TEST_LINE.findall(r.stdout) if k == tid]
            g[tid] = r.returncode == 0 and ok == ["passed"]
        finally:
            f.unlink()
    return g


def diff_stats(ws: Path) -> dict:
    r = sh(["git", "diff", "--shortstat", "HEAD"], ws)
    nums = [int(x) for x in re.findall(r"(\d+) (?:files? changed|insertions?|deletions?)", r.stdout)]
    untracked = sh(["git", "ls-files", "--others", "--exclude-standard"], ws).stdout.split()
    return {"shortstat": r.stdout.strip(), "numbers": nums, "new_files": untracked}


# ---------------------------------------------------------------- run

def prepare_workspace(run_dir: Path, fiches: list[list[str]]) -> Path:
    ws = run_dir / "ws"
    shutil.copytree(FIXTURE, ws, ignore=shutil.ignore_patterns(".build", ".swiftpm"))
    (ws / "AGENTS.md").write_text(AGENTS_MD)
    (ws / ".gitignore").write_text(".build/\n.swiftpm/\n.pi/\n")
    plan = ws / "plan"
    plan.mkdir()
    for i, group in enumerate(fiches, 1):
        body = "\n\n".join((TASKS / f"{t}.md").read_text().strip() for t in group)
        intro = (f"# Fiche F{i}\n\nÀ réaliser dans l'ordre"
                 + (" ; chaque partie suppose les précédentes faites." if len(group) > 1 else ".")
                 + "\n\n")
        (plan / f"F{i}.md").write_text(intro + body + "\n")
    sh(["git", "init", "-q"], ws)
    sh(["git", "add", "-A"], ws)
    sh(["git", "-c", "user.name=bench", "-c", "user.email=bench@local", "commit", "-qm", "fixture"], ws)
    # Premier build hors chronomètre : l'agent ne paie pas le cache de SwiftPM.
    sh(["swift", "build", "--build-tests"], ws)
    return ws


def one_run(args, model: str, rep: int) -> dict:
    fiches = PLANS[args.plan]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    run_id = f"{stamp}-{args.plan}-{safe_model}-r{rep}"
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True)
    ws = prepare_workspace(run_dir, fiches)
    agent_dir = run_dir / "pi-agent"
    write_pi_config(agent_dir, args.provider, model, args.base_url, args.context_window, args.max_tokens)
    session_dir = run_dir / "sessions"
    session_dir.mkdir()

    result = {"run_id": run_id, "plan": args.plan, "provider": args.provider, "model": model,
              "thinking": args.thinking, "rep": rep, "label": args.label,
              "started": dt.datetime.now().isoformat(timespec="seconds"), "fiches": []}
    print(f"== {run_id}", flush=True)
    continued = args.plan.endswith("-continue")
    if args.measure_power:
        # Puissance au repos, modèle chargé et serveur inactif : ce que le Mac
        # consomme de toute façon, retranché pour le coût marginal.
        idle = PowerSampler().start()
        time.sleep(args.idle_seconds)
        result["idle"] = idle.stop()
        print(f"   repos : {result['idle'].get('mean_w', '?')} W", flush=True)
    done: list[str] = []
    seen_entries = 0
    for i, group in enumerate(fiches, 1):
        fiche = f"plan/F{i}.md"
        sampler = PowerSampler().start() if args.measure_power else None
        pi = run_pi(ws, agent_dir, session_dir, args.provider, model, args.thinking, fiche,
                    cont=continued and i > 1, timeout=args.fiche_timeout,
                    log=run_dir / f"F{i}.events.jsonl")
        if sampler:
            pi.update(sampler.stop())
        sessions = sorted(session_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        metrics = {}
        if sessions:
            since = seen_entries if continued else 0
            metrics, seen_entries = session_metrics(sessions[-1], since)
        done += group
        g = grade(ws, done)
        fr = {"fiche": i, "tasks": group, **pi, **metrics, "grade": g}
        result["fiches"].append(fr)
        print(f"   F{i} {'+'.join(group)} : {pi['wall_s']:.0f} s, {metrics.get('turns', 0)} tours, "
              f"{metrics.get('tool_calls', 0)} outils, {metrics.get('compactions', 0)} compactions, "
              f"build={g['build']} tests={g['tests']} "
              + " ".join(f"{t}={'ok' if g[t] else 'KO'}" for t in done)
              + (" TIMEOUT" if pi["timed_out"] else ""), flush=True)

    final = result["fiches"][-1]["grade"]
    result["tasks_passed"] = sum(1 for t in TASK_IDS if final.get(t))
    result["success"] = result["tasks_passed"] == len(TASK_IDS) and final["build"] and final["tests"]
    result["wall_s"] = round(sum(f["wall_s"] for f in result["fiches"]), 1)
    for k in ("turns", "tool_calls", "tool_errors", "input_tokens", "cached_tokens",
              "output_tokens", "compactions", "test_runs"):
        result[k] = sum(f.get(k, 0) for f in result["fiches"])
    if args.measure_power:
        result["energy_wh"] = round(sum(f.get("energy_wh", 0) for f in result["fiches"]), 2)
        result["mean_w"] = round(result["energy_wh"] * 3600 / result["wall_s"], 1)
    result["max_context"] = max(f.get("max_context", 0) for f in result["fiches"])
    result["diff"] = diff_stats(ws)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    with open(RESULTS, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"   => {result['tasks_passed']}/4 tâches, succès={result['success']}, "
          f"{result['wall_s']:.0f} s, {result['turns']} tours", flush=True)
    return result


def dry_check() -> None:
    """Vérifie la notation : fixture brute (0/4) et solution de référence (4/4)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for name, patch in (("fixture", None), ("référence", HERE / "reference" / "solution.patch")):
            ws = Path(tmp) / name
            shutil.copytree(FIXTURE, ws, ignore=shutil.ignore_patterns(".build", ".swiftpm"))
            if patch:
                r = subprocess.run(["patch", "-p1", "-s"], cwd=ws, stdin=open(patch),
                                   capture_output=True, text=True)
                assert r.returncode == 0, r.stdout + r.stderr
                test = ws / "Tests/AgentKitTests/AgentKitTests.swift"
                test.write_text(test.read_text().replace("tools.count == 4", "tools.count == 5"))
            print(name, grade(ws, TASK_IDS))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", choices=PLANS, default="split4")
    ap.add_argument("--provider", choices=["qwen38-local", "ollama"], default="qwen38-local")
    ap.add_argument("--model", help="identifiant du modèle ; par défaut celui chargé par qwen38 serve")
    ap.add_argument("--base-url", default="http://127.0.0.1:8848")
    ap.add_argument("--thinking", default="low")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--fiche-timeout", type=int, default=2700, help="secondes par fiche (défaut 45 min)")
    ap.add_argument("--context-window", type=int, default=65536)
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--label", default="")
    ap.add_argument("--measure-power", action="store_true",
                    help="mesurer l'énergie du Mac pendant chaque fiche (télémétrie ioreg)")
    ap.add_argument("--idle-seconds", type=int, default=60)
    ap.add_argument("--dry-check", action="store_true")
    args = ap.parse_args()

    if args.dry_check:
        dry_check()
        return
    model = args.model
    if args.provider == "qwen38-local":
        h = health(args.base_url)
        loaded = h.get("model")
        if not loaded:
            sys.exit("qwen38 serve ne rapporte aucun modèle chargé (voir /healthz)")
        model = model or loaded
    elif not model:
        sys.exit("--model est requis avec --provider ollama")
    for rep in range(1, args.reps + 1):
        one_run(args, model, rep)


if __name__ == "__main__":
    main()
