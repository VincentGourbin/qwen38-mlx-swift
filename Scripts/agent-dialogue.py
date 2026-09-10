#!/usr/bin/env python3
"""Deux agents (deux conversations distinctes sur le même serveur Flash-Next)
dialoguent : A dit « Hello », B répond, A répond à B, etc. Chaque agent voit
les messages de l'autre comme des messages utilisateur. Journal lisible dans
results/dialogue/dialogue.log (à suivre avec `tail -f`), données par tour dans
dialogue.jsonl. S'arrête sur erreur HTTP (limite de contexte, mémoire), sur
--max-turns ou --max-minutes.

    python3 Scripts/agent-dialogue.py --base http://127.0.0.1:8848 --max-minutes 180

Note : par défaut, chaque agent a son propre `conversation_id` et profite du
LRU de conversations du serveur (P5.2) — TTFT plat. Avec --no-conversation-id
(P6.1), aucun agent n'envoie de conversation_id : chaque tour renvoie tout
l'historique, comme le ferait Open WebUI ou le SDK openai ordinaire ; le
serveur doit alors reconnaître la continuation via son cache de préfixe
implicite (comparaison sur les IDs de tokens rendus) pour rester au même TTFT
plat au lieu de rejouer l'historique complet à chaque tour.
"""
import argparse, json, os, sys, time, urllib.request, urllib.error
from datetime import datetime

AGENTS = {
    "A": ("Aria", "Tu es Aria, une exploratrice curieuse qui voyage à travers le monde. Tu discutes avec un ami, "
                  "de manière naturelle, en français, en 2 à 4 phrases par message. Tu poses souvent une question "
                  "pour relancer la conversation et tu rebondis sur ce que dit ton interlocuteur."),
    "B": ("Bastien", "Tu es Bastien, un historien un peu pince-sans-rire, passionné de sciences. Tu discutes avec une amie, "
                     "en français, en 2 à 4 phrases par message. Tu réponds à ses questions avec des faits précis et tu "
                     "relances avec une anecdote ou une question."),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8848")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-turns", type=int, default=10_000)
    ap.add_argument("--max-minutes", type=float, default=180)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default="results/dialogue")
    ap.add_argument("--no-conversation-id", action="store_true",
                     help="P6.1 : n'envoie jamais conversation_id — chaque agent renvoie tout son "
                          "historique à chaque tour, comme Open WebUI ou le SDK openai ordinaire. "
                          "Sert à valider le cache de préfixe implicite du serveur.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "dialogue.log"), "a", encoding="utf-8")
    jl = open(os.path.join(args.out, "dialogue.jsonl"), "a", encoding="utf-8")

    def say(text):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {text}"
        print(line, flush=True); log.write(line + "\n"); log.flush()

    def get(path):
        with urllib.request.urlopen(args.base + path, timeout=30) as r:
            return json.loads(r.read())

    model = args.model or next(m["id"] for m in get("/v1/models")["data"] if m.get("loaded"))
    say(f"=== dialogue A/B sur {args.base} · modèle {model} · max_tokens {args.max_tokens} · T={args.temperature}")

    # historiques vus par chaque agent (rôles depuis son point de vue)
    hist = {"A": [{"role": "system", "content": AGENTS["A"][1]}],
            "B": [{"role": "system", "content": AGENTS["B"][1]}]}
    # A ouvre : on lui fait dire « Hello » sans génération, pour que B réponde
    first = "Hello"
    hist["A"].append({"role": "assistant", "content": first})
    hist["B"].append({"role": "user", "content": first})
    say(f"A (Aria) : {first}")
    speaker, last_text = "B", first
    started = time.time(); turn = 0; total_generated = 0
    while turn < args.max_turns and (time.time() - started) < args.max_minutes * 60:
        turn += 1
        body = {"model": model, "temperature": args.temperature, "top_p": 0.8, "max_tokens": args.max_tokens,
                "enable_thinking": False, "mtp": False, "messages": hist[speaker]}
        if not args.no_conversation_id:
            body["conversation_id"] = f"dialogue-{speaker}"
        req = urllib.request.Request(args.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            say(f"!!! HTTP {e.code} au tour {turn} ({speaker}) : {e.read()[:300]!r} — arrêt")
            break
        except Exception as e:
            say(f"!!! erreur au tour {turn} ({speaker}) : {e!r} — arrêt")
            break
        dt = time.time() - t0
        text = (resp["choices"][0]["message"].get("content") or "").strip()
        finish = resp["choices"][0].get("finish_reason")
        # métriques serveur de la dernière session
        try:
            sessions = get("/metrics").get("sessions", [])
            s = sessions[-1] if sessions else {}
        except Exception:
            s = {}
        prompt_tokens = s.get("promptTokens"); gen = s.get("generatedTokens"); ttft = s.get("timeToFirstToken"); tps = s.get("tokensPerSecond")
        total_generated += gen or 0
        name = AGENTS[speaker][0]
        say(f"{speaker} ({name}) [tour {turn} · prompt {prompt_tokens} tok · {gen} tok · TTFT {ttft if ttft is None else round(ttft,2)} s · "
            f"{tps if tps is None else round(tps,1)} tok/s · {dt:.1f} s · finish {finish} · cumul {total_generated} tok, {(time.time()-started)/60:.1f} min] : {text}")
        jl.write(json.dumps({"turn": turn, "speaker": speaker, "prompt_tokens": prompt_tokens, "generated": gen, "ttft": ttft,
                             "tok_s": tps, "wall_s": round(dt, 2), "finish": finish, "elapsed_s": round(time.time() - started, 1),
                             "text": text}, ensure_ascii=False) + "\n"); jl.flush()
        if not text:
            say("!!! réponse vide — arrêt"); break
        hist[speaker].append({"role": "assistant", "content": text})
        other = "A" if speaker == "B" else "B"
        hist[other].append({"role": "user", "content": text})
        speaker, last_text = other, text
    say(f"=== fin : {turn} tours, {total_generated} tokens générés, {(time.time()-started)/60:.1f} min")

if __name__ == "__main__":
    main()
