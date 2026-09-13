#!/usr/bin/env python3
"""P12.3 — banc multi-clients : N requêtes simultanées sur le serveur.

Mesure ce qu'un utilisateur ressent (latence de bout en bout par client) et
ce que la machine produit (débit agrégé). À lancer contre un serveur déjà
chargé, une fois par valeur de --batch-size, pour comparer.

    python3 Scripts/bench-concurrent.py --clients 4 --max-tokens 96

Les prompts sont de longueurs VOISINES par défaut : P12.2 a mesuré qu'un lot
de longueurs mêlées coûte +24 % par pas, donc mélanger les tailles mesurerait
l'ordonnanceur et la pénalité de remplissage en même temps. --mixed force le
cas hétérogène, à mesurer séparément.
"""
import argparse, json, statistics as st, sys, threading, time, urllib.request

HOMOGENES = [
    "Explique en français le rôle du président de la République française.",
    "Explique en français le rôle du chancelier de la République fédérale.",
    "Explique en français le rôle du premier ministre du Royaume-Uni.",
    "Explique en français le rôle du président du Conseil italien.",
    "Explique en français le rôle du chef du gouvernement espagnol.",
    "Explique en français le rôle du premier ministre canadien.",
    "Explique en français le rôle du président de la Confédération suisse.",
    "Explique en français le rôle du premier ministre belge.",
]
MELANGES = [
    "Bonjour",
    "Explique en français qui est le président de la Chine et quel est son rôle.",
    "Rédige un essai détaillé de dix paragraphes sur l'histoire de l'astronomie moderne, "
    "de Copernic à nos jours, en insistant sur les instruments et les controverses.",
    "Quelle heure est-il ?",
    "Décris en français le cycle de l'eau.",
    "Résume en une phrase la théorie de la relativité restreinte.",
    "Écris une fonction Swift qui inverse une chaîne de caractères.",
    "Traduis en anglais : le chat dort sur le canapé.",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8848")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--mixed", action="store_true", help="prompts de longueurs hétérogènes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    def get(p):
        with urllib.request.urlopen(args.base + p, timeout=30) as r:
            return json.loads(r.read())

    health = get("/healthz")
    model = health["model"]
    print(f"modèle {model} · batch-size annoncé par le serveur : {health.get('batch_size_configured', '?')}")

    prompts = (MELANGES if args.mixed else HOMOGENES)[: args.clients]
    if len(prompts) < args.clients:
        prompts = (prompts * ((args.clients // len(prompts)) + 1))[: args.clients]

    results = [None] * args.clients
    barrier = threading.Barrier(args.clients)

    def worker(i):
        body = {"model": model, "messages": [{"role": "user", "content": prompts[i]}],
                "temperature": 0, "max_tokens": args.max_tokens,
                "enable_thinking": False, "mtp": False, "stream": False}
        req = urllib.request.Request(args.base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        barrier.wait()                      # départ simultané, c'est tout l'enjeu
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                resp = json.loads(r.read())
            dt = time.time() - t0
            # La réponse ne porte pas de champ `usage` : on compte les jetons
            # côté serveur, via /metrics, après coup (constaté le 2026-09-13).
            results[i] = {"ok": True, "seconds": dt, "tokens": None,
                          "finish": resp["choices"][0].get("finish_reason")}
        except Exception as e:
            results[i] = {"ok": False, "seconds": time.time() - t0, "error": repr(e)}

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.clients)]
    wall0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - wall0

    ok = [r for r in results if r and r["ok"]]
    if not ok:
        print("aucune requête aboutie :", results[:2]); sys.exit(1)
    sessions = []
    try:
        sessions = [s for s in get("/metrics").get("sessions", []) if isinstance(s, dict)][-args.clients:]
    except Exception as e:
        print("  (métriques indisponibles :", repr(e), ")")
    toks = sum(s.get("generatedTokens") or 0 for s in sessions)
    lat = sorted(r["seconds"] for r in ok)
    print(f"\n{len(ok)}/{args.clients} requêtes abouties · prompts "
          f"{'hétérogènes' if args.mixed else 'de longueurs voisines'}")
    print(f"  temps total (mur)        : {wall:.2f} s")
    print(f"  jetons produits          : {toks}")
    print(f"  DÉBIT AGRÉGÉ             : {toks/wall:.2f} jetons/s")
    print(f"  latence par client       : médiane {st.median(lat):.2f} s · min {lat[0]:.2f} · max {lat[-1]:.2f}")
    bs = {s.get("batchSizeServed") for s in sessions}
    if bs and bs != {None}:
        print(f"  taille de lot vue par le serveur : {sorted(x for x in bs if x is not None)}")
    if args.out:
        json.dump({"clients": args.clients, "mixed": args.mixed, "wall": wall,
                   "tokens": toks, "aggregate_tok_s": toks/wall,
                   "latencies": lat}, open(args.out, "w"), indent=1)
        print(f"  → {args.out}")

if __name__ == "__main__":
    main()
