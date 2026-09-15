#!/usr/bin/env python3
"""P13.2 — mesure la capacité du modèle local à mener une tâche avec des outils.

Boucle d'agent minimale : trois outils réels en lecture seule, une tâche, et on
compte ce qui compte — appels bien formés, appels valides (outil connu,
paramètres requis présents), et tâche aboutie ou non.

    python3 Scripts/agent-loop.py --task 1 --max-steps 12
"""
import argparse, json, os, subprocess, sys, time, urllib.request

TOOLS = [
 {"type":"function","function":{"name":"list_files","description":"Liste les fichiers d'un dossier du dépôt.",
  "parameters":{"type":"object","properties":{"path":{"type":"string","description":"chemin relatif au dépôt"}},"required":["path"]}}},
 {"type":"function","function":{"name":"read_file","description":"Lit un fichier du dépôt. Renvoie au plus 200 lignes.",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},"start_line":{"type":"integer"}},"required":["path"]}}},
 {"type":"function","function":{"name":"grep","description":"Cherche un motif dans le dépôt et renvoie les lignes correspondantes.",
  "parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"}},"required":["pattern"]}}},
 {"type":"function","function":{"name":"final_answer","description":"Donne la réponse finale à l'utilisateur et termine.",
  "parameters":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}}},
]
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def safe(p):
    q = os.path.abspath(os.path.join(ROOT, p or "."))
    if not q.startswith(ROOT): raise ValueError("chemin hors dépôt")
    return q

def run_tool(name, args):
    try:
        if name == "list_files":
            d = safe(args.get("path","."))
            return "\n".join(sorted(os.listdir(d))[:80]) or "(vide)"
        if name == "read_file":
            f = safe(args["path"]); s = int(args.get("start_line", 1))
            lines = open(f, errors="replace").read().split("\n")
            return "\n".join(f"{i+1}\t{l}" for i,l in enumerate(lines[s-1:s+199], start=s-1))[:6000]
        if name == "grep":
            pat = args["pattern"]; d = safe(args.get("path","."))
            r = subprocess.run(["grep","-rn","--include=*.swift","--include=*.md",pat,d],
                               capture_output=True, text=True, timeout=30)
            out = r.stdout or "(aucune correspondance)"
            return "\n".join(out.split("\n")[:60])[:6000]
        return f"outil inconnu : {name}"
    except Exception as e:
        return f"ERREUR : {e!r}"

TASKS = {
 1: ("Dans ce dépôt Swift, quel est le nom exact du type Swift qui implémente le "
     "cache clé/valeur de l'attention pleine pour Flash-Next ? Réponds par le nom du type."),
 2: ("Combien de niveaux de fusion l'énumération Qwen4ExpFusionLevel définit-elle, "
     "et quel est le niveau utilisé par défaut en production ?"),
 3: ("Quelle option en ligne de commande permet de régler le nombre d'experts routés, "
     "et quelle est sa valeur par défaut ?"),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8848")
    ap.add_argument("--task", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=400)
    a = ap.parse_args()
    model = json.load(urllib.request.urlopen(a.base+"/healthz"))["model"]
    task = TASKS[a.task]
    print(f"modèle {model}\ntâche  {task}\n" + "-"*70)
    msgs = [{"role":"system","content":
             "Tu es un assistant de code. Tu explores le dépôt avec les outils fournis, "
             "puis tu appelles final_answer avec ta réponse. Ne devine jamais : vérifie dans le code."},
            {"role":"user","content":task}]
    stats = {"tours":0,"appels":0,"bien_formes":0,"valides":0,"texte_sans_appel":0}
    t0=time.time()
    for step in range(a.max_steps):
        body={"model":model,"messages":msgs,"tools":TOOLS,"temperature":0,
              "max_tokens":a.max_tokens,"enable_thinking":False,"mtp":False}
        req=urllib.request.Request(a.base+"/v1/chat/completions",data=json.dumps(body).encode(),
                                   headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=1800) as r: d=json.loads(r.read())
        stats["tours"]+=1
        m=d["choices"][0]["message"]; fr=d["choices"][0].get("finish_reason")
        tcs=m.get("tool_calls") or []
        if not tcs:
            stats["texte_sans_appel"]+=1
            print(f"[{step}] pas d'appel · finish={fr} · {(m.get('content') or '')[:150]}")
            if fr=="stop": break
            msgs.append({"role":"assistant","content":m.get("content") or ""})
            continue
        msgs.append({"role":"assistant","content":m.get("content") or "","tool_calls":tcs})
        for tc in tcs:
            stats["appels"]+=1; stats["bien_formes"]+=1
            fn=tc["function"]["name"]
            try: args=json.loads(tc["function"]["arguments"])
            except Exception: args={}
            spec=next((t for t in TOOLS if t["function"]["name"]==fn), None)
            req_ok = spec is not None and all(k in args for k in spec["function"]["parameters"].get("required",[]))
            if req_ok: stats["valides"]+=1
            print(f"[{step}] {fn}({json.dumps(args,ensure_ascii=False)[:110]}) {'' if req_ok else '  ⚠ invalide'}")
            if fn=="final_answer":
                print("-"*70); print("RÉPONSE FINALE :", args.get("answer","")[:400])
                stats["fini"]=True
                dt=time.time()-t0
                print(f"\n{stats}\n  durée {dt:.1f} s")
                return
            msgs.append({"role":"tool","tool_call_id":tc["id"],"content":run_tool(fn,args)[:4000]})
    print("-"*70); print("PAS DE RÉPONSE FINALE dans le budget de pas")
    print(f"\n{stats}\n  durée {time.time()-t0:.1f} s")

if __name__=="__main__": main()
