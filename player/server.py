"""Servidor local do player (um so para todos os videos).

  python player/server.py            # abre http://127.0.0.1:8765 no navegador
  python player/server.py --port 9000 --no-browser

No navegador voce escolhe um video e uma rede (as opcoes de player/catalog.py). O que ja foi processado (player/prepare.py
ou uma escolha anterior) abre na hora; o que ainda nao foi, o servidor processa com o pipeline e mostra o progresso.
So escuta em 127.0.0.1 (esta maquina) e so processa o que esta no catalogo: nao le pastas arbitrarias.
"""
import argparse
import importlib
import json
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402
import catalog  # noqa: E402
sys.path.insert(0, str(catalog.ROOT))
from celegans_seg import load_config  # noqa: E402

JOBS = {}  # nome do cache -> {state, done, total, error}
LOCK = threading.Lock()


def stage_labels():
    """Nomes das etapas lidos de build.py AGORA (o arquivo e recarregado a cada chamada), e nao do data.json do cache:
    assim mudar o texto em build.stage_defs aparece no player com um F5, sem reprocessar nem reiniciar o servidor."""
    try:
        importlib.reload(build)
        cfg = load_config(catalog.ROOT / "config.yaml")
        return [dict(id=s[0], label=s[1], ext=s[2]) for s in build.stage_defs(cfg)]
    except Exception:  # erro de sintaxe em build.py: o player usa os nomes que ja estao no cache
        return []


def info():
    vids, mods = catalog.videos(), catalog.models()
    ready = {}
    for v in vids:
        for m in mods:
            name = catalog.cache_name(v["images"], m["path"])
            if catalog.is_ready(name):
                ready[f"{v['id']}|{m['id']}"] = name
    return dict(videos=[{k: v[k] for k in ("id", "label", "frames")} for v in vids],
                models=[{k: m[k] for k in ("id", "label")} for m in mods], ready=ready, stages=stage_labels())


def start_job(video_id: str, model_id: str, max_frames=None) -> dict:
    v = next((x for x in catalog.videos() if x["id"] == video_id), None)
    m = next((x for x in catalog.models() if x["id"] == model_id), None)
    if not v or not m:
        return dict(error="opcao desconhecida")
    name = catalog.cache_name(v["images"], m["path"], max_frames)
    out = catalog.CACHE / name
    with LOCK:
        if JOBS.get(name, {}).get("state") == "running":
            return dict(name=name)
        if catalog.is_ready(name):
            JOBS[name] = dict(state="done", done=1, total=1)
            return dict(name=name)
        JOBS[name] = dict(state="running", done=0, total=1)

    def run():
        try:
            build.build(v["images"], out, str(catalog.ROOT / m["path"]) if m["path"] else "", max_frames,
                        lambda d, t: JOBS[name].update(done=d, total=t))
            JOBS[name]["state"] = "done"
        except Exception as e:  # mostra o erro no navegador
            JOBS[name].update(state="error", error=f"{type(e).__name__}: {e}")

    threading.Thread(target=run, daemon=True).start()
    return dict(name=name)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE), **k)

    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/info":
            return self.send_json(info())
        if u.path == "/api/job":
            return self.send_json(JOBS.get(parse_qs(u.query).get("name", [""])[0], dict(state="unknown")))
        if u.path.startswith("/cache/") and ".." in u.path:
            return self.send_error(HTTPStatus.FORBIDDEN)
        return super().do_GET()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if urlparse(self.path).path == "/api/process":
            return self.send_json(start_job(body.get("video", ""), body.get("model", "")))
        self.send_error(HTTPStatus.NOT_FOUND)

    def end_headers(self):
        if self.path.startswith("/cache/"):
            self.send_header("Cache-Control", "max-age=3600")
        super().end_headers()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    catalog.CACHE.mkdir(exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print("Player em", url, "(Ctrl+C para encerrar)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
