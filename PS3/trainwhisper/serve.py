"""Local preview server for the web app.

Same as `python -m http.server`, but tells the browser not to cache, so an
edited JS module is never mixed with a stale copy of another one.

    python serve.py            # http://localhost:8000
    python serve.py 8123       # another port
"""
import functools
import http.server
import pathlib
import sys

PUBLIC = pathlib.Path(__file__).resolve().parent / "public"


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    handler = functools.partial(NoCacheHandler, directory=str(PUBLIC))
    with http.server.ThreadingHTTPServer(("", port), handler) as httpd:
        print(f"Serving {PUBLIC} at http://localhost:{port}  (Ctrl+C to stop)")
        httpd.serve_forever()
