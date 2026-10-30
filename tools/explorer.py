"""Open tools/explorer.html with live access to results/.

Browsers do not let a page opened from disk read other files, so this serves the
repository on localhost (only this computer can reach it) and opens the page.
The page reads the result CSVs directly; nothing is copied or embedded.

    python tools/explorer.py                 # free port, opens the browser
    python tools/explorer.py --port 8765 --no-browser

Stop with Ctrl-C.
"""
import argparse
import functools
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = "tools/explorer.html"


class NoCacheHandler(SimpleHTTPRequestHandler):
    """Always serve the current files, so Reload in the page picks up a new run."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=0, help="port to use (default: any free port)")
    parser.add_argument("--no-browser", action="store_true", help="print the URL without opening a browser")
    args = parser.parse_args()

    handler = functools.partial(NoCacheHandler, directory=str(ROOT))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/{PAGE}"
        print(f"Serving {ROOT}\nOpen {url}\nStop with Ctrl-C.", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
