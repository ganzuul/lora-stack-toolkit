from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


CONTROLLER_PATH = (
    Path(__file__).resolve().parents[1]
    / ".pi"
    / "agent"
    / "skills"
    / "lora-router"
    / "scripts"
    / "lora_ctl.py"
)
spec = importlib.util.spec_from_file_location("lora_ctl", CONTROLLER_PATH)
assert spec is not None and spec.loader is not None
lora_ctl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lora_ctl)


class AdapterServer(ThreadingHTTPServer):
    adapters: list[dict[str, Any]]


class AdapterHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = json.dumps(self.server.adapters).encode()  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        scales = {int(entry["id"]): float(entry["scale"]) for entry in payload}
        for adapter in self.server.adapters:  # type: ignore[attr-defined]
            adapter["scale"] = scales.get(int(adapter["id"]), 0.0)
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: Any) -> None:
        pass


class LoraControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = AdapterServer(("127.0.0.1", 0), AdapterHandler)
        self.server.adapters = [
            {
                "id": 0,
                "path": "/models/Qwen3.8-27B-core-pi-qwopus-maximin-r32-Q8_0.gguf",
                "scale": 1.0,
            },
            {
                "id": 1,
                "path": "/models/Bonsai-2-27B-blender-c7-gn-sft-Q8_0.gguf",
                "scale": 0.0,
            },
            {
                "id": 2,
                "path": "/models/Bonsai-2-27B-blender-c5-unified-sft-v2-Q8_0.gguf",
                "scale": 0.0,
            },
        ]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def run_controller(self, *args: str) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return lora_ctl.main(["--direct", self.base_url, *args])

    def scales(self) -> list[float]:
        return [float(adapter["scale"]) for adapter in self.server.adapters]

    def test_default_load_and_switch_scales(self) -> None:
        self.assertEqual(self.run_controller("load", "geonodes"), 0)
        self.assertEqual(self.scales(), [0.8, 0.35, 0.0])
        self.assertEqual(self.run_controller("load", "blenderworld"), 0)
        self.assertEqual(self.scales(), [0.8, 0.0, 0.35])

    def test_last_specialist_unload_restores_core_baseline(self) -> None:
        self.run_controller("load", "geonodes")
        self.assertEqual(self.run_controller("unload", "geonodes"), 0)
        self.assertEqual(self.scales(), [1.0, 0.0, 0.0])

    def test_custom_scales_and_global_unload(self) -> None:
        self.assertEqual(
            self.run_controller("load", "geonodes", "--core-scale", "0.75", "--scale", "0.5"),
            0,
        )
        self.assertEqual(self.scales(), [0.75, 0.5, 0.0])
        self.assertEqual(self.run_controller("unload"), 0)
        self.assertEqual(self.scales(), [1.0, 0.0, 0.0])

    def test_raw_payload_cannot_omit_preloaded_adapters(self) -> None:
        self.assertEqual(
            self.run_controller("raw", "--json", '[{"id":0,"scale":1.0}]'),
            1,
        )
        self.assertEqual(self.scales(), [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
