#!/usr/bin/env python3
"""Read-Modify-Write LoRA hot-swap controller for llama-server / llama-swap.

Always sends the full list of loaded adapter IDs on POST /lora-adapters so
toggling a specialist never zeroes out the always-on core adapters (pi-lora,
qwopus-r32), and checks /running before hitting llama-swap /upstream/<model>/
so an unrequested model is never auto-loaded by accident.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_SWAP_URL = os.environ.get("LLAMA_SWAP_URL", "http://127.0.0.1:8080").rstrip("/")
DEFAULT_MODEL = os.environ.get("PI_MODEL") or "qwen27b"
DEFAULT_CORE_PATTERNS = tuple(
    p.strip().lower()
    for p in os.environ.get("LORA_CORE_PATTERNS", "pi-lora,qwopus,thinkingcap").split(",")
    if p.strip()
)

BUILTIN_ALIASES: dict[str, tuple[str, ...]] = {
    "geonodes": ("c4-gn", "geonodes", "gn-rl", "gn-sft"),
    "blenderworld": ("sft-v2", "blenderworld", "strata", "currc"),
    "blender": ("blender",),
}


def _http_json(
    url: str,
    method: str = "GET",
    body: Any | None = None,
    timeout: float = 15.0,
) -> tuple[int, float, Any]:
    """Perform an HTTP JSON request and return (status_code, elapsed_ms, parsed_body)."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            raw = resp.read().decode("utf-8", errors="replace").strip()
            parsed = json.loads(raw) if raw else None
            return int(resp.status), dt_ms, parsed
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"HTTP {exc.code} on {method} {url}: {err_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed on {method} {url}: {exc.reason}") from exc


def _resolve_endpoint(args: argparse.Namespace) -> str:
    """Resolve base /lora-adapters URL and verify /running state when using llama-swap."""
    if args.direct:
        base = args.direct.rstrip("/")
        if base.endswith("/lora-adapters"):
            return base
        return f"{base}/lora-adapters"

    swap_url = args.swap_url.rstrip("/")
    model = args.model
    if not args.autoload:
        try:
            _, _, running_data = _http_json(f"{swap_url}/running", method="GET", timeout=5.0)
        except RuntimeError as exc:
            print(f"[lora_ctl] ERROR checking {swap_url}/running: {exc}", file=sys.stderr)
            sys.exit(2)
        running_models = [
            str(item.get("model", ""))
            for item in (running_data or {}).get("running", [])
            if isinstance(item, dict)
        ]
        if model not in running_models:
            shown = ", ".join(running_models) if running_models else "(none)"
            print(
                f"[lora_ctl] REFUSING to touch '{model}': not currently running in llama-swap "
                f"(running: {shown}). Pass --autoload to force swap-in.",
                file=sys.stderr,
            )
            sys.exit(2)

    return f"{swap_url}/upstream/{model}/lora-adapters"


def is_core_adapter(path: str, core_patterns: tuple[str, ...]) -> bool:
    """Return True when adapter filename matches an always-on core pattern."""
    name = Path(path).name.lower()
    return any(pat in name for pat in core_patterns)


def _core_preserved_scale(adapter: dict[str, Any]) -> float:
    cur = float(adapter.get("scale", 0.0))
    return cur if cur > 0.0 else 1.0


def _format_adapter_table(adapters: list[dict[str, Any]], core_patterns: tuple[str, ...]) -> str:
    if not adapters:
        return "(no LoRA adapters pre-loaded on server)"
    lines = ["ID  ROLE        SCALE  FILE"]
    for a in adapters:
        aid = int(a["id"])
        path = str(a.get("path", ""))
        role = "core" if is_core_adapter(path, core_patterns) else "specialist"
        scale = float(a.get("scale", 0.0))
        lines.append(f"{aid:<3d} {role:<11s} {scale:<5.2f}  {Path(path).name} ({path})")
    return "\n".join(lines)


def _resolve_selector(
    adapters: list[dict[str, Any]],
    selector: str,
    core_patterns: tuple[str, ...],
) -> dict[str, Any]:
    """Resolve an ID, built-in alias, or filename substring to exactly one adapter."""
    sel_clean = selector.strip()
    if sel_clean.isdigit():
        target_id = int(sel_clean)
        for a in adapters:
            if int(a["id"]) == target_id:
                return a
        print(
            f"[lora_ctl] No adapter with id={target_id}.\n{_format_adapter_table(adapters, core_patterns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    sel_low = sel_clean.lower()
    patterns = BUILTIN_ALIASES.get(sel_low, (sel_low,))
    matches: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for a in adapters:
        aid = int(a["id"])
        path = str(a.get("path", ""))
        bname = Path(path).name.lower()
        full = path.lower()
        if any(pat in bname or pat in full for pat in patterns):
            if aid not in seen_ids:
                matches.append(a)
                seen_ids.add(aid)

    if len(matches) == 1:
        return matches[0]
    if not matches:
        print(
            f"[lora_ctl] Selector '{selector}' matched 0 adapters.\n{_format_adapter_table(adapters, core_patterns)}",
            file=sys.stderr,
        )
    else:
        names = ", ".join(f"id={m['id']}:{Path(m['path']).name}" for m in matches)
        print(
            f"[lora_ctl] Selector '{selector}' is ambiguous ({len(matches)} matches: {names}). "
            f"Pass numeric ID or a tighter substring.\n{_format_adapter_table(adapters, core_patterns)}",
            file=sys.stderr,
        )
    sys.exit(1)


def _apply_and_verify(
    endpoint: str,
    payload: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    _, dt_ms, _ = _http_json(endpoint, method="POST", body=payload)
    _, _, verified = _http_json(endpoint, method="GET")
    if not isinstance(verified, list):
        raise RuntimeError(f"Expected list from GET {endpoint}, got {type(verified).__name__}")
    verified_by_id = {
        int(adapter["id"]): float(adapter.get("scale", 0.0))
        for adapter in verified
        if "id" in adapter
    }
    for entry in payload:
        aid = int(entry["id"])
        requested_scale = float(entry["scale"])
        if aid not in verified_by_id:
            raise RuntimeError(f"POST verification failed: adapter id={aid} is missing after write")
        if abs(verified_by_id[aid] - requested_scale) > 1e-6:
            raise RuntimeError(
                f"POST verification failed: adapter id={aid} scale is "
                f"{verified_by_id[aid]:g}, expected {requested_scale:g}"
            )
    return dt_ms, verified


def _emit_result(
    action: str,
    adapters: list[dict[str, Any]],
    core_patterns: tuple[str, ...],
    as_json: bool,
    latency_ms: float | None = None,
) -> None:
    if as_json:
        out: dict[str, Any] = {
            "action": action,
            "adapters": [
                {
                    "id": int(a["id"]),
                    "role": "core" if is_core_adapter(str(a.get("path", "")), core_patterns) else "specialist",
                    "scale": float(a.get("scale", 0.0)),
                    "path": str(a.get("path", "")),
                }
                for a in adapters
            ],
        }
        if latency_ms is not None:
            out["post_ms"] = round(latency_ms, 2)
        print(json.dumps(out, indent=2))
        return

    if latency_ms is not None:
        print(f"[lora_ctl] {action} applied & verified in {latency_ms:.2f} ms")
    print(_format_adapter_table(adapters, core_patterns))


def cmd_list(args: argparse.Namespace, endpoint: str, core_patterns: tuple[str, ...]) -> int:
    _, dt_ms, adapters = _http_json(endpoint, method="GET")
    if not isinstance(adapters, list):
        raise RuntimeError(f"Unexpected /lora-adapters response: {adapters!r}")
    _emit_result("list", adapters, core_patterns, args.json, latency_ms=None)
    return 0


def cmd_load(args: argparse.Namespace, endpoint: str, core_patterns: tuple[str, ...]) -> int:
    if args.scale is not None and float(args.scale) <= 0.0:
        print("[lora_ctl] --scale must be > 0.0 (use 'unload' to set scale=0.0).", file=sys.stderr)
        return 1
    if float(args.core_scale) <= 0.0:
        print("[lora_ctl] --core-scale must be > 0.0.", file=sys.stderr)
        return 1

    _, _, adapters = _http_json(endpoint, method="GET")
    if not isinstance(adapters, list) or not adapters:
        print("[lora_ctl] No adapters pre-loaded on server; cannot load.", file=sys.stderr)
        return 1

    target = _resolve_selector(adapters, args.selector, core_patterns)
    target_id = int(target["id"])
    target_is_core = is_core_adapter(str(target.get("path", "")), core_patterns)
    if target_is_core and not args.force_core:
        print(
            f"[lora_ctl] '{Path(target['path']).name}' (id={target_id}) is a core adapter. "
            f"Pass --force-core if you intentionally want to modify its scale.",
            file=sys.stderr,
        )
        return 1
    target_scale = float(args.scale) if args.scale is not None else (1.0 if target_is_core else 0.35)

    payload: list[dict[str, Any]] = []
    for a in adapters:
        aid = int(a["id"])
        path = str(a.get("path", ""))
        cur_scale = float(a.get("scale", 0.0))
        if aid == target_id:
            new_scale = target_scale
        elif is_core_adapter(path, core_patterns):
            new_scale = (
                float(args.core_scale)
                if not is_core_adapter(str(target.get("path", "")), core_patterns)
                else _core_preserved_scale(a)
            )
        else:
            new_scale = cur_scale if args.stack else 0.0
        payload.append({"id": aid, "scale": new_scale})

    dt_ms, verified = _apply_and_verify(endpoint, payload)
    _emit_result(f"load({Path(target['path']).name}:{target_scale:g})", verified, core_patterns, args.json, dt_ms)
    return 0


def cmd_unload(args: argparse.Namespace, endpoint: str, core_patterns: tuple[str, ...]) -> int:
    _, _, adapters = _http_json(endpoint, method="GET")
    if not isinstance(adapters, list):
        raise RuntimeError(f"Unexpected /lora-adapters response: {adapters!r}")
    if not adapters:
        _emit_result("unload(empty)", [], core_patterns, args.json)
        return 0

    payload: list[dict[str, Any]] = []
    if args.selector:
        target = _resolve_selector(adapters, args.selector, core_patterns)
        target_id = int(target["id"])
        target_is_core = is_core_adapter(str(target.get("path", "")), core_patterns)
        if target_is_core and not args.force_core:
            print(
                f"[lora_ctl] Refusing to unload core adapter '{Path(target['path']).name}' (id={target_id}). "
                f"Pass --force-core to override.",
                file=sys.stderr,
            )
            return 1
        for a in adapters:
            aid = int(a["id"])
            path = str(a.get("path", ""))
            if aid == target_id:
                new_scale = 0.0
            elif is_core_adapter(path, core_patterns):
                new_scale = _core_preserved_scale(a)
            else:
                new_scale = float(a.get("scale", 0.0))
            payload.append({"id": aid, "scale": new_scale})
        if not target_is_core and not any(
            not is_core_adapter(str(a.get("path", "")), core_patterns)
            and float(a.get("scale", 0.0)) > 0.0
            and int(a["id"]) != target_id
            for a in adapters
        ):
            for entry, adapter in zip(payload, adapters):
                if is_core_adapter(str(adapter.get("path", "")), core_patterns):
                    entry["scale"] = 1.0
        label = f"unload({Path(target['path']).name})"
    else:
        for a in adapters:
            aid = int(a["id"])
            path = str(a.get("path", ""))
            new_scale = 1.0 if is_core_adapter(path, core_patterns) else 0.0
            payload.append({"id": aid, "scale": new_scale})
        label = "unload(all_specialists)"

    dt_ms, verified = _apply_and_verify(endpoint, payload)
    _emit_result(label, verified, core_patterns, args.json, dt_ms)
    return 0


def cmd_raw(args: argparse.Namespace, endpoint: str, core_patterns: tuple[str, ...]) -> int:
    try:
        raw_payload = json.loads(args.raw_json)
    except json.JSONDecodeError as exc:
        print(f"[lora_ctl] Invalid JSON for raw --json: {exc}", file=sys.stderr)
        return 1

    if not isinstance(raw_payload, list):
        print("[lora_ctl] raw --json must be a JSON array of {\"id\": int, \"scale\": float}.", file=sys.stderr)
        return 1

    _, _, adapters = _http_json(endpoint, method="GET")
    if not isinstance(adapters, list):
        raise RuntimeError(f"Unexpected /lora-adapters response: {adapters!r}")
    if not adapters:
        print("[lora_ctl] No adapters pre-loaded on server; cannot apply raw payload.", file=sys.stderr)
        return 1

    known_ids = {int(adapter["id"]) for adapter in adapters}
    clean_payload: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for idx, entry in enumerate(raw_payload):
        if not isinstance(entry, dict) or "id" not in entry or "scale" not in entry:
            print(f"[lora_ctl] Invalid entry at index {idx}: expected {{\"id\", \"scale\"}}.", file=sys.stderr)
            return 1
        aid = int(entry["id"])
        if aid in seen_ids:
            print(f"[lora_ctl] Invalid raw payload: duplicate adapter id={aid}.", file=sys.stderr)
            return 1
        if aid not in known_ids:
            print(f"[lora_ctl] Invalid raw payload: unknown adapter id={aid}.", file=sys.stderr)
            return 1
        seen_ids.add(aid)
        clean_payload.append({"id": aid, "scale": float(entry["scale"])})

    missing_ids = sorted(known_ids - seen_ids)
    if missing_ids:
        print(
            "[lora_ctl] Refusing incomplete raw payload: omitted adapter "
            f"id(s) {missing_ids}. Send every pre-loaded ID.",
            file=sys.stderr,
        )
        return 1

    dt_ms, verified = _apply_and_verify(endpoint, clean_payload)
    _emit_result("raw", verified, core_patterns, args.json, dt_ms)
    return 0


def _build_shared_parser(with_defaults: bool, include_json: bool = True) -> argparse.ArgumentParser:
    d_sup = argparse.SUPPRESS
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", default=DEFAULT_MODEL if with_defaults else d_sup, help=f"llama-swap model id (default: {DEFAULT_MODEL}).")
    p.add_argument("--swap-url", default=DEFAULT_SWAP_URL if with_defaults else d_sup, help=f"llama-swap base URL (default: {DEFAULT_SWAP_URL}).")
    p.add_argument("--direct", default=None if with_defaults else d_sup, help="Direct llama-server base URL (bypasses llama-swap /running check).")
    p.add_argument("--autoload", action="store_true", default=False if with_defaults else d_sup, help="Allow llama-swap to auto-load --model if not currently running.")
    if include_json:
        p.add_argument("--json", action="store_true", default=False if with_defaults else d_sup, help="Emit structured JSON output.")
    p.add_argument("--core-pattern", default=None if with_defaults else d_sup, help="Comma-separated filename substrings identifying always-on core LoRAs.")
    return p


def build_parser() -> argparse.ArgumentParser:
    top_shared = _build_shared_parser(with_defaults=True, include_json=True)
    sub_shared = _build_shared_parser(with_defaults=False, include_json=True)
    sub_shared_no_json = _build_shared_parser(with_defaults=False, include_json=False)

    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[top_shared],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", aliases=["status"], parents=[sub_shared], help="Show pre-loaded LoRA IDs, roles, and active scales.")
    p_list.set_defaults(func=cmd_list)

    p_load = sub.add_parser("load", aliases=["on", "enable"], parents=[sub_shared], help="Enable one specialist LoRA while keeping core adapters active.")
    p_load.add_argument("selector", help="Adapter numeric id, alias (geonodes, blenderworld, blender), or filename substring.")
    p_load.add_argument("--scale", type=float, default=None, help="Target scale > 0.0 (default: specialist 0.35, core 1.0).")
    p_load.add_argument("--core-scale", type=float, default=0.8, help="Scale for always-on core adapters during specialist use (default: 0.8).")
    p_load.add_argument("--stack", action="store_true", help="Keep other currently active specialists active instead of zeroing them.")
    p_load.add_argument("--force-core", action="store_true", help="Allow modifying a core adapter's scale.")
    p_load.set_defaults(func=cmd_load)

    p_unload = sub.add_parser("unload", aliases=["off", "disable", "reset"], parents=[sub_shared], help="Unload one specialist (or all specialists if omitted), preserving core adapters.")
    p_unload.add_argument("selector", nargs="?", default=None, help="Optional adapter id/alias/substring; omit to unload all non-core specialists.")
    p_unload.add_argument("--force-core", action="store_true", help="Allow unloading a core adapter.")
    p_unload.set_defaults(func=cmd_unload)

    p_raw = sub.add_parser("raw", parents=[sub_shared_no_json], help="Send an explicit JSON array to POST /lora-adapters.")
    p_raw.add_argument("--json", dest="raw_json", required=True, help="JSON array string, e.g. '[{\"id\":0,\"scale\":1.0}]'.")
    p_raw.set_defaults(func=cmd_raw)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.core_pattern:
        core_patterns = tuple(p.strip().lower() for p in args.core_pattern.split(",") if p.strip())
    else:
        core_patterns = DEFAULT_CORE_PATTERNS

    endpoint = _resolve_endpoint(args)
    try:
        return int(args.func(args, endpoint, core_patterns))
    except RuntimeError as exc:
        print(f"[lora_ctl] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
