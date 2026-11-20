#!/usr/bin/env python3
"""Benchmark Ternary-Bonsai-2-27B (+ optional LoRA GGUFs) on held-out Qwen/BlenderWorld text tasks."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import blender_verifier as bv  # noqa: E402

LLAMA_SERVER_BIN = Path(os.environ.get(
    "LLAMA_SERVER_BIN",
    "/run/media/nos/games/inference_engines/Bonsai-demo/bin/cuda/llama-server",
))
CHAT_TEMPLATE_PATH = Path(os.environ.get(
    "QWEN_CHAT_TEMPLATE",
    str(SCRIPT_DIR.parent / "qwen_chat_template.jinja"),
))
DEFAULT_REASONING_BUDGET_MESSAGE = (
    "\n\n[Thinking budget reached. Stop planning and immediately output the complete, executable Blender Python script now.]\n"
)


def ensure_llama_swap_free() -> None:
    """Unload llama-swap if running or wait until VRAM is free."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/running", timeout=2) as resp:
            running = json.loads(resp.read().decode("utf-8")).get("running", [])
        if running:
            req = urllib.request.Request("http://127.0.0.1:8080/api/models/unload", method="POST", data=b"")
            with urllib.request.urlopen(req, timeout=10):
                pass
            time.sleep(2.0)
    except Exception:
        pass


def start_llama_server(
    model_gguf: Path,
    lora_specs: list[str],
    port: int = 18089,
    ctx_size: int = 16384,
    reasoning_budget: int = 2048,
    reasoning_budget_message: str | None = DEFAULT_REASONING_BUDGET_MESSAGE,
) -> subprocess.Popen[str]:
    """Launch Bonsai-demo llama-server cleanly for inference/rollout generation."""
    ensure_llama_swap_free()
    cmd = [
        str(LLAMA_SERVER_BIN),
        "-m", str(model_gguf),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-ngl", "999",
        "-ot", "token_embd.weight=CPU",
        "-c", str(ctx_size),
        "-np", "1",
        "-ub", "1024",
        "-t", "12",
        "-tb", "12",
        "-fa", "on",
        "--cache-type-k", "q4_0",
        "--cache-type-v", "q4_0",
        "--jinja",
        "--chat-template-file", str(CHAT_TEMPLATE_PATH),
        "--temp", "0.6",
        "--top-p", "0.95",
        "--top-k", "20",
        "--min-p", "0.05",
    ]
    if reasoning_budget >= 0:
        cmd.extend([
            "--reasoning-effort", "medium" if reasoning_budget > 0 else "none",
            "--reasoning-budget", str(reasoning_budget),
            "--reasoning-preserve",
        ])
        if reasoning_budget > 0 and reasoning_budget_message:
            cmd.extend(["--reasoning-budget-message", reasoning_budget_message])
    if lora_specs:
        cmd.extend(["--lora-scaled", ",".join(lora_specs)])

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "FASTEST_FIRST"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            err = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"llama-server failed ({proc.returncode}): {err[-1200:]}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                if r.status == 200:
                    return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise TimeoutError("Timed out waiting for llama-server /health")


def stop_llama_server(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _post_chat_completion(
    port: int,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    include_logprobs: bool = False,
) -> dict[str, Any]:
    """Send a chat completion request to local llama-server and parse JSON response."""
    payload: dict[str, Any] = {
        "model": "local",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 20,
    }
    if include_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 1
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat_complete(
    port: int,
    prompt_text: str,
    max_tokens: int = 4096,
    temperature: float = 0.6,
    system_prompt: str = (
        "You are an expert Blender 4.2 Python programmer. Generate clean, self-contained, executable `bpy` code "
        "compatible with Blender 4.2.10."
    ),
) -> tuple[str, str]:
    """Query llama-server /v1/chat/completions and return (content, reasoning_content)."""
    res = _post_chat_completion(port, prompt_text, max_tokens, temperature, system_prompt, include_logprobs=False)
    msg = res["choices"][0]["message"]
    reasoning = str(msg.get("reasoning_content") or msg.get("reasoning") or "")
    return str(msg.get("content") or ""), reasoning


def chat_complete_with_logprobs(
    port: int,
    prompt_text: str,
    max_tokens: int = 4096,
    temperature: float = 0.6,
    system_prompt: str = (
        "You are an expert Blender 4.2 Python programmer. Generate clean, self-contained, executable `bpy` code "
        "compatible with Blender 4.2.10."
    ),
) -> tuple[str, str, str, list[int], list[float]]:
    """Query llama-server with token logprobs and return (content, reasoning, finish_reason, tok_ids, tok_logps)."""
    res = _post_chat_completion(port, prompt_text, max_tokens, temperature, system_prompt, include_logprobs=True)
    choice = res["choices"][0]
    msg = choice["message"]
    content = str(msg.get("content") or "")
    reasoning = str(msg.get("reasoning_content") or msg.get("reasoning") or "")
    finish_reason = str(choice.get("finish_reason") or "")
    lp_items = (choice.get("logprobs") or {}).get("content") or []
    has_all_ids = bool(lp_items) and all(
        isinstance(x.get("id"), int) and not isinstance(x.get("id"), bool) for x in lp_items
    )
    tok_ids = [int(x["id"]) for x in lp_items] if has_all_ids else []
    tok_logps = [float(x["logprob"]) for x in lp_items if "logprob" in x]
    return content, reasoning, finish_reason, tok_ids, tok_logps


def run_benchmark(
    model_gguf: Path,
    lora_specs: list[str],
    out_json: Path,
    task_limit: int = 8,
    max_tokens: int = 4096,
    reasoning_budget: int = 2048,
    temperature: float = 0.6,
    reasoning_budget_message: str | None = DEFAULT_REASONING_BUDGET_MESSAGE,
) -> dict[str, Any]:
    """Evaluate model + LoRA(s) on the held-out BlenderWorld text split and save metrics."""
    split = bv.load_blenderworld_text_tasks()["split"]
    eval_ids = split["eval"][:task_limit]
    proc = start_llama_server(
        model_gguf,
        lora_specs,
        reasoning_budget=reasoning_budget,
        reasoning_budget_message=reasoning_budget_message,
    )
    rows = []
    try:
        for tid in eval_ids:
            bundle = bv.get_blenderworld_task(tid)
            prompt = bv.format_task_prompt(bundle, include_rubrics=False)
            t0 = time.perf_counter()
            content, reasoning = chat_complete(18089, prompt, max_tokens=max_tokens, temperature=temperature)
            dt_gen = time.perf_counter() - t0
            vr = bv.verify_script(content, task_bundle=bundle)
            rows.append({
                "task_id": tid,
                "gen_sec": round(dt_gen, 2),
                "content_chars": len(content),
                "reasoning_chars": len(reasoning),
                "syntax_ok": vr.syntax_ok,
                "exec_ok": vr.exec_ok,
                "r_syntax": vr.r_syntax,
                "r_exec": vr.r_exec,
                "r_scene": vr.r_scene,
                "r_rubric": vr.r_rubric,
                "reward": vr.reward,
                "rubric_hits": vr.rubric_hits,
                "rubric_total": vr.rubric_total,
                "error": vr.error[:600],
            })
            print(
                f"[{tid}] syntax={vr.syntax_ok} exec={vr.exec_ok} scene={vr.r_scene:.2f} "
                f"rubric={vr.rubric_hits}/{vr.rubric_total} reward={vr.reward:.3f} ({dt_gen:.1f}s)",
                flush=True,
            )
    finally:
        stop_llama_server(proc)

    n = max(1, len(rows))
    summary = {
        "model": str(model_gguf),
        "loras": lora_specs,
        "tasks": len(rows),
        "syntax_rate": round(sum(1 for r in rows if r["syntax_ok"]) / n, 4),
        "exec_rate": round(sum(1 for r in rows if r["exec_ok"]) / n, 4),
        "mean_r_scene": round(sum(r["r_scene"] for r in rows) / n, 4),
        "mean_r_rubric": round(sum(r["r_rubric"] for r in rows) / n, 4),
        "mean_reward": round(sum(r["reward"] for r in rows) / n, 4),
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("BASE_GGUF", "/home/nos/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf")),
    )
    parser.add_argument("--lora", action="append", default=[], help="LoRA spec path:scale (repeatable)")
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "work" / "baseline_eval.json")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2560)
    parser.add_argument("--reasoning-budget", type=int, default=0)
    parser.add_argument("--reasoning-budget-message", type=str, default=DEFAULT_REASONING_BUDGET_MESSAGE)
    parser.add_argument("--temp", type=float, default=0.6)
    args = parser.parse_args()

    run_benchmark(
        args.model,
        args.lora,
        args.out,
        task_limit=args.limit,
        max_tokens=args.max_tokens,
        reasoning_budget=args.reasoning_budget,
        temperature=args.temp,
        reasoning_budget_message=args.reasoning_budget_message,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
