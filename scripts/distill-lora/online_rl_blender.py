#!/usr/bin/env python3
"""Online Verifiable-Reward RL (GRPO + Best-of-G RFT) for Ternary-Bonsai-2-27B on Qwen/BlenderWorld.

Alternates cleanly inside 8 GB VRAM:
  1) Launch Bonsai-demo llama-server (+ current LoRA) to sample G rollouts per task
  2) Grade rollouts online in headless Blender 4.2.10 via blender_verifier.py
  3) Unload llama-server, compute exact ref_logps + old_logps, update primal LoRA via
     clipped GRPO + k3 KL penalty in State-Carrying Tiled-BPTT, and export Q8_0 GGUF.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import bitsandbytes as bnb
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_SCRIPTS = SCRIPT_DIR.parent
DEFAULT_MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/home/nos/models"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PARENT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PARENT_SCRIPTS))

import benchmark_blenderworld as bbw  # noqa: E402
import blender_verifier as bv  # noqa: E402
import bonsai_ptq_model as bpm  # noqa: E402
import train_lora as tlora  # noqa: E402


BLENDERWORLD_SYS_PROMPT = (
    "You are an expert Blender 4.2 Python programmer. Generate clean, self-contained, executable `bpy` code "
    "compatible with Blender 4.2.10."
)
GEONODES_SYS_PROMPT = (
    "You are an expert Blender 5.4 Python programmer. Generate clean, self-contained, executable `bpy` code."
)


def _align_rollout_logprobs(
    comp_ids: list[int],
    roll_tok_ids: list[int],
    roll_logps: list[float],
    device: torch.device,
) -> torch.Tensor | None:
    """Align llama-server rollout token logprobs to tokenized completion IDs for DPPO bypass_mode."""
    n_comp = len(comp_ids)
    if n_comp == 0 or not roll_tok_ids or not roll_logps or len(roll_tok_ids) != len(roll_logps):
        return None
    n_roll = len(roll_tok_ids)
    if 0 <= n_comp - n_roll <= 2:
        matches = sum(1 for a, b in zip(roll_tok_ids, comp_ids[:n_roll]) if int(a) == int(b))
        if matches / float(max(1, n_roll)) >= 0.98:
            aligned = [float(x) for x in roll_logps] + [0.0] * (n_comp - n_roll)
            return torch.tensor(aligned, dtype=torch.float32, device=device)
    elif n_roll >= n_comp:
        matches = sum(1 for a, b in zip(roll_tok_ids[:n_comp], comp_ids) if int(a) == int(b))
        if matches / float(max(1, n_comp)) >= 0.98:
            return torch.tensor([float(x) for x in roll_logps[:n_comp]], dtype=torch.float32, device=device)
    return None


def _build_rollout_sequence(
    item: dict[str, Any],
    tok: Any,
    max_seq_len: int,
    reasoning_budget: int,
    algo: str,
    device: torch.device,
) -> tuple[list[int], int, torch.Tensor | None]:
    """Build (ids, plen, exact_rollout_logps) preserving exact llama-server token stream in DPPO bypass_mode."""
    roll_ids = item.get("rollout_token_ids") or []
    roll_lps = item.get("rollout_logprobs") or []
    if algo == "dppo" and roll_ids and roll_lps and len(roll_ids) == len(roll_lps):
        sys_p = str(item.get("system") or GEONODES_SYS_PROMPT).strip()
        usr_p = str(item.get("prompt") or "").strip()
        p_txt = tlora.render_chat_prefix(sys_p, usr_p, enable_thinking=(reasoning_budget > 0))
        p_ids = tok.encode(p_txt, add_special_tokens=False).ids
        full_ids = (p_ids + [int(x) for x in roll_ids])[:max_seq_len]
        plen = len(p_ids)
        n_comp = len(full_ids) - plen
        if n_comp > 8:
            lp_t = torch.tensor([float(x) for x in roll_lps[:n_comp]], dtype=torch.float32, device=device)
            return full_ids, plen, lp_t
    ids, plen = tlora.format_conversation(item, tok, max_seq_len=max_seq_len)
    aligned = (
        _align_rollout_logprobs(ids[plen:], roll_ids, roll_lps, device)
        if algo == "dppo"
        else None
    )
    return ids, plen, aligned


def _strip_rollout_debug(item: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a rollout dict without bulky per-token logprob arrays for disk logs."""
    return {k: v for k, v in item.items() if k not in ("rollout_logprobs", "rollout_token_ids")}


def collect_online_rollouts(
    base_gguf: Path,
    lora_gguf: Path | None,
    task_items: list[tuple[str, str, dict[str, Any] | None, str]],
    group_size: int = 4,
    max_tokens: int = 2560,
    reasoning_budget: int = 0,
    reasoning_budget_message: str | None = bbw.DEFAULT_REASONING_BUDGET_MESSAGE,
    temperature: float = 0.8,
    require_geonodes: bool = False,
    algo: str = "dppo",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Sample G rollouts per task with llama-server (+ token logprobs) and verify in headless Blender."""
    lora_specs = [f"{lora_gguf}:1.0"] if (lora_gguf and lora_gguf.is_file()) else []
    proc = bbw.start_llama_server(
        base_gguf,
        lora_specs,
        port=18089,
        ctx_size=12288,
        reasoning_budget=reasoning_budget,
        reasoning_budget_message=reasoning_budget_message,
    )
    grouped_items: list[tuple[float, float, str, list[dict[str, Any]]]] = []
    all_rewards: list[float] = []
    all_exec: list[bool] = []
    all_gn_pass: list[bool] = []
    try:
        for tid, prompt, bundle, domain in task_items:
            eff_domain = domain or ("geonodes" if require_geonodes else "blenderworld")
            sys_prompt = GEONODES_SYS_PROMPT if eff_domain == "geonodes" else BLENDERWORLD_SYS_PROMPT
            grp_rewards: list[float] = []
            grp_items: list[dict[str, Any]] = []

            for g_idx in range(group_size):
                content, reasoning, finish_reason, roll_tok_ids, roll_logps = bbw.chat_complete_with_logprobs(
                    18089,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=sys_prompt,
                )
                vr = bv.verify_script(content, task_bundle=bundle)
                gn_pass = bv.is_geonodes_pass(vr)
                ss = vr.scene_summary or {}
                eff_reward = vr.reward
                if eff_domain == "geonodes" and vr.exec_ok and not gn_pass:
                    # Strictly penalize fallback scripts that omit GeometryNodeTree when Geometry Nodes are required.
                    eff_reward = min(vr.reward, 0.25)
                grp_rewards.append(eff_reward)
                all_rewards.append(eff_reward)
                all_exec.append(vr.exec_ok)
                all_gn_pass.append(gn_pass)
                grp_items.append({
                    "id": f"{tid}_g{g_idx}",
                    "task_id": tid,
                    "domain": eff_domain,
                    "preserve_prompt": (eff_domain == "geonodes"),
                    "system": sys_prompt,
                    "prompt": prompt,
                    "reasoning": reasoning,
                    "response": content,
                    "finish_reason": finish_reason,
                    "reward": eff_reward,
                    "raw_reward": vr.reward,
                    "exec_ok": vr.exec_ok,
                    "gn_pass": gn_pass,
                    "geonode_trees_count": ss.get("geonode_trees_count", 0),
                    "geonode_nodes_total": ss.get("geonode_nodes_total", 0),
                    "total_verts": ss.get("total_verts", 0),
                    "syntax_ok": vr.syntax_ok,
                    "error": vr.error[:300],
                    "rollout_token_ids": roll_tok_ids,
                    "rollout_logprobs": roll_logps,
                })

            mean_r = float(np.mean(grp_rewards))
            std_r = float(np.std(grp_rewards))
            print(
                f"  [rollout {tid} ({eff_domain})] rewards={[round(x, 3) for x in grp_rewards]} "
                f"exec={[x['exec_ok'] for x in grp_items]} gn={[x['gn_pass'] for x in grp_items]} "
                f"mean={mean_r:.3f} std={std_r:.3f}",
                flush=True,
            )
            grouped_items.append((mean_r, std_r, eff_domain, grp_items))
    finally:
        bbw.stop_llama_server(proc)
        time.sleep(1.0)

    round_mean = float(np.mean(all_rewards)) if all_rewards else 0.0
    rollouts: list[dict[str, Any]] = []
    all_items: list[dict[str, Any]] = []

    for mean_r, std_r, eff_domain, grp_items in grouped_items:
        tid = grp_items[0]["task_id"] if grp_items else "?"
        gate_key = "gn_pass" if eff_domain == "geonodes" else "exec_ok"
        pass_items = [it for it in grp_items if it[gate_key]]
        if not pass_items:
            print(
                f"  [rollout {tid}] skipped group (0/{len(grp_items)} {gate_key} — gating out noise)",
                flush=True,
            )
            for item in grp_items:
                item["advantage"] = 0.0
                all_items.append(item)
            continue

        if algo == "dppo":
            # Dr.GRPO: unnormalized reward centering (no / std difficulty bias, mild floor for uniform all-pass groups).
            all_passed = len(pass_items) == len(grp_items)
            for item in grp_items:
                adv = float(item["reward"] - mean_r)
                if all_passed and abs(adv) < 1e-4:
                    adv = 0.25
                item["advantage"] = max(-1.0, min(1.0, adv))
                all_items.append(item)
                if abs(item["advantage"]) >= 1e-4:
                    rollouts.append(item)
        else:
            best_grp_r = max(it["reward"] for it in pass_items)
            for item in grp_items:
                if item[gate_key]:
                    if std_r >= 1e-4:
                        adv = float((item["reward"] - mean_r) / (std_r + 1e-4))
                    else:
                        adv = 0.6
                    if abs(item["reward"] - best_grp_r) < 1e-6:
                        adv = max(adv, 0.6)
                else:
                    raw_neg = float((item["reward"] - mean_r) / (std_r + 1e-4)) * 0.35 if std_r >= 1e-4 else -0.35
                    adv = max(-0.4, min(-0.15, raw_neg))

                item["advantage"] = max(-2.0, min(2.0, adv))
                all_items.append(item)
                if abs(item["advantage"]) >= 1e-4:
                    rollouts.append(item)

    stats = {
        "n_rollouts": len(all_rewards),
        "mean_reward": round(round_mean, 4),
        "exec_rate": round(float(sum(1 for x in all_exec if x)) / max(1, len(all_exec)), 4),
        "gn_pass_rate": round(float(sum(1 for x in all_gn_pass if x)) / max(1, len(all_gn_pass)), 4),
        "n_nondegenerate": len(rollouts),
    }
    return rollouts, all_items, stats


def _harvest_winners(all_items: list[dict[str, Any]], winners_path: Path, min_reward: float = 0.53) -> int:
    """Persist highest-reward exec_ok rollouts per task_id into a durable Tier 4 SFT pool."""
    by_task: dict[str, dict[str, Any]] = {}
    if winners_path.is_file():
        for line in winners_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                by_task[str(obj.get("task_id") or obj.get("id"))] = obj
    added = 0
    for it in all_items:
        if not it.get("exec_ok") or float(it.get("reward", 0.0)) < min_reward:
            continue
        tid = str(it.get("task_id") or it.get("id"))
        prev = by_task.get(tid)
        if prev is None or float(it["reward"]) > float(prev.get("reward", 0.0)):
            by_task[tid] = it
            added += 1
    winners_path.parent.mkdir(parents=True, exist_ok=True)
    winners_path.write_text(
        "\n".join(json.dumps(by_task[k], ensure_ascii=False) for k in sorted(by_task.keys())) + ("\n" if by_task else ""),
        encoding="utf-8",
    )
    return len(by_task)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-gguf", type=Path, default=tlora.DEFAULT_BASE_GGUF)
    parser.add_argument("--init-peft", type=Path, required=True, help="Warm-up PEFT adapter dir.")
    parser.add_argument("--init-gguf", type=Path, required=True, help="Warm-up Q8_0 GGUF adapter for round 0 rollouts.")
    parser.add_argument("--peft-out", type=Path, default=SCRIPT_DIR / "work" / "peft_rl")
    parser.add_argument(
        "--gguf-out",
        type=Path,
        default=DEFAULT_MODEL_DIR / "Bonsai-2-27B-blenderworld-rl-Q8_0.gguf",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--tasks-per-round", type=int, default=6)
    parser.add_argument("--task-offset", type=int, default=0, help="Starting index into train_ids for multi-cycle RL.")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2560)
    parser.add_argument("--reasoning-budget", type=int, default=0)
    parser.add_argument("--reasoning-budget-message", type=str, default=bbw.DEFAULT_REASONING_BUDGET_MESSAGE)
    parser.add_argument("--start-layer", type=int, default=32)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--kl-beta", type=float, default=0.015)
    parser.add_argument("--algo", choices=["dppo", "grpo"], default="dppo")
    parser.add_argument("--dppo-variant", choices=["binary_tv", "binary_kl"], default="binary_tv")
    parser.add_argument("--dppo-clip", type=float, default=None, help="DPPO divergence threshold (default 0.15 TV / 0.05 KL).")
    parser.add_argument("--dppo-is-clip", type=float, default=3.0)
    parser.add_argument("--dppo-norm-tokens", type=float, default=2048.0)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=8)
    parser.add_argument("--ref-anchor", choices=["sft", "base"], default="sft")
    parser.add_argument("--mix-blenderworld-tasks", type=int, default=0, help="Append N BlenderWorld train tasks/round when --tasks-jsonl is set.")
    parser.add_argument("--bw-task-offset", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=3072)
    parser.add_argument("--window-len", type=int, default=512)
    parser.add_argument("--tasks-jsonl", type=Path, default=None, help="Optional JSONL of custom {id, prompt} RL tasks.")
    parser.add_argument("--require-geonodes", action="store_true", help="Gate positive RL advantage strictly on is_geonodes_pass().")
    parser.add_argument("--run-tag", type=str, default="", help="Optional suffix tag for rollout/history/intermediate filenames.")
    args = parser.parse_args()

    custom_tasks: list[dict[str, Any]] | None = None
    bw_train_ids = bv.load_blenderworld_text_tasks()["split"]["train"]
    if args.tasks_jsonl is not None:
        custom_tasks = [json.loads(l) for l in args.tasks_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        train_ids = [str(r["id"]) for r in custom_tasks]
    else:
        train_ids = bw_train_ids
    tok = tlora.load_qwen35_tokenizer()
    cur_peft = args.init_peft
    cur_gguf = args.init_gguf
    history = []
    dppo_clip = (
        float(args.dppo_clip)
        if args.dppo_clip is not None
        else (0.15 if args.dppo_variant == "binary_tv" else 0.05)
    )
    tag_part = f"_{args.run_tag}" if args.run_tag else ""

    for r_idx in range(args.rounds):
        batch_indices = [
            (args.task_offset + r_idx * args.tasks_per_round + k) % len(train_ids)
            for k in range(args.tasks_per_round)
        ]
        batch_ids = [train_ids[i] for i in batch_indices]
        task_items: list[tuple[str, str, dict[str, Any] | None, str]] = []
        if custom_tasks is not None:
            for i in batch_indices:
                r = custom_tasks[i]
                dom = "geonodes" if args.require_geonodes else str(r.get("domain") or "blenderworld")
                task_items.append((str(r["id"]), str(r["prompt"]), None, dom))
            if args.mix_blenderworld_tasks > 0:
                for k in range(args.mix_blenderworld_tasks):
                    bw_tid = bw_train_ids[(args.bw_task_offset + r_idx * args.mix_blenderworld_tasks + k) % len(bw_train_ids)]
                    bundle = bv.get_blenderworld_task(bw_tid)
                    task_items.append((bw_tid, bv.format_task_prompt(bundle, include_rubrics=False), bundle, "blenderworld"))
                    batch_ids.append(bw_tid)
        else:
            dom = "geonodes" if args.require_geonodes else "blenderworld"
            for tid in batch_ids:
                bundle = bv.get_blenderworld_task(tid)
                task_items.append((tid, bv.format_task_prompt(bundle, include_rubrics=False), bundle, dom))

        print(
            f"\n=== ONLINE RL ROUND {r_idx + 1}/{args.rounds} tasks={batch_ids} "
            f"(algo={args.algo}, ref={args.ref_anchor}, require_gn={args.require_geonodes}) ===",
            flush=True,
        )
        rollouts, all_items, r_stats = collect_online_rollouts(
            args.base_gguf,
            cur_gguf,
            task_items,
            group_size=args.group_size,
            max_tokens=args.max_tokens,
            reasoning_budget=args.reasoning_budget,
            reasoning_budget_message=args.reasoning_budget_message,
            require_geonodes=args.require_geonodes,
            algo=args.algo,
        )
        print(f"[round {r_idx + 1} rollout stats] {r_stats}", flush=True)
        history.append({
            "round": r_idx + 1,
            "task_offset": args.task_offset,
            "algo": args.algo,
            "ref_anchor": args.ref_anchor,
            "require_geonodes": args.require_geonodes,
            **r_stats,
        })

        prefix = "rl_gn" if args.require_geonodes else "rl"
        roll_log = SCRIPT_DIR / "work" / f"{prefix}{tag_part}_round_{r_idx + 1}_offset_{args.task_offset}_rollouts.jsonl"
        roll_log.parent.mkdir(parents=True, exist_ok=True)
        clean_items = [_strip_rollout_debug(x) for x in all_items]
        roll_log.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in clean_items), encoding="utf-8")

        gn_items = [it for it in clean_items if it.get("domain") == "geonodes" and it.get("gn_pass")]
        bw_items = [it for it in clean_items if it.get("domain") == "blenderworld"]
        total_gn_w = _harvest_winners(gn_items, SCRIPT_DIR / "work" / "rl_gn_harvested_winners.jsonl") if gn_items else 0
        total_bw_w = _harvest_winners(bw_items, SCRIPT_DIR / "work" / "rl_harvested_winners.jsonl") if bw_items else 0
        print(
            f"[round {r_idx + 1}] saved {len(clean_items)} rollouts -> {roll_log.name}; "
            f"harvested winners (gn_pool={total_gn_w}, bw_pool={total_bw_w})"
        )

        if not rollouts:
            print(f"[round {r_idx + 1}] no execution-verified groups; preserving input checkpoint without regression.")
            if r_idx == args.rounds - 1:
                if Path(cur_peft).resolve() != args.peft_out.resolve():
                    shutil.copytree(cur_peft, args.peft_out, dirs_exist_ok=True)
                if Path(cur_gguf).resolve() != args.gguf_out.resolve():
                    args.gguf_out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cur_gguf, args.gguf_out)
            continue

        # Train Phase: load BonsaiPTQStudent, compute ref_logps + bypass/old_logps, apply DPPO/GRPO update
        torch.cuda.empty_cache()
        model = bpm.BonsaiPTQStudent(
            gguf_path=args.base_gguf,
            start_grad_layer=args.start_layer,
            lora_rank=args.rank,
            lora_alpha=args.alpha,
        )

        # Reference anchor logprobs (either init SFT LoRA or frozen Base PTQ1_0)
        staged: list[tuple[dict[str, Any], list[int], int, float, torch.Tensor, torch.Tensor | None]] = []
        if args.ref_anchor == "sft":
            model.load_peft_dir(args.init_peft)
            model.set_adapters_enabled(True)
        else:
            model.load_peft_dir(cur_peft)
            model.set_adapters_enabled(False)

        for item in rollouts:
            ids, plen, pre_aligned = _build_rollout_sequence(
                item, tok, args.max_seq_len, args.reasoning_budget, args.algo, model.device
            )
            adv_val = float(item["advantage"])
            if len(ids) <= plen + 8 or abs(adv_val) < 1e-4:
                continue
            ref_lp = tlora.compute_sequence_logps(model, ids, plen, window_len=args.window_len)
            if ref_lp.numel() > 0:
                staged.append((item, ids, plen, adv_val, ref_lp, pre_aligned))

        if args.ref_anchor == "sft":
            if Path(cur_peft).resolve() != Path(args.init_peft).resolve():
                model.load_peft_dir(cur_peft)
        else:
            model.set_adapters_enabled(True)

        prepared: list[tuple[list[int], int, float, torch.Tensor, torch.Tensor, str]] = []
        bypass_used = 0
        fallback_used = 0
        for item, ids, plen, adv_val, ref_lp, pre_aligned in staged:
            aligned = pre_aligned if args.algo == "dppo" else None
            if aligned is not None and aligned.numel() == ref_lp.numel():
                roll_or_old = aligned
                bypass_used += 1
            else:
                roll_or_old = tlora.compute_sequence_logps(model, ids, plen, window_len=args.window_len)
                fallback_used += 1
            if roll_or_old.numel() == ref_lp.numel():
                prepared.append((ids, plen, adv_val, ref_lp, roll_or_old, item["id"]))

        print(
            f"[round {r_idx + 1} ref={args.ref_anchor}] prepared={len(prepared)} "
            f"bypass_mode={bypass_used} fallback_old_lp={fallback_used}",
            flush=True,
        )

        if prepared:
            params = [p for p in model.parameters() if p.requires_grad]
            opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.01)
            model.train()
            opt.zero_grad(set_to_none=True)
            n_epochs = args.ppo_epochs if args.algo == "dppo" else 1
            mb_size = max(1, int(args.minibatch_size))

            for ep in range(n_epochs):
                epoch_items = list(prepared)
                random.Random(42 + r_idx * 100 + ep).shuffle(epoch_items)
                for mb_idx, mb_start in enumerate(range(0, len(epoch_items), mb_size)):
                    chunk = epoch_items[mb_start : mb_start + mb_size]
                    grad_scale = 1.0 / float(len(chunk))
                    sum_pg, sum_kl, sum_mask = 0.0, 0.0, 0.0
                    for ids, plen, adv, ref_lp, roll_or_old_lp, rid in chunk:
                        if args.algo == "dppo":
                            pg_v, kl_v, mask_r, n_tok = tlora.backward_dppo_sequence(
                                model,
                                ids,
                                plen,
                                advantage=adv,
                                ref_logps=ref_lp,
                                rollout_logps=roll_or_old_lp,
                                kl_beta=args.kl_beta,
                                dppo_variant=args.dppo_variant,
                                clip_low=dppo_clip,
                                clip_high=dppo_clip,
                                is_clip_c=args.dppo_is_clip,
                                norm_tokens=args.dppo_norm_tokens,
                                window_len=args.window_len,
                                grad_scale=grad_scale,
                            )
                            sum_mask += mask_r * grad_scale
                            print(
                                f"  [dppo ep={ep} mb={mb_idx}] {rid} adv={adv:+.2f} toks={n_tok} "
                                f"pg={pg_v:.4f} kl={kl_v:.4f} mask={mask_r:.3f}",
                                flush=True,
                            )
                        else:
                            pg_v, kl_v, n_tok = tlora.backward_grpo_sequence(
                                model,
                                ids,
                                plen,
                                advantage=adv,
                                ref_logps=ref_lp,
                                old_logps=roll_or_old_lp,
                                kl_beta=args.kl_beta,
                                window_len=args.window_len,
                                grad_scale=grad_scale,
                            )
                            print(f"  [grpo ep={ep} mb={mb_idx}] {rid} adv={adv:+.2f} toks={n_tok} pg={pg_v:.4f} kl={kl_v:.4f}", flush=True)
                        sum_pg += pg_v * grad_scale
                        sum_kl += kl_v * grad_scale

                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    print(
                        f"[round {r_idx + 1} ep={ep} mb={mb_idx} update] algo={args.algo} n={len(chunk)} "
                        f"pg={sum_pg:.4f} kl={sum_kl:.4f}" + (f" masked={sum_mask:.3f}" if args.algo == "dppo" else ""),
                        flush=True,
                    )

        round_peft = args.peft_out if r_idx == args.rounds - 1 else SCRIPT_DIR / "work" / f"peft_rl{tag_part}_r{r_idx + 1}"
        round_gguf = args.gguf_out if r_idx == args.rounds - 1 else SCRIPT_DIR / "work" / f"rl{tag_part}_r{r_idx + 1}_Q8_0.gguf"
        tlora.export_peft_and_gguf(model, round_peft, round_gguf, outtype="q8_0")
        cur_peft = round_peft
        cur_gguf = round_gguf
        del model
        torch.cuda.empty_cache()

    hist_path = SCRIPT_DIR / "work" / f"rl_history{tag_part}.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[online_rl] DONE -> {args.gguf_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
