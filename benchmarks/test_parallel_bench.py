import asyncio
import json
import subprocess
import time
import urllib.request
from datetime import datetime

SERVER_URL = "http://192.168.1.240:8086/v1/chat/completions"
METRICS_URL = "http://192.168.1.240:8086/metrics"
SSH_HOST = "hp"
MODEL = "qwen3.8-flash-next-ud-q4_k_xl"
MAX_TOKENS = 800

PROMPTS = [
    (
        "conv-A",
        "Write a detailed, comprehensive technical guide on designing a lock-free multi-producer multi-consumer ring buffer in C++20 for low-latency trading systems, including memory ordering semantics, cache-line padding, and false sharing mitigation.",
    ),
    (
        "conv-B",
        "Write a detailed, comprehensive architectural breakdown of modern distributed consensus algorithms (Raft, Paxos, and Viewstamped Replication), comparing leader election, log replication, network partition recovery, and read linearizability.",
    ),
]


def fetch_metrics():
    with urllib.request.urlopen(METRICS_URL, timeout=10) as resp:
        text = resp.read().decode("utf-8")
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def stream_request(conv_id, prompt):
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.6,
            "stream": True,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        SERVER_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Strata-Conversation": conv_id,
        },
    )
    t0 = time.time()
    first_token_t = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    if first_token_t is None:
                        first_token_t = time.time()
                    tokens += 1
            except Exception:
                pass
    t1 = time.time()
    total_s = t1 - t0
    decode_s = (t1 - first_token_t) if first_token_t else total_s
    ttft_s = (first_token_t - t0) if first_token_t else total_s
    return {
        "conv_id": conv_id,
        "tokens": tokens,
        "ttft_s": round(ttft_s, 3),
        "decode_s": round(decode_s, 3),
        "total_s": round(total_s, 3),
        "e2e_tps": round(tokens / total_s, 2) if total_s > 0 else 0,
        "decode_tps": round(tokens / decode_s, 2) if decode_s > 0 else 0,
    }


async def run_parallel_round(round_idx):
    m_before = fetch_metrics()
    t0 = time.time()
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(None, stream_request, f"{cid}-r{round_idx}", prompt)
        for cid, prompt in PROMPTS
    ]
    results = await asyncio.gather(*tasks)
    wall_s = time.time() - t0
    m_after = fetch_metrics()

    total_tokens = sum(r["tokens"] for r in results)
    agg_e2e_tps = round(total_tokens / wall_s, 2)
    drafts_offered = int(
        m_after.get("strata_drafts_offered_total", 0)
        - m_before.get("strata_drafts_offered_total", 0)
    )
    drafts_accepted = int(
        m_after.get("strata_drafts_accepted_total", 0)
        - m_before.get("strata_drafts_accepted_total", 0)
    )
    accept_pct = (
        round(100.0 * drafts_accepted / drafts_offered, 1)
        if drafts_offered > 0
        else 0.0
    )

    print(f"\n--- Parallel Round {round_idx} (2 concurrent streams) ---")
    for r in results:
        print(
            f"  [{r['conv_id']}] {r['tokens']} tokens | TTFT: {r['ttft_s']}s | "
            f"Decode: {r['decode_tps']} tok/s | E2E: {r['e2e_tps']} tok/s ({r['total_s']}s)"
        )
    print(
        f"  => Aggregate: {total_tokens} tokens in {wall_s:.2f}s = {agg_e2e_tps} tok/s "
        f"| Batch MTP drafts accepted: {drafts_accepted}/{drafts_offered} ({accept_pct}%)"
    )
    return {
        "round": round_idx,
        "wall_s": round(wall_s, 2),
        "total_tokens": total_tokens,
        "aggregate_tps": agg_e2e_tps,
        "drafts_offered": drafts_offered,
        "drafts_accepted": drafts_accepted,
        "draft_accept_pct": accept_pct,
        "streams": results,
    }


async def main():
    print("======================================================================")
    print("STRATA PARALLEL 2-CONVERSATION BENCHMARK (Dual Tesla V100 64GB)")
    print("======================================================================")
    rounds = []
    for r in range(1, 4):
        res = await run_parallel_round(r)
        rounds.append(res)
    avg_agg = sum(x["aggregate_tps"] for x in rounds) / len(rounds)
    avg_per_stream = sum(
        s["e2e_tps"] for x in rounds for s in x["streams"]
    ) / (2 * len(rounds))
    print("\n======================================================================")
    print(
        f"SUMMARY (3 rounds x 2 parallel streams): "
        f"Avg Aggregate = {avg_agg:.1f} tok/s | Avg Per-Stream E2E = {avg_per_stream:.1f} tok/s"
    )
    print("======================================================================")


if __name__ == "__main__":
    asyncio.run(main())
