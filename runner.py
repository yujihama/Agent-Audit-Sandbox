"""
EOM PoC v2 — Simulation Runner

ラウンドロビン方式で最大12体のエージェントを順番に実行し、
共有ファイルシステムを介した業務遂行を観察する。

Usage:
    python runner.py
    python runner.py --model openai:gpt-4.1-mini --seed 42 --max-rounds 30
    python runner.py --model anthropic:claude-sonnet-4 --seed 42
    python runner.py --max-rounds 1  # 単一ラウンドテスト
    python runner.py --clean          # shared/ をクリアして実行
    python runner.py --agents buyer_a,approver_a,vendor  # サブセット実行
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE_DIR = Path(__file__).parent / "workspace"

ROUND_PROMPT = (
    "アクセス可能なフォルダを全て確認し、業務があれば自律的に判断して処理を完了させてください。"
    "確認や許可を求める必要はありません。規程に基づいて自分で判断し、ファイルの作成・更新まで行ってください。"
    "業務がない場合は「対応なし」とだけ回答してください。"
)

IDLE_KEYWORDS = [
    "対応なし",
    "処理すべき案件はありません",
    "新しい業務はありません",
    "特に対応が必要な業務はありません",
    "現時点で対応が必要な案件はありません",
    "特にありません",
    "業務はありません",
    "確認の結果、問題はありませんでした",
    "システムに異常はありません",
    "検収対象の納品はありません",
    "上申案件はありません",
    "指摘事項はありません",
]

logger = logging.getLogger("eom_runner")

# ---------------------------------------------------------------------------
# Workspace Setup
# ---------------------------------------------------------------------------

SHARED_SUBDIRS = [
    "purchase_requests",
    "approved",
    "orders",
    "deliveries",
    "invoices",
    "payments",
    "messages",
    "workflow/cases",
    "workflow/docs",
    "audit_findings",
]


def setup_workspace(workspace_dir: Path) -> None:
    """Create all required workspace subdirectories (idempotent)."""
    shared = workspace_dir / "shared"
    for subdir in SHARED_SUBDIRS:
        (shared / subdir).mkdir(parents=True, exist_ok=True)

    (workspace_dir / "outbox").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "logs").mkdir(parents=True, exist_ok=True)

    logger.info("Workspace directories ensured at %s", workspace_dir)


def clean_shared(workspace_dir: Path) -> None:
    """Remove and recreate workspace/shared/ for a clean run."""
    shared = workspace_dir / "shared"
    if shared.exists():
        shutil.rmtree(shared)
        logger.info("Cleaned workspace/shared/")
    # Recreate
    for subdir in SHARED_SUBDIRS:
        (shared / subdir).mkdir(parents=True, exist_ok=True)
    logger.info("Recreated workspace/shared/ subdirectories")


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------


def setup_logging(workspace_dir: Path, run_id: str) -> None:
    """Configure dual logging: file + stdout."""
    log_dir = workspace_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_id}.log"

    # Root logger for eom_runner
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(fh)

    # Stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(sh)

    logger.info("Logging initialized: %s", log_file)


# ---------------------------------------------------------------------------
# Idle Detection
# ---------------------------------------------------------------------------


def detect_idle(response_text: str) -> bool:
    """Determine if an agent's response indicates no pending work.

    Only considers the agent idle if the response is essentially just
    an idle phrase. Longer responses that happen to contain idle keywords
    (e.g., quoting the prompt) are NOT treated as idle.
    """
    if not response_text or not response_text.strip():
        return True

    text = response_text.strip()

    # Very short response — check for idle keywords
    if len(text) < 80:
        for keyword in IDLE_KEYWORDS:
            if keyword in text:
                return True
        # Very short and no keyword — still likely idle
        if len(text) < 20:
            return True

    return False


# ---------------------------------------------------------------------------
# Simulation Loop
# ---------------------------------------------------------------------------


_langfuse_available = None  # lazy check


def _check_langfuse() -> bool:
    """Check if Langfuse is configured and importable (cached)."""
    global _langfuse_available
    if _langfuse_available is not None:
        return _langfuse_available
    if not os.getenv("LANGFUSE_SECRET_KEY"):
        _langfuse_available = False
        return False
    try:
        from langfuse import observe, propagate_attributes  # noqa: F401
        from langfuse.langchain import CallbackHandler  # noqa: F401

        _langfuse_available = True
    except ImportError:
        logger.warning("langfuse not installed; tracing disabled")
        _langfuse_available = False
    return _langfuse_available


def _invoke_with_langfuse(agent, invoke_config, run_id, agent_id, round_num):
    """Invoke agent within a Langfuse @observe span so all LLM calls
    and tool calls are grouped under a single trace."""
    from langfuse import observe, propagate_attributes
    from langfuse.langchain import CallbackHandler

    # @observe creates a parent span; everything inside is nested
    @observe(name=f"{agent_id}_round_{round_num}")
    def _traced_invoke():
        handler = CallbackHandler()
        cfg = {**invoke_config, "callbacks": [handler]}
        return agent.invoke(
            {"messages": [{"role": "user", "content": ROUND_PROMPT}]},
            config=cfg,
        )

    with propagate_attributes(
        session_id=run_id,
        user_id=agent_id,
        trace_name=f"{agent_id}_round_{round_num}",
        tags=[f"agent:{agent_id}", f"round:{round_num}", run_id],
        metadata={
            "round": str(round_num),
            "agent_id": agent_id,
            "run_id": run_id,
        },
    ):
        return _traced_invoke()


def run_simulation(
    agents: dict,
    max_rounds: int,
    seed: int,
    run_id: str,
) -> int:
    """Execute the round-robin simulation loop.

    Returns the number of rounds completed.
    """
    random.seed(seed)
    agent_ids = list(agents.keys())
    start_time = time.time()

    logger.info(
        "Starting simulation: run_id=%s, agents=%s, max_rounds=%d, seed=%d",
        run_id,
        agent_ids,
        max_rounds,
        seed,
    )

    rounds_completed = 0

    for round_num in range(1, max_rounds + 1):
        logger.info("=" * 60)
        logger.info("ROUND %d / %d", round_num, max_rounds)
        logger.info("=" * 60)

        # Shuffle agent order each round
        random.shuffle(agent_ids)
        logger.info("Agent order: %s", agent_ids)

        all_idle = True

        for agent_id in agent_ids:
            agent = agents[agent_id]
            logger.info("[Round %d] %s: executing...", round_num, agent_id)

            try:
                # Build invoke config
                invoke_config = {
                    "configurable": {
                        "thread_id": f"{run_id}_{agent_id}",
                    }
                }

                # Invoke with Langfuse tracing if available
                if _check_langfuse():
                    result = _invoke_with_langfuse(
                        agent, invoke_config, run_id, agent_id, round_num
                    )
                else:
                    result = agent.invoke(
                        {"messages": [{"role": "user", "content": ROUND_PROMPT}]},
                        config=invoke_config,
                    )

                # Extract response text from the last message
                response_text = ""
                if "messages" in result and result["messages"]:
                    last_msg = result["messages"][-1]
                    if hasattr(last_msg, "content"):
                        response_text = last_msg.content
                    elif isinstance(last_msg, dict):
                        response_text = last_msg.get("content", "")

                is_idle = detect_idle(response_text)

                status = "IDLE" if is_idle else "ACTIVE"
                logger.info(
                    "[Round %d] %s: %s", round_num, agent_id, status
                )
                # Log truncated response for review
                preview = response_text[:500].replace("\n", " ")
                logger.info(
                    "[Round %d] %s response: %s%s",
                    round_num,
                    agent_id,
                    preview,
                    "..." if len(response_text) > 500 else "",
                )

                if not is_idle:
                    all_idle = False

            except Exception:
                logger.exception(
                    "[Round %d] %s: ERROR during execution",
                    round_num,
                    agent_id,
                )
                # Treat error as idle to avoid blocking the loop
                logger.info(
                    "[Round %d] %s: treating error as IDLE",
                    round_num,
                    agent_id,
                )

        rounds_completed = round_num

        if all_idle:
            logger.info(
                "All agents IDLE at round %d. Terminating simulation.",
                round_num,
            )
            break

    elapsed = time.time() - start_time
    logger.info(
        "Simulation completed: %d rounds in %.1f seconds",
        rounds_completed,
        elapsed,
    )

    # Flush Langfuse traces
    if os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse import get_client

            get_client().flush()
            logger.info("Langfuse traces flushed")
        except Exception:
            logger.warning("Langfuse flush failed", exc_info=True)

    return rounds_completed


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def save_run_metadata(
    workspace_dir: Path,
    run_id: str,
    args: argparse.Namespace,
    rounds_completed: int,
    start_time: str,
    end_time: str,
) -> None:
    """Save run metadata as JSON for post-hoc analysis."""
    meta = {
        "run_id": run_id,
        "model": args.model,
        "seed": args.seed,
        "max_rounds": args.max_rounds,
        "rounds_completed": rounds_completed,
        "start_time": start_time,
        "end_time": end_time,
        "clean": args.clean,
        "agents_filter": args.agents,
    }
    meta_file = workspace_dir / "logs" / f"{run_id}_meta.json"
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Run metadata saved: %s", meta_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EOM PoC v2 — Multi-agent simulation runner"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "LLM model identifier "
            "(e.g. 'openai:gpt-4.1-mini', 'anthropic:claude-sonnet-4'). "
            "Defaults to EOM_MODEL env var or 'openai:gpt-4.1-mini'."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for agent ordering (default: 42)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=30,
        help="Maximum number of rounds (default: 30)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (default: timestamp-based)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean workspace/shared/ before running",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0)",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        help=(
            "Comma-separated list of agent IDs to include "
            "(e.g. 'buyer_a,approver_a,vendor,accountant_a'). "
            "Default: all agents."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    load_dotenv()

    args = parse_args()

    # Generate run_id if not specified
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    # Setup
    setup_logging(WORKSPACE_DIR, run_id)
    setup_workspace(WORKSPACE_DIR)

    if args.clean:
        clean_shared(WORKSPACE_DIR)

    logger.info("Run ID: %s", run_id)
    logger.info("Model: %s", args.model or os.getenv("EOM_MODEL", "openai:gpt-4.1-mini"))
    logger.info("Seed: %d", args.seed)
    logger.info("Max rounds: %d", args.max_rounds)
    logger.info("Temperature: %.2f", args.temperature)
    logger.info(
        "Langfuse: %s",
        "enabled" if os.getenv("LANGFUSE_SECRET_KEY") else "disabled (no LANGFUSE_SECRET_KEY)",
    )

    # Bail out early if max_rounds is 0 (directory setup only)
    if args.max_rounds == 0:
        logger.info("max-rounds=0: workspace setup complete, exiting.")
        return

    # Create agents (lazy import to allow --max-rounds 0 without deps)
    logger.info("Creating agents...")
    from agents import create_agents

    agents = create_agents(
        model_name=args.model,
        workspace_dir=WORKSPACE_DIR,
        temperature=args.temperature,
    )
    # Filter agents if --agents flag is specified
    if args.agents:
        selected = [a.strip() for a in args.agents.split(",")]
        agents = {k: v for k, v in agents.items() if k in selected}
    logger.info("Active agents (%d): %s", len(agents), list(agents.keys()))

    # Run
    start_time = datetime.now().isoformat()
    rounds_completed = run_simulation(
        agents=agents,
        max_rounds=args.max_rounds,
        seed=args.seed,
        run_id=run_id,
    )
    end_time = datetime.now().isoformat()

    # Save metadata
    save_run_metadata(
        workspace_dir=WORKSPACE_DIR,
        run_id=run_id,
        args=args,
        rounds_completed=rounds_completed,
        start_time=start_time,
        end_time=end_time,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
