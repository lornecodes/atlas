#!/usr/bin/env python
"""Demo: Knowledge-backed agent workflow.

Populates a file-backed knowledge store, then queries it to show how
agents access structured knowledge during execution.

Usage:
    python demos/knowledge_agent.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas.knowledge.file_provider import FileKnowledgeProvider
from atlas.knowledge.provider import KnowledgeEntry
from atlas.memory.file_provider import FileMemoryProvider


async def main():
    # --- Set up providers ---
    tmp_dir = Path(tempfile.mkdtemp())
    knowledge_dir = tmp_dir / "knowledge"
    memory_path = tmp_dir / "memory.md"

    knowledge = FileKnowledgeProvider(knowledge_dir)
    memory = FileMemoryProvider(memory_path)

    print("=== Knowledge Store Demo ===\n")

    # --- Populate knowledge ---
    entries = [
        KnowledgeEntry(
            id="pac-theory",
            content=(
                "PAC (Primal Assertion of Constraint) is the foundational axiom of "
                "Dawn Field Theory. It states that the universe minimizes informational "
                "entropy subject to recursive boundary conditions."
            ),
            domain="physics",
            tags=["pac", "dft", "entropy", "axiom"],
        ),
        KnowledgeEntry(
            id="sec-mechanism",
            content=(
                "SEC (Self-Encoding Compression) describes how physical systems "
                "compress their state space through recursive self-reference. "
                "This is the mechanism by which PAC produces observable dynamics."
            ),
            domain="physics",
            tags=["sec", "compression", "recursion"],
        ),
        KnowledgeEntry(
            id="rbf-topology",
            content=(
                "RBF (Recursive Boundary Folding) generates the topological structure "
                "of spacetime via iterative self-similar boundary conditions. "
                "The Möbius manifold emerges from RBF at 6 recursion depths."
            ),
            domain="physics",
            tags=["rbf", "topology", "mobius", "spacetime"],
        ),
        KnowledgeEntry(
            id="grim-architecture",
            content=(
                "GRIM is an AI companion built on the Agent SDK. It uses "
                "18 skills, 12 MCP tools, and a Kronos knowledge graph with "
                "116+ FDOs. The architecture follows Phoenix patterns: "
                "two MCP services behind Cloudflare Tunnel."
            ),
            domain="ai-systems",
            tags=["grim", "architecture", "mcp", "kronos"],
        ),
        KnowledgeEntry(
            id="atlas-pool",
            content=(
                "Atlas execution pool manages agent lifecycle: warm slots, "
                "hardware-aware scheduling, event-driven metrics, and "
                "persistent job storage. Supports chains, triggers, "
                "and MCP federation."
            ),
            domain="ai-systems",
            tags=["atlas", "pool", "execution", "hardware"],
        ),
    ]

    print("--- Populating knowledge store ---")
    for entry in entries:
        created = await knowledge.create(entry)
        print(f"  Created: {created.id} (domain={created.domain})")

    # --- Search queries ---
    print("\n--- Searching knowledge ---")

    queries = [
        ("entropy", None, None),
        ("recursion", "physics", None),
        ("architecture", None, ["grim"]),
        ("pool", "ai-systems", None),
    ]

    for query, domain, tags in queries:
        results = await knowledge.search(
            query, domain=domain, tags=tags, limit=5
        )
        domain_str = f" domain={domain}" if domain else ""
        tags_str = f" tags={tags}" if tags else ""
        print(f"\n  Query: '{query}'{domain_str}{tags_str}")
        for r in results:
            print(f"    - {r.id}: {r.content[:80]}...")

    # --- CRUD operations ---
    print("\n--- CRUD operations ---")

    # Update
    updated = await knowledge.update(
        "pac-theory",
        tags=["pac", "dft", "entropy", "axiom", "fundamental"],
        metadata={"importance": "critical"},
    )
    print(f"  Updated pac-theory: tags={updated.tags}")

    # Get
    entry = await knowledge.get("rbf-topology")
    print(f"  Got rbf-topology: {entry.content[:60]}...")

    # List by domain
    physics = await knowledge.list_entries(domain="physics")
    print(f"  Physics entries: {len(physics)}")

    # Delete
    deleted = await knowledge.delete("sec-mechanism")
    print(f"  Deleted sec-mechanism: {deleted}")

    # --- Memory provider ---
    print("\n--- Shared Memory ---")

    await memory.write("# Agent Memory\n\nSession started.")
    await memory.append("Learned: PAC is the foundational axiom")
    await memory.append("Learned: GRIM uses 18 skills and 12 MCP tools")
    await memory.append("Learned: Atlas pool supports hardware-aware scheduling")

    content = await memory.read()
    print(f"  Memory contents ({len(content)} chars):")
    for line in content.strip().split("\n"):
        print(f"    {line}")

    # --- Persistence check ---
    print("\n--- Persistence verification ---")
    knowledge2 = FileKnowledgeProvider(knowledge_dir)
    check = await knowledge2.get("pac-theory")
    print(f"  Re-loaded pac-theory: {'OK' if check else 'MISSING'}")
    print(f"  Tags after reload: {check.tags if check else 'N/A'}")

    memory2 = FileMemoryProvider(memory_path)
    mem_check = await memory2.read()
    print(f"  Re-loaded memory: {len(mem_check)} chars")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
