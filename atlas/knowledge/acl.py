"""KnowledgeACL — domain-scoped read/write permissions for agents.

Enforcement happens in AgentContext methods, not in the provider.
The provider is domain-unaware; the context wraps it with ACL checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KnowledgeACL:
    """Access control for knowledge domains.

    - read_domains: ["*"] = read all, ["physics", "ai"] = only those
    - write_domains: ["*"] = write all (except protected), ["ai"] = only "ai"
    - protected_domains: require explicit listing in write_domains
      (a "*" wildcard does NOT grant access to protected domains)
    """

    read_domains: list[str] = field(default_factory=lambda: ["*"])
    write_domains: list[str] = field(default_factory=list)
    protected_domains: frozenset[str] = field(default_factory=frozenset)

    def can_read(self, domain: str) -> bool:
        """Check if this ACL grants read access to the given domain."""
        if "*" in self.read_domains:
            return True
        return domain in self.read_domains

    def can_write(self, domain: str) -> bool:
        """Check if this ACL grants write access to the given domain."""
        if "*" in self.write_domains:
            return domain not in self.protected_domains
        return domain in self.write_domains

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> KnowledgeACL:
        if not d:
            return KnowledgeACL()
        return KnowledgeACL(
            read_domains=d.get("read_domains", ["*"]),
            write_domains=d.get("write_domains", []),
            protected_domains=frozenset(d.get("protected_domains", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "read_domains": list(self.read_domains),
            "write_domains": list(self.write_domains),
            "protected_domains": sorted(self.protected_domains),
        }
