"""Atlas structured knowledge — searchable, domain-scoped, access-controlled."""

from atlas.knowledge.acl import KnowledgeACL
from atlas.knowledge.file_provider import FileKnowledgeProvider
from atlas.knowledge.http_provider import HttpKnowledgeProvider
from atlas.knowledge.mcp_provider import MCPKnowledgeProvider
from atlas.knowledge.provider import KnowledgeEntry, KnowledgeProvider

__all__ = [
    "KnowledgeACL",
    "KnowledgeEntry",
    "KnowledgeProvider",
    "FileKnowledgeProvider",
    "HttpKnowledgeProvider",
    "MCPKnowledgeProvider",
]
