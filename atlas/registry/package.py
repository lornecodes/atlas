"""AgentPackage — pack and unpack agent tar.gz archives.

Security: tar paths must not escape target dir, total size capped at 10MB.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas.contract.schema import ContractError, load_contract
from atlas.contract.types import AgentContract
from atlas.registry.provider import PackageMetadata

# 10 MB max package size
MAX_PACKAGE_BYTES = 10 * 1024 * 1024


class PackageError(Exception):
    """Raised when packing or unpacking fails."""


def pack(agent_dir: Path | str) -> tuple[PackageMetadata, bytes]:
    """Pack an agent directory into a tar.gz archive.

    The directory must contain an agent.yaml. All files in the directory
    are included in the archive.

    Returns:
        (metadata, archive_bytes)

    Raises:
        PackageError: If agent.yaml is missing, invalid, or dir too large.
    """
    agent_dir = Path(agent_dir)
    yaml_path = agent_dir / "agent.yaml"

    if not yaml_path.exists():
        raise PackageError(f"No agent.yaml found in {agent_dir}")

    try:
        contract = load_contract(yaml_path)
    except ContractError as e:
        raise PackageError(f"Invalid agent contract: {e}") from e

    # Collect files
    buf = io.BytesIO()
    total_size = 0
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for file_path in sorted(agent_dir.rglob("*")):
            if not file_path.is_file():
                continue
            # Skip __pycache__ and hidden files
            rel = file_path.relative_to(agent_dir)
            if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                continue
            total_size += file_path.stat().st_size
            if total_size > MAX_PACKAGE_BYTES:
                raise PackageError(
                    f"Package exceeds {MAX_PACKAGE_BYTES // (1024*1024)}MB limit"
                )
            tar.add(file_path, arcname=str(rel))

    archive_bytes = buf.getvalue()
    sha = hashlib.sha256(archive_bytes).hexdigest()

    metadata = PackageMetadata(
        name=contract.name,
        version=contract.version,
        description=contract.description,
        capabilities=list(contract.capabilities),
        sha256=sha,
        published_at=datetime.now(timezone.utc).isoformat(),
        size_bytes=len(archive_bytes),
    )

    return metadata, archive_bytes


def unpack(data: bytes, target_dir: Path | str) -> AgentContract:
    """Unpack a tar.gz archive into a target directory.

    Validates that:
    - No paths escape the target directory (path traversal protection)
    - Archive size is within limits
    - The unpacked directory contains a valid agent.yaml

    Returns:
        The parsed AgentContract from the unpacked agent.yaml.

    Raises:
        PackageError: If extraction fails or contract is invalid.
    """
    target_dir = Path(target_dir)

    if len(data) > MAX_PACKAGE_BYTES:
        raise PackageError(
            f"Package exceeds {MAX_PACKAGE_BYTES // (1024*1024)}MB limit"
        )

    buf = io.BytesIO(data)
    try:
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            # Security: check all paths before extraction
            for member in tar.getmembers():
                member_path = (target_dir / member.name).resolve()
                if not str(member_path).startswith(str(target_dir.resolve())):
                    raise PackageError(
                        f"Path traversal detected in archive: {member.name}"
                    )
            tar.extractall(target_dir, filter="data")
    except tarfile.TarError as e:
        raise PackageError(f"Failed to extract archive: {e}") from e

    yaml_path = target_dir / "agent.yaml"
    if not yaml_path.exists():
        raise PackageError("Archive does not contain agent.yaml")

    try:
        return load_contract(yaml_path)
    except ContractError as e:
        raise PackageError(f"Invalid agent contract in archive: {e}") from e
