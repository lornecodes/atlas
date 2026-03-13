"""HttpRegistryProvider — REST client for remote agent registries."""

from __future__ import annotations

from atlas.registry.provider import PackageMetadata


class HttpRegistryError(Exception):
    """Raised when an HTTP registry request fails."""


class HttpRegistryProvider:
    """HTTP-based registry provider delegating to a remote REST API.

    Endpoints:
        GET  {url}/search?q=...&limit=...
        GET  {url}/agents/{name}/versions
        GET  {url}/agents/{name}/{version}/metadata
        GET  {url}/agents/{name}/{version}/download
        POST {url}/agents/{name}/{version}
    """

    def __init__(self, url: str, *, auth_token: str | None = None) -> None:
        self._url = url.rstrip("/")
        self._headers: dict[str, str] = {}
        if auth_token:
            self._headers["Authorization"] = f"Bearer {auth_token}"

    async def search(
        self, query: str, *, limit: int = 20
    ) -> list[PackageMetadata]:
        import aiohttp

        params = {"q": query, "limit": str(limit)}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._url}/search", params=params, headers=self._headers
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return [PackageMetadata.from_dict(d) for d in data]
        except aiohttp.ClientError as e:
            raise HttpRegistryError(f"Registry search failed: {e}") from e

    async def list_versions(self, name: str) -> list[PackageMetadata]:
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._url}/agents/{name}/versions",
                    headers=self._headers,
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return [PackageMetadata.from_dict(d) for d in data]
        except aiohttp.ClientError as e:
            raise HttpRegistryError(
                f"Registry list_versions failed: {e}"
            ) from e

    async def get_metadata(
        self, name: str, version: str
    ) -> PackageMetadata | None:
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._url}/agents/{name}/{version}/metadata",
                    headers=self._headers,
                ) as resp:
                    if resp.status == 404:
                        return None
                    resp.raise_for_status()
                    data = await resp.json()
                    return PackageMetadata.from_dict(data)
        except aiohttp.ClientError as e:
            raise HttpRegistryError(
                f"Registry get_metadata failed: {e}"
            ) from e

    async def download(self, name: str, version: str) -> bytes | None:
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._url}/agents/{name}/{version}/download",
                    headers=self._headers,
                ) as resp:
                    if resp.status == 404:
                        return None
                    resp.raise_for_status()
                    return await resp.read()
        except aiohttp.ClientError as e:
            raise HttpRegistryError(
                f"Registry download failed: {e}"
            ) from e

    async def publish(self, metadata: PackageMetadata, data: bytes) -> bool:
        import aiohttp

        try:
            form = aiohttp.FormData()
            form.add_field("metadata", metadata.to_dict().__str__())
            form.add_field(
                "package",
                data,
                filename="package.tar.gz",
                content_type="application/gzip",
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._url}/agents/{metadata.name}/{metadata.version}",
                    data=form,
                    headers=self._headers,
                ) as resp:
                    resp.raise_for_status()
                    return resp.status == 200
        except aiohttp.ClientError as e:
            raise HttpRegistryError(
                f"Registry publish failed: {e}"
            ) from e
