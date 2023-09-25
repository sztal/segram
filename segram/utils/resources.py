"""Utilities for accessing package resources."""
from __future__ import annotations
from typing import Any, Literal, Iterator, Iterable, Mapping
from pathlib import Path
import os
import gzip
import bz2
import lzma
import json
from types import MappingProxyType
from importlib.resources import files


class Resource:
    """Resource handler class.

    Attributes
    ----------
    path
        Path to the resource file.
    compression
        Compression being used.
        Deduced from the filename extension
        if not specified directly.
    """
    __compressions__ = MappingProxyType({
        ".gz": gzip,
        ".bz2": bz2,
        ".xz": lzma
    })

    def __init__(
        self,
        path: str | bytes | os.PathLike,
        *,
        compression: Literal[*__compressions__] | None = None
    ) -> None:
        self.path = Path(path)
        self._compression = compression

    # Properties --------------------------------------------------------------

    @property
    def compression(self) -> str:
        if (comp := self._compression) is None:
            suffix = Path(self.path).suffix
            if suffix in self.__compressions__:
                return suffix
        return comp

    # Methods -----------------------------------------------------------------

    @classmethod
    def from_package(cls, package: str, filename: str) -> Resource:
        """Construct from a package/filename specification.

        Parameters
        ----------
        package
            Package/module name using standard dot notation.
        filename
            Name of the resource file.
            Compression extension can be omitted.
            However, if it is specified then only files
            with a given extension are considered.
        """
        path = cls.get_path(package, filename)
        return cls(path)

    @staticmethod
    def get_path(
        package: str,
        filename: str | bytes | os.PathLike
    ) -> os.PathLike:
        """Get file path from package and file names."""
        paths = []
        for path in files(package).iterdir():
            if str(filename) in (path.stem, path.parts[-1]):
                paths.append(path)
        if not paths:
            raise FileNotFoundError("no matching resources found")
        if len(paths) > 1:
            raise FileExistsError(f"multiple matching resources: {paths}")
        return paths.pop()

    def open(
        self,
        path: os.PathLike,
        mode: str,
        **kwds: Any
    ) -> Any:
        """Open resource file.

        Parameters
        ----------
        path
            File path.
        mode
            File opening mode.
        **kwds
            Other keyword arguments passed to an appropriate
            ``open`` function depending on the compression method.
        """
        if self.compression not in self.__compressions__:
            _open = open
            kwds = { "encoding": "utf-8", **kwds }
        else:
            if "b" not in mode:
                mode += "b"
            _open = self.__compressions__[self.compression].open
        return _open(str(path), mode, **kwds)

    def get(self, **kwds: Any) -> str | bytes:
        """Get resource data.

        Parameters
        ----------
        **kwds
            Passed to :func:`open`, :func:`gzip.open`,
            :func:`bz2.open` or :func:`lzma.open` depending
            on file compression.
        """
        with self.open(self.path, "r", **kwds) as fh:
            return fh.read()

    def write(
        self,
        data: str | bytes,
        mode: str = "x",
        **kwds: Any
    ) -> None:
        """Write resource."""
        if isinstance(data, str) and not data.endswith("\n"):
            data += "\n"
        if self.compression:
            data = data.encode()
        with self.open(self.path, mode, **kwds) as fh:
            fh.write(data)


class JSONResource(Resource):
    """JSON resource handler class.

    See Also
    --------
    Resource : base resource accessor class.
    """
    def get(
        self,
        *,
        json_kws: Mapping | None = None,
        **kwds: Any
    ) -> dict | list:
        """Get JSON resource.

        Parameters
        ----------
        json_kws
            Dictionary with keyword arguments passed to
            :func:`json.loads`.
        **kwds
            Passed to :meth:`Resource.get`.
        """
        json_kws = { "object_hook": self.obj_hook, **(json_kws or {}) }
        return json.loads(super().get(**kwds), **json_kws)

    def write(
        self,
        data: str | list | dict,
        mode: str = "w",
        json_kws: Mapping | None = None,
        **kwds: Any
    ) -> None:
        """Write JSON resource."""
        json_kws = json_kws or {}
        data = json.dumps(data, **json_kws)
        super().write(data, mode, **kwds)

    @staticmethod
    def obj_hook(obj: Mapping) -> dict:
        """Custom deserialization of JSON objects."""
        return {
            (int(k) if k.isdigit() else k): v
            for k, v in obj.items()
        }


class JSONLinesResource(JSONResource):
    """JSON lines resource handler class.

    See also
    --------
    JSONResource : JSON resource handler.
    """
    def iter(
        self,
        *,
        json_kws: Mapping | None,
        **kwds: Any
    ) -> Iterator[str | list | dict]:
        """Get resource data line by line.

        Parameters
        ----------
        json_kws
            Dictionary with keyword arguments passed to
            :func:`json.loads`.
        **kwds
            Passed to :meth:`Resource.get`.
        """
        json_kws = { "object_hook": self.obj_hook, **(json_kws or {}) }
        with self.open(self.path, "r", **kwds) as fh:
            for line in fh:
                if self.compression:
                    line = line.decode()
                yield json.loads(line.strip())

    def get(
        self,
        *,
        json_kws: Mapping | None,
        **kwds: Any
    ) -> Iterator[str | list | dict]:
        """Get resource data line by line.

        Parameters
        ----------
        json_kws
            Dictionary with keyword arguments passed to
            :func:`json.loads`.
        **kwds
            Passed to :meth:`Resource.get`.
        """
        return list(self.iter(json_kws=json_kws, **kwds))

    def write(
        self,
        data: Iterable[str | list | dict],
        mode: str = "x",
        json_kws: Mapping | None = None,
        **kwds: Any
    ) -> None:
        """Write JSON resource."""
        json_kws = json_kws or {}
        with self.open(self.path, mode, **kwds) as fh:
            for record in data:
                line = json.dumps(record, **json_kws).strip()+"\n"
                if self.compression:
                    line = line.encode()
                fh.write(line)
