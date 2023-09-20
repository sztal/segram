from __future__ import annotations
from typing import Any, ClassVar
from collections import ChainMap
from collections.abc import Iterable, Sequence
from .abc import Semantic, FrameABC
from ..nlp import DocABC


class Story(Sequence, Semantic):
    """Story semantic class."""
    alias: ClassVar[str] = "Story"
    __slots__ = ("_doc", "frames", "emap", "pmap")

    def __init__(
        self,
        doc: DocABC,
        frames: Iterable[FrameABC] = ()
    ) -> None:
        self._doc = doc
        self.frames = tuple(frames)
        self.emap = {}
        self.pmap = ChainMap()

    def __repr__(self) -> str:
        nframes = len(self)
        text = "frame" if nframes == 1 else "frames"
        return f"<{self.ppath()} with {nframes} {text} at {hex(id(self))}>"

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Story) -> bool:
        if (res := super().__eq__(other)) is NotImplemented:
            return res
        return res and self.frames == other.frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int | slice) -> FrameABC | tuple[FrameABC, ...]:
        return self.frames[idx]

    # Properties --------------------------------------------------------------

    @property
    def doc(self) -> DocABC:
        return self._doc

    @property
    def hashdata(self) -> tuple[Any, ...]:
        return (*super().hashdata, id(self))

    # Methods -----------------------------------------------------------------

    def copy(self, **kwds: Any) -> Story:
        return self.__class__(**{ "doc": self.doc, **self.data, **kwds })

    def is_comparable_with(self, other: Story) -> bool:
        return isinstance(other, Story)
