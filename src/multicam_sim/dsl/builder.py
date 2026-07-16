"""SceneBuilder: the fluent top that ties rig + motion + occlusion into a Scene.

Compiles the DSL down to the existing :class:`multicam_sim.scene.Scene` — the
contract type — so the manifest is produced by the unchanged
:func:`multicam_sim.manifest.build_manifest`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cameras import Camera
from ..entities import Entity, EntityFrame
from ..occluders import OccluderUnion
from ..scene import Scene
from .motion import PathUnion
from .occlusion import Occlusion


@dataclass(frozen=True)
class _EntitySpec:
    id: str
    path: PathUnion
    name: str
    edges: list[tuple[str, str]] | None


class SceneBuilder:
    """Fluent builder: add cameras, entities (as motion paths), and occlusions,
    then :meth:`build` a :class:`Scene`."""

    def __init__(self, fps: float, num_frames: int) -> None:
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        if fps <= 0.0:
            raise ValueError("fps must be > 0")
        self.fps = fps
        self.num_frames = num_frames
        self._cameras: list[Camera] = []
        self._entities: list[_EntitySpec] = []
        self._occlusions: list[Occlusion] = []

    def cameras(self, cameras: list[Camera]) -> SceneBuilder:
        """Set the camera array (e.g. from :class:`multicam_sim.dsl.CameraRig`)."""
        self._cameras = list(cameras)
        return self

    def entity(
        self,
        id: str,
        path: PathUnion,
        *,
        name: str = "center",
        edges: list[tuple[str, str]] | None = None,
    ) -> SceneBuilder:
        """Add a moving entity whose motion is a compiled :class:`Path`."""
        self._entities.append(_EntitySpec(id=id, path=path, name=name, edges=edges))
        return self

    def occlude(self, occlusion: Occlusion) -> SceneBuilder:
        """Add a declarative occlusion pattern (compiled to real geometry)."""
        self._occlusions.append(occlusion)
        return self

    def build(self) -> Scene:
        """Compile the DSL into a :class:`Scene` (cameras, entities, occluders)."""
        if not self._cameras:
            raise ValueError("no cameras; call .cameras(...)")
        if not self._entities:
            raise ValueError("no entities; call .entity(...)")

        entities: list[Entity] = []
        frames_by_id: dict[str, list[EntityFrame]] = {}
        for spec in self._entities:
            frames = spec.path.compile_frames(self.fps, self.num_frames, name=spec.name)
            frames_by_id[spec.id] = frames
            entities.append(Entity(id=spec.id, edges=spec.edges, frames=frames))

        occluders: list[OccluderUnion] = []
        for occ in self._occlusions:
            if occ.frames is not None and occ.seconds is not None:
                raise ValueError("occlusion has both frames and seconds windows; use one schedule")
            if occ.seconds is not None:
                t0, t1 = occ.seconds
                f0 = int(round(t0 * self.fps))
                f1 = int(round(t1 * self.fps))
                occ = occ.model_copy(update={"frames": (f0, f1), "seconds": None})
            target_id = occ.entity if occ.entity is not None else self._entities[0].id
            if target_id not in frames_by_id:
                raise ValueError(f"occlusion targets unknown entity {target_id!r}")
            occluders.append(occ.realize(self._cameras, frames_by_id[target_id]))

        return Scene(
            fps=self.fps,
            num_frames=self.num_frames,
            cameras=self._cameras,
            entities=entities,
            occluders=occluders,
        )
