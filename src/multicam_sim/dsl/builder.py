"""SceneBuilder: the fluent top that ties rig + motion + occlusion into a Scene.

Compiles the DSL down to the existing :class:`multicam_sim.scene.Scene` — the
contract type — so the manifest is produced by the unchanged
:func:`multicam_sim.manifest.build_manifest`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..activity import ActivitySegment, ActivityState, ActivityTimeline
from ..cameras import Camera
from ..entities import Entity, EntityFrame
from ..occluders import OccluderUnion
from ..possession import InteractionEvent, PossessionSegment, PossessionTimeline
from ..randomization import Background, Light, RandomizationRecord, RandomizationSpec
from ..scene import Scene
from .behavior import Behavior, PathBehavior
from .motion import LinearPath, PathUnion
from .occlusion import HandSweep, Occlusion

Vec3 = tuple[float, float, float]

#: What ``entity`` / ``distractor`` accept: a raw motion ``Path`` (wrapped in a
#: :class:`~multicam_sim.dsl.behavior.PathBehavior` for backward compatibility)
#: or any :class:`~multicam_sim.dsl.behavior.Behavior` policy. Open-closed: the
#: builder only ever calls ``.rollout(...)``.
Motion = PathUnion | Behavior


def _as_behavior(motion: Motion) -> Behavior:
    """Normalise a ``Path`` or a ``Behavior`` to a ``Behavior`` (open-closed)."""
    return motion if isinstance(motion, Behavior) else PathBehavior(motion)


@dataclass(frozen=True)
class _EntitySpec:
    id: str
    behavior: Behavior
    name: str
    edges: list[tuple[str, str]] | None
    seed: int


@dataclass(frozen=True)
class _AttachmentSpec:
    object_id: str
    holder_id: str
    start: int
    end: int
    offset: Vec3
    holder_point: str
    object_point: str


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
        self._distractors: list[_EntitySpec] = []
        self._occlusions: list[Occlusion] = []
        self._hand_sweeps: list[HandSweep] = []
        self._attachments: list[_AttachmentSpec] = []
        self._interactions: list[InteractionEvent] = []
        self._activity_segments: list[ActivitySegment] = []
        self._background: Background | None = None
        self._light: Light | None = None
        self._randomization: RandomizationRecord | None = None

    def cameras(self, cameras: list[Camera]) -> SceneBuilder:
        """Set the camera array (e.g. from :class:`multicam_sim.dsl.CameraRig`)."""
        self._cameras = list(cameras)
        return self

    def entity(
        self,
        id: str,
        path: Motion,
        *,
        name: str = "center",
        edges: list[tuple[str, str]] | None = None,
        seed: int = 0,
    ) -> SceneBuilder:
        """Add a moving entity whose motion is a compiled :class:`Path` or a
        :class:`~multicam_sim.dsl.behavior.Behavior` policy.

        ``seed`` is threaded into the behaviour's ``rollout`` so a stochastic
        behaviour (e.g. a jittered :class:`WaypointBehavior`) is deterministic and
        can be swept over >= 3 seeds; a plain path ignores it.
        """
        self._entities.append(
            _EntitySpec(id=id, behavior=_as_behavior(path), name=name, edges=edges, seed=seed)
        )
        return self

    def distractor(
        self,
        id: str,
        path: Motion,
        *,
        name: str = "center",
        seed: int = 0,
    ) -> SceneBuilder:
        """Add a non-target entity that appears in the manifest but never affects
        the visibility or ground truth of the primary entities."""
        self._distractors.append(
            _EntitySpec(id=id, behavior=_as_behavior(path), name=name, edges=None, seed=seed)
        )
        return self

    def randomize(self, spec: RandomizationSpec, *, seed: int = 0) -> SceneBuilder:
        """Sample ``spec`` under ``seed`` and apply the result to the scene.

        The builder does not own the randomness: :meth:`RandomizationSpec.sample`
        draws one concrete sample (deterministic for a given spec + seed), and
        this entry point applies it — the sampled ``background`` / ``light`` are
        stored on the built :class:`Scene`, each sampled distractor position is
        added as a static distractor through the existing :meth:`distractor`
        entry point (id ``rand_distractor_{i}``), and a
        :class:`~multicam_sim.randomization.RandomizationRecord` (spec + seed)
        rides on the scene as provenance. Calling it again re-samples and
        appends its distractors on top.
        """
        sample = spec.sample(seed)
        self._background = sample.background
        self._light = sample.light
        self._randomization = RandomizationRecord(spec=spec, seed=seed)
        for i, position in enumerate(sample.distractor_positions):
            self.distractor(f"rand_distractor_{i}", LinearPath(a=position, b=position))
        return self

    def occlude(self, occlusion: Occlusion) -> SceneBuilder:
        """Add a declarative occlusion pattern (compiled to real geometry)."""
        self._occlusions.append(occlusion)
        return self

    def occlude_hand(self, sweep: HandSweep) -> SceneBuilder:
        """Add a moving hand-proxy sweep (compiled to a HandOccluder)."""
        self._hand_sweeps.append(sweep)
        return self

    def attach(
        self,
        object_id: str,
        holder_id: str,
        start: int,
        end: int,
        *,
        offset: Vec3 = (0.0, 0.0, 0.0),
        holder_point: str = "center",
        object_point: str = "center",
    ) -> SceneBuilder:
        """Attach ``object_id`` to ``holder_id`` over ``[start, end)`` frames.

        At build time the object's ``object_point`` is overwritten by the
        holder's ``holder_point`` plus ``offset`` for every frame in the
        half-open interval; the object detaches at ``end`` and resumes the
        frames defined by its own path. A possession-GT sidecar recording the
        attached interval is baked into the returned :class:`Scene`.
        """
        if start < 0 or end > self.num_frames or end <= start:
            raise ValueError(
                f"invalid attachment window [{start}, {end}) for num_frames={self.num_frames}"
            )
        self._attachments.append(
            _AttachmentSpec(
                object_id=object_id,
                holder_id=holder_id,
                start=start,
                end=end,
                offset=offset,
                holder_point=holder_point,
                object_point=object_point,
            )
        )
        return self

    def handoff(
        self,
        object_id: str,
        giver_id: str,
        receiver_id: str,
        frame: int,
        *,
        start: int = 0,
        end: int | None = None,
        offset: Vec3 = (0.0, 0.0, 0.0),
        holder_point: str = "center",
        object_point: str = "center",
    ) -> SceneBuilder:
        """Stage a two-agent hand-off of ``object_id`` at ``frame``.

        ``giver_id`` carries the object over ``[start, frame)`` and
        ``receiver_id`` carries it over ``[frame, end)`` (``end`` defaults to
        ``num_frames``), so ``frame`` is the single frame at which the holder
        id changes. Geometry is baked exactly like :meth:`attach` — the object
        tracks its current holder plus ``offset`` on every held frame — and an
        :class:`~multicam_sim.possession.InteractionEvent` (frame, ``time =
        frame / fps``, giver, receiver, object) is recorded in the possession
        sidecar of the built :class:`Scene`.
        """
        end = self.num_frames if end is None else end
        if giver_id == receiver_id:
            raise ValueError("giver and receiver must be different entities")
        if not start < frame < end:
            raise ValueError(
                f"hand-off frame {frame} must satisfy start ({start}) < frame < end ({end})"
            )
        self.attach(
            object_id,
            giver_id,
            start,
            frame,
            offset=offset,
            holder_point=holder_point,
            object_point=object_point,
        )
        self.attach(
            object_id,
            receiver_id,
            frame,
            end,
            offset=offset,
            holder_point=holder_point,
            object_point=object_point,
        )
        self._interactions.append(
            InteractionEvent(
                frame=frame,
                time=frame / self.fps,
                giver_id=giver_id,
                receiver_id=receiver_id,
                object_id=object_id,
            )
        )
        return self

    def activity(
        self,
        entity_id: str,
        state: ActivityState,
        start: int,
        end: int,
    ) -> SceneBuilder:
        """Label ``entity_id`` as being in activity ``state`` over ``[start, end)``.

        Pure ground truth: it records an
        :class:`~multicam_sim.activity.ActivitySegment` in the activity GT
        sidecar of the built :class:`Scene` and never touches the entity's
        motion or geometry, so the byte-golden manifest is unchanged. Frames
        outside all of an entity's segments are unlabeled.
        """
        if start < 0 or end > self.num_frames or end <= start:
            raise ValueError(
                f"invalid activity window [{start}, {end}) for num_frames={self.num_frames}"
            )
        self._activity_segments.append(
            ActivitySegment(entity_id=entity_id, state=state, start_frame=start, end_frame=end)
        )
        return self

    def build(self) -> Scene:
        """Compile the DSL into a :class:`Scene` (cameras, entities, occluders)."""
        if not self._cameras:
            raise ValueError("no cameras; call .cameras(...)")
        if not self._entities:
            raise ValueError("no entities; call .entity(...)")

        specs = self._entities + self._distractors
        entities: list[Entity] = []
        frames_by_id: dict[str, list[EntityFrame]] = {}
        for spec in specs:
            frames = spec.behavior.rollout(
                self.fps, self.num_frames, seed=spec.seed, name=spec.name
            )
            frames_by_id[spec.id] = frames
            entities.append(Entity(id=spec.id, edges=spec.edges, frames=frames))

        possession_segments: list[PossessionSegment] = []
        for att in self._attachments:
            if att.object_id not in frames_by_id:
                raise ValueError(f"attachment references unknown object {att.object_id!r}")
            if att.holder_id not in frames_by_id:
                raise ValueError(f"attachment references unknown holder {att.holder_id!r}")

            object_frames = frames_by_id[att.object_id]
            holder_frames = frames_by_id[att.holder_id]
            holder_by_frame = {f.frame: f.points for f in holder_frames}

            for f in range(att.start, att.end):
                holder_points = holder_by_frame[f]
                if att.holder_point not in holder_points:
                    raise ValueError(
                        f"holder {att.holder_id!r} has no point {att.holder_point!r} at frame {f}"
                    )
                hx, hy, hz = holder_points[att.holder_point]
                ox, oy, oz = att.offset
                carried = [hx + ox, hy + oy, hz + oz]

                object_frame = object_frames[f]
                if att.object_point not in object_frame.points:
                    raise ValueError(
                        f"object {att.object_id!r} has no point {att.object_point!r} at frame {f}"
                    )
                object_frame.points[att.object_point] = carried

            possession_segments.append(
                PossessionSegment(
                    object_id=att.object_id,
                    holder_id=att.holder_id,
                    start_frame=att.start,
                    end_frame=att.end,
                )
            )

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
            occluders.append(
                occ.realize(
                    self._cameras,
                    frames_by_id[target_id],
                    fps=self.fps,
                    num_frames=self.num_frames,
                )
            )

        for sweep in self._hand_sweeps:
            if sweep.frames is not None and sweep.seconds is not None:
                raise ValueError("hand sweep has both frames and seconds windows; use one")
            if sweep.seconds is not None:
                t0, t1 = sweep.seconds
                f0 = int(round(t0 * self.fps))
                f1 = int(round(t1 * self.fps))
                sweep = sweep.model_copy(update={"frames": (f0, f1), "seconds": None})
            target_id = sweep.entity if sweep.entity is not None else self._entities[0].id
            if target_id not in frames_by_id:
                raise ValueError(f"hand sweep targets unknown entity {target_id!r}")
            occluders.append(
                sweep.realize(
                    self._cameras,
                    frames_by_id[target_id],
                    fps=self.fps,
                    num_frames=self.num_frames,
                )
            )

        possession = (
            PossessionTimeline(segments=possession_segments, events=self._interactions)
            if possession_segments
            else None
        )

        for seg in self._activity_segments:
            if seg.entity_id not in frames_by_id:
                raise ValueError(f"activity segment references unknown entity {seg.entity_id!r}")
        activity = (
            ActivityTimeline(segments=self._activity_segments) if self._activity_segments else None
        )

        return Scene(
            fps=self.fps,
            num_frames=self.num_frames,
            cameras=self._cameras,
            entities=entities,
            occluders=occluders,
            possession=possession,
            activity=activity,
            background=self._background,
            light=self._light,
            randomization=self._randomization,
        )
