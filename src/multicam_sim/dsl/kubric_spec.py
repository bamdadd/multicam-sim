"""Pure, Blender-free translation of a :class:`~multicam_sim.scene.Scene` into a
typed *Kubric spec*.

This module is the coordinate-math heart of the Kubric backend, and it is kept
**deliberately free of any ``kubric``/Blender import** so the conversion is
unit-testable on a plain CI box. :mod:`multicam_sim.dsl.kubric_backend` consumes
this spec to build the actual ``kb.Scene`` (which *does* need Blender).

Coordinate conversion (our OpenCV RDF, world Z-up -> Kubric/Blender camera)
--------------------------------------------------------------------------

Both our world and Blender's world are right-handed and Z-up, so world points
copy across unchanged. Only the **camera basis** differs:

* Ours (OpenCV RDF): the camera looks along **+Z**, with **+X right** and
  **+Y down**. ``R`` maps world -> camera, its rows are ``[right, down, forward]``
  in world coordinates, and the centre is ``C = -R^T @ t``.
* Blender/Kubric: the camera looks along **-Z**, with **+X right** and **+Y up**.

So the camera-to-world rotation Blender wants has columns ``[right, -down,
-forward]``::

    R_c2w = R.T @ diag(1, -1, -1)

which is exactly the OpenCV->OpenGL pose flip the tested pyrender backend already
uses (``render.py``: ``pose[:3, :3] = cam.rotation().T @ flip``,
``pose[:3, 3] = cam.centre()``). ``diag(1, -1, -1)`` has determinant ``+1`` here
because ``[right, down, forward]`` is right-handed (``right x down = forward``),
so ``R_c2w`` is a proper rotation and converts cleanly to a unit quaternion.

Intrinsics conversion (our ``fx/fy/cx/cy`` -> Kubric ``focal_length`` mm)
-------------------------------------------------------------------------

``kb.PerspectiveCamera`` is parameterised by ``focal_length`` and
``sensor_width`` (both in mm), and derives (verified against
``kubric/core/cameras.py``)::

    sensor_height = sensor_width * height / width
    f_x[px] = focal_length / sensor_width  * width
    f_y[px] = focal_length / sensor_height * height   ==  f_x[px]

so **Kubric can only represent square pixels** (``fx == fy``), with the principal
point pinned at the image centre (``shift_x``/``shift_y`` exist but its own source
notes they are "currently not supported"). We therefore *guard* both invariants
and raise a clear :class:`ValueError` for a camera Kubric cannot express (e.g. a
custom rig with ``fx != fy`` or an off-centre principal point). ``sensor_width``
is a free constant: it cancels in the ``f_x`` round-trip, so any positive value
gives the same pixels. We keep Kubric's default ``36.0`` mm and derive::

    focal_length = fx * sensor_width / width
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..cameras import Camera
from ..geometry import FloatArray

#: Kubric/Blender's default sensor width (mm). Cancels in the pixel round-trip
#: (``f_x = focal_length / sensor_width * width``), so the value is arbitrary; we
#: keep the default so a spec is easy to eyeball against Blender.
DEFAULT_SENSOR_WIDTH_MM: float = 36.0

#: OpenCV(RDF, +Z fwd/+Y down) -> Blender(RUB, -Z fwd/+Y up) camera-axis flip.
_RDF_TO_RUB: FloatArray = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

#: Relative tolerance for the ``fx == fy`` / centred-principal-point guards.
_INTRINSICS_RTOL: float = 1e-9


class KubricCameraSpec(BaseModel):
    """A Kubric ``PerspectiveCamera`` in the exact parameters Kubric consumes.

    ``position`` is the camera centre in world coordinates and ``quaternion`` is
    the camera-to-world rotation as ``(w, x, y, z)`` (Kubric's order, verified
    against ``kubric/core/objects.py``). ``focal_length``/``sensor_width`` are in
    mm; ``width``/``height`` are the image resolution in pixels.
    """

    model_config = ConfigDict(frozen=True)

    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    focal_length: float = Field(gt=0.0)
    sensor_width: float = Field(gt=0.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class KubricObjectSpec(BaseModel):
    """One renderable primitive: a sphere at ``position`` with an RGB ``color``.

    Each named point of an entity present at the frame becomes one sphere, so a
    1-point object and a 17-joint pose entity both lower without a schema fork.
    ``name`` is ``"<entity_id>/<point_name>"`` and ``color`` is a stable RGB in
    ``[0, 1]`` derived from the entity id.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    position: tuple[float, float, float]
    radius: float = Field(gt=0.0)
    color: tuple[float, float, float]


class KubricSceneSpec(BaseModel):
    """The full Blender-free translation of one ``(camera_id, frame)`` view."""

    model_config = ConfigDict(frozen=True)

    camera: KubricCameraSpec
    objects: list[KubricObjectSpec]
    frame: int


def _rotation_matrix_to_quaternion(rot: FloatArray) -> tuple[float, float, float, float]:
    """Proper 3x3 rotation -> unit quaternion ``(w, x, y, z)`` (Kubric's order).

    Shepperd's method (numerically stable across the four cases); the result is
    sign-normalised to ``w >= 0`` so the mapping is deterministic.
    """
    m = np.asarray(rot, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def camera_to_kubric_spec(
    camera: Camera, *, sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM
) -> KubricCameraSpec:
    """Translate one :class:`~multicam_sim.cameras.Camera` to a Kubric camera spec.

    Raises :class:`ValueError` if the camera cannot be represented by
    ``kb.PerspectiveCamera``: Kubric forces ``fx == fy`` and a centred principal
    point (see the module docstring). ``from_focal``/``from_fov`` cameras satisfy
    both; a general custom-rig camera may not, and is rejected loudly rather than
    rendered wrong.
    """
    intr = camera.intrinsics
    if not np.isclose(intr.fx, intr.fy, rtol=_INTRINSICS_RTOL):
        raise ValueError(
            "Kubric PerspectiveCamera requires square pixels (fx == fy); "
            f"got fx={intr.fx}, fy={intr.fy}. Use square-pixel intrinsics "
            "(Intrinsics.from_focal / from_fov without fov_y_deg)."
        )
    cx_centre = intr.width / 2.0
    cy_centre = intr.height / 2.0
    if not (
        np.isclose(intr.cx, cx_centre, rtol=_INTRINSICS_RTOL, atol=1e-9)
        and np.isclose(intr.cy, cy_centre, rtol=_INTRINSICS_RTOL, atol=1e-9)
    ):
        raise ValueError(
            "Kubric PerspectiveCamera requires a centred principal point "
            f"(cx == width/2, cy == height/2); got cx={intr.cx}, cy={intr.cy} "
            f"vs centre ({cx_centre}, {cy_centre}). Kubric does not support "
            "shift_x/shift_y."
        )

    # focal_length (mm) from fx (px): f_x = focal_length / sensor_width * width.
    focal_length = intr.fx * sensor_width_mm / intr.width

    # camera-to-world rotation Blender wants: R.T @ diag(1, -1, -1).
    rot_c2w = camera.rotation().T @ _RDF_TO_RUB
    quaternion = _rotation_matrix_to_quaternion(rot_c2w)
    centre = camera.centre()

    return KubricCameraSpec(
        position=(float(centre[0]), float(centre[1]), float(centre[2])),
        quaternion=quaternion,
        focal_length=focal_length,
        sensor_width=sensor_width_mm,
        width=intr.width,
        height=intr.height,
    )


def entity_color(entity_id: str) -> tuple[float, float, float]:
    """A stable RGB colour in ``[0, 1]`` derived from an entity id.

    Deterministic (hash-free, so it is stable across processes): the id's bytes
    are folded into three channels, then lifted away from black so every entity
    is visible against the default dark background.
    """
    acc = 2166136261
    r_ch = g_ch = b_ch = 0
    for i, byte in enumerate(entity_id.encode("utf-8")):
        acc = (acc ^ byte) * 16777619 & 0xFFFFFFFF
        channel = i % 3
        mixed = (acc >> 8) & 0xFF
        if channel == 0:
            r_ch ^= mixed
        elif channel == 1:
            g_ch ^= mixed
        else:
            b_ch ^= mixed
    # lift into [0.25, 1.0] so no entity renders as pure black.
    return (
        0.25 + 0.75 * (r_ch / 255.0),
        0.25 + 0.75 * (g_ch / 255.0),
        0.25 + 0.75 * (b_ch / 255.0),
    )


def scene_to_kubric_spec(
    scene: object,
    camera_id: int,
    frame: int,
    *,
    point_radius: float = 0.05,
    sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM,
) -> KubricSceneSpec:
    """Translate a :class:`~multicam_sim.scene.Scene` at ``(camera_id, frame)``.

    Pure and Blender-free: builds the camera spec (with the exact coordinate
    conversion above) and one sphere per entity named point present at ``frame``,
    coloured by the entity's stable id. Does **not** consult analytic
    ``in_view``/``visible`` — pixels never couple to the manifest.
    """
    from ..scene import Scene

    assert isinstance(scene, Scene)
    camera_spec = camera_to_kubric_spec(scene.cameras[camera_id], sensor_width_mm=sensor_width_mm)

    objects: list[KubricObjectSpec] = []
    for entity in scene.entities:
        match = next((f for f in entity.frames if f.frame == frame), None)
        if match is None:
            continue
        color = entity_color(entity.id)
        for point_name, xyz in match.points.items():
            objects.append(
                KubricObjectSpec(
                    name=f"{entity.id}/{point_name}",
                    position=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                    radius=point_radius,
                    color=color,
                )
            )

    return KubricSceneSpec(camera=camera_spec, objects=objects, frame=frame)
