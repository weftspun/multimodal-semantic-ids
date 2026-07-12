"""Headless mesh -> image rendering via **viser** (three.js / WebGL), captured through a **Playwright**
Chromium client — the NVIDIA kimodo-style approach the team picked over OpenGL/OSMesa/nvdiffrast.

Why this stack: three.js renders the scene *client-side* in WebGL, so ``ClientHandle.get_render`` needs a
connected browser. We connect a headless Chromium (SwiftShader = software WebGL, no GPU, no system GL libs)
so the whole path is FOSS-clean (viser Apache-2.0, Playwright Apache-2.0, SwiftShader Apache-2.0) and runs
anywhere. One server + one browser render the whole corpus; the mesh is swapped per asset.
"""

from __future__ import annotations

import time

import numpy as np
import trimesh

# --- Render constants (no magic numbers) --------------------------------------------------------------
RENDER_SIZE = 512                # square output; <= browser viewport
CAMERA_DISTANCE_FACTOR = 2.4     # camera distance as a multiple of the mesh's bounding radius
CAMERA_DIRECTION = (1.0, -1.0, 0.7)  # canonical 3/4 view (right / front / slightly-above)
CLIENT_CONNECT_TIMEOUT_S = 30.0  # how long to wait for the headless browser to attach
CLIENT_POLL_INTERVAL_S = 0.1
GLB_LOAD_DEBOUNCE_S = 2.0        # debounce: wait out the async three.js GLB download+decode before capture

_HOST = "127.0.0.1"
# SwiftShader gives software WebGL in headless Chromium (no GPU needed); the blocklist override forces it.
_CHROMIUM_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--ignore-gpu-blocklist", "--headless=new"]


def _frame_camera(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Return (camera_position, look_at_target) that frames ``mesh`` from the canonical 3/4 view."""
    center = mesh.bounds.mean(axis=0)
    radius = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])) / 2.0

    # Guard against a degenerate (zero-extent) mesh so we never divide by zero.
    radius = radius if radius > 1e-6 else 1.0

    direction = np.asarray(CAMERA_DIRECTION, dtype=float)
    direction /= np.linalg.norm(direction)

    position = center + direction * radius * CAMERA_DISTANCE_FACTOR
    return position, center


class MeshRenderer:
    """Render trimesh geometry to RGB images through a headless viser+Chromium session.

    Use as a context manager so the viser server and the browser are always torn down::

        with MeshRenderer() as r:
            rgb = r.render(trimesh.load("player.glb"))  # (H, W, 3) uint8
    """

    def __init__(self, size: int = RENDER_SIZE, port: int = 8080):
        self._size = size
        self._port = port
        self._server = None
        self._playwright = None
        self._browser = None
        self._client = None

    def __enter__(self) -> "MeshRenderer":
        import viser
        from playwright.sync_api import sync_playwright

        # Start the viser server (its own background threads/event loop).
        self._server = viser.ViserServer(host=_HOST, port=self._port)
        self._server.scene.set_up_direction("+y")  # glTF assets are Y-up

        # Launch a headless Chromium that connects as a render client.
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(args=_CHROMIUM_ARGS)
        page = self._browser.new_page(viewport={"width": self._size, "height": self._size})
        page.goto(f"http://{_HOST}:{self._port}")

        self._client = self._wait_for_client()
        return self

    def __exit__(self, *exc) -> None:
        # Tear everything down in reverse order; ignore shutdown races.
        for close in (
            lambda: self._browser and self._browser.close(),
            lambda: self._playwright and self._playwright.stop(),
            lambda: self._server and self._server.stop(),
        ):
            try:
                close()
            except Exception:
                pass

    def _wait_for_client(self):
        """Block until the headless browser has attached, or raise on timeout."""
        deadline = time.monotonic() + CLIENT_CONNECT_TIMEOUT_S
        while time.monotonic() < deadline:
            clients = self._server.get_clients()
            if clients:
                return next(iter(clients.values()))
            time.sleep(CLIENT_POLL_INTERVAL_S)
        raise TimeoutError("no viser client connected within the timeout")

    def render(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Render one mesh from the canonical view; return an (H, W, 3) uint8 RGB array."""
        # Swap the scene contents for this asset only.
        self._server.scene.reset()
        self._server.scene.add_mesh_trimesh("asset", mesh)

        # Aim the client camera so the mesh fills the frame (position first, then orient at the target).
        position, target = _frame_camera(mesh)
        self._client.camera.position = tuple(position)
        self._client.camera.look_at = tuple(target)

        # Debounce the async GLB load: flush the camera/scene messages, then wait out the browser's
        # download + decode before we grab the frame (get_render captures whatever is on screen NOW).
        self._client.flush()
        time.sleep(GLB_LOAD_DEBOUNCE_S)

        # Request a lossless frame from the browser, then drop the alpha channel.
        rgba = self._client.get_render(self._size, self._size, transport_format="png")
        return rgba[..., :3]
