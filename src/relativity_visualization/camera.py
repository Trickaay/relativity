"""
Orbit camera for the pygame-ce 3D renderer.

Mouse-drag rotates around a target point, the scroll wheel zooms. Pure
numpy math — no OpenGL, projection is done by hand for pygame's 2D surface.
"""

import math
import numpy as np

try:
    import pygame
except ImportError:  # camera math is usable without pygame installed
    pygame = None


class OrbitCamera:
    def __init__(self, target=(0.0, 0.0, 0.0), distance=12.0, yaw=0.0,
                 pitch=0.35, fov_deg=60.0, width=1000, height=700,
                 near=0.1):
        self.target = np.array(target, dtype=float)
        self.distance = distance
        self.yaw = yaw
        self.pitch = pitch
        self.fov_deg = fov_deg
        self.width = width
        self.height = height
        self.near = near

        self._dragging = False
        self._last_mouse = (0, 0)

        self.min_distance = 1.0
        self.max_distance = 200.0
        self.pitch_limit = math.radians(89.0)

    # ------------------------------------------------------------------
    # View basis
    # ------------------------------------------------------------------
    def position(self):
        cp = math.cos(self.pitch)
        x = self.distance * cp * math.sin(self.yaw)
        y = self.distance * math.sin(self.pitch)
        z = self.distance * cp * math.cos(self.yaw)
        return self.target + np.array([x, y, z])

    def basis(self):
        eye = self.position()
        forward = self.target - eye
        forward = forward / np.linalg.norm(forward)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        return eye, forward, right, up

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    def project(self, points):
        """Project (N, 3) world points to screen space.

        Returns (screen_xy (N, 2) float, cam_depth (N,) float, visible (N,) bool)
        """
        points = np.asarray(points, dtype=float)
        eye, forward, right, up = self.basis()
        rel = points - eye

        cx = rel @ right
        cy = rel @ up
        cz = rel @ forward

        visible = cz > self.near
        safe_cz = np.where(visible, cz, self.near)

        f = 1.0 / math.tan(math.radians(self.fov_deg) / 2.0)
        half_h = self.height / 2.0
        sx = self.width / 2.0 + (cx * f / safe_cz) * half_h
        sy = self.height / 2.0 - (cy * f / safe_cz) * half_h

        screen = np.stack([sx, sy], axis=1)
        return screen, cz, visible

    # ------------------------------------------------------------------
    # Unprojection (for click-to-place / drag in the scene editor)
    # ------------------------------------------------------------------
    def unproject_ray(self, screen_pos):
        """Inverse of project(): a screen pixel -> a world-space ray
        (origin, direction). Exact algebraic inverse of the cx/cy/cz ->
        sx/sy formulas in project() (solved for cx, cy at cz=1)."""
        eye, forward, right, up = self.basis()
        sx, sy = screen_pos
        half_h = self.height / 2.0
        f = 1.0 / math.tan(math.radians(self.fov_deg) / 2.0)

        cx = (sx - self.width / 2.0) / (f * half_h)
        cy = (self.height / 2.0 - sy) / (f * half_h)

        direction = cx * right + cy * up + forward
        direction = direction / np.linalg.norm(direction)
        return eye, direction

    def screen_to_ground(self, screen_pos, plane_y=0.0):
        """World-space point where the ray through this screen pixel hits
        the y=plane_y plane, or None if the ray is parallel to it or the
        intersection is behind the camera."""
        eye, direction = self.unproject_ray(screen_pos)
        if abs(direction[1]) < 1e-8:
            return None
        t = (plane_y - eye[1]) / direction[1]
        if t < 0:
            return None
        return eye + t * direction

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def handle_event(self, event):
        if pygame is None:
            return
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self._dragging = True
                self._last_mouse = event.pos
            elif event.button == 4:
                self.zoom(-1)
            elif event.button == 5:
                self.zoom(1)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self._dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self._dragging:
                dx = event.pos[0] - self._last_mouse[0]
                dy = event.pos[1] - self._last_mouse[1]
                self._last_mouse = event.pos
                self.orbit(dx, dy)
        elif event.type == pygame.MOUSEWHEEL:
            self.zoom(-event.y)

    def orbit(self, dx_pixels, dy_pixels, sensitivity=0.007):
        self.yaw += dx_pixels * sensitivity
        self.pitch += dy_pixels * sensitivity
        self.pitch = max(-self.pitch_limit, min(self.pitch_limit, self.pitch))

    def zoom(self, steps, factor=1.1):
        self.distance *= factor ** steps
        self.distance = max(self.min_distance, min(self.max_distance, self.distance))
