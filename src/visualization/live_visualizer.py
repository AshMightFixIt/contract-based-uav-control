"""
Live 3D drone visualizer using Pygame with manual perspective projection.

Usage:
    from visualization.live_visualizer import LiveVisualizer
    viz = LiveVisualizer(waypoints)
    for step in sim_loop:
        ...
        if not viz.update(state_dict):
            break
    viz.close()
"""

import math
import time
from collections import deque

import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


def ned_to_display(ned):
    """Convert NED coordinates to display coordinates (X-right, Y-up, Z-into-screen)."""
    return np.array([ned[0], -ned[2], ned[1]], dtype=float)


class OrbitalCamera:
    """Orbital camera with spherical coordinates and perspective projection."""

    def __init__(self, center=None, azimuth=45.0, elevation=30.0, distance=40.0, fov=60.0):
        self.center = np.array(center if center is not None else [0.0, 5.0, 0.0], dtype=float)
        self.azimuth = math.radians(azimuth)
        self.elevation = math.radians(elevation)
        self.distance = distance
        self.fov = math.radians(fov)
        self.near_clip = 0.5

    def rotate(self, daz, dele):
        self.azimuth += daz
        self.elevation = max(math.radians(5), min(math.radians(85), self.elevation + dele))

    def zoom(self, delta):
        self.distance = max(5.0, min(200.0, self.distance - delta * 2.0))

    def set_center(self, pos):
        self.center = np.array(pos, dtype=float)

    def _get_camera_position(self):
        cx = self.center[0] + self.distance * math.cos(self.elevation) * math.sin(self.azimuth)
        cy = self.center[1] + self.distance * math.sin(self.elevation)
        cz = self.center[2] + self.distance * math.cos(self.elevation) * math.cos(self.azimuth)
        return np.array([cx, cy, cz])

    def world_to_screen(self, point, screen_size):
        """Project a 3D world point to 2D screen coordinates.

        Returns (x, y) tuple or None if point is behind camera.
        """
        cam_pos = self._get_camera_position()

        # Look-at basis vectors
        forward = self.center - cam_pos
        forward_len = np.linalg.norm(forward)
        if forward_len < 1e-6:
            return None
        forward = forward / forward_len

        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right_len = np.linalg.norm(right)
        if right_len < 1e-6:
            return None
        right = right / right_len

        up = np.cross(right, forward)

        # Transform to camera space
        diff = np.array(point, dtype=float) - cam_pos
        x = np.dot(diff, right)
        y = np.dot(diff, up)
        z = np.dot(diff, forward)

        if z < self.near_clip:
            return None

        # Perspective divide
        f = 1.0 / math.tan(self.fov / 2.0)
        aspect = screen_size[0] / screen_size[1]

        sx = (x * f / (z * aspect)) * screen_size[0] / 2 + screen_size[0] / 2
        sy = -(y * f / z) * screen_size[1] / 2 + screen_size[1] / 2

        return (int(sx), int(sy))


class LiveVisualizer:
    """Live 3D drone visualization window.

    Call update(state_dict) every simulation step. Renders at target_fps,
    but always processes events and records trail data at full sim rate.
    """

    # Colors
    BG_COLOR = (20, 20, 30)
    GRID_COLOR = (50, 50, 60)
    AXIS_RED = (200, 60, 60)
    AXIS_GREEN = (60, 200, 60)
    WP_VISITED = (80, 200, 80)
    WP_UPCOMING = (220, 60, 60)
    WP_LINE = (100, 100, 120)
    ALTITUDE_LINE = (150, 150, 170)
    HUD_BG = (0, 0, 0, 180)
    HUD_TEXT = (220, 220, 230)
    CBF_OK = (80, 200, 80)
    CBF_ACTIVE = (255, 165, 0)
    DRONE_BODY = (220, 220, 240)
    VELOCITY_ARROW = (255, 255, 100)

    CONTROLLER_COLORS = {
        'PID': (80, 200, 80),
        'MPC': (80, 130, 255),
        'Hinf': (220, 60, 60),
    }

    def __init__(self, waypoints, window_size=(1280, 800), target_fps=30,
                 trail_length=3000, follow_drone=True):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for LiveVisualizer")

        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("UAV Live Visualizer")
        self.clock = pygame.time.Clock()
        self.window_size = window_size
        self.target_fps = target_fps
        self.follow_drone = follow_drone

        # Fonts
        self.font_large = pygame.font.SysFont("consolas", 18)
        self.font_small = pygame.font.SysFont("consolas", 14)
        self.font_label = pygame.font.SysFont("consolas", 12)

        # Waypoints in display coords
        self.waypoints = [ned_to_display(wp) for wp in waypoints]

        # Camera
        if len(self.waypoints) > 0:
            center = np.mean(self.waypoints, axis=0)
        else:
            center = np.array([0.0, 5.0, 0.0])
        self.camera = OrbitalCamera(center=center, distance=35.0)

        # Trail: stores (display_pos, controller_name)
        self.trail = deque(maxlen=trail_length)

        # Mouse state
        self._dragging = False
        self._last_mouse = (0, 0)

        # Frame timing
        self._last_render_time = 0.0
        self._min_render_interval = 1.0 / target_fps

        # Pulse animation for CBF
        self._cbf_pulse = 0.0

        self._running = True

    def update(self, state):
        """Process one simulation step. Returns False if window was closed."""
        if not self._running:
            return False

        # Always record trail
        pos_ned = state.get('position', np.zeros(3))
        pos_disp = ned_to_display(pos_ned)
        controller = state.get('active_controller', 'PID')
        self.trail.append((pos_disp.copy(), controller))

        # Always process events
        if not self._process_events():
            return False

        # Track drone
        if self.follow_drone:
            # Smooth follow
            target = pos_disp.copy()
            self.camera.center = 0.95 * self.camera.center + 0.05 * target

        # Frame-rate limit rendering
        now = time.monotonic()
        if now - self._last_render_time < self._min_render_interval:
            return True

        self._last_render_time = now
        self._render(state, pos_disp)
        return True

    def close(self):
        if self._running:
            self._running = False
            pygame.quit()

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self._dragging = True
                    self._last_mouse = event.pos
                elif event.button == 4:  # Scroll up
                    self.camera.zoom(1)
                elif event.button == 5:  # Scroll down
                    self.camera.zoom(-1)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self._dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self._dragging:
                    dx = event.pos[0] - self._last_mouse[0]
                    dy = event.pos[1] - self._last_mouse[1]
                    self.camera.rotate(-dx * 0.005, dy * 0.005)
                    self._last_mouse = event.pos
            elif event.type == pygame.MOUSEWHEEL:
                self.camera.zoom(event.y)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    self.follow_drone = not self.follow_drone
                elif event.key == pygame.K_ESCAPE:
                    self._running = False
                    return False
        return True

    def _render(self, state, drone_pos):
        self.screen.fill(self.BG_COLOR)

        wp_idx = state.get('waypoint_idx', 0)
        attitude = state.get('attitude', np.zeros(3))
        controller = state.get('active_controller', 'PID')
        cbf = state.get('cbf_intervened', False)
        velocity = state.get('velocity', np.zeros(3))

        self._draw_ground_grid()
        self._draw_waypoints(wp_idx)
        self._draw_trail()
        self._draw_altitude_line(drone_pos)
        self._draw_drone(drone_pos, attitude, controller, cbf, velocity)
        self._draw_hud(state)

        pygame.display.flip()

    # -- Drawing routines --

    def _draw_ground_grid(self):
        """Draw a ground plane grid with axis markers."""
        grid_extent = 50  # meters each direction
        grid_step = 5

        for i in range(-grid_extent, grid_extent + 1, grid_step):
            # Lines along X (North)
            p1 = self.camera.world_to_screen(np.array([i, 0, -grid_extent], dtype=float), self.window_size)
            p2 = self.camera.world_to_screen(np.array([i, 0, grid_extent], dtype=float), self.window_size)
            if p1 and p2:
                pygame.draw.line(self.screen, self.GRID_COLOR, p1, p2, 1)

            # Lines along Z (East in display)
            p1 = self.camera.world_to_screen(np.array([-grid_extent, 0, i], dtype=float), self.window_size)
            p2 = self.camera.world_to_screen(np.array([grid_extent, 0, i], dtype=float), self.window_size)
            if p1 and p2:
                pygame.draw.line(self.screen, self.GRID_COLOR, p1, p2, 1)

        # Axis markers at origin
        origin = np.array([0, 0, 0], dtype=float)
        o_screen = self.camera.world_to_screen(origin, self.window_size)
        nx = self.camera.world_to_screen(np.array([8, 0, 0], dtype=float), self.window_size)
        ny = self.camera.world_to_screen(np.array([0, 0, 8], dtype=float), self.window_size)
        if o_screen and nx:
            pygame.draw.line(self.screen, self.AXIS_RED, o_screen, nx, 2)
            label = self.font_label.render("N", True, self.AXIS_RED)
            self.screen.blit(label, (nx[0] + 3, nx[1] - 6))
        if o_screen and ny:
            pygame.draw.line(self.screen, self.AXIS_GREEN, o_screen, ny, 2)
            label = self.font_label.render("E", True, self.AXIS_GREEN)
            self.screen.blit(label, (ny[0] + 3, ny[1] - 6))

    def _draw_waypoints(self, current_wp_idx):
        """Draw waypoint markers, labels, altitude lines, and connecting path."""
        # Connecting path
        for i in range(len(self.waypoints) - 1):
            p1 = self.camera.world_to_screen(self.waypoints[i], self.window_size)
            p2 = self.camera.world_to_screen(self.waypoints[i + 1], self.window_size)
            if p1 and p2:
                pygame.draw.line(self.screen, self.WP_LINE, p1, p2, 1)

        for i, wp in enumerate(self.waypoints):
            screen_pos = self.camera.world_to_screen(wp, self.window_size)
            if screen_pos is None:
                continue

            # Vertical altitude reference line
            ground_pos = np.array([wp[0], 0, wp[2]])
            ground_screen = self.camera.world_to_screen(ground_pos, self.window_size)
            if ground_screen:
                pygame.draw.line(self.screen, self.ALTITUDE_LINE, screen_pos, ground_screen, 1)

            # Diamond marker
            color = self.WP_VISITED if i < current_wp_idx else self.WP_UPCOMING
            size = 8
            diamond = [
                (screen_pos[0], screen_pos[1] - size),
                (screen_pos[0] + size, screen_pos[1]),
                (screen_pos[0], screen_pos[1] + size),
                (screen_pos[0] - size, screen_pos[1]),
            ]
            pygame.draw.polygon(self.screen, color, diamond)
            pygame.draw.polygon(self.screen, (255, 255, 255), diamond, 1)

            # Label
            label = self.font_label.render(f"WP{i + 1}", True, (220, 220, 220))
            self.screen.blit(label, (screen_pos[0] + 10, screen_pos[1] - 8))

    def _draw_trail(self):
        """Draw flight trail colored by controller type."""
        if len(self.trail) < 2:
            return

        # Determine stride based on trail length
        n = len(self.trail)
        if n > 1500:
            stride = 3
        elif n > 500:
            stride = 2
        else:
            stride = 1

        # Group consecutive points by controller for batched drawing
        segments = {}  # controller -> list of point pairs
        prev_screen = None
        prev_ctrl = None

        for i in range(0, n, stride):
            pos, ctrl = self.trail[i]
            screen_pos = self.camera.world_to_screen(pos, self.window_size)

            if screen_pos is not None and prev_screen is not None and ctrl == prev_ctrl:
                if ctrl not in segments:
                    segments[ctrl] = []
                segments[ctrl].append((prev_screen, screen_pos))

            prev_screen = screen_pos
            prev_ctrl = ctrl

        # Draw each controller's segments
        for ctrl, pairs in segments.items():
            color = self.CONTROLLER_COLORS.get(ctrl, (180, 180, 180))
            for p1, p2 in pairs:
                pygame.draw.line(self.screen, color, p1, p2, 2)

    def _draw_altitude_line(self, pos):
        """Draw vertical dashed line from drone to ground plane."""
        screen_pos = self.camera.world_to_screen(pos, self.window_size)
        ground_pos = np.array([pos[0], 0, pos[2]])
        ground_screen = self.camera.world_to_screen(ground_pos, self.window_size)

        if screen_pos is None or ground_screen is None:
            return

        # Dashed line
        dx = ground_screen[0] - screen_pos[0]
        dy = ground_screen[1] - screen_pos[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 2:
            return

        dash_len = 6
        gap_len = 4
        num_segments = int(length / (dash_len + gap_len))

        for j in range(num_segments + 1):
            t1 = j * (dash_len + gap_len) / length
            t2 = min(1.0, (j * (dash_len + gap_len) + dash_len) / length)
            x1 = int(screen_pos[0] + dx * t1)
            y1 = int(screen_pos[1] + dy * t1)
            x2 = int(screen_pos[0] + dx * t2)
            y2 = int(screen_pos[1] + dy * t2)
            pygame.draw.line(self.screen, self.ALTITUDE_LINE, (x1, y1), (x2, y2), 1)

    def _draw_drone(self, pos, attitude, controller, cbf, velocity):
        """Draw a quadrotor shape at the drone's position."""
        screen_pos = self.camera.world_to_screen(pos, self.window_size)
        if screen_pos is None:
            return

        # Scale arm length based on apparent distance
        arm_len = 18
        body_radius = 6

        # Yaw rotation for arm orientation
        yaw = attitude[2] if len(attitude) > 2 else 0.0

        color = self.CONTROLLER_COLORS.get(controller, self.DRONE_BODY)

        # Draw 4 arms rotated by yaw
        for k in range(4):
            angle = yaw + k * math.pi / 2
            dx = int(arm_len * math.cos(angle))
            dy = int(arm_len * math.sin(angle))
            end = (screen_pos[0] + dx, screen_pos[1] + dy)
            pygame.draw.line(self.screen, self.DRONE_BODY, screen_pos, end, 2)
            # Motor circle at end
            pygame.draw.circle(self.screen, color, end, 4)

        # Center body
        pygame.draw.circle(self.screen, color, screen_pos, body_radius)
        pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, body_radius, 1)

        # Velocity arrow
        vel_disp = ned_to_display(velocity)
        vel_mag = np.linalg.norm(vel_disp)
        if vel_mag > 0.3:
            vel_dir = vel_disp / vel_mag
            # Project velocity direction into screen space (approximate)
            tip_world = pos + vel_dir * 3.0
            tip_screen = self.camera.world_to_screen(tip_world, self.window_size)
            if tip_screen:
                pygame.draw.line(self.screen, self.VELOCITY_ARROW, screen_pos, tip_screen, 2)

        # CBF pulsing ring
        if cbf:
            self._cbf_pulse = (self._cbf_pulse + 0.15) % (2 * math.pi)
            pulse_radius = int(body_radius + 12 + 4 * math.sin(self._cbf_pulse))
            pulse_alpha = int(180 + 75 * math.sin(self._cbf_pulse))
            cbf_color = (255, 165, 0)
            pygame.draw.circle(self.screen, cbf_color, screen_pos, pulse_radius, 2)

    def _draw_hud(self, state):
        """Draw heads-up display with flight information."""
        sim_time = state.get('time', 0.0)
        controller = state.get('active_controller', 'PID')
        flight_mode = state.get('flight_mode', 'track')
        wind_speed = state.get('wind_speed', 0.0)
        cbf = state.get('cbf_intervened', False)
        wp_idx = state.get('waypoint_idx', 0)
        pos_ned = state.get('position', np.zeros(3))
        altitude = -pos_ned[2]  # NED to altitude

        # Distance to current waypoint
        if wp_idx < len(self.waypoints):
            wp_disp = self.waypoints[wp_idx]
            drone_disp = ned_to_display(pos_ned)
            dist_wp = np.linalg.norm(wp_disp - drone_disp)
        else:
            dist_wp = 0.0

        # Top-left info panel
        panel_x, panel_y = 10, 10
        lines = [
            f"Controller: {controller}",
            f"Mode:       {flight_mode.upper()}",
            f"Wind:       {wind_speed:.1f} m/s",
            f"Altitude:   {altitude:.1f} m",
            f"Dist to WP: {dist_wp:.1f} m",
            f"Waypoint:   {wp_idx + 1}/{len(self.waypoints)}",
            f"Time:       {sim_time:.1f} s",
        ]

        # Background panel
        line_height = 20
        panel_w = 210
        panel_h = len(lines) * line_height + 12
        panel_surface = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 160))
        self.screen.blit(panel_surface, (panel_x, panel_y))

        for i, line in enumerate(lines):
            text_surf = self.font_small.render(line, True, self.HUD_TEXT)
            self.screen.blit(text_surf, (panel_x + 8, panel_y + 6 + i * line_height))

        # Top-right CBF indicator
        cbf_x = self.window_size[0] - 130
        cbf_y = 10
        cbf_w, cbf_h = 120, 30
        cbf_surface = pygame.Surface((cbf_w, cbf_h), pygame.SRCALPHA)
        cbf_surface.fill((0, 0, 0, 160))
        self.screen.blit(cbf_surface, (cbf_x, cbf_y))

        if cbf:
            cbf_text = self.font_small.render("CBF ACTIVE", True, self.CBF_ACTIVE)
            pygame.draw.circle(self.screen, self.CBF_ACTIVE, (cbf_x + 14, cbf_y + 15), 5)
        else:
            cbf_text = self.font_small.render("CBF OK", True, self.CBF_OK)
            pygame.draw.circle(self.screen, self.CBF_OK, (cbf_x + 14, cbf_y + 15), 5)
        self.screen.blit(cbf_text, (cbf_x + 24, cbf_y + 6))

        # Bottom-center controller legend strip
        legend_w = 300
        legend_h = 26
        legend_x = (self.window_size[0] - legend_w) // 2
        legend_y = self.window_size[1] - 36

        legend_surface = pygame.Surface((legend_w, legend_h), pygame.SRCALPHA)
        legend_surface.fill((0, 0, 0, 160))
        self.screen.blit(legend_surface, (legend_x, legend_y))

        items = [("PID", self.CONTROLLER_COLORS['PID']),
                 ("MPC", self.CONTROLLER_COLORS['MPC']),
                 ("H-inf", self.CONTROLLER_COLORS['Hinf'])]
        offset = 10
        for label, color in items:
            pygame.draw.rect(self.screen, color, (legend_x + offset, legend_y + 7, 14, 12))
            text_surf = self.font_label.render(label, True, self.HUD_TEXT)
            self.screen.blit(text_surf, (legend_x + offset + 18, legend_y + 6))
            offset += 95

        # Controls hint
        hint = self.font_label.render("Drag: orbit | Scroll: zoom | F: follow | Esc: exit", True, (120, 120, 140))
        self.screen.blit(hint, (10, self.window_size[1] - 20))
