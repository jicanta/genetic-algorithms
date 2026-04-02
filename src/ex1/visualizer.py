"""
Real-time pygame visualizer for Exercise 1: ASCII Art GA

Layout:
┌─────────────────────────────────────────────────┐
│  TARGET IMAGE  │  BEST INDIVIDUAL  │  FITNESS   │
│   (rendered)   │  (current best)   │   GRAPH    │
│                │                   │            │
│                │  Gen: 42          │            │
│                │  Best: 0.9612     │            │
│                │  Avg:  0.9340     │            │
└─────────────────────────────────────────────────┘

The fitness graph scrolls as generations accumulate.
Press Q or close the window to stop early.
"""

import numpy as np
import pygame
from PIL import Image


PANEL_W = 300          # width of each of the 3 panels
PANEL_H = 350          # height of image panels
GRAPH_H = 150          # height of fitness graph
SIDEBAR_W = 300        # sidebar for stats + graph
TOTAL_W = PANEL_W * 2 + SIDEBAR_W
TOTAL_H = PANEL_H + GRAPH_H + 40  # +40 for bottom bar

BG_COLOR = (18, 18, 24)
TEXT_COLOR = (220, 220, 220)
BEST_COLOR = (80, 160, 255)
AVG_COLOR = (255, 180, 60)
GRID_COLOR = (40, 40, 50)
ACCENT = (100, 220, 100)


def pil_to_surface(pil_img: Image.Image, size: tuple) -> pygame.Surface:
    """Convert a PIL Image to a pygame Surface, scaling to `size`."""
    pil_img = pil_img.resize(size, Image.NEAREST)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    raw = pil_img.tobytes()
    return pygame.image.fromstring(raw, size, "RGB")


class AsciiGAVisualizer:
    """
    Pygame window that shows the GA evolving in real time.

    Usage:
        vis = AsciiGAVisualizer(target_pil)
        vis.update(gen, best_genome, best_fitness, avg_fitness, grid_n, cell_size)
        # Returns False if user closed the window (stop signal)
        vis.close()
    """

    def __init__(self, target_pil: Image.Image, title: str = "ASCII Art GA"):
        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((TOTAL_W, TOTAL_H))
        self.clock = pygame.time.Clock()

        self.font_lg = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_sm = pygame.font.SysFont("monospace", 12)
        self.font_xs = pygame.font.SysFont("monospace", 10)

        # Pre-render the target image surface (stays fixed)
        self.target_surf = pil_to_surface(target_pil, (PANEL_W, PANEL_H))

        self.best_fitnesses: list[float] = []
        self.avg_fitnesses: list[float] = []
        self.running = True

    def _draw_panel_border(self, x: int, y: int, w: int, h: int, label: str):
        pygame.draw.rect(self.screen, GRID_COLOR, (x, y, w, h), 1)
        label_surf = self.font_sm.render(label, True, TEXT_COLOR)
        self.screen.blit(label_surf, (x + 6, y + 4))

    def _draw_fitness_graph(self, x: int, y: int, w: int, h: int):
        """Draw a scrolling fitness graph (best=blue, avg=orange)."""
        pygame.draw.rect(self.screen, (25, 25, 35), (x, y, w, h))
        self._draw_panel_border(x, y, w, h, "Fitness")

        if len(self.best_fitnesses) < 2:
            return

        def to_px(val, mn, mx):
            """Map a fitness value to a y-pixel (inverted: higher = higher on screen)."""
            if mx == mn:
                return y + h // 2
            return y + h - int((val - mn) / (mx - mn) * (h - 30)) - 10

        all_vals = self.best_fitnesses + self.avg_fitnesses
        mn, mx = min(all_vals), max(all_vals)
        # small padding so the line doesn't hug the edges
        pad = (mx - mn) * 0.05 or 0.01
        mn -= pad
        mx += pad

        n = len(self.best_fitnesses)
        # Only show the last `max_pts` points so graph doesn't get too dense
        max_pts = w - 20
        start = max(0, n - max_pts)
        bests = self.best_fitnesses[start:]
        avgs  = self.avg_fitnesses[start:]

        def draw_line(vals, color):
            pts = []
            for i, v in enumerate(vals):
                px = x + 10 + int(i * (w - 20) / max(len(vals) - 1, 1))
                py = to_px(v, mn, mx)
                pts.append((px, py))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, color, False, pts, 2)

        draw_line(avgs, AVG_COLOR)
        draw_line(bests, BEST_COLOR)

        # Legend
        pygame.draw.line(self.screen, BEST_COLOR, (x + w - 80, y + 12), (x + w - 60, y + 12), 2)
        self.screen.blit(self.font_xs.render("best", True, BEST_COLOR), (x + w - 57, y + 7))
        pygame.draw.line(self.screen, AVG_COLOR, (x + w - 80, y + 26), (x + w - 60, y + 26), 2)
        self.screen.blit(self.font_xs.render("avg", True, AVG_COLOR), (x + w - 57, y + 21))

    def _draw_stats(self, x: int, y: int, w: int, gen: int, best_f: float, avg_f: float):
        """Draw generation stats text."""
        lines = [
            ("Generation", f"{gen}"),
            ("Best fitness", f"{best_f:.5f}"),
            ("Avg fitness",  f"{avg_f:.5f}"),
            ("Improvement",  f"{best_f - self.avg_fitnesses[0]:.5f}" if self.avg_fitnesses else "—"),
        ]
        for i, (label, val) in enumerate(lines):
            lbl_surf = self.font_sm.render(label + ":", True, (150, 150, 170))
            val_surf = self.font_sm.render(val, True, ACCENT)
            self.screen.blit(lbl_surf, (x + 10, y + i * 24))
            self.screen.blit(val_surf, (x + 130, y + i * 24))

    def update(
        self,
        gen: int,
        best_rendered_pil: Image.Image,
        best_fitness: float,
        avg_fitness: float,
    ) -> bool:
        """
        Refresh the display with the current best individual.

        Returns True if the GA should continue, False if the user closed the window.
        """
        # Handle events (close button, Q key)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                self.running = False

        if not self.running:
            return False

        self.best_fitnesses.append(best_fitness)
        self.avg_fitnesses.append(avg_fitness)

        self.screen.fill(BG_COLOR)

        # ── Panel 1: Target image ──
        self.screen.blit(self.target_surf, (0, 30))
        self._draw_panel_border(0, 30, PANEL_W, PANEL_H, "Target")

        # ── Panel 2: Best individual rendered ──
        best_surf = pil_to_surface(best_rendered_pil, (PANEL_W, PANEL_H))
        self.screen.blit(best_surf, (PANEL_W, 30))
        self._draw_panel_border(PANEL_W, 30, PANEL_W, PANEL_H, f"Best (gen {gen})")

        # ── Sidebar: stats + graph ──
        sidebar_x = PANEL_W * 2
        self._draw_stats(sidebar_x, 40, SIDEBAR_W, gen, best_fitness, avg_fitness)
        self._draw_fitness_graph(sidebar_x, 150, SIDEBAR_W, PANEL_H - 120)

        # ── Bottom bar: hint ──
        hint = self.font_xs.render("Press Q to stop early", True, (80, 80, 100))
        self.screen.blit(hint, (10, TOTAL_H - 18))

        pygame.display.flip()
        self.clock.tick(60)
        return True

    def close(self):
        pygame.quit()
