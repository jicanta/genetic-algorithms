"""
Real-time pygame visualizer for Exercise 1: ASCII Art GA

Layout:
┌──────────────────────────────────────────────────┐
│  TARGET IMAGE  │     ASCII ART      │  FITNESS   │
│  (original,    │  (actual chars,    │   GRAPH    │
│   scaled)      │   current best)    │  + stats   │
└──────────────────────────────────────────────────┘

Press Q or close the window to stop early.
"""

import numpy as np
import pygame
from PIL import Image


PANEL_W  = 320
PANEL_H  = 360
SIDEBAR_W = 280
LABEL_H  = 22          # space above each panel for the label
TOTAL_W  = PANEL_W * 2 + SIDEBAR_W
TOTAL_H  = PANEL_H + LABEL_H + 30   # +30 for bottom hint bar

BG_COLOR    = (18, 18, 24)
TEXT_COLOR  = (220, 220, 220)
BEST_COLOR  = (80, 160, 255)
AVG_COLOR   = (255, 180, 60)
BORDER_COLOR = (50, 50, 65)
ACCENT      = (100, 220, 100)
ASCII_BG    = (10, 10, 15)
ASCII_FG    = (180, 230, 180)   # greenish tint for the ASCII chars


def pil_to_surface(pil_img: Image.Image, size: tuple) -> pygame.Surface:
    pil_img = pil_img.resize(size, Image.LANCZOS).convert("RGB")
    return pygame.image.fromstring(pil_img.tobytes(), size, "RGB")


class AsciiGAVisualizer:
    def __init__(self, target_pil: Image.Image, grid_n: int, title: str = "ASCII Art GA"):
        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((TOTAL_W, TOTAL_H))
        self.clock  = pygame.time.Clock()
        self.grid_n = grid_n

        self.font_label = pygame.font.SysFont("monospace", 12, bold=True)
        self.font_stat  = pygame.font.SysFont("monospace", 12)
        self.font_hint  = pygame.font.SysFont("monospace", 10)

        # Pick the largest monospace font size where grid_n chars fit in the panel
        self.char_font, self.char_w, self.char_h = self._fit_font(grid_n)

        # Pre-render target image (fixed throughout the run)
        self.target_surf = pil_to_surface(target_pil, (PANEL_W, PANEL_H))

        self.best_fitnesses: list[float] = []
        self.avg_fitnesses:  list[float] = []
        self.running = True

    # ── Font sizing ───────────────────────────────────────────────

    def _fit_font(self, grid_n: int):
        """Find the biggest monospace font size where NxN chars fit in PANEL_W x PANEL_H."""
        for size in range(20, 3, -1):
            f = pygame.font.SysFont("monospace", size)
            cw, ch = f.size("@")
            if cw * grid_n <= PANEL_W and ch * grid_n <= PANEL_H:
                return f, cw, ch
        f = pygame.font.SysFont("monospace", 4)
        cw, ch = f.size("@")
        return f, cw, ch

    # ── Drawing helpers ───────────────────────────────────────────

    def _label(self, x: int, y: int, text: str):
        surf = self.font_label.render(text, True, TEXT_COLOR)
        self.screen.blit(surf, (x + 6, y + 4))
        pygame.draw.rect(self.screen, BORDER_COLOR, (x, y, PANEL_W, PANEL_H + LABEL_H), 1)

    def _draw_ascii_panel(self, x: int, y: int, genome: np.ndarray):
        """Render the ASCII grid as actual characters inside the panel."""
        from src.ex1.renderer import CHARSET
        pygame.draw.rect(self.screen, ASCII_BG, (x, y, PANEL_W, PANEL_H))

        grid = genome.reshape(self.grid_n, self.grid_n)

        # Center the text block inside the panel
        block_w = self.char_w * self.grid_n
        block_h = self.char_h * self.grid_n
        x_off = x + (PANEL_W - block_w) // 2
        y_off = y + (PANEL_H - block_h) // 2

        for row in range(self.grid_n):
            for col in range(self.grid_n):
                ch = CHARSET[grid[row, col]]
                surf = self.char_font.render(ch, True, ASCII_FG)
                self.screen.blit(surf, (x_off + col * self.char_w,
                                        y_off + row * self.char_h))

    def _draw_fitness_graph(self, x: int, y: int, w: int, h: int):
        pygame.draw.rect(self.screen, (22, 22, 32), (x, y, w, h))
        pygame.draw.rect(self.screen, BORDER_COLOR, (x, y, w, h), 1)
        self.screen.blit(self.font_label.render("Fitness", True, TEXT_COLOR), (x + 6, y + 4))

        if len(self.best_fitnesses) < 2:
            return

        all_v = self.best_fitnesses + self.avg_fitnesses
        mn, mx = min(all_v), max(all_v)
        pad = (mx - mn) * 0.05 or 0.005
        mn -= pad; mx += pad

        max_pts = w - 20
        start = max(0, len(self.best_fitnesses) - max_pts)
        bests = self.best_fitnesses[start:]
        avgs  = self.avg_fitnesses[start:]

        def to_px(v):
            return y + h - int((v - mn) / (mx - mn) * (h - 30)) - 10

        def draw_line(vals, color):
            n = len(vals)
            if n < 2:
                return
            pts = [(x + 10 + int(i * (w - 20) / (n - 1)), to_px(v)) for i, v in enumerate(vals)]
            pygame.draw.lines(self.screen, color, False, pts, 2)

        draw_line(avgs, AVG_COLOR)
        draw_line(bests, BEST_COLOR)

        pygame.draw.line(self.screen, BEST_COLOR, (x + w - 85, y + 14), (x + w - 65, y + 14), 2)
        self.screen.blit(self.font_hint.render("best", True, BEST_COLOR), (x + w - 62, y + 9))
        pygame.draw.line(self.screen, AVG_COLOR,  (x + w - 85, y + 28), (x + w - 65, y + 28), 2)
        self.screen.blit(self.font_hint.render("avg",  True, AVG_COLOR),  (x + w - 62, y + 23))

    def _draw_stats(self, x: int, y: int, gen: int, best_f: float, avg_f: float):
        rows = [
            ("Generation",   str(gen)),
            ("Best fitness", f"{best_f:.5f}"),
            ("Avg fitness",  f"{avg_f:.5f}"),
            ("Delta",        f"+{best_f - self.best_fitnesses[0]:.5f}" if len(self.best_fitnesses) > 1 else "—"),
        ]
        for i, (lbl, val) in enumerate(rows):
            self.screen.blit(self.font_stat.render(lbl + ":", True, (140, 140, 160)), (x + 8, y + i * 24))
            self.screen.blit(self.font_stat.render(val,        True, ACCENT),          (x + 145, y + i * 24))

    # ── Public API ────────────────────────────────────────────────

    def update(self, gen: int, genome: np.ndarray, best_fitness: float, avg_fitness: float) -> bool:
        """
        Redraw the window with the current best genome.
        Returns False if the user closed the window (signal GA to stop).
        """
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

        top = LABEL_H  # panels start below label row

        # ── Left panel: original image ──
        self.screen.blit(self.target_surf, (0, top))
        self._label(0, 0, "Target Image")

        # ── Right panel: ASCII art as actual characters ──
        self._draw_ascii_panel(PANEL_W, top, genome)
        self._label(PANEL_W, 0, f"ASCII Art  (gen {gen})")

        # ── Sidebar: stats + graph ──
        sx = PANEL_W * 2
        pygame.draw.rect(self.screen, BG_COLOR, (sx, 0, SIDEBAR_W, TOTAL_H))
        self._draw_stats(sx, top + 10, gen, best_fitness, avg_fitness)
        graph_y = top + 120
        self._draw_fitness_graph(sx, graph_y, SIDEBAR_W, TOTAL_H - graph_y - 30)

        # ── Bottom hint ──
        hint = self.font_hint.render("Q  close early", True, (70, 70, 90))
        self.screen.blit(hint, (10, TOTAL_H - 18))

        pygame.display.flip()
        self.clock.tick(60)
        return True

    def close(self):
        pygame.quit()
