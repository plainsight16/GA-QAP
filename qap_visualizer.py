
# qap_visualizer.py
#
# Turtle-based visualizer for the Quadratic Assignment Problem (QAP).
# Drop this file next to your GA .py files and import QAPVisualizer.
#
# Example usage (inside your GA loop):
# ------------------------------------------------------------
# from qap_visualizer import QAPVisualizer
# viz = QAPVisualizer(draw_every=5)  # draw every 5 generations
# ...
# for gen in range(iters):
#     # ... evolve population, compute best_cost, best_perm ...
#     viz.draw_generation(best_perm, gen, best_cost, F, D)
# viz.hold()  # keep the window open at the end
# ------------------------------------------------------------

import math
import sys
import turtle
import time
from typing import Dict, Tuple, Iterable

import numpy as np


class QAPVisualizer:
    """
    Visualizes a QAP chromosome (permutation) on a circle using turtle graphics.

    - Nodes are placed evenly around a circle.
    - Each node is labeled with the facility assigned to that location.
    - Each pair of nodes is connected with a line:
        * Pen thickness ~ flow between the two facilities.
        * Color (green -> red) ~ distance between the two locations.

    Parameters
    ----------
    show_turtle : bool
        If True, shows the turtle cursor.
    point_radius : int
        Radius of each node circle.
    font : tuple[str, int, str]
        Font used for captions and node labels.
    distance_between_points : int
        Chord length between adjacent nodes on the circle.
    caption_margin : int
        Margin for the text captions at the top-left.
    draw_every : int
        Only draw every Nth generation (to speed up very long runs).
    flow_pen_scale : float
        Multiplier to scale line thickness derived from flow values.
    """

    def __init__(
        self,
        show_turtle: bool = False,
        point_radius: int = 18,
        font: Tuple[str, int, str] = ("Arial", 14, "normal"),
        distance_between_points: int = 100,
        caption_margin: int = 50,
        draw_every: int = 1,
        flow_pen_scale: float = 1.5,
        sleep_seconds: float = 0.15,
    ) -> None:
        self.point_radius = point_radius
        self.font = font
        self.distance_between_points = distance_between_points
        self.caption_margin = caption_margin
        self.draw_every = max(1, int(draw_every))
        self.flow_pen_scale = float(flow_pen_scale)
        self.sleep_seconds = float(sleep_seconds)

        # Turtle setup
        self.t = turtle.Turtle(shape="classic")
        self.screen = self.t.screen
        self.screen.tracer(0, 0)  # manual redraws
        self.screen.colormode(255)
        self.t.speed("fastest")
        if not show_turtle:
            self.t.hideturtle()

        # Re-created each frame
        self._coords: Dict[int, Tuple[float, float]] = {}

        # Internal counter if the caller doesn't pass generation
        self._frame = 0

    # ---------------------------- Public API ---------------------------- #
    def draw_generation(
        self,
        chromosome: Iterable[int],
        generation_number: int,
        best_score: float,
        flow_matrix: np.ndarray,
        distance_matrix: np.ndarray,
    ) -> None:
        # Draw a single generation frame.")
        chromosome = list(chromosome)
        if (generation_number % self.draw_every) != 0:
            return

        self._frame += 1
        self._clear_and_reset()

        # Draw captions first
        self._draw_generation_info(generation_number, best_score, chromosome)

        # Draw nodes and remember the coordinates
        self._draw_points(chromosome)

        # Draw all pairwise connections
        self._draw_connections(chromosome, flow_matrix, distance_matrix)

        self.screen.update()
        if self.sleep_seconds > 0.0:
            time.sleep(self.sleep_seconds)

    def hold(self) -> None:
        #Keep the turtle window open (blocks)
        turtle.done()

    # --------------------------- Drawing helpers ----------------------- #
    def _clear_and_reset(self) -> None:
        self.t.clear()
        self.t.penup()
        self.t.home()
        self.t.pendown()
        self._coords.clear()

    def _move_to_circle_center(self, circle_radius: float, polygon_angle: float) -> None:
        self.t.penup()
        self.t.left(90)
        self.t.forward(circle_radius)
        self.t.right(90 - (polygon_angle / 2.0))
        self.t.pendown()

    def _draw_points(self, chromosome: Iterable[int]) -> None:
        chrom = list(chromosome)
        n = len(chrom)
        if n == 0:
            return

        polygon_angle = 360.0 / n
        # chord length c between adjacent points is distance_between_points
        # Radius R for a regular n-gon: c = 2 R sin(pi/n) => R = c / (2 sin(pi/n))
        circle_radius = self.distance_between_points / (2.0 * math.sin(math.pi / n))

        self._move_to_circle_center(circle_radius, polygon_angle)
        for idx in range(n):
            # Node
            self.t.circle(self.point_radius)
            self._coords[idx] = self.t.pos()

            # Label
            self.t.penup()
            self.t.left(90)
            self.t.forward(self.point_radius + 4)
            self.t.write(chrom[idx], font=self.font, align="center")
            self.t.backward(self.point_radius + 4)
            self.t.right(90)

            # Step to next vertex of the polygon
            self.t.right(polygon_angle)
            self.t.forward(self.distance_between_points)
            self.t.pendown()

    def _draw_connections(
        self,
        chromosome: Iterable[int],
        flow_matrix: np.ndarray,
        distance_matrix: np.ndarray,
    ) -> None:
        chrom = list(chromosome)
        n = len(chrom)
        if n == 0:
            return

        # Detect whether chromosome is 0-based or 1-based (facility labels)
        # If max == n => likely 1-based [1..n]; else assume 0-based [0..n-1].
        max_val = max(chrom)
        one_based = (max_val == n)

        # Precompute min/max distance (ignore zeros which often appear on diagonal)
        # If all distances are zero (degenerate), avoid division by zero.
        distances_no_zero = distance_matrix[distance_matrix > 0]
        if distances_no_zero.size == 0:
            min_d = 0.0
            max_d = 1.0
        else:
            min_d = float(np.min(distances_no_zero))
            max_d = float(np.max(distances_no_zero))
            if abs(max_d - min_d) < sys.float_info.epsilon:
                # Prevent division by zero in color mapping
                max_d = min_d + 1.0

        # Scale pen width gently so visuals stay readable
        max_flow = float(np.max(flow_matrix)) if flow_matrix.size else 1.0
        if max_flow < 1e-9:
            max_flow = 1.0

        for x in range(n):
            for y in range(n):
                # facilities at locations x and y
                fx = chrom[x] - 1 if one_based else chrom[x]
                fy = chrom[y] - 1 if one_based else chrom[y]

                flow = float(flow_matrix[fx, fy])
                if flow == 0.0:
                    continue

                dist = float(distance_matrix[x, y])
                pen_color = self._color_for_distance(dist, min_d, max_d)
                pen_size = self._pen_width_for_flow(flow, max_flow)
                self._draw_line(x, y, pen_color, pen_size)

    def _draw_line(self, from_idx: int, to_idx: int, pen_color=(0, 0, 0), pen_size: float = 1.0) -> None:
        p_from = self._coords[from_idx]
        p_to = self._coords[to_idx]
        self.t.penup()
        self.t.goto(p_from[0], p_from[1])
        self.t.pendown()
        self.t.pensize(pen_size)
        self.t.pencolor(pen_color)
        self.t.goto(p_to[0], p_to[1])
        self.t.pensize(1)

    def _draw_generation_info(self, generation_number: int, best_score: float, chromosome) -> None:
        start_pos = self.t.pos()
        self.t.penup()
        self.t.goto(-self.screen.window_width() / 2 + self.caption_margin,
                    self.screen.window_height() / 2 - self.caption_margin)
        self.t.write(f"Best chromosome: {list(chromosome)}", font=self.font)
        self.t.right(90)
        self.t.forward(24)
        self.t.write(f"Generation number: {generation_number}", font=self.font)
        self.t.forward(24)
        self.t.write(f"Best score: {best_score}", font=self.font)

        self.t.left(90)
        self.t.goto(start_pos[0], start_pos[1])
        self.t.pendown()

    # --------------------------- Color/width helpers -------------------- #
    def _color_for_distance(self, val: float, min_d: float, max_d: float) -> Tuple[int, int, int]:
        # Map distance to a green->red gradient
        return self._lerp_rgb((0, 255, 0), (255, 0, 0), self._normalize(val, min_d, max_d))

    def _pen_width_for_flow(self, flow_val: float, max_flow: float) -> float:
        # Smoothly map flow to a pen size in [1.0, 1.0 + 6*scale]
        # The multiplier self.flow_pen_scale controls the range.
        frac = flow_val / max_flow
        return 1.0 + 6.0 * self.flow_pen_scale * frac

    @staticmethod
    def _normalize(val: float, lo: float, hi: float) -> float:
        if hi - lo < sys.float_info.epsilon:
            return 0.0
        x = (val - lo) / (hi - lo)
        if x < 0.0: return 0.0
        if x > 1.0: return 1.0
        return x

    @staticmethod
    def _lerp_rgb(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
        r1, g1, b1 = c1; r2, g2, b2 = c2
        r = int(r1 + t * (r2 - r1))
        g = int(g1 + t * (g2 - g1))
        b = int(b1 + t * (b2 - b1))
        return (r, g, b)


# ----------------------------- Optional demo -----------------------------
if __name__ == "__main__":
    # Small self-test with random symmetric matrices
    import random

    def make_symmetric(n: int, max_val: int, zero_diag=True) -> np.ndarray:
        M = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i, n):
                if i == j and zero_diag:
                    v = 0
                else:
                    v = random.randint(0, max_val)
                M[i, j] = v
                M[j, i] = v
        return M

    n = 8
    F = make_symmetric(n, 7, zero_diag=True)
    D = make_symmetric(n, 100, zero_diag=True)
    # 1-based permutation to match many QAP instances (comment next line and uncomment 0-based to test)
    perm = list(range(1, n + 1))
    random.shuffle(perm)
    # 0-based sample:
    # perm = list(range(n)); random.shuffle(perm)

    viz = QAPVisualizer(draw_every=1, flow_pen_scale=1.2)
    viz.draw_generation(perm, generation_number=1, best_score=12345, flow_matrix=F, distance_matrix=D)
    viz.hold()
