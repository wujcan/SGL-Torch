"""
Reference: https://github.com/agile-geoscience/striplog/blob/master/striplog/hatches.py
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.hatch import Shapes
from matplotlib.path import Path
from matplotlib.lines import Line2D
import os

__all__ = ["colors", "markers", "patterns"]

markers = Line2D.filled_markers

colors = ["#5ad8a6", "#e8684a", "#5b8ff9", "#ff9d4d", "#6dc8ec",
          "#f6bd16", "#5d7092", "#ff99c3", "#269a99", "#9270ca"]

patterns = ['x', '.', '\\', "v", '/', "l", "!", "c", "=", "p"]


def use(mplstyle):  # TODO
    style_path = os.path.join(os.path.dirname(__file__), mplstyle, ".mplstyle")
    plt.style.use(style_path)


class CrossHatch(Shapes):
    """
    Attempt at USGS pattern 712
    """
    def __init__(self, hatch, density):
        verts = [
            (0.8, 0.8),
            (0.0, 0.0),
            (0.0, 0.8),
            (0.8, 0.0),
            ]

        codes = [Path.MOVETO,
                 Path.LINETO,
                 Path.MOVETO,
                 Path.LINETO,
                 ]

        path = Path(verts, codes, closed=False)

        self.shape_vertices = path.vertices
        self.shape_codes = path.codes
        self.num_rows = hatch.count('c') * density
        self.size = 0.5

        super().__init__(hatch, density)


class PlusHatch(Shapes):
    """
    Attempt at USGS pattern 721, 327
    """
    def __init__(self, hatch, density):
        verts = [
            (0.4, 0.8),
            (0.4, 0.0),
            (0.0, 0.4),
            (0.8, 0.4),
            ]

        codes = [Path.MOVETO,
                 Path.LINETO,
                 Path.MOVETO,
                 Path.LINETO,
                 ]

        path = Path(verts, codes, closed=False)

        self.shape_vertices = path.vertices
        self.shape_codes = path.codes
        self.num_rows = hatch.count('p') * density
        self.size = 0.5

        super().__init__(hatch, density)


class DashHatch(Shapes):
    """
    Attempt at USGS pattern 620
    """
    def __init__(self, hatch, density):
        verts = [
            (0., 0.),  # left, bottom
            (1., 0.),  # right, top
            ]

        codes = [Path.MOVETO,
                 Path.LINETO,
                 ]

        path = Path(verts, codes)

        self.shape_vertices = path.vertices
        self.shape_codes = path.codes
        self.num_rows = hatch.count('=') * density
        self.size = 0.5

        super().__init__(hatch, density)


class TickHatch(Shapes):
    """
    Attempt at USGS pattern 230
    """
    def __init__(self, hatch, density):
        verts = [
            (0.0, 0.0),
            (0.0, 1.0),
            ]

        codes = [Path.MOVETO,
                 Path.LINETO,
                 ]

        path = Path(verts, codes, closed=False)

        self.shape_vertices = path.vertices
        self.shape_codes = path.codes
        self.num_rows = hatch.count("!") * density
        self.size = 1.0

        super().__init__(hatch, density)


class EllHatch(Shapes):
    """
    Attempt at USGS pattern 412
    """
    def __init__(self, hatch, density):
        verts = [
            (0.0, 0.0),
            (0.0, 0.5),
            (0.0, 0.0),
            (0.5, 0.0),
            ]

        codes = [Path.MOVETO,
                 Path.LINETO,
                 Path.MOVETO,
                 Path.LINETO,
                 ]

        path = Path(verts, codes, closed=False)

        self.shape_vertices = path.vertices
        self.shape_codes = path.codes
        self.num_rows = hatch.count("l") * density
        self.size = 1.0

        super().__init__(hatch, density)


class VeeHatch(Shapes):
    """
    Attempt at USGS pattern 731
    """
    def __init__(self, hatch, density):
        verts = [
            (0.250, 0.0),
            (0., 0.5),
            (0.25, 0.),
            (0.5, 0.5),
            ]

        codes = [Path.MOVETO,
                 Path.LINETO,
                 Path.MOVETO,
                 Path.LINETO,
                 ]

        path = Path(verts, codes, closed=False)

        self.shape_vertices = path.vertices
        self.shape_codes = path.codes
        self.num_rows = hatch.count("v") * density
        self.size = 1.0

        super().__init__(hatch, density)


# Register custom hatches
mpl.hatch._hatch_types.append(CrossHatch)
mpl.hatch._hatch_types.append(PlusHatch)
mpl.hatch._hatch_types.append(DashHatch)
mpl.hatch._hatch_types.append(TickHatch)
mpl.hatch._hatch_types.append(EllHatch)
mpl.hatch._hatch_types.append(VeeHatch)
