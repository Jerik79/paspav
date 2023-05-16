from enum import StrEnum
from dataclasses import dataclass, replace

import click
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import RadioButtons, Slider


class Norm(StrEnum):
    L1 = "$L_1$"
    L2 = "$L_2$"
    LINF = "$L_\infty$"
    HEX = "Hex"
    CHILI = "Chili"

    @classmethod
    def supported_values(cls, dimension: int) -> list[str]:
        if dimension < 1:
            return ValueError("The dimension must be at least 1.")
        elif dimension == 1:
            return [cls.L1.value]
        else:
            return [norm.value for norm in cls if dimension == 2 or norm.value.startswith("$L_")]

    @staticmethod
    def _regular_polygon_norm(vectors: np.ndarray, k: int, align_vertex: bool, project_vectors: bool) -> np.ndarray:
        if k < 3 or k % 2 != 0:
            raise ValueError("The number of polygon vertices must be an even number greater than 2.")

        apothem = np.cos(np.pi / k)
        angles = np.arctan2(vectors[..., 1], vectors[..., 0])

        offset1, offset2 = int(not align_vertex) / 2, int(align_vertex) / 2
        angle_offsets = (2 * np.pi / k) * (np.floor((k * angles) / (2 * np.pi) + offset1) + offset2)

        if not project_vectors:
            norm_values = np.linalg.norm(vectors, ord=2, axis=-1)
            projected_norm_values = apothem / np.cos(angles - angle_offsets)    # Note: Divide by 0 can't happen here.
        else:
            opposite_sides = apothem * np.tan(angles - angle_offsets)
            cos_values, sin_values = np.cos(angle_offsets), np.sin(angle_offsets)

            projected_vectors = np.zeros_like(vectors, dtype=np.float64)
            projected_vectors[..., 0] = cos_values * apothem - sin_values * opposite_sides
            projected_vectors[..., 1] = sin_values * apothem + cos_values * opposite_sides

            norm_values = np.linalg.norm(vectors, ord=1, axis=-1)
            projected_norm_values = np.linalg.norm(projected_vectors, ord=1, axis=-1)

        return np.divide(norm_values, projected_norm_values, where=projected_norm_values != 0.0)


@dataclass
class Config:
    levels: int
    norm: Norm
    align_norm_vertex: bool
    project_vectors: bool

    def apply_norm(self, vectors: np.ndarray) -> np.ndarray:
        match self.norm:
            case Norm.L1:
                return np.linalg.norm(vectors, ord=1, axis=-1)
            case Norm.L2:
                return np.linalg.norm(vectors, ord=2, axis=-1)
            case Norm.LINF:
                return np.linalg.norm(vectors, ord=np.inf, axis=-1)
            case Norm.HEX:
                return Norm._regular_polygon_norm(vectors, 6, self.align_norm_vertex, self.project_vectors)
            case Norm.CHILI:
                return Norm._regular_polygon_norm(vectors, 1000, self.align_norm_vertex, self.project_vectors)


class PaSpaV:
    MAX_SEGMENTS = 25
    MAX_DIMENSION = 10
    MIN_LEVELS = 10
    MAX_LEVELS = 250

    def __init__(self, curve1_vertices: np.ndarray, curve2_vertices: np.ndarray, config: Config):
        curve_dimension = None
        if curve1_vertices.ndim == curve2_vertices.ndim == 2 and curve1_vertices.shape[1] == curve2_vertices.shape[1]:
            curve_dimension = curve1_vertices.shape[1]
        elif curve1_vertices.ndim == curve2_vertices.ndim == 1:
            curve_dimension = 1
            curve1_vertices = curve1_vertices[:, np.newaxis]
            curve2_vertices = curve2_vertices[:, np.newaxis]

        if curve_dimension is None or curve_dimension == 0:
            raise ValueError("The input arrays have unsupported or incompatible shapes.")
        elif curve_dimension > self.MAX_DIMENSION:
            raise ValueError("The dimension of the curve vertices is too large.")
        elif not (1 <= len(curve1_vertices) - 1 <= self.MAX_SEGMENTS):
            raise ValueError("The first curve has too few or too many vertices/segments.")
        elif not (1 <= len(curve2_vertices) - 1 <= self.MAX_SEGMENTS):
            raise ValueError("The second curve has too few or too many vertices/segments.")
        elif config.norm.value not in Norm.supported_values(curve_dimension):
            raise ValueError(f"The norm '{config.norm.name}' isn't supported in dimension {curve_dimension}.")

        self._curve1_vertices = curve1_vertices
        self._curve2_vertices = curve2_vertices
        self._config = replace(config, levels=np.clip(config.levels, self.MIN_LEVELS, self.MAX_LEVELS))

        self._init_figure(curve_dimension)

    def _init_figure(self, curve_dimension: int):
        self._fig, ((self._main_ax, self._colorbar_ax), (self._slider_ax, self._buttons_ax)) = plt.subplots(
            2, 2, width_ratios=(0.9, 0.1), height_ratios=(0.8, 0.2), figsize=(12.8, 7.2)
        )

        self._main_ax.set_title("Parameter Space")
        self._main_ax.set_xlabel("$x$")
        self._main_ax.set_ylabel("$y$", labelpad=10.0, rotation="horizontal")
        self._main_ax.set_aspect("equal")
        self._main_ax.grid(which="minor")

        self._contour = None
        self._adjust_height_grid_and_ticks()
        self._adjust_contour_plot()

        self._colorbar_ax.set_title("$h(x,y)$")
        self._colorbar = plt.colorbar(cax=self._colorbar_ax, mappable=ScalarMappable())
        self._adjust_colorbar()

        self._slider = Slider(
            ax=self._slider_ax,
            label="# of Levels",
            valmin=self.MIN_LEVELS,
            valmax=self.MAX_LEVELS,
            valinit=self._config.levels,
            valstep=1,
            dragging=False,
            orientation="horizontal"
        )
        self._slider.on_changed(self._update_levels)

        self._buttons_ax.set_title("Norm")
        self._buttons = RadioButtons(
            ax=self._buttons_ax,
            labels=Norm.supported_values(curve_dimension),
            active=Norm.supported_values(curve_dimension).index(self._config.norm.value),
            label_props={"fontsize": ["x-large"]}
        )
        self._buttons.on_clicked(self._update_norm)

    def _adjust_height_grid_and_ticks(self):
        curve1_samples, self._x_param_samples = self._sample_curve_and_adjust_ticks(x=True)
        curve2_samples, self._y_param_samples = self._sample_curve_and_adjust_ticks(x=False)
        difference_grid = curve1_samples - curve2_samples[:, np.newaxis]
        self._height_grid = self._config.apply_norm(difference_grid)

    def _sample_curve_and_adjust_ticks(self, x: bool) -> tuple[np.ndarray, np.ndarray]:
        vertices = self._curve1_vertices if x else self._curve2_vertices
        lengths = self._config.apply_norm(np.diff(vertices, axis=0))
        boundary_values = np.zeros(len(lengths) + 1, dtype=np.float64)
        np.cumsum(lengths, out=boundary_values[1:])

        set_ticks = self._main_ax.set_xticks if x else self._main_ax.set_yticks
        set_ticks([0.0, boundary_values[-1]], minor=False)
        set_ticks(boundary_values[1:-1], minor=True)

        sample_numbers = np.maximum((10 * lengths).astype(int), 10)
        def calculate_samples(array: np.ndarray) -> np.ndarray:
            segment_samples = []
            for i in range(1, len(lengths)):
                segment_samples.append(np.linspace(array[i-1], array[i], sample_numbers[i-1], endpoint=False))
            segment_samples.append(np.linspace(array[-2], array[-1], sample_numbers[-1], endpoint=True))
            return np.concatenate(segment_samples)

        return calculate_samples(vertices), calculate_samples(boundary_values)

    def _adjust_contour_plot(self):
        if self._contour is not None:
            while self._contour.collections:
                self._contour.collections.pop().remove()

        self._contour = self._main_ax.contourf(
            self._x_param_samples, self._y_param_samples, self._height_grid,
            self._config.levels, cmap="YlOrRd_r"
        )
        self._main_ax.set_xbound(self._x_param_samples[0], self._x_param_samples[-1])
        self._main_ax.set_ybound(self._y_param_samples[0], self._y_param_samples[-1])

    def _adjust_colorbar(self):
        self._colorbar.update_normal(ScalarMappable(norm=self._contour.norm, cmap=self._contour.get_cmap()))
        self._colorbar.set_ticks(MaxNLocator(10, steps=[1, 2, 5, 10], integer=True))

    def _update_levels(self, value: float):
        self._config.levels = int(value)
        self._adjust_contour_plot()

        self._fig.tight_layout()
        self._fig.canvas.draw_idle()

    def _update_norm(self, label: str):
        self._config.norm = Norm(label)
        self._adjust_height_grid_and_ticks()
        self._adjust_contour_plot()
        self._adjust_colorbar()

        self._fig.tight_layout()
        self._fig.canvas.draw_idle()

    def show(self):
        self._fig.tight_layout()
        plt.show()


@click.group(context_settings={"show_default": True})
@click.option(
    "--levels", "-l", type=click.IntRange(PaSpaV.MIN_LEVELS, PaSpaV.MAX_LEVELS), default=50,
    help="Number of levels for contour plot drawings."
)
@click.option(
    "--norm", "-n", type=click.Choice([norm.name.capitalize() for norm in Norm], case_sensitive=False), default="L1",
    help="Norm for arc length parametrisation and height calculation."
)
@click.option(
    "--align-norm-vertex/--no-align-norm-vertex", "-a/-na", default=True,
    help="For polygon norms: " \
         "If true, a polygon vertex is aligned to the point (1, 0). " \
         "If false, an edge center is aligned to that point instead."
)
@click.option(
    "--project-vectors/--no-project-vectors", "-p/-np", default=False,
    help="For polygon norms: " \
         "If true, vectors are projected to the unit polygon during norm calculation. " \
         "If false, a shortcut is used that doesn't require explicit projection."
)
@click.pass_context
def paspav(ctx: click.Context, levels: int, norm: str, align_norm_vertex: bool, project_vectors: bool):
    """PaSpaV: Parameter Space Visualiser for polygonal curves."""
    ctx.obj = Config(levels, Norm[norm.upper()], align_norm_vertex, project_vectors)

@paspav.command()
@click.option(
    "--segments", "-s", type=click.IntRange(1, PaSpaV.MAX_SEGMENTS), default=5,
    help="Number of segments per curve."
)
@click.option(
    "--dimension", "-d", type=click.IntRange(1, PaSpaV.MAX_DIMENSION), default=2,
    help="Dimension of curve vertices."
)
@click.pass_obj
def random(config: Config, segments: int, dimension: int):
    """Visualise parameter space of randomly generated polygonal curves."""
    bound = 100.0 / segments / (2 * dimension)
    array_shape = (segments + 1, dimension)
    curve1_vertices = np.random.uniform(-bound, bound, array_shape)
    curve2_vertices = np.random.uniform(-bound, bound, array_shape)

    paspav = PaSpaV(curve1_vertices, curve2_vertices, config)
    paspav.show()

@paspav.command()
@click.argument("filepath1", type=click.Path(exists=True))
@click.argument("filepath2", type=click.Path(exists=True))
@click.pass_obj
def load(config: Config, filepath1: str, filepath2: str):
    """Visualise parameter space of polygonal curves loaded from files.
    The files at FILEPATH1 and FILEPATH2 have to be in CSV format with comma delimiters."""
    curve1_vertices = np.loadtxt(filepath1, delimiter=",", dtype=np.float64)
    curve2_vertices = np.loadtxt(filepath2, delimiter=",", dtype=np.float64)

    paspav = PaSpaV(curve1_vertices, curve2_vertices, config)
    paspav.show()

if __name__ == "__main__":
    paspav()
