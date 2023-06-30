from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Callable

import click
import numpy as np
from tkinter import filedialog

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.figure import SubFigure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from matplotlib.widgets import Button, RadioButtons, Slider


class Norm(StrEnum):
    P_1 = "1-norm"
    INNER_K_GON = "Inner $k$-gon-norm"
    P_2 = "2-norm"
    OUTER_K_GON = "Outer $k$-gon-norm"
    P_INF = "$\infty$-norm"

    @classmethod
    def supported_enum_values(cls, dimension: int) -> list[str]:
        if dimension < 1:
            raise ValueError("The dimension must be at least 1.")
        
        return [norm.value for norm in cls if dimension == 2 or norm.name.startswith("P")]
    
    @staticmethod
    def p_norm(vectors: np.ndarray, p: int | float) -> np.ndarray:
        if p < 1:
            raise ValueError("The value of p must be at least 1.")

        return np.linalg.norm(vectors, ord=p, axis=-1)

    @staticmethod
    def k_gon_norm(vectors: np.ndarray, k: int, outer: bool, project_vectors: bool) -> np.ndarray:
        if k < 3 or k % 2 != 0:
            raise ValueError("The number k of polygon vertices must be an even integer greater than 2.")

        apothem = 1.0 if outer else np.cos(np.pi / k)
        angles = np.arctan2(vectors[..., 1], vectors[..., 0])

        term1, term2 = int(outer) / 2, int(not outer) / 2
        angle_offsets = (2 * np.pi / k) * (np.floor((k * angles) / (2 * np.pi) + term1) + term2)

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

        return norm_values / projected_norm_values    # Note: Divide by 0 can't happen here either.


@dataclass
class Config:
    levels: int
    k: int
    norm: Norm
    euclidean_arc_lengths: bool
    project_vectors: bool

    def apply_norm(self, vectors: np.ndarray) -> np.ndarray:
        match self.norm:
            case Norm.P_1:
                return Norm.p_norm(vectors, 1)
            case Norm.P_2:
                return Norm.p_norm(vectors, 2)
            case Norm.P_INF:
                return Norm.p_norm(vectors, np.inf)
            case Norm.INNER_K_GON:
                return Norm.k_gon_norm(vectors, self.k, False, self.project_vectors)
            case Norm.OUTER_K_GON:
                return Norm.k_gon_norm(vectors, self.k, True, self.project_vectors)


class PaSpaV:
    MIN_LEVELS = 10
    MAX_LEVELS = 250
    MIN_K = 4
    MAX_K = 100
    MAX_SEGMENTS = 25
    MAX_DIMENSION = 10

    @staticmethod
    def random_curve(segments: int, dimension: int) -> np.ndarray:
        if segments < 1 or dimension < 1:
            raise ValueError("The number of segments and the dimension must be at least 1.")

        bound = 100.0 / segments / (2 * dimension)
        array_shape = (segments + 1, dimension)

        return np.random.uniform(-bound, bound, array_shape)
    
    @staticmethod
    def load_curve(filepath: str) -> np.ndarray:
        return np.loadtxt(filepath, delimiter=",", dtype=np.float64)

    def show(self):
        plt.show()

    def __init__(self, curve1_vertices: np.ndarray, curve2_vertices: np.ndarray, config: Config):
        self._curve_dimension = None
        if curve1_vertices.ndim == curve2_vertices.ndim == 2 and curve1_vertices.shape[1] == curve2_vertices.shape[1]:
            self._curve_dimension = curve1_vertices.shape[1]
            self._curve1_vertices = curve1_vertices
            self._curve2_vertices = curve2_vertices
        elif curve1_vertices.ndim == curve2_vertices.ndim == 1:
            self._curve_dimension = 1
            self._curve1_vertices = curve1_vertices[:, np.newaxis]
            self._curve2_vertices = curve2_vertices[:, np.newaxis]
        
        self._curve1_segments = len(self._curve1_vertices) - 1
        self._curve2_segments = len(self._curve2_vertices) - 1
        self._config = replace(
            config,
            levels=np.clip(config.levels, self.MIN_LEVELS, self.MAX_LEVELS),
            k=np.clip(config.k, self.MIN_K, self.MAX_K) & (~0 << 1)
        )

        if self._curve_dimension is None or self._curve_dimension == 0:
            raise ValueError("The input arrays have unsupported or incompatible shapes.")
        elif self._curve_dimension > self.MAX_DIMENSION:
            raise ValueError("The dimension of the curve vertices is too large.")
        elif not (1 <= self._curve1_segments <= self.MAX_SEGMENTS):
            raise ValueError("The first curve has too few or too many vertices/segments.")
        elif not (1 <= self._curve2_segments <= self.MAX_SEGMENTS):
            raise ValueError("The second curve has too few or too many vertices/segments.")
        elif self._config.norm.value not in Norm.supported_enum_values(self._curve_dimension):
            raise ValueError(f"The norm '{self._config.norm.name}' isn't supported in {self._curve_dimension}D.")

        self._init_figure()

    def _init_figure(self):
        self._tick_formatter = FormatStrFormatter("%.2f")
        self._fig = plt.figure("PaSpaV", figsize=(12.8, 7.2))
        plot_subfig, ui_subfig = self._fig.subfigures(1, 2, width_ratios=(2, 1))
        plot_grid_spec = GridSpec(2, 1, figure=plot_subfig, height_ratios=(15, 1), hspace=0.5)
        ui_grid_spec = GridSpec(3, 3, figure=ui_subfig, height_ratios=(2, 4, 2), hspace=0.375, width_ratios=(1, 1, 6))

        self._contour_ax = plot_subfig.add_subplot(plot_grid_spec[0], title="Parameter Space")
        self._contour_ax.set_aspect("equal")
        self._contour_ax.set_xlabel("$x_1$")
        self._contour_ax.set_ylabel("$x_2$", rotation="horizontal")
        self._contour_ax.xaxis.set_major_formatter(self._tick_formatter)
        self._contour_ax.yaxis.set_major_formatter(self._tick_formatter)
        self._contour_ax.grid(which="minor")

        self._contour = None
        self._colorbar = Colorbar(
            ax=plot_subfig.add_subplot(plot_grid_spec[1], title="height$(x_1, x_2)$"),
            mappable=ScalarMappable(),
            orientation="horizontal"
        )
        self._adjust_all()

        self._init_basic_ui(ui_subfig, ui_grid_spec)
        if self._curve_dimension != 1:
            self._init_additional_ui(ui_subfig, ui_grid_spec)

    def _init_basic_ui(self, ui_subfig: SubFigure, ui_grid_spec: GridSpec):
        self._level_slider = Slider(
            ax=ui_subfig.add_subplot(ui_grid_spec[:, 0], title="Levels"),
            label="",
            valmin=self.MIN_LEVELS,
            valmax=self.MAX_LEVELS,
            valinit=self._config.levels,
            valstep=1,
            dragging=False,
            orientation="vertical"
        )
        def update_levels(value: float):
            self._config.levels = int(value)
            self._adjust_contour_plot()
        self._level_slider.on_changed(update_levels)

        buttons_grid_spec = GridSpecFromSubplotSpec(2, 1, subplot_spec=ui_grid_spec[0, 2], hspace=0.25)

        self._random_button = Button(
            ax=ui_subfig.add_subplot(buttons_grid_spec[0], title="Curves"),
            label="Generate Random Curves"
        )
        def generate_random_curves(_):
            self._curve1_vertices = self.random_curve(self._curve1_segments, self._curve_dimension)
            self._curve2_vertices = self.random_curve(self._curve2_segments, self._curve_dimension)
            self._adjust_all()
        self._random_button.on_clicked(generate_random_curves)

        self._save_button = Button(ax=ui_subfig.add_subplot(buttons_grid_spec[1]), label="Save Curves as CSV files")
        def save_curves(_):
            for i, curve_vertices in enumerate([self._curve1_vertices, self._curve2_vertices], start=1):
                filepath = filedialog.asksaveasfilename(filetypes=[("CSV files", "*.csv")], title=f"Save curve {i}")
                if not filepath:
                    return
                np.savetxt(filepath, curve_vertices, delimiter=",", fmt="%.2f")
        self._save_button.on_clicked(save_curves)

    def _init_additional_ui(self, ui_subfig: SubFigure, ui_grid_spec: GridSpec):
        if self._curve_dimension == 2:
            self._k_slider = Slider(
                ax=ui_subfig.add_subplot(ui_grid_spec[:, 1], title="$k$"),
                label="",
                valmin=self.MIN_K,
                valmax=self.MAX_K,
                valinit=self._config.k,
                valstep=2,
                dragging=False,
                orientation="vertical"
            )
            def update_k(value: float):
                self._config.k = int(value)
                if self._config.norm is Norm.INNER_K_GON or self._config.norm is Norm.OUTER_K_GON:
                    self._adjust_all()
            self._k_slider.on_changed(update_k)

        supported_norms = Norm.supported_enum_values(self._curve_dimension)
        self._norm_radio_buttons = RadioButtons(
            ax=ui_subfig.add_subplot(ui_grid_spec[1, 2], title="Norm"),
            labels=supported_norms,
            active=supported_norms.index(self._config.norm.value),
            label_props={"fontsize": ["large"]}
        )
        def update_norm(label: str):
            self._config.norm = Norm(label)
            self._adjust_all()
        self._norm_radio_buttons.on_clicked(update_norm)

        self._arc_length_radio_buttons = RadioButtons(
            ax=ui_subfig.add_subplot(ui_grid_spec[2, 2], title="Arc Lengths"),
            labels=["Norm-dependent", "Euclidean"],
            active=int(self._config.euclidean_arc_lengths),
            label_props={"fontsize": ["large"]}
        )
        def update_arc_lengths(label: str):
            self._config.euclidean_arc_lengths = label == "Euclidean"
            self._adjust_all()
        self._arc_length_radio_buttons.on_clicked(update_arc_lengths)

    def _adjust_all(self):
        self._adjust_height_grid_and_ticks()
        self._adjust_contour_plot()
        self._adjust_colorbar()
        self._fig.canvas.draw_idle()

    def _adjust_height_grid_and_ticks(self):
        curve1_samples, self._x1_param_samples = self._sample_curve_and_set_ticks(
            self._curve1_vertices, self._contour_ax.set_xticks
        )
        curve2_samples, self._x2_param_samples = self._sample_curve_and_set_ticks(
            self._curve2_vertices, self._contour_ax.set_yticks
        )
        difference_vector_grid = curve1_samples - curve2_samples[:, np.newaxis]
        self._height_grid = self._config.apply_norm(difference_vector_grid)

    def _sample_curve_and_set_ticks(self, vertices: np.ndarray, set_ticks: Callable) -> tuple[np.ndarray, np.ndarray]:
        difference_vectors = np.diff(vertices, axis=0)
        if self._config.euclidean_arc_lengths:
            lengths = Norm.p_norm(difference_vectors, 2)
        else:
            lengths = self._config.apply_norm(difference_vectors)

        boundary_values = np.zeros(len(vertices), dtype=np.float64)
        np.cumsum(lengths, out=boundary_values[1:])
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

        self._contour = self._contour_ax.contourf(
            self._x1_param_samples, self._x2_param_samples, self._height_grid,
            self._config.levels, cmap="YlOrRd_r"
        )
        self._contour_ax.set_xbound(self._x1_param_samples[0], self._x1_param_samples[-1])
        self._contour_ax.set_ybound(self._x2_param_samples[0], self._x2_param_samples[-1])

    def _adjust_colorbar(self):
        self._colorbar.update_normal(ScalarMappable(norm=self._contour.norm, cmap=self._contour.get_cmap()))
        self._colorbar.set_ticks(LinearLocator(6))
        self._colorbar.formatter = self._tick_formatter


@click.group(context_settings={"show_default": True})
@click.option(
    "--levels", "-l", type=click.IntRange(PaSpaV.MIN_LEVELS, PaSpaV.MAX_LEVELS), default=50,
    help="Number of levels for contour plot drawings."
)
@click.option(
    "-k", type=click.IntRange(PaSpaV.MIN_K, PaSpaV.MAX_K), default=10,
    help="Number of vertices for polygonal norms (k-gon-norms)."
)
@click.option(
    "--norm", "-n", type=click.Choice([norm.name.lower() for norm in Norm], case_sensitive=False),
    default=Norm.P_1.name.lower(), help="Norm used for height calculations."
)
@click.option(
    "--euclidean-arc-lengths/--no-euclidean-arc-lengths", "-e/-ne", default=False,
    help="If true, arc lengths are always measured using the Euclidean 2-norm. " \
         "If false, they depend on the same norm used for height calculations."
)
@click.option(
    "--project-vectors/--no-project-vectors", "-p/-np", default=False,
    help="For polygonal norms (k-gon-norms): " \
         "If true, vectors are projected onto the unit polygon during norm computations. " \
         "If false, a shortcut is used that doesn't require explicit projection."
)
@click.pass_context
def paspav(ctx: click.Context, levels: int, k: int, norm: str, euclidean_arc_lengths: bool, project_vectors: bool):
    """PaSpaV: Parameter Space Visualiser for polygonal curves."""
    ctx.obj = Config(levels, k, Norm[norm.upper()], euclidean_arc_lengths, project_vectors)

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
    curve1_vertices = PaSpaV.random_curve(segments, dimension)
    curve2_vertices = PaSpaV.random_curve(segments, dimension)
    paspav = PaSpaV(curve1_vertices, curve2_vertices, config)
    paspav.show()

@paspav.command()
@click.argument("filepath1", type=click.Path(exists=True))
@click.argument("filepath2", type=click.Path(exists=True))
@click.pass_obj
def load(config: Config, filepath1: str, filepath2: str):
    """Visualise parameter space of polygonal curves loaded from files.
    The files at FILEPATH1 and FILEPATH2 have to be in CSV format with comma delimiters."""
    curve1_vertices = PaSpaV.load_curve(filepath1)
    curve2_vertices = PaSpaV.load_curve(filepath2)
    paspav = PaSpaV(curve1_vertices, curve2_vertices, config)
    paspav.show()

if __name__ == "__main__":
    paspav()
