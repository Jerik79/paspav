from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from enum import StrEnum
from typing import Callable

import click
import numpy as np
from tkinter import filedialog

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from matplotlib.figure import SubFigure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from matplotlib.widgets import Button, RadioButtons, Slider


class ComputationApproach(StrEnum):
    LT = "Linear transformations"
    EP = "Explicit projections"
    IP = "Implicit projections"


class Norm(StrEnum):
    P_1 = "$1$-norm"
    INNER_K_GON = "Inner $k$-gon-norm"
    P_2 = "$2$-norm"
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
    def k_gon_norm(vectors: np.ndarray, k: int, outer: bool, approach: ComputationApproach) -> np.ndarray:
        if k < 3 or k % 2 != 0:
            raise ValueError("The number k of polygon vertices must be an even integer greater than 2.")

        match approach:
            case ComputationApproach.LT:
                return Norm._k_gon_norm_via_transformations(vectors, k, outer)
            case ComputationApproach.EP:
                return Norm._k_gon_norm_via_projections(vectors, k, outer, False)
            case ComputationApproach.IP:
                return Norm._k_gon_norm_via_projections(vectors, k, outer, True)

    @staticmethod
    def _k_gon_norm_via_transformations(vectors: np.ndarray, k: int, outer: bool) -> np.ndarray:
        sector_angle = 2 * np.pi / k
        if outer:
            cos_value, sin_value = np.cos(sector_angle / 2), np.sin(sector_angle / 2)
            isometry = cos_value * np.array([[cos_value, -sin_value], [sin_value, cos_value]])
            vectors = np.tensordot(vectors, isometry, axes=(-1, -1))

        angles = np.arctan2(vectors[..., 1], vectors[..., 0])
        upper_angles = sector_angle * np.ceil(angles / sector_angle)
        lower_angles = upper_angles - sector_angle

        summands1 = (np.sin(upper_angles) - np.sin(lower_angles)) * vectors[..., 0]
        summands2 = (np.cos(lower_angles) - np.cos(upper_angles)) * vectors[..., 1]

        return (summands1 + summands2) / np.sin(sector_angle)

    @staticmethod
    def _k_gon_norm_via_projections(vectors: np.ndarray, k: int, outer: bool, implicit: bool) -> np.ndarray:
        sector_angle = 2 * np.pi / k
        apothem = 1.0 if outer else np.cos(sector_angle / 2)

        angles = np.arctan2(vectors[..., 1], vectors[..., 0])
        term1, term2 = int(outer) / 2, int(not outer) / 2
        angle_offsets = sector_angle * (np.floor(angles / sector_angle + term1) + term2)

        if implicit:
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
    norm: Norm
    approach: ComputationApproach
    k: int
    levels: int
    euclidean_arc_lengths: bool

    def get_norm_symbol(self) -> str:
        match self.norm:
            case Norm.P_1:
                return "\| \cdot \|_{1}"
            case Norm.P_2:
                return "\| \cdot \|_{2}"
            case Norm.P_INF:
                return "\| \cdot \|_{\infty}"
            case Norm.INNER_K_GON:
                return f"\| \cdot \|_{{ {self.k} \\text{{-}} \mathrm{{gon}} }}^{{\mathrm{{inner}}}}"
            case Norm.OUTER_K_GON:
                return f"\| \cdot \|_{{ {self.k} \\text{{-}} \mathrm{{gon}} }}^{{\mathrm{{outer}}}}"

    def apply_norm(self, vectors: np.ndarray) -> np.ndarray:
        match self.norm:
            case Norm.P_1:
                return Norm.p_norm(vectors, 1)
            case Norm.P_2:
                return Norm.p_norm(vectors, 2)
            case Norm.P_INF:
                return Norm.p_norm(vectors, np.inf)
            case Norm.INNER_K_GON:
                return Norm.k_gon_norm(vectors, self.k, False, self.approach)
            case Norm.OUTER_K_GON:
                return Norm.k_gon_norm(vectors, self.k, True, self.approach)


class PaSpaV:
    MIN_LEVELS = 10
    MAX_LEVELS = 250
    MIN_K = 4
    MAX_K = 100
    MAX_SEGMENTS = 25
    MAX_DIMENSION = 10

    IMG_FONT_SIZE = 12
    MIN_IMG_WIDTH = 2.0
    MAX_IMG_WIDTH = 8.3
    MIN_IMG_DPI = 100.0
    MAX_IMG_DPI = 1000.0
    MAX_IMG_LABELPAD = 20.0

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

        self._contour = None
        self._tick_formatter = FormatStrFormatter("$%.2f$")
        plt.rcParams.update({"backend": "TkAgg", "mathtext.fontset": "cm"})

    def show(self):
        plt.rcParams.update({"font.size": 14})
        self._fig = plt.figure("PaSpaV", figsize=(19.2, 10.8))

        plot_subfig, ui_subfig = self._fig.subfigures(1, 2, width_ratios=(2, 1))
        plot_subfig.subplots_adjust(hspace=0.5)

        self._contour_ax, colorbar_ax = plot_subfig.subplots(2, 1, height_ratios=(15, 1))
        self._colorbar = Colorbar(ax=colorbar_ax, mappable=ScalarMappable(), orientation="horizontal")
        self._adjust_all()

        labelpad = 10.0
        colorbar_ax.set_title("$\mathrm{height}_{\| \cdot \|}(x_1, x_2)$", pad=labelpad)
        colorbar_ax.tick_params(pad=labelpad)
        self._contour_ax.set_title("Parameter Space")
        self._init_contour_ax(labelpad)

        ui_grid_spec = GridSpec(3, 3, figure=ui_subfig, width_ratios=(1, 1, 6), height_ratios=(2, 4, 2), hspace=0.375)
        self._init_basic_ui(ui_subfig, ui_grid_spec)
        if self._curve_dimension != 1:
            self._init_additional_ui(ui_subfig, ui_grid_spec)

        plt.show()

    def saveimg(self, filepath: str, width: float, dpi: float, labelpad: float):
        plt.rcParams.update({"font.size": self.IMG_FONT_SIZE})
        self._fig = plt.figure("PaSpaV")

        self._contour_ax, colorbar_ax = self._fig.subplots(1, 2, width_ratios=(20, 1))
        self._colorbar = Colorbar(ax=colorbar_ax, mappable=ScalarMappable(), orientation="vertical")
        self._adjust_all()

        if self._x1_param_samples[-1] == 0.0:
            raise RuntimeError("The first curve has arc length 0.")
        elif self._x2_param_samples[-1] == 0.0:
            raise RuntimeError("The second curve has arc length 0.")

        parameter_space_aspect = self._x2_param_samples[-1] / self._x1_param_samples[-1]
        self._fig.set_size_inches((width, width * parameter_space_aspect))
        colorbar_ax.set_box_aspect(17.5 * parameter_space_aspect)

        colorbar_ax.yaxis.set_label_position("left")
        colorbar_ax.set_ylabel(
            f"$\mathrm{{height}}_{{ {self._config.get_norm_symbol()} }}(x_1, x_2)$",
            labelpad=labelpad
        )
        colorbar_ax.tick_params(pad=labelpad)
        self._init_contour_ax(labelpad)

        plt.tight_layout(pad=0.0, w_pad=1.0)
        plt.savefig(filepath, transparent=True, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
        plt.close()

    def _init_contour_ax(self, labelpad: float):
        self._contour_ax.set_aspect("equal")
        self._contour_ax.set_xlabel("$x_1$", labelpad=labelpad)
        self._contour_ax.set_ylabel("$x_2$", labelpad=labelpad, rotation="horizontal")
        self._contour_ax.xaxis.set_major_formatter(self._tick_formatter)
        self._contour_ax.yaxis.set_major_formatter(self._tick_formatter)
        self._contour_ax.tick_params(pad=labelpad)
        self._contour_ax.tick_params(which="minor", color="0.7")
        self._contour_ax.grid(which="minor", color="0.7")

    def _init_basic_ui(self, ui_subfig: SubFigure, ui_grid_spec: GridSpec):
        self._level_slider = Slider(
            ax=ui_subfig.add_subplot(ui_grid_spec[:, 0], title="Levels"), label="", valfmt="$%d$",
            valmin=self.MIN_LEVELS, valmax=self.MAX_LEVELS, valinit=self._config.levels, valstep=1,
            dragging=False, orientation="vertical"
        )
        def update_levels(value: float):
            if self._config.levels != (levels := int(value)):
                self._config.levels = levels
                self._adjust_contour_plot()
        self._level_slider.on_changed(update_levels)

        buttons_grid_spec = GridSpecFromSubplotSpec(2, 1, subplot_spec=ui_grid_spec[0, 2], hspace=0.25)

        self._random_button = Button(
            ax=ui_subfig.add_subplot(buttons_grid_spec[0], title="Curves"),
            label="Generate random curves"
        )
        def generate_random_curves(_):
            self._curve1_vertices = self.random_curve(self._curve1_segments, self._curve_dimension)
            self._curve2_vertices = self.random_curve(self._curve2_segments, self._curve_dimension)
            self._adjust_all()
        self._random_button.on_clicked(generate_random_curves)

        self._save_button = Button(ax=ui_subfig.add_subplot(buttons_grid_spec[1]), label="Save curves as CSV files")
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
                ax=ui_subfig.add_subplot(ui_grid_spec[:, 1], title="$k$"), label="", valfmt="$%d$",
                valmin=self.MIN_K, valmax=self.MAX_K, valinit=self._config.k, valstep=2,
                dragging=False, orientation="vertical"
            )
            def update_k(value: float):
                if self._config.k != (k := int(value)):
                    self._config.k = k
                    if self._config.norm is Norm.INNER_K_GON or self._config.norm is Norm.OUTER_K_GON:
                        self._adjust_all()
            self._k_slider.on_changed(update_k)

        supported_norms = Norm.supported_enum_values(self._curve_dimension)
        self._norm_radio_buttons = RadioButtons(
            ax=ui_subfig.add_subplot(ui_grid_spec[1, 2], title="Norm $\| \cdot \|$"),
            labels=supported_norms,
            active=supported_norms.index(self._config.norm.value),
            label_props={"fontsize": ["large"]}
        )
        def update_norm(label: str):
            if self._config.norm is not (norm := Norm(label)):
                self._config.norm = norm
                self._adjust_all()
        self._norm_radio_buttons.on_clicked(update_norm)

        self._arc_length_radio_buttons = RadioButtons(
            ax=ui_subfig.add_subplot(ui_grid_spec[2, 2], title="Arc Lengths"),
            labels=["Norm-dependent", "Euclidean"],
            active=int(self._config.euclidean_arc_lengths),
            label_props={"fontsize": ["large"]}
        )
        def update_arc_lengths(label: str):
            if self._config.euclidean_arc_lengths != (euclidean_arc_lengths := label == "Euclidean"):
                self._config.euclidean_arc_lengths = euclidean_arc_lengths
                if self._config.norm is not Norm.P_2:
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

        sample_numbers = 2 * np.maximum((10 * lengths).astype(int), 10)
        def calculate_samples(array: np.ndarray) -> np.ndarray:
            segment_samples = []
            for i in range(1, len(lengths)):
                segment_samples.append(np.linspace(array[i-1], array[i], sample_numbers[i-1], endpoint=False))
            segment_samples.append(np.linspace(array[-2], array[-1], sample_numbers[-1] + 1, endpoint=True))
            return np.concatenate(segment_samples)

        return calculate_samples(vertices), calculate_samples(boundary_values)

    def _adjust_contour_plot(self):
        if self._contour is not None:
            self._contour.remove()

        height_lb, height_ub = self._get_height_bounds()
        self._contour = self._contour_ax.contourf(
            self._x1_param_samples[::2], self._x2_param_samples[::2], self._height_grid[::2, ::2],
            self._config.levels, norm=Normalize(height_lb, height_ub), cmap="inferno_r"
        )
        self._contour_ax.set_xbound(0.0, self._x1_param_samples[-1])
        self._contour_ax.set_ybound(0.0, self._x2_param_samples[-1])

    def _get_height_bounds(self, exp=Decimal("0.1")) -> tuple[float, float]:
        lower_bound = Decimal(np.min(self._height_grid)).quantize(exp, rounding=ROUND_DOWN)
        upper_bound = Decimal(np.max(self._height_grid)).quantize(exp, rounding=ROUND_UP)
        return float(lower_bound), float(upper_bound)

    def _adjust_colorbar(self):
        self._colorbar.update_normal(ScalarMappable(norm=self._contour.norm, cmap=self._contour.get_cmap()))
        self._colorbar.set_ticks(LinearLocator(6))
        self._colorbar.formatter = self._tick_formatter


@click.group(context_settings={"show_default": True})
@click.option(
    "--norm", "-n", type=click.Choice([norm.name for norm in Norm], case_sensitive=False),
    default=Norm.P_1.name, help="Norm used for height calculations."
)
@click.option(
    "--approach", "-a", type=click.Choice([approach.name for approach in ComputationApproach], case_sensitive=False),
    default=ComputationApproach.LT.name, help="""Approach used to compute polygonal norms (k-gon-norms).
    The supported approaches are based on linear transformations and on explicit or implicit projections."""
)
@click.option(
    "-k", type=click.IntRange(PaSpaV.MIN_K, PaSpaV.MAX_K), default=10,
    help="Number of sides for polygonal norms (k-gon-norms)."
)
@click.option(
    "--levels", "-l", type=click.IntRange(PaSpaV.MIN_LEVELS, PaSpaV.MAX_LEVELS), default=50,
    help="Number of levels for contour plot drawings."
)
@click.option(
    "--euclidean-arc-lengths/--no-euclidean-arc-lengths", "-e/-ne", default=False,
    help="""If true, arc lengths are always measured using the Euclidean 2-norm.
    If false, they depend on the norm that is used for height calculations."""
)
@click.pass_context
def paspav(ctx: click.Context, norm: str, approach: str, k: int, levels: int, euclidean_arc_lengths: bool):
    """PaSpaV: Parameter Space Visualiser for polygonal curves."""
    ctx.obj = Config(Norm[norm], ComputationApproach[approach], k, levels, euclidean_arc_lengths)

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

@paspav.group(invoke_without_command=True, subcommand_metavar="[COMMAND [ARGS]...]")
@click.argument("CURVE1_PATH", type=click.Path(exists=True))
@click.argument("CURVE2_PATH", type=click.Path(exists=True))
@click.pass_context
def load(ctx: click.Context, curve1_path: str, curve2_path: str):
    """Visualise parameter space of polygonal curves loaded from files.
    The files at CURVE1_PATH and CURVE2_PATH need to have CSV format with comma delimiters,
    where each row specifies coordinates of a curve vertex."""
    curve1_vertices = PaSpaV.load_curve(curve1_path)
    curve2_vertices = PaSpaV.load_curve(curve2_path)
    paspav = PaSpaV(curve1_vertices, curve2_vertices, ctx.obj)

    if ctx.invoked_subcommand is None:
        paspav.show()
    else:
        ctx.obj = paspav

@load.command()
@click.argument("IMAGE_PATH", type=click.Path())
@click.option(
    "--width", "-w", type=click.FloatRange(PaSpaV.MIN_IMG_WIDTH, PaSpaV.MAX_IMG_WIDTH), default=6.2,
    help=f"""Width of the image file in inches. For smaller image widths, label texts appear larger in
    comparison to graphical elements because the font size is fixed to {PaSpaV.IMG_FONT_SIZE} points."""
)
@click.option(
    "--dpi", "-d", type=click.FloatRange(PaSpaV.MIN_IMG_DPI, PaSpaV.MAX_IMG_DPI), default=100.0,
    help="DPI (dots per inch) of the image file. Together with the width, this determines resolution of raster images."
)
@click.option(
    "--labelpad", "-l", type=click.FloatRange(0.0, PaSpaV.MAX_IMG_LABELPAD), default=6.5,
    help="Padding for label texts in points."
)
@click.pass_obj
def saveimg(paspav: PaSpaV, image_path: str, width: float, dpi: float, labelpad: float):
    """Save parameter space of loaded polygonal curves as an image file.
    The file is stored at IMAGE_PATH, which needs to have a format extension supported by Matplotlib.
    This includes *.eps, *.pdf, *.pgf, *.png and more."""
    paspav.saveimg(image_path, width, dpi, labelpad)

if __name__ == "__main__":
    paspav()
