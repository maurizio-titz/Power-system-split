#!/usr/bin/env python3
"""
Plot Style Configuration
Defines consistent styling constants and helper functions for all plots in the power-system-split project.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

from loguru import logger


# === FONT SIZES ===
# Main text sizes
TITLE_FONTSIZE = 20  # Main plot titles
SUBTITLE_FONTSIZE = 18  # Subplot titles
AXIS_LABEL_FONTSIZE = 16  # Axis labels (xlabel, ylabel)
TICK_LABEL_FONTSIZE = 14  # Tick labels
LEGEND_FONTSIZE = 14  # Legend text
COLORBAR_LABEL_FONTSIZE = 14  # Colorbar labels
COLORBAR_TICK_FONTSIZE = 12  # Colorbar tick labels

# Panel labels (A, B, C, D...)
PANEL_LABEL_FONTSIZE = 24  # Large bold panel labels
PANEL_LABEL_WEIGHT = "bold"

# Text annotations
ANNOTATION_FONTSIZE = 12  # General text annotations
SMALL_TEXT_FONTSIZE = 10  # Small explanatory text


# === FIGURE SIZES ===
# Standard figure sizes for different plot types
SINGLE_PLOT_SIZE = (8, 6)  # Single plot
TWO_PANEL_SIZE = (16, 6)  # Two side-by-side plots
THREE_PANEL_SIZE = (18, 6)  # Three side-by-side plots
FOUR_PANEL_SIZE = (16, 12)  # 2x2 grid
SIX_PANEL_SIZE = (18, 12)  # 2x3 or 3x2 grid
LARGE_COMBINED_SIZE = (24, 18)  # Large multi-panel figures
MAP_SINGLE_SIZE = (10, 8)  # Single map
MAP_MULTI_SIZE = (20, 10)  # Multiple maps


# === COLORS ===
# Standard color palette for consistency
PRIMARY_COLORS = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "olive": "#bcbd22",
    "cyan": "#17becf",
}

# CO2 level colors (consistent across plots)
cmap_lvls = plt.get_cmap("cividis")
co2ls = np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0])
CO2_COLORS = {
    0.0: cmap_lvls(np.where(co2ls == 0.0)[0][0] / (len(co2ls) - 1)),
    0.05: cmap_lvls(np.where(co2ls == 0.05)[0][0] / (len(co2ls) - 1)),
    0.1: cmap_lvls(np.where(co2ls == 0.1)[0][0] / (len(co2ls) - 1)),
    0.2: cmap_lvls(np.where(co2ls == 0.2)[0][0] / (len(co2ls) - 1)),
    0.3: cmap_lvls(np.where(co2ls == 0.3)[0][0] / (len(co2ls) - 1)),
    0.4: cmap_lvls(np.where(co2ls == 0.4)[0][0] / (len(co2ls) - 1)),
    0.5: cmap_lvls(np.where(co2ls == 0.5)[0][0] / (len(co2ls) - 1)),
    0.6: cmap_lvls(np.where(co2ls == 0.6)[0][0] / (len(co2ls) - 1)),
}

# Technology colors (for generation/storage)
TECH_COLORS = {
    "solar": "#ffb347",
    "onwind": "#74add1",
    "offwind": "#4575b4",
    "offwind-dc": "#4575b4",
    "gas": "#cc0099",
    "OCGT": "#cc0099",
    "CCGT": "#990066",
    "coal": "#545454",
    "nuclear": "#ff9000",
    "hydro": "#298c81",
    "battery": "#b3de69",
    "H2": "#2ca02c",
    "PHS": "#7fb3d3",
}

# Map colors
MAP_LINE_COLOR = "black"
MAP_LINK_COLOR = "black"
MAP_NODE_COLOR = "black"


# === LINE STYLES ===
LINE_WIDTH = 2.0  # Standard line width
THIN_LINE_WIDTH = 1.0  # Thin lines
THICK_LINE_WIDTH = 3.0  # Thick emphasis lines
MAP_LINE_WIDTH = 0.5  # Map transmission lines
MAP_LINK_WIDTH = 0.5  # Map DC links

# Marker sizes
MARKER_SIZE = 100  # Standard scatter plot markers
SMALL_MARKER_SIZE = 50  # Small markers
LARGE_MARKER_SIZE = 200  # Large emphasis markers


# === GRID AND LAYOUT ===
GRID_ALPHA = 0.3  # Grid transparency
WSPACE = 0.3  # Default horizontal spacing between subplots
HSPACE = 0.3  # Default vertical spacing between subplots

# Map limits (for European power system)
MAP_XLIM = (-14, 35)  # Longitude range
MAP_YLIM = (35, 70)  # Latitude range


# === PANEL LABELS ===
PANEL_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


# === ENVIRONMENT VARIABLE CONFIGURATION ===
def get_plot_config():
    """Get plot configuration from environment variables."""
    panel_lowercase_env = os.getenv("PLOT_PANEL_LOWERCASE")
    plot_style = os.getenv("PLOT_STYLE", "").strip().lower()
    if panel_lowercase_env is None:
        if plot_style == "joule":
            panel_lowercase = False
        else:
            panel_lowercase = False
    else:
        panel_lowercase = panel_lowercase_env.lower() == "true"

    config = {
        "panel_lowercase": panel_lowercase,
        "save_with_config": os.getenv("PLOT_SAVE_WITH_CONFIG", "false").lower()
        == "true",
        "dpi": int(os.getenv("PLOT_DPI", "300")),
        "output_dir": os.getenv("PLOT_OUTPUT_DIR", ""),
    }
    return config


def apply_plot_profile(profile_name, output_dir=None):
    """Apply a named plot profile and optionally set the output directory.

    Supported profiles:
        - "joule": uppercase panel labels
    """
    profile = profile_name.strip().lower()
    if profile == "joule":
        os.environ["PLOT_PANEL_LOWERCASE"] = "false"
    else:
        raise ValueError(
            f"Unknown plot profile: {profile_name}. Supported: ['joule']"
        )

    if output_dir is not None:
        os.environ["PLOT_OUTPUT_DIR"] = output_dir


def get_config_suffix():
    """Generate filename suffix based on current configuration."""
    config = get_plot_config()
    suffix_parts = []

    # # Add panel label style indicator
    # if config["panel_lowercase"]:
    #     suffix_parts.append("pLC")
    # else:
    #     suffix_parts.append("pUC")

    # Add DPI if different from default
    if config["dpi"] != 300:
        suffix_parts.append(f"dpi{config['dpi']}")

    return "_" + "_".join(suffix_parts) if suffix_parts else ""


def get_panel_label(index, bold=True, lowercase=False):
    """Get formatted panel label."""
    label = PANEL_LABELS[index] if index < len(PANEL_LABELS) else f"Panel{index}"
    if lowercase:
        label = label.lower()
    return rf"\textbf{{{label}}}" if bold else label


# === MATPLOTLIB SETUP FUNCTIONS ===
def setup_matplotlib_style():
    """Configure matplotlib with consistent settings."""
    # Use default style as base
    plt.style.use("default")

    # Enable LaTeX text rendering
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}\usepackage{bm}")

    # Set default font sizes
    plt.rc("font", size=TICK_LABEL_FONTSIZE)
    plt.rc("axes", titlesize=SUBTITLE_FONTSIZE)
    plt.rc("axes", labelsize=AXIS_LABEL_FONTSIZE)
    plt.rc("xtick", labelsize=TICK_LABEL_FONTSIZE)
    plt.rc("ytick", labelsize=TICK_LABEL_FONTSIZE)
    plt.rc("legend", fontsize=LEGEND_FONTSIZE)
    plt.rc("figure", titlesize=TITLE_FONTSIZE)

    # Set default line widths and marker sizes
    plt.rc("lines", linewidth=LINE_WIDTH)
    plt.rc("lines", markersize=8)

    # Grid settings
    plt.rc("axes", grid=True)
    plt.rc("grid", alpha=GRID_ALPHA)


def setup_colormap_scientific_notation(colorbar, power_limits=(-3, -3)):
    """Format colorbar with scientific notation."""
    
    colorbar.ax.ticklabel_format(style="scientific", axis="y", scilimits=power_limits)
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE, width=1.0, which="both")


def add_panel_label(
    ax, label_index, x_offset=-0.1, y_offset=0.05, lowercase=None, **kwargs
):
    """Add panel label to subplot.

    Args:
        ax: matplotlib axes object
        label_index: index for panel label (0=A, 1=B, etc.)
        x_offset: horizontal offset for label position
        y_offset: vertical offset for label position
        lowercase: if True, use lowercase labels; if False, use uppercase;
                  if None, use environment variable PLOT_PANEL_LOWERCASE
        **kwargs: additional text formatting arguments
    """
    # Use environment variable if lowercase not explicitly set
    if lowercase is None:
        config = get_plot_config()
        lowercase = config["panel_lowercase"]

    default_kwargs = {
        "fontsize": PANEL_LABEL_FONTSIZE,
        "weight": PANEL_LABEL_WEIGHT,
        "verticalalignment": "center",
        "transform": ax.transAxes,
    }
    default_kwargs.update(kwargs)

    ax.text(
        x_offset,
        1 + y_offset,
        get_panel_label(label_index, lowercase=lowercase),
        **default_kwargs,
    )
    
    
def add_panel_label_fig_position(fig: mpl.figure.Figure, 
                                   label_index: int, 
                                   fig_pos_xx: float, fig_pos_yy: float, 
                                   lowercase: bool | None = None, **kwargs):
    """Add panel label to position on the fig supplied.

    Args:
        fig: pyplot Figure obejct
        label_index: index for panel label (0=A, 1=B, etc.)
        fig_pos_xx, fig_pos_yy ( both float): x and y position in figure coordinates
        lowercase: if True, use lowercase labels; if False, use uppercase;
                  if None, use environment variable PLOT_PANEL_LOWERCASE
        **kwargs: additional text formatting arguments
    """

    # Use environment variable if lowercase not explicitly set
    if lowercase is None:
        config = get_plot_config()
        lowercase = config["panel_lowercase"]

    default_kwargs = {
        "fontsize": PANEL_LABEL_FONTSIZE,
        "weight": PANEL_LABEL_WEIGHT,
        "verticalalignment": "center"
    }
    default_kwargs.update(kwargs)
    
    fig.text(fig_pos_xx, fig_pos_yy,
        get_panel_label(label_index, lowercase=lowercase),
        **default_kwargs)


def setup_map_axes(ax, xlim=MAP_XLIM, ylim=MAP_YLIM):
    """Configure map axes with consistent limits and styling."""
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")


def get_co2_color(co2_level):
    """Get consistent color for CO2 level."""
    # Round to nearest standard level
    available_levels = np.array(list(CO2_COLORS.keys()))
    closest_level = available_levels[np.argmin(np.abs(available_levels - co2_level))]
    return CO2_COLORS[closest_level]


def get_tech_color(technology):
    """Get consistent color for technology type."""
    tech_lower = technology.lower()
    for tech_key, color in TECH_COLORS.items():
        if tech_key.lower() in tech_lower:
            return color
    return PRIMARY_COLORS["gray"]  # Default color


def create_co2_colormap(co2_levels, colormap_name="viridis"):
    """Create consistent colormap for CO2 levels."""
    from matplotlib.colors import LinearSegmentedColormap

    colors = [get_co2_color(level) for level in sorted(co2_levels)]
    return LinearSegmentedColormap.from_list("co2_cmap", colors)


# === LEGEND HELPERS ===
def create_unified_legend(
    axes_list, loc="center", bbox_to_anchor=None, ncol=2, **kwargs
):
    """Create unified legend from multiple axes."""
    handles, labels = [], []
    for ax in axes_list:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Remove duplicates while preserving order
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    default_kwargs = {"fontsize": LEGEND_FONTSIZE, "ncol": ncol, "loc": loc}
    if bbox_to_anchor:
        default_kwargs["bbox_to_anchor"] = bbox_to_anchor
    default_kwargs.update(kwargs)

    return plt.legend(unique_handles, unique_labels, **default_kwargs)


# === SAVE FUNCTIONS ===
def _extract_plot_base_name(filename):
    """Extract base name from filename by removing parametric suffixes.

    Removes common parametric suffixes like:
    - _co2l{value} or _co2{value}
    - _conditioning_{value}
    - _{numeric_id}
    - _GSS, _normalized, _picked, etc. (common boolean/categorical suffixes)
    - _same_corridor, _different_corridor
    - other common conditional suffixes

    This allows plots that are variations of the same type (differing by parameters)
    to be grouped in the same subdirectory.

    Examples:
        weather_regime_cluster_comparison_co2l0.6 -> weather_regime_cluster_comparison
        regime_1d_histograms_co2l0.5 -> regime_1d_histograms
        initial_network_cascade_ES_FR_1 -> initial_network_cascade_ES_FR
        weather_regime_blackout_frequency_GSS -> weather_regime_blackout_frequency
        renewable_cap_blackout_statistics_same_corridor -> renewable_cap_blackout_statistics
        renewable_capacity_factor_grid_normalized_by_frequency -> renewable_capacity_factor_grid
    """
    import re

    name = filename

    # Apply removals iteratively until no more changes
    # This handles cases with multiple suffixes
    prev_name = None
    max_iterations = 5
    iteration = 0

    while name != prev_name and iteration < max_iterations:
        prev_name = name
        iteration += 1

        # Remove trailing CO2 level specifications (co2l, co2)
        name = re.sub(r"_(?:co2l?)[0-9.]+$", "", name, flags=re.IGNORECASE)
        # Remove conditioning specifications
        name = re.sub(r"_conditioning_\w+$", "", name, flags=re.IGNORECASE)
        # Remove common boolean flag suffixes
        name = re.sub(
            r"_(?:GSS|normalized|picked|normalized_by_\w+)$",
            "",
            name,
            flags=re.IGNORECASE,
        )
        # Remove corridor and similar categorical suffixes
        name = re.sub(r"_(?:same|different)_corridor$", "", name, flags=re.IGNORECASE)
        name = re.sub(r"_(?:co2_)?picked$", "", name, flags=re.IGNORECASE)
        # Remove trailing numeric indices (but be careful: only remove if last part is purely numeric)
        name = re.sub(r"_\d+$", "", name)

    return name


def save_figure(
    fig,
    filename,
    save_path=None,
    formats=None,
    dpi=None,
    bbox_inches="tight",
    clear_fig=True,
    organize_plots=True,
):
    """Save figure with consistent settings and optional configuration suffix.

    Signature:
        save_figure(fig, filename, save_path=None, formats=None, dpi=None, bbox_inches="tight",
                   clear_fig=True, organize_plots=True)

    Args:
        fig: matplotlib figure object
        filename: base filename without extension
        save_path: directory to save into; if None, uses PLOT_OUTPUT_DIR
        formats: list of formats to save (default: ["pdf"])
        dpi: figure DPI (if None, uses environment variable PLOT_DPI)
        bbox_inches: bounding box setting for saving
        clear_fig: if True, clear and close the figure after saving
        organize_plots: if True, organize plots into subdirectories based on base name
    """
    import os

    if save_path is None:
        config = get_plot_config()
        if not config["output_dir"]:
            raise ValueError("save_figure requires save_path or set PLOT_OUTPUT_DIR")
        save_path = config["output_dir"]

    # Get configuration
    config = get_plot_config()

    # Use environment variable defaults if not specified
    if formats is None:
        formats = ["pdf"]
    if dpi is None:
        dpi = config["dpi"]

    # Add config suffix to filename if enabled
    if config["save_with_config"]:
        config_suffix = get_config_suffix()
        base_filename, ext = os.path.splitext(filename)
        ext = ext.lstrip(".")
        format_set = {fmt.lstrip(".") for fmt in formats}
        if ext and ext in format_set:
            filename = base_filename + config_suffix
        else:
            filename = filename + config_suffix

    # Organize plots into subdirectories if enabled
    if organize_plots:
        base_name = _extract_plot_base_name(filename)
        if base_name != filename:  # Only create subdirectory if name was modified
            save_path = os.path.join(save_path, base_name)

    os.makedirs(save_path, exist_ok=True)

    for fmt in formats:
        full_path = os.path.join(save_path, f"{filename}.{fmt}")
        fig.savefig(full_path, format=fmt, dpi=dpi, bbox_inches=bbox_inches)
        if clear_fig:
            fig.clear()
            plt.close(fig)

        assert os.path.isfile(full_path), f"Failed to save figure: {full_path}"
        logger.info(f"Saved: {full_path}")


def savefig_organized(fig, file_path, organize_plots=True, **savefig_kwargs):
    """Save figure with optional plot organization into subdirectories.

    Wrapper around plt.savefig that organizes plots into subdirectories based on base name.
    Useful for scripts using plt.savefig directly instead of save_figure().

    Args:
        fig: matplotlib figure object
        file_path: full file path or filename (can include directory and extension)
        organize_plots: if True, organize into subdirectory based on base name
        **savefig_kwargs: additional arguments passed to fig.savefig() (dpi, bbox_inches, etc.)

    Example:
        savefig_organized(fig, "/path/plots/regime_1d_histograms_co2l0.5.pdf")
        # Saves to: /path/plots/regime_1d_histograms/regime_1d_histograms_co2l0.5.pdf
    """
    import os

    file_path = str(file_path)

    if organize_plots:
        # Split path into directory and filename
        dir_path = os.path.dirname(file_path)
        filename_with_ext = os.path.basename(file_path)
        filename_base, ext = os.path.splitext(filename_with_ext)

        # Extract base name (removes parametric suffixes)
        base_name = _extract_plot_base_name(filename_base)

        # Create subdirectory based on base name
        if base_name != filename_base:  # Only organize if name was modified
            dir_path = os.path.join(dir_path, base_name)

        file_path = os.path.join(dir_path, f"{filename_base}{ext}")

    # Ensure directory exists
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # Save figure
    fig.savefig(file_path, **savefig_kwargs)
    logger.info(f"Saved: {file_path}")


# === UTILITY FUNCTIONS ===
def format_co2_title(co2_level, n_nodes=600, percent=True):
    """Format CO2 level for plot titles."""
    from utils.data_handling import get_actual_co2_level

    actual_level = get_actual_co2_level(co2_level, n_nodes, percent=percent)
    return (
        rf"CO$_2$ = {int(actual_level)}\%"
        if percent
        else rf"CO$_2$ = {actual_level:.1f}"
    )


def set_axis_style(
    ax,
    xlabel=None,
    ylabel=None,
    title=None,
    grid=True,
    tick_labelsize=None,
    label_fontsize=None,
    title_fontsize=None,
):
    """Apply consistent axis styling."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fontsize or AXIS_LABEL_FONTSIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize or AXIS_LABEL_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=title_fontsize or SUBTITLE_FONTSIZE)

    if grid:
        ax.grid(True, alpha=GRID_ALPHA)

    ax.tick_params(
        axis="both", which="both", labelsize=tick_labelsize or TICK_LABEL_FONTSIZE
    )


# === CUSTOM COLORMAPS ===
def create_failure_probability_colormap():
    """Create colormap specifically for failure probabilities."""
    from matplotlib.colors import LinearSegmentedColormap

    colors = ["gainsboro", "#440154", "#31688e", "#35b779", "#fde725"]
    return LinearSegmentedColormap.from_list("failure_prob", colors)


# Initialize matplotlib with our style when imported
setup_matplotlib_style()
