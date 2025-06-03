import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patheffects import Stroke, Normal


def draw_polygon(
    ax, proj, lon, lat, edgecolor="red", linestyle="solid", facecolor=None, **kwargs
):
    """Plot a polygon from a list of lon, lat coordinates.

    This routine is a convenience wrapper around plot() and fill(), both
    of which work in geodesic coordinates.

    Parameters
    ----------
    lon : `np.ndarray`
        Array of longitude points in polygon.
    lat : `np.ndarray`
        Array of latitude points in polygon.
    edgecolor : `str`, optional
        Color of polygon boundary.  Set to None for no boundary.
    linestyle : `str`, optional
        Line style for boundary.
    facecolor : `str`, optional
        Color of polygon face.  Set to None for no fill color.
    **kwargs : `dict`, optional
        Additional keywords passed to plot.
    """
    plt.sca(ax)
    lon, lat = np.append(lon, lon[0]), np.append(lat, lat[0])
    x, y = proj.ang2xy(lon, lat, lonlat=True)
    if linestyle is not None and edgecolor is not None:
        plt.plot(x, y, color=edgecolor, linestyle=linestyle, **kwargs)
    if facecolor is not None:
        plt.fill(x, y, color=facecolor, **kwargs)


def load_lsst_line(filepath="../data/annotations/lsst_boundary.csv"):
    # get ra/dec coordinates of the LSST footprint from a CSV file
    lsst_coords = pd.read_csv(filepath)
    lsst_coords = lsst_coords.sort_values("wrap_order")
    lsst_ra = lsst_coords["ra"].values
    lsst_ra[lsst_ra > 180] -= 360
    lsst_dec = lsst_coords["dec"].values
    return lsst_ra, lsst_dec


def load_des_polygon(filepath="../data/annotations/des-round19-poly.txt"):
    # get ra/dec coordinates of the DES polygon from a text file
    des_coords = np.genfromtxt(filepath, names=["ra", "dec"])
    des_ra = np.append(des_coords["ra"], des_coords["ra"][0])
    des_dec = np.append(des_coords["dec"], des_coords["dec"][0])
    return des_ra, des_dec


def plot_stream_annotations(
    ax,
    annotations_dict,
    proj,
    color="white",
    fontsize=12,
    arrow_style="->",
    arrow_color="white",
    arrow_linewidth=1.2,
    outline=False,
    outline_width=2,
    ha="center",
    va="center",
):
    """
    Plots stream annotations with RA/Dec coordinates on a given Axes.

    Parameters:
        ax: matplotlib.axes.Axes
            The axes to draw on
        annotations_dict: dict
            Dictionary with stream names and their RA/Dec coordinates
        color: str
            Text color (default: "white")
        fontsize: int
            Font size (default: 12)
        arrow_style: str
            Arrow style (default: "->")
        arrow_color: str
            Arrow color (default: "white")
        arrow_linewidth: float
            Arrow line width (default: 1.2)
        outline: bool
            Add text outline (default: False)
        outline_width: int
            Outline stroke width (default: 2)
        ha/va: str
            Horizontal/vertical alignment (default: "center")
        projection: callable
            Function to convert (ra,dec) → (x,y)
    """
    # Create arrowprops if any arrow style is specified
    arrowprops = None
    if arrow_style:
        arrowprops = dict(
            arrowstyle=arrow_style,
            color=arrow_color,
            lw=arrow_linewidth,
            # Add arrow outline effect
            path_effects=(
                [Stroke(linewidth=arrow_linewidth + 1, foreground="black"), Normal()]
                if outline
                else None
            ),  # Only add if outline is True
        )

    for text_label, params in annotations_dict.items():
        # Convert RA/Dec to plot coordinates
        x, y = proj.ang2xy(params["ra"], params["dec"], lonlat=True)

        # Convert offset to text position
        text_ra = params["ra"] + params.get("ra_offset", 0)
        text_dec = params["dec"] + params.get("dec_offset", 0)
        x_text, y_text = proj.ang2xy(text_ra, text_dec, lonlat=True)

        # Create annotation
        ann = ax.annotate(
            text_label,
            xy=(x, y),
            xytext=(x_text, y_text),
            color=color,
            fontsize=fontsize,
            ha=ha,
            va=va,
            arrowprops=arrowprops,
        )

        # Add outline effect if specified
        if outline:
            ann.set_path_effects(
                [Stroke(linewidth=outline_width, foreground="black"), Normal()]
            )


def plot_fos(
    ax,
    image,
    proj,
    DES=True,
    LSST=True,
    annotations=True,
    annotations_dict=None,
    white_background=False,
    grid=False,
    grid_color="white",
    grid_alpha=0.5,
    name=True,
):
    """
    Plots the Field of Streams (FoS) image with survey footprints and annotations.

    Parameters:
        ax: matplotlib.axes.Axes
            The axes to draw on
        image: array-like
            Image data to display
        proj: callable
            Projection object with ang2xy() for coordinate conversion
        DES: bool
            Plot DES survey footprint (default: True)
        LSST: bool
            Plot LSST survey footprint (default: True)
        annotations: bool
            Add stream annotations (default: True)
        annotations_dict: dict
            Stream annotation data (required if annotations=True)
        oval: bool
            Draw black oval around image edges (default: True)
        white_background: bool
            Use black text for white backgrounds (default: False)
        grid: bool
            Draw coordinate grid (default: False)
        grid_color: str
            Grid line color (default: "white")
        grid_alpha: float
            Grid transparency (0-1) (default: 0.5)
        name: bool
            Display author names (default: True)
    """
    proj_name = proj.name
    if proj_name == "Orthographic":
        oval = False  # No oval for orthographic projection
        annotations = False  # No annotations for orthographic projection
        LSST = False  # No LSST for orthographic projection
        DES = False  # No DES for orthographic projection
        name_y = -0.05  # Adjust name position for orthographic projection
    elif proj_name == "Mollweide":
        oval = True 
        name_y = 0
    else:
        print(f"Warning: Unknown projection {proj_name}")
        name_y = 0

    ax.imshow(image, origin="lower", extent=proj.get_extent())
    if LSST:
        lsst_ra, lsst_dec = load_lsst_line()
        x_lsst, y_lsst = proj.ang2xy(lsst_ra, lsst_dec, lonlat=True)
        ax.plot(x_lsst, y_lsst, c="white", ls="dashed", label="LSST")
    if DES:
        des_ra, des_dec = load_des_polygon()
        x_des, y_des = proj.ang2xy(des_ra, des_dec, lonlat=True)
        ax.plot(x_des, y_des, c="white", ls="solid", label="DES", lw=3)

    if annotations:
        plot_stream_annotations(
            ax,
            annotations_dict,
            proj,
            color="white",
            fontsize=10,
            arrow_style="->",
            outline=True,
        )
    # create black oval around image
    if oval:
        dec_edge = np.linspace(-90, 90, 300)
        ra_edge = np.ones_like(dec_edge) * 180.3
        x_edge, y_edge = proj.ang2xy(ra_edge, dec_edge, lonlat=True)
        ax.plot(x_edge, y_edge, c="k", lw=3, zorder=8)
        x_edge, y_edge = proj.ang2xy(ra_edge * 0 + 179.7, dec_edge, lonlat=True)
        ax.plot(x_edge, y_edge, c="k", lw=3, zorder=8)

    if grid:
        # Draw Dec grid lines
        for dec in np.arange(-80, 90, 10):
            ra_grid = np.linspace(0, 360, 500)
            dec_grid = np.ones_like(ra_grid) * dec
            xg, yg = proj.ang2xy(ra_grid, dec_grid, lonlat=True)
            ax.plot(xg, yg, color=grid_color, lw=0.8, alpha=grid_alpha, zorder=1)
        # Draw RA grid lines
        for ra in np.arange(0, 360, 30):
            dec_grid = np.linspace(-90, 90, 500)
            ra_grid = np.ones_like(dec_grid) * ra
            xg, yg = proj.ang2xy(ra_grid, dec_grid, lonlat=True)
            ax.plot(xg, yg, color=grid_color, lw=0.8, alpha=grid_alpha, zorder=1)

        # Add RA tick labels at Dec=0
        ra_ticks = np.arange(0, 360, 30)
        dec0 = np.zeros_like(ra_ticks)
        x_ticks, y_ticks = proj.ang2xy(ra_ticks, dec0, lonlat=True)
        for x, y, ra in zip(x_ticks, y_ticks, ra_ticks):
            ax.text(
                x,
                y - 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"{int(ra)}°",
                color=grid_color,
                fontsize=10,
                ha="center",
                va="top",
            )

        # Add Dec tick labels at RA=0
        dec_ticks = np.arange(-80, 90, 10)
        ra0 = np.zeros_like(dec_ticks)
        x_ticks, y_ticks = proj.ang2xy(ra0, dec_ticks, lonlat=True)
        for x, y, dec in zip(x_ticks, y_ticks, dec_ticks):
            ax.text(
                x - 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                y,
                f"{int(dec)}°",
                color=grid_color,
                fontsize=10,
                ha="right",
                va="center",
            )

    if DES or LSST:
        leg = ax.legend(
            loc=(0.7, 0.73),
            labelcolor="white",
            facecolor="None",
            edgecolor="None",
            fontsize=20,
        )
        leg.set_zorder(10)

    if name:
        if white_background:
            text_color = "k"
        else:
            text_color = "white"

        ax.text(
            0.01,
            name_y,
            r"Peter Ferguson \& Nora Shipp",
            color=text_color,
            fontsize=18,
            zorder=9,
            ha="left",
            va="bottom",
            transform=ax.transAxes,
        )

    ax.set_axis_off()
    ax.grid(False)


# ToDo: add option to plot galstreams footprints
# mws = galstreams.MWStreams(verbose=False, implement_Off=False,
# print_topcat_friendly_files=False)
# from matplotlib.colors import ListedColormap
# tab10 = plt.get_cmap("tab10").colors
# custom_tab = [tab10[i] for i in range(10) if i not in [7]]
# custom_cmap = ListedColormap(custom_tab)
# stream_keys=['GD-1-I21', 'Jhelum-a-B19']
# for i,key in enumerate(stream_keys):
#     stream_ra = mws[key].track.ra.deg
#     stream_dec = mws[key].track.dec.deg
#     offset=0.02
#     x, y = proj.ang2xy(stream_ra , stream_dec, lonlat=True)
#     plt.plot(x + offset, y + offset, label=key, lw=1,
#              alpha=0.75, c=custom_cmap(i))
#     plt.plot(x - offset, y - offset, lw=1,
#              alpha=0.75, c=custom_cmap(i))
