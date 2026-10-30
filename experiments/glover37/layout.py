"""
Network layout: PowerWorld .axd display parsing and the one-line diagram.

The parser reads the display objects of a PowerWorld .axd file (DisplayBus,
DisplayBusField, DisplayGen, DisplayTransmissionLine, DisplayTransformer). For
lines and transformers, <SUBDATA Line> coordinates are preserved so routed
bends from the original one-line diagram remain visible. The diagram uses that
geometry; it never modifies the AXD data.

Usage
-----
    python -m experiments.glover37.layout          # figures/glover37/network.pdf
    python -m experiments.glover37.layout --svg

Inputs
------
cases/glover37.json
data/powerworld/glover37/base/DesignCase2_2010.axd
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from experiments.glover37.config import AXD_FILE, CASE_FILE, FIGURES_DIR as FIGURES_ROOT


# =====================================================================
# AXD parsing
# =====================================================================

POINT_RE = re.compile(
    r"\[\s*([-+]?\d+(?:[\.,]\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:[\.,]\d+)?(?:[eE][-+]?\d+)?)\s*\]"
)

GENERATOR_BODY_SIZE = 3.96


def pw_float(text: str) -> float:
    """Convert PowerWorld numeric text, including decimal commas."""
    return float(text.strip().replace(",", "."))


def _tokenize(line: str) -> list[str]:
    """Split one AXD record while preserving quoted fields."""
    return shlex.split(
        line,
        comments=False,
        posix=True,
    )


@dataclass
class Bus:
    number: int
    auxiliary_id: str
    x: float
    y: float
    thickness: float
    size: float
    width: float
    orientation: str
    style: str


@dataclass
class BusField:
    """
    DisplayBusField geometry and basic text formatting.

    Only fields needed for geometry/rendering are parsed. The original
    AXD record contains additional PowerWorld display-state properties.
    """
    bus: int
    auxiliary_id: str
    variable_name: str
    x: float
    y: float
    thickness: float
    orientation: str
    font_name: str
    font_size: float
    font_styles: str
    rotation_angle: float


@dataclass
class Generator:
    bus: int
    gen_id: str
    auxiliary_id: str
    x: float
    y: float
    thickness: float
    size: float
    width: float
    orientation: str
    style: str


@dataclass
class Edge:
    kind: str
    from_bus: int
    to_bus: int
    circuit: str
    auxiliary_id: str
    x: float
    y: float
    thickness: float
    coordinates: list[tuple[float, float]]
    symbol_segment: Optional[int] = None


@dataclass
class AxdModel:
    buses: list[Bus]
    bus_fields: list[BusField]
    generators: list[Generator]
    lines: list[Edge]
    transformers: list[Edge]

    @property
    def edges(self) -> list[Edge]:
        return self.lines + self.transformers

    def bus_field(
        self,
        bus: int,
        variable_name: str,
    ) -> BusField | None:
        """
        Return the first DisplayBusField matching bus and variable name.
        Variable-name matching is case-insensitive.
        """
        target = variable_name.strip().lower()

        return next(
            (
                field
                for field in self.bus_fields
                if field.bus == bus
                and field.variable_name.strip().lower() == target
            ),
            None,
        )

    def bus_field_position(
        self,
        bus: int,
        variable_name: str = "Name",
    ) -> tuple[float, float] | None:
        """
        Return the AXD anchor position of a displayed bus field.
        """
        field = self.bus_field(bus, variable_name)

        if field is None:
            return None

        return field.x, field.y


def _find_block(
    lines: list[str],
    name: str,
) -> tuple[int, int]:
    """
    Return inclusive content bounds inside a named top-level AXD block.
    """
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if line.lstrip().startswith(name + " (")
        ),
        None,
    )

    if start is None:
        raise ValueError(f"Block {name!r} not found")

    i = start

    while i < len(lines) and "{" not in lines[i]:
        i += 1

    if i >= len(lines):
        raise ValueError(
            f"Opening '{{' for block {name!r} not found"
        )

    content_start = i + 1
    depth = 1
    i = content_start

    while i < len(lines):
        depth += lines[i].count("{")
        depth -= lines[i].count("}")

        if depth == 0:
            return content_start, i - 1

        i += 1

    raise ValueError(f"Closing '}}' for block {name!r} not found")


def _parse_bus_record(line: str) -> Bus:
    t = _tokenize(line)

    if len(t) < 16:
        raise ValueError(
            "Unexpected DisplayBus record: "
            f"{line.rstrip()}"
        )

    return Bus(
        number=int(t[0]),
        auxiliary_id=t[1].strip(),
        x=pw_float(t[2]),
        y=pw_float(t[3]),
        thickness=pw_float(t[4]),
        size=pw_float(t[8]),
        width=pw_float(t[9]),
        orientation=t[10].strip(),
        style=t[15].strip(),
    )


def _parse_bus_field_record(line: str) -> BusField:
    """
    Parse the leading geometry/text-format columns of DisplayBusField.

    AXD column order used here:
      0 BusNum
      1 SOAuxiliaryID
      2 VariableName
      3 SOX
      4 SOY
      5 SOThickness
     12 SOOrientation
     17 FontName
     19 SOFontSize
     20 FontStyles
     23 RotationAngle
    """
    t = _tokenize(line)

    if len(t) < 24:
        raise ValueError(
            "Unexpected DisplayBusField record: "
            f"{line.rstrip()}"
        )

    return BusField(
        bus=int(t[0]),
        auxiliary_id=t[1].strip(),
        variable_name=t[2].strip(),
        x=pw_float(t[3]),
        y=pw_float(t[4]),
        thickness=pw_float(t[5]),
        orientation=t[12].strip(),
        font_name=t[17].strip(),
        font_size=pw_float(t[19]),
        font_styles=t[20].strip(),
        rotation_angle=pw_float(t[23]),
    )


def _parse_generator_record(line: str) -> Generator:
    t = _tokenize(line)

    if len(t) < 20:
        raise ValueError(
            "Unexpected DisplayGen record: "
            f"{line.rstrip()}"
        )

    return Generator(
        bus=int(t[0]),
        gen_id=t[1].strip(),
        auxiliary_id=t[2].strip(),
        x=pw_float(t[3]),
        y=pw_float(t[4]),
        thickness=pw_float(t[5]),
        size=pw_float(t[11]),
        width=pw_float(t[12]),
        orientation=t[14].strip(),
        style=t[19].strip(),
    )


def _parse_edge_record(
    line: str,
    kind: str,
) -> Edge:
    t = _tokenize(line)

    if len(t) < 8:
        raise ValueError(
            "Unexpected edge record: "
            f"{line.rstrip()}"
        )

    symbol_segment = None

    if kind == "transformer":
        try:
            symbol_segment = int(t[-1])
        except (ValueError, TypeError):
            symbol_segment = None

    return Edge(
        kind=kind,
        from_bus=int(t[0]),
        to_bus=int(t[1]),
        circuit=t[2].strip(),
        auxiliary_id=t[3].strip(),
        x=pw_float(t[4]),
        y=pw_float(t[5]),
        thickness=1.0,
        coordinates=[],
        symbol_segment=symbol_segment,
    )


def _parse_display_bus(
    lines: list[str],
) -> list[Bus]:
    a, b = _find_block(lines, "DisplayBus")
    buses = []

    for raw in lines[a : b + 1]:
        s = raw.strip()

        if not s or s.startswith("//") or s.startswith("<"):
            continue

        buses.append(_parse_bus_record(raw))

    return buses


def _parse_display_bus_field(
    lines: list[str],
) -> list[BusField]:
    a, b = _find_block(lines, "DisplayBusField")
    fields = []

    for raw in lines[a : b + 1]:
        s = raw.strip()

        if not s or s.startswith("//") or s.startswith("<"):
            continue

        fields.append(_parse_bus_field_record(raw))

    return fields


def _parse_display_gen(
    lines: list[str],
) -> list[Generator]:
    a, b = _find_block(lines, "DisplayGen")
    generators = []

    for raw in lines[a : b + 1]:
        s = raw.strip()

        if not s or s.startswith("//") or s.startswith("<"):
            continue

        generators.append(_parse_generator_record(raw))

    return generators


def _parse_edge_block(
    lines: list[str],
    block_name: str,
    kind: str,
) -> list[Edge]:
    a, b = _find_block(
        lines,
        block_name,
    )

    edges = []
    i = a

    while i <= b:
        s = lines[i].strip()

        if not s or s.startswith("//"):
            i += 1
            continue

        if s.startswith("<SUBDATA") or s.startswith("</SUBDATA"):
            i += 1
            continue

        edge = _parse_edge_record(
            lines[i],
            kind,
        )

        i += 1

        while i <= b and not lines[i].strip():
            i += 1

        if (
            i <= b
            and lines[i]
            .strip()
            .lower()
            .startswith("<subdata line")
        ):
            i += 1
            coords = []

            while (
                i <= b
                and not lines[i]
                .strip()
                .lower()
                .startswith("</subdata")
            ):
                for mx, my in POINT_RE.findall(lines[i]):
                    coords.append(
                        (
                            pw_float(mx),
                            pw_float(my),
                        )
                    )
                i += 1

            edge.coordinates = coords

            if i <= b:
                i += 1

        edges.append(edge)

    return edges


def parse_axd(
    path: str | Path,
) -> AxdModel:
    path = Path(path)

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = text.splitlines()

    buses = _parse_display_bus(lines)
    bus_fields = _parse_display_bus_field(lines)
    generators = _parse_display_gen(lines)

    lines_out = _parse_edge_block(
        lines,
        "DisplayTransmissionLine",
        "line",
    )

    transformers = _parse_edge_block(
        lines,
        "DisplayTransformer",
        "transformer",
    )

    bus_xy = {
        bus.number: (bus.x, bus.y)
        for bus in buses
    }

    for edge in lines_out + transformers:
        if not edge.coordinates:
            p0 = bus_xy.get(
                edge.from_bus,
                (edge.x, edge.y),
            )
            p1 = bus_xy.get(
                edge.to_bus,
                p0,
            )
            edge.coordinates = [p0, p1]

    return AxdModel(
        buses=buses,
        bus_fields=bus_fields,
        generators=generators,
        lines=lines_out,
        transformers=transformers,
    )


def _segment_midpoint(
    points: list[tuple[float, float]],
    one_based_segment: Optional[int],
) -> tuple[float, float, float, float]:
    """
    Return midpoint and unit direction of a selected polyline segment.
    """
    if len(points) < 2:
        x, y = (
            points[0]
            if points
            else (0.0, 0.0)
        )
        return x, y, 1.0, 0.0

    if (
        one_based_segment is not None
        and 1 <= one_based_segment < len(points)
    ):
        i = one_based_segment - 1
        (x0, y0), (x1, y1) = points[i], points[i + 1]
    else:
        lengths = [
            math.hypot(
                x1 - x0,
                y1 - y0,
            )
            for (x0, y0), (x1, y1)
            in zip(points, points[1:])
        ]

        total = sum(lengths)
        target = total / 2.0
        acc = 0.0
        i = 0

        for i, seg_len in enumerate(lengths):
            if acc + seg_len >= target:
                break
            acc += seg_len

        (x0, y0), (x1, y1) = points[i], points[i + 1]

    dx = x1 - x0
    dy = y1 - y0
    norm = math.hypot(dx, dy) or 1.0

    return (
        (x0 + x1) / 2.0,
        (y0 + y1) / 2.0,
        dx / norm,
        dy / norm,
    )


def draw_generator_symbol(
    ax,
    generator: Generator,
    edgecolor="0.35",
    facecolor="white",
    linewidth: float = 1.2,
    zorder: int = 5,
) -> None:
    """
    Draw a PowerWorld DisplayGen using its AXD anchor,
    orientation, and total size.

    All generator bodies use the same diameter as G28
    for visual consistency.
    """
    from matplotlib.patches import Circle

    orientation = generator.orientation.lower()
    x = generator.x
    y = generator.y

    total_length = generator.size
    body_size = GENERATOR_BODY_SIZE
    radius = body_size / 2.0
    center_offset = total_length - radius
    stem_length = max(
        total_length - body_size,
        0.0,
    )

    if orientation == "right":
        cx = x + center_offset
        cy = y
        ax.plot(
            [x, x + stem_length],
            [y, y],
            color=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )

    elif orientation == "left":
        cx = x - center_offset
        cy = y
        ax.plot(
            [x, x - stem_length],
            [y, y],
            color=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )

    elif orientation == "up":
        cx = x
        cy = y + center_offset
        ax.plot(
            [x, x],
            [y, y + stem_length],
            color=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )

    elif orientation == "down":
        cx = x
        cy = y - center_offset
        ax.plot(
            [x, x],
            [y, y - stem_length],
            color=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )

    else:
        raise ValueError(
            f"Unsupported generator orientation "
            f"{generator.orientation!r} for "
            f"generator {generator.bus}:{generator.gen_id}"
        )

    ax.add_patch(
        Circle(
            (cx, cy),
            radius=radius,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )
    )


# =====================================================================
# Publication settings
# =====================================================================

# Measured from the supplied IEEE conference template.
IEEE_COLUMN_WIDTH_IN = 3.49
IEEE_TEXT_WIDTH_IN = 7.14

# IEEE template guidance for text embedded in figures.
FIGURE_FONT_SIZE_PT = 8.0

# Display-only widening of bus rectangles. This does not modify AXD data.
BUS_WIDTH_SCALE = 2.5

# One restrained highlight style for all study modifications.
HIGHLIGHT_COLOR = "#2166ac"
HIGHLIGHT_LINESTYLE = "--"



# =====================================================================
# Case helpers
# =====================================================================

def active_generator_keys() -> set[tuple[int, str]]:
    """
    Return generators that are active in the actual JSON case.

    AXD files may still contain display objects for equipment that is
    out of service, so AXD presence alone is not an activity test.
    """
    case = json.loads(
        CASE_FILE.read_text(encoding="utf-8")
    )

    return {
        (
            int(generator["from_bus"]),
            generator["name"].split("-", 1)[1].strip(),
        )
        for generator in case["elements"]["generators"]
        if int(generator.get("status", 1)) == 1
    }


# =====================================================================
# AXD lookup helpers
# =====================================================================

def find_generator(
    model: AxdModel,
    *,
    bus: int,
    gen_id: str,
) -> Generator:
    gen_id = str(gen_id).strip()

    generator = next(
        (
            item
            for item in model.generators
            if item.bus == bus
            and item.gen_id.strip() == gen_id
        ),
        None,
    )

    if generator is None:
        raise ValueError(
            f"Generator {bus}:{gen_id} not found in AXD."
        )

    return generator


def find_edge(
    model: AxdModel,
    *,
    kind: str,
    from_bus: int,
    to_bus: int,
    circuit: str,
):
    if kind == "line":
        candidates = model.lines
    elif kind == "transformer":
        candidates = model.transformers
    else:
        raise ValueError(
            f"Unsupported edge kind: {kind}"
        )

    edge = next(
        (
            item
            for item in candidates
            if {
                item.from_bus,
                item.to_bus,
            }
            == {
                from_bus,
                to_bus,
            }
            and str(item.circuit).strip()
            == str(circuit).strip()
        ),
        None,
    )

    if edge is None:
        raise ValueError(
            f"{kind} {from_bus}-{to_bus} "
            f"ckt {circuit} not found in AXD."
        )

    return edge


def bus_label_position(
    model: AxdModel,
    bus_number: int,
) -> tuple[float, float] | None:
    """
    Use the original AXD DisplayBusField position.

    This requires the updated axd_grid_viewer.py in which AxdModel
    exposes bus_field_position(). The Name-field anchor is reused for
    the bus number shown in the paper figure.
    """
    getter = getattr(
        model,
        "bus_field_position",
        None,
    )

    if getter is None:
        raise RuntimeError(
            "AxdModel does not expose bus_field_position(). "
            "Use the updated tools/axd_grid_viewer.py that parses "
            "DisplayBusField."
        )

    return getter(bus_number, "Name")


# =====================================================================
# Drawing primitives
# =====================================================================

def draw_network_background(
    ax,
    model: AxdModel,
) -> None:
    """
    Draw passive network geometry and active generators in subdued gray.
    """
    # Lines
    for edge in model.lines:
        xs = [p[0] for p in edge.coordinates]
        ys = [p[1] for p in edge.coordinates]

        ax.plot(
            xs,
            ys,
            color="0.78",
            linewidth=max(
                0.7,
                0.7 * edge.thickness,
            ),
            zorder=1,
        )

    # Transformers
    for edge in model.transformers:
        xs = [p[0] for p in edge.coordinates]
        ys = [p[1] for p in edge.coordinates]

        ax.plot(
            xs,
            ys,
            color="0.70",
            linewidth=max(
                0.8,
                0.8 * edge.thickness,
            ),
            zorder=2,
        )

        mx, my, ux, uy = _segment_midpoint(
            edge.coordinates,
            edge.symbol_segment,
        )

        radius = 0.65
        offset = 0.55

        for sign in (-1.0, 1.0):
            cx = mx + sign * offset * ux
            cy = my + sign * offset * uy

            ax.add_patch(
                Circle(
                    (cx, cy),
                    radius=radius,
                    facecolor="white",
                    edgecolor="0.70",
                    linewidth=1.0,
                    zorder=4,
                )
            )

    # Only generators that are active in the actual case.
    active = active_generator_keys()

    for generator in model.generators:
        key = (
            generator.bus,
            generator.gen_id.strip(),
        )

        if key not in active:
            continue

        draw_generator_symbol(
            ax,
            generator,
            edgecolor="0.55",
            facecolor="white",
            linewidth=1.0,
            zorder=4,
        )


def draw_bus(
    ax,
    model: AxdModel,
    bus,
    *,
    facecolor: str,
    value: float | None = None,
    show_value: bool = False,
    font_size: float = FIGURE_FONT_SIZE_PT,
    label_color: str = "black",
    label_weight: str = "normal",
) -> None:
    """
    Draw one bus rectangle and its paper label.

    Bus geometry comes from DisplayBus.
    Label placement comes from DisplayBusField(Name).
    """
    orientation = bus.orientation.lower()

    length = bus.size
    width = bus.width * BUS_WIDTH_SCALE

    if orientation == "right":
        x0 = bus.x
        y0 = bus.y - width / 2.0
        rect_width = length
        rect_height = width

    elif orientation == "left":
        x0 = bus.x - length
        y0 = bus.y - width / 2.0
        rect_width = length
        rect_height = width

    elif orientation == "up":
        x0 = bus.x - width / 2.0
        y0 = bus.y
        rect_width = width
        rect_height = length

    elif orientation == "down":
        x0 = bus.x - width / 2.0
        y0 = bus.y - length
        rect_width = width
        rect_height = length

    else:
        raise ValueError(
            f"Unsupported bus orientation "
            f"{bus.orientation!r} for bus {bus.number}."
        )

    ax.add_patch(
        Rectangle(
            (x0, y0),
            rect_width,
            rect_height,
            facecolor=facecolor,
            edgecolor="0.35",
            linewidth=0.8,
            zorder=9,
        )
    )

    label_xy = bus_label_position(
        model,
        bus.number,
    )

    if label_xy is None:
        # Defensive fallback only. With the current AXD all displayed
        # buses are expected to have a Name field.
        label_xy = (bus.x, bus.y)

    if show_value and value is not None:
        label = f"{bus.number}\n{value:+.1f}%"
    else:
        label = str(bus.number)

    ax.text(
        label_xy[0],
        label_xy[1],
        label,
        fontsize=font_size,
        ha="left",
        va="bottom",
        color=label_color,
        fontweight=label_weight,
        clip_on=False,
        zorder=30,
    )


def draw_highlighted_generator(
    ax,
    generator: Generator,
    *,
    tag: str | None,
) -> None:
    draw_generator_symbol(
        ax,
        generator,
        edgecolor=HIGHLIGHT_COLOR,
        facecolor="white",
        linewidth=2.2,
        zorder=15,
    )

    if tag is not None:
        ax.annotate(
            tag,
            xy=(generator.x, generator.y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=FIGURE_FONT_SIZE_PT,
            fontweight="bold",
            color=HIGHLIGHT_COLOR,
            ha="left",
            va="bottom",
            annotation_clip=False,
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": HIGHLIGHT_COLOR,
                "linewidth": 0.6,
            },
            zorder=40,
        )


def draw_highlighted_edge(
    ax,
    edge,
    *,
    tag: str | None,
) -> None:
    xs = [p[0] for p in edge.coordinates]
    ys = [p[1] for p in edge.coordinates]

    ax.plot(
        xs,
        ys,
        color=HIGHLIGHT_COLOR,
        linewidth=2.0,
        linestyle=HIGHLIGHT_LINESTYLE,
        zorder=15,
    )

    if tag is None:
        return

    mx, my, _, _ = _segment_midpoint(
        edge.coordinates,
        edge.symbol_segment,
    )

    ax.annotate(
        tag,
        xy=(mx, my),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=FIGURE_FONT_SIZE_PT,
        fontweight="bold",
        color=HIGHLIGHT_COLOR,
        ha="left",
        va="bottom",
        annotation_clip=False,
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": HIGHLIGHT_COLOR,
            "linewidth": 0.6,
        },
        zorder=40,
    )


# =====================================================================
# Figure sizing / cropping
# =====================================================================

def fit_axes_to_drawn_network(
    fig,
    ax,
    *,
    target_width_in: float,
    data_margin_fraction: float = 0.008,
) -> None:
    """
    Fit the axes tightly around the actual network artists.

    Important:
    - matplotlib line/patch artists determine the primary-element bounds;
    - text/annotations are deliberately NOT used to enlarge the data
      limits, but bbox_inches='tight' includes them on export;
    - figure height is derived from the resulting data aspect ratio;
    - therefore the network fills the requested width without clipping
      the primary elements or leaving avoidable left/right whitespace.
    """
    ax.relim()
    ax.autoscale_view(tight=True)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)

    xpad = data_margin_fraction * xspan
    ypad = data_margin_fraction * yspan

    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    xspan = xmax - xmin
    yspan = ymax - ymin

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.0)
    ax.set_axis_off()

    target_height_in = (
        target_width_in
        * yspan
        / xspan
    )

    fig.set_size_inches(
        target_width_in,
        target_height_in,
        forward=True,
    )

    fig.subplots_adjust(
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1.0,
    )


def save_figure(
    fig,
    stem: Path,
    *,
    save_svg: bool = False,
) -> list[Path]:
    """Save the publication PDF and, optionally, an SVG copy."""
    FIGURES_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    written: list[Path] = []

    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.01,
    )
    written.append(pdf_path)

    if save_svg:
        svg_path = stem.with_suffix(".svg")
        fig.savefig(
            svg_path,
            format="svg",
            bbox_inches="tight",
            pad_inches=0.01,
        )
        written.append(svg_path)

    return written

# =====================================================================
# Network diagram
# =====================================================================

def draw_network(ax, items=(), bus_colors=None):
    """Draw the benchmark topology onto `ax`; `bus_colors` {bus number: color} fills and
    labels those buses in that color."""
    bus_colors = bus_colors or {}
    model = parse_axd(AXD_FILE)
    draw_network_background(ax, model)
    for bus in model.buses:
        color = bus_colors.get(bus.number)
        draw_bus(ax, model, bus, facecolor=color or "white", font_size=FIGURE_FONT_SIZE_PT,
                 label_color=color or "black", label_weight="bold" if color else "normal")
    for item in items:
        if item["kind"] == "generator":
            generator = find_generator(model, bus=item["bus"], gen_id=item["gen_id"])
            draw_highlighted_generator(ax, generator, tag=item["tag"])
        else:
            edge = find_edge(model, kind=item["kind"], from_bus=item["from_bus"],
                             to_bus=item["to_bus"], circuit=item["circuit"])
            draw_highlighted_edge(ax, edge, tag=item["tag"])
    return model


def render_network(
    items=(),
    *,
    stem: str = "network",
    width_in: float = IEEE_COLUMN_WIDTH_IN,
    save_svg: bool = False,
    show: bool = False,
) -> list[Path]:
    """
    Render the benchmark topology from the PowerWorld display geometry.

    `items` are elements to highlight, each a dict with "tag" and "kind"
    ("generator": bus, gen_id; "line" / "transformer": from_bus, to_bus, circuit).
    """
    model = parse_axd(AXD_FILE)

    fig, ax = plt.subplots()

    draw_network_background(
        ax,
        model,
    )

    for bus in model.buses:
        draw_bus(
            ax,
            model,
            bus,
            facecolor="white",
            font_size=FIGURE_FONT_SIZE_PT,
        )

    for item in items:
        if item["kind"] == "generator":
            generator = find_generator(model, bus=item["bus"], gen_id=item["gen_id"])
            draw_highlighted_generator(ax, generator, tag=item["tag"])
        else:
            edge = find_edge(model, kind=item["kind"], from_bus=item["from_bus"],
                             to_bus=item["to_bus"], circuit=item["circuit"])
            draw_highlighted_edge(ax, edge, tag=item["tag"])

    fit_axes_to_drawn_network(
        fig,
        ax,
        target_width_in=width_in,
    )

    written = save_figure(
        fig,
        FIGURES_ROOT / stem,
        save_svg=save_svg,
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    for path in written:
        print(f"Wrote: {path}")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Glover 37-bus one-line diagram.")
    parser.add_argument("--width", type=float, default=IEEE_COLUMN_WIDTH_IN,
                        help="Output width in inches (default: one IEEE column).")
    parser.add_argument("--svg", action="store_true", help="Also save an SVG copy.")
    parser.add_argument("--show", action="store_true", help="Open the matplotlib window.")
    args = parser.parse_args()
    render_network(width_in=args.width, save_svg=args.svg, show=args.show)


if __name__ == "__main__":
    main()
