"""
D3 backend for ordinary ConceptFlow lattice diagrams.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any

from conceptflow.visualization._checks import (
    check_max_label_items,
    html_page_layout,
    object_formatter,
)
from conceptflow.visualization.graph_data import GraphData, needs_rotation
from conceptflow.visualization.html_figure import HTMLFigure, script_json


def graph_data_to_d3_data(graph_data: GraphData) -> dict[str, Any]:
    """
    Convert GraphData to plain JSON-like data for D3.
    """
    return {
        "nodes": [
            {
                "id": node.node_id,
                "label": node.label,
                "hover": node.hover_text,
                "x": node.x,
                "y": node.y,
                "metadata": node.metadata,
            }
            for node in graph_data.nodes
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "metadata": edge.metadata,
            }
            for edge in graph_data.edges
        ],
        "metadata": graph_data.metadata,
        "rotate": needs_rotation(graph_data),
    }


def render_graph_data_html(
    graph_data: GraphData,
    width: int = 900,
    height: int = 700,
    title: str = "Concept lattice",
    show_attribute_labels: bool = True,
    show_object_labels: bool = True,
    max_label_items: int | None = 6,
    object_label: Callable[[str], str] | None = None,
) -> str:
    """
    Render GraphData as a standalone D3 HTML string.

    ``width`` includes the 280 px details panel, which moves below the
    drawing in pages narrower than 560 px.

    Labels follow reduced labelling, as in the other backends: attributes
    above the node that introduces them, objects below. The full extent and
    intent of a concept appear in its tooltip and, on click, in the details
    panel.
    """
    from conceptflow.visualization.static import _join_labels, _reduced_labels

    page = html_page_layout(width, height)
    check_max_label_items(max_label_items)
    data = graph_data_to_d3_data(graph_data)
    attributes_at, objects_at = _reduced_labels(graph_data)
    format_object = object_formatter(object_label)
    for node in data["nodes"]:
        attributes = sorted(attributes_at[node["id"]]) if show_attribute_labels else []
        objects = sorted(map(format_object, objects_at[node["id"]])) if show_object_labels else []
        node["attribute_text"] = _join_labels(attributes, max_label_items)
        node["object_text"] = _join_labels(objects, max_label_items)
    data_json = script_json(data)

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<style>
  body {{
    margin: 0;
    font-family: sans-serif;
    background: #ffffff;
  }}

  .conceptflow-container {{
    width: {width}px;
    max-width: 100%;
    border: 1px solid #ddd;
    border-radius: 10px;
    overflow: hidden;
    background: #fafafa;
  }}

  .conceptflow-header {{
    padding: 10px 14px;
    background: #f2f2f2;
    border-bottom: 1px solid #ddd;
  }}

  .conceptflow-title {{
    font-size: 16px;
    font-weight: 600;
  }}

  .conceptflow-subtitle {{
    font-size: 12px;
    color: #666;
    margin-top: 2px;
  }}

  .conceptflow-body {{
    display: grid;
    grid-template-columns: {page['columns']};
    min-height: {height}px;
  }}

  .conceptflow-details {{
    {page['panel_border']}: 1px solid #ddd;
    background: #fafafa;
    padding: 12px;
    font-size: 12px;
    overflow: auto;
  }}

  .details-title {{
    font-weight: 700;
    margin-bottom: 8px;
  }}

  .details-section {{
    margin-top: 10px;
  }}

  .details-label {{
    font-weight: 600;
    color: #555;
  }}

  svg {{
    display: block;
    background: white;
    cursor: grab;
  }}

  svg:active {{
    cursor: grabbing;
  }}

  .edge {{
    stroke: #555;
    stroke-width: 1.5;
    opacity: 0.65;
  }}

  .edge.highlighted {{
    stroke: #1f4e79;
    stroke-width: 3;
    opacity: 1;
  }}

  /* --node-scale shrinks the outline with the radius (see render). */
  .node {{
    fill: white;
    stroke: black;
    stroke-width: calc(1.6px * var(--node-scale, 1));
    cursor: pointer;
  }}

  .node:hover {{
    stroke: #d97706;
    stroke-width: calc(3px * var(--node-scale, 1));
  }}

  .node.selected {{
    stroke: #d97706;
    stroke-width: calc(3.5px * var(--node-scale, 1));
  }}

  /* Reduced labels: attributes above the node, objects below. They shrink
     with the nodes (--node-scale) and grow back when zooming in. */
  .attr-label, .obj-label {{
    font-size: calc(10px * var(--node-scale, 1));
    text-anchor: middle;
    pointer-events: none;
    paint-order: stroke;
    stroke: white;
    stroke-width: calc(3px * var(--node-scale, 1));
    stroke-linejoin: round;
  }}

  .attr-label {{
    dominant-baseline: auto;
    fill: #1a1a2e;
    font-weight: 600;
  }}

  .obj-label {{
    dominant-baseline: hanging;
    fill: #444;
    font-style: italic;
  }}
</style>
</head>
<body>
<div class="conceptflow-container">
  <div class="conceptflow-header">
    <div class="conceptflow-title">{html.escape(title)}</div>
    <div id="conceptflow-subtitle" class="conceptflow-subtitle"></div>
  </div>

  <div class="conceptflow-body">
    <svg id="conceptflow-svg" width="{page['drawing_width']}" height="{height}"></svg>

    <div id="conceptflow-details" class="conceptflow-details">
      <div class="details-title">Details</div>
      <div>Click a concept node to inspect its extent and intent.</div>
    </div>
  </div>
</div>

<script>
const graphData = {data_json};
const svgWidth = {page['drawing_width']};
const svgHeight = {height};

const svg = d3.select("#conceptflow-svg");
const details = d3.select("#conceptflow-details");
const subtitle = d3.select("#conceptflow-subtitle");

subtitle.text(
  `objects: ${{graphData.metadata?.n_objects ?? "?"}} · ` +
  `attributes: ${{graphData.metadata?.n_attributes ?? "?"}} · ` +
  `concepts: ${{graphData.metadata?.n_concepts ?? graphData.nodes.length}} · ` +
  `layout: ${{graphData.metadata?.layout ?? "none"}}`
);

let rootLayer = svg.append("g").attr("class", "root-layer");

const zoom = d3.zoom()
  .scaleExtent([0.3, 8])
  .on("zoom", (event) => {{
    rootLayer.attr("transform", event.transform);
  }});

svg.call(zoom);

function escapeHtml(text) {{
  return String(text).replace(/[&<>"']/g, c => ({{
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }})[c]);
}}

function formatList(values) {{
  if (!values || values.length === 0) return "∅";
  return escapeHtml(values.join(", "));
}}

function labelForNode(d) {{
  if (d.metadata?.intent?.length > 0) {{
    return d.metadata.intent.join(", ");
  }}
  if (d.metadata?.extent_size === 0) {{
    return "⊥";
  }}
  return "⊤";
}}

function showDetails(d) {{
  details.html(`
    <div class="details-title">Concept details</div>

    <div class="details-section">
      <div class="details-label">Label</div>
      <div>${{escapeHtml(labelForNode(d))}}</div>
    </div>

    <div class="details-section">
      <div class="details-label">Extent (${{d.metadata?.extent_size ?? 0}})</div>
      <div>${{formatList(d.metadata?.extent)}}</div>
    </div>

    <div class="details-section">
      <div class="details-label">Intent (${{d.metadata?.intent_size ?? 0}})</div>
      <div>${{formatList(d.metadata?.intent)}}</div>
    </div>

    <div class="details-section">
      <div class="details-label">Coordinates</div>
      <div>x=${{Number(d.x ?? 0).toFixed(2)}}, y=${{Number(d.y ?? 0).toFixed(2)}}</div>
    </div>
  `);
}}

function closestDistance(nodes) {{
  const points = nodes.map(d => [d.screenX, d.screenY]).sort((a, b) => a[0] - b[0]);
  let best = Infinity;
  for (let i = 0; i < points.length; i++) {{
    for (let j = i + 1; j < points.length && points[j][0] - points[i][0] < best; j++) {{
      best = Math.min(best, Math.hypot(points[j][0] - points[i][0], points[j][1] - points[i][1]));
    }}
  }}
  return best;
}}

function computeTransform(nodes, targetWidth, targetHeight, padding, rotate) {{
  const topNode = nodes.reduce((best, d) =>
    (d.metadata?.extent_size ?? 0) > (best.metadata?.extent_size ?? 0) ? d : best
  , nodes[0]);

  const bottomNode = nodes.reduce((best, d) =>
    (d.metadata?.extent_size ?? 0) < (best.metadata?.extent_size ?? 0) ? d : best
  , nodes[0]);

  const bx = bottomNode.x ?? 0;
  const by = bottomNode.y ?? 0;
  const tx = topNode.x ?? 0;
  const ty = topNode.y ?? 0;

  const vx = tx - bx;
  const vy = ty - by;

  const currentAngle = Math.atan2(vy, vx);
  const desiredAngle = -Math.PI / 2;
  const angle = rotate ? desiredAngle - currentAngle : 0;

  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);

  function rotatePoint(d) {{
    const x = d.x ?? 0;
    const y = d.y ?? 0;

    return {{
      x: x * cosA - y * sinA,
      y: x * sinA + y * cosA
    }};
  }}

  const rotatedById = new Map(
    nodes.map(d => [d.id, rotatePoint(d)])
  );

  const rotated = nodes.map(d => rotatedById.get(d.id));

  const xs = rotated.map(d => d.x);
  const ys = rotated.map(d => d.y);

  const minX = d3.min(xs);
  const maxX = d3.max(xs);
  const minY = d3.min(ys);
  const maxY = d3.max(ys);

  const spanX = Math.max(maxX - minX, 1);
  const spanY = Math.max(maxY - minY, 1);

  const scale = Math.min(
    (targetWidth - 2 * padding) / spanX,
    (targetHeight - 2 * padding) / spanY
  );

  return function(d) {{
    const p = rotatedById.get(d.id);

    return {{
      x: padding + (p.x - minX) * scale
        + (targetWidth - 2 * padding - spanX * scale) / 2,

      y: padding + (p.y - minY) * scale
        + (targetHeight - 2 * padding - spanY * scale) / 2
    }};
  }};
}}

function render() {{
  rootLayer.selectAll("*").remove();

  const transform = computeTransform(
    graphData.nodes, svgWidth, svgHeight, Math.min(80, svgWidth / 6, svgHeight / 6),
    graphData.rotate !== false
  );
  const nodeById = new Map(graphData.nodes.map(d => [d.id, d]));

  graphData.nodes.forEach(d => {{
    const p = transform(d);
    d.screenX = p.x;
    d.screenY = p.y;
  }});
  // Up to 18 px, but neighbouring circles never overlap; zooming in can
  // always bring small nodes up to full size.
  const radius = Math.min(18, 0.4 * closestDistance(graphData.nodes));
  svg.style("--node-scale", radius / 18);
  zoom.scaleExtent([0.3, Math.max(8, 18 / radius)]);

  const edgeSelection = rootLayer
    .append("g")
    .selectAll("line.edge")
    .data(graphData.edges)
    .join("line")
    .attr("class", "edge")
    .attr("x1", d => nodeById.get(d.source).screenX)
    .attr("y1", d => nodeById.get(d.source).screenY)
    .attr("x2", d => nodeById.get(d.target).screenX)
    .attr("y2", d => nodeById.get(d.target).screenY);

  const nodeGroup = rootLayer
    .append("g")
    .selectAll("g.node-group")
    .data(graphData.nodes)
    .join("g")
    .attr("class", "node-group")
    .attr("transform", d => `translate(${{d.screenX}}, ${{d.screenY}})`);

  const circles = nodeGroup
    .append("circle")
    .attr("class", "node")
    .attr("r", radius)
    .on("mouseover", function(event, d) {{
      d3.select(this).classed("selected", true);

      edgeSelection.classed("highlighted", e =>
        e.source === d.id || e.target === d.id
      );
    }})
    .on("mouseout", function(event, d) {{
      d3.select(this).classed("selected", false);
      edgeSelection.classed("highlighted", false);
    }})
    .on("click", function(event, d) {{
      event.stopPropagation();
      showDetails(d);
    }});

  circles
    .append("title")
    .text(d => d.hover);

  nodeGroup
    .filter(d => d.attribute_text)
    .append("text")
    .attr("class", "attr-label")
    .attr("y", -radius * 1.3)
    .text(d => d.attribute_text);

  nodeGroup
    .filter(d => d.object_text)
    .append("text")
    .attr("class", "obj-label")
    .attr("y", radius * 1.3)
    .text(d => d.object_text);
}}

render();
</script>
</body>
</html>
"""


def render_with_d3(
    graph_data: GraphData,
    width: int = 900,
    height: int = 700,
    title: str = "Concept lattice",
    **label_options,
) -> HTMLFigure:
    """
    Render GraphData with the built-in D3 backend.

    Nodes without coordinates get the layered layout first.
    ``label_options`` (``show_attribute_labels``, ``show_object_labels``,
    ``max_label_items``, ``object_label``) go to ``render_graph_data_html``.
    """
    if any(node.x is None or node.y is None for node in graph_data.nodes):
        from conceptflow.visualization.layered_layout import layered_layout

        graph_data = layered_layout(graph_data)

    return HTMLFigure(
        render_graph_data_html(
            graph_data,
            width=width,
            height=height,
            title=title,
            **label_options,
        ),
        height=height,
    )
