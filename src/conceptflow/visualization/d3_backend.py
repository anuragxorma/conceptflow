"""
D3 backend for ordinary ConceptFlow lattice diagrams.
"""

from __future__ import annotations

import json
from typing import Any

from conceptflow.visualization.graph_data import GraphData
from conceptflow.visualization.html_figure import HTMLFigure


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
    }


def render_graph_data_html(
    graph_data: GraphData,
    width: int = 900,
    height: int = 700,
    title: str = "Concept lattice",
) -> str:
    """
    Render GraphData as a standalone D3 HTML string.
    """
    data_json = json.dumps(graph_data_to_d3_data(graph_data))

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
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
    grid-template-columns: 1fr 280px;
    min-height: {height}px;
  }}

  .conceptflow-details {{
    border-left: 1px solid #ddd;
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

  .node {{
    fill: white;
    stroke: black;
    stroke-width: 1.6;
    cursor: pointer;
  }}

  .node:hover {{
    stroke: #d97706;
    stroke-width: 3;
  }}

  .node.selected {{
    stroke: #d97706;
    stroke-width: 3.5;
  }}

  .node-label {{
    font-size: 10px;
    text-anchor: middle;
    dominant-baseline: middle;
    pointer-events: none;
  }}
</style>
</head>
<body>
<div class="conceptflow-container">
  <div class="conceptflow-header">
    <div class="conceptflow-title">{title}</div>
    <div id="conceptflow-subtitle" class="conceptflow-subtitle"></div>
  </div>

  <div class="conceptflow-body">
    <svg id="conceptflow-svg" width="{width - 280}" height="{height}"></svg>

    <div id="conceptflow-details" class="conceptflow-details">
      <div class="details-title">Details</div>
      <div>Click a concept node to inspect its extent and intent.</div>
    </div>
  </div>
</div>

<script>
const graphData = {data_json};
const svgWidth = {width - 280};
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

function formatList(values) {{
  if (!values || values.length === 0) return "∅";
  return values.join(", ");
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
      <div>${{labelForNode(d)}}</div>
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

function computeTransform(nodes, targetWidth, targetHeight, padding) {{
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
  const angle = desiredAngle - currentAngle;

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

  const transform = computeTransform(graphData.nodes, svgWidth, svgHeight, 80);
  const nodeById = new Map(graphData.nodes.map(d => [d.id, d]));

  graphData.nodes.forEach(d => {{
    const p = transform(d);
    d.screenX = p.x;
    d.screenY = p.y;
  }});

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
    .attr("r", 18)
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
    .append("text")
    .attr("class", "node-label")
    .attr("y", 30)
    .text(d => labelForNode(d));
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
) -> HTMLFigure:
    """
    Render GraphData with the built-in D3 backend.
    """
    return HTMLFigure(
      render_graph_data_html(
          graph_data,
          width=width,
          height=height,
          title=title,
      )
  )