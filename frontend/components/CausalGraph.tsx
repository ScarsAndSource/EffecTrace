"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  useReactFlow,
  type Edge,
  type NodeMouseHandler,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import type {
  CausalGraphOutput,
  CausalNode as CausalNodeType,
  ConfidenceTier,
  OutcomeMap,
} from "@/lib/types";
import { TIER_LABEL, polarityColor, domainHue } from "@/lib/visualTokens";
import { formatSigned } from "@/lib/format";
import CausalNodeView, { type CausalFlowNode } from "./CausalNode";

/** Imperative view controls handed to GraphControls.tsx (Session 4 addition).
 * CausalGraph still owns all ReactFlow state internally — this is read-only
 * from the parent's perspective, just zoom/fit triggers. */
export interface GraphViewApi {
  zoomIn: () => void;
  zoomOut: () => void;
  fitView: () => void;
}

interface CausalGraphProps {
  graph: CausalGraphOutput;
  outcomes: OutcomeMap;
  /** Session 4 additions — driven by GraphControls.tsx (Section 11.1: "filter by
   * horizon, filter by confidence tier"). Both optional and undefined by default,
   * so Session 3's page.tsx needs zero changes to keep working exactly as before. */
  activeTiers?: ConfidenceTier[];
  maxHorizonDays?: number | null;
  onReady?: (api: GraphViewApi) => void;
}

// ──────────────────────────────────────────────
//  Manual radial/concentric layout — Section 11.2.
//  decision_root sits dead center; direct effects form ring 1, second-order
//  ring 2, third-order ring 3. No dagre, no force-directed: deterministic,
//  count-agnostic, and it literally IS the "ripples expanding outward" metaphor
//  instead of approximating it with a generic graph-layout algorithm.
// ──────────────────────────────────────────────

const RING_RADIUS = [260, 460, 660]; // ring 1 (direct effects), ring 2, ring 3

function ringIndexFor(node: CausalNodeType): -1 | 0 | 1 | 2 {
  if (node.id === "decision_root") return -1;
  if (node.layer === 0) return 0;
  if (node.layer === 1) return 1;
  return 2;
}

/** Visual reveal stage — distinct from the data model's 0/1/2 `layer` field.
 * decision_root and direct effects are both `layer: 0` in the data, but they are
 * two different rings visually, so the reveal sequence treats them as two
 * separate stages to match what's actually drawn on screen. */
function revealStageFor(node: CausalNodeType): number {
  const ring = ringIndexFor(node);
  return ring === -1 ? 0 : ring + 1;
}

function computeRadialLayout(nodes: CausalNodeType[]): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();
  const root = nodes.find((n) => n.id === "decision_root");
  if (root) positions.set(root.id, { x: 0, y: 0 });

  for (let ring = 0; ring < 3; ring++) {
    const ringNodes = nodes.filter((n) => ringIndexFor(n) === ring);
    // Group angularly by domain (stable hue order) so same-domain effects cluster
    // together around the ring rather than scattering randomly.
    const sorted = [...ringNodes].sort((a, b) => domainHue(a.domain) - domainHue(b.domain));
    const n = sorted.length;
    if (n === 0) continue;
    // Stagger each ring's start angle slightly so spokes don't all line up radially,
    // which reads as more organic and avoids edges stacking visually.
    const startAngle = -Math.PI / 2 + (ring * Math.PI) / 7;
    sorted.forEach((node, i) => {
      const angle = startAngle + (i / n) * 2 * Math.PI;
      const radius = RING_RADIUS[ring];
      positions.set(node.id, {
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
      });
    });
  }

  return positions;
}

const nodeTypes = { causal: CausalNodeView };

function CausalGraphInner({ graph, outcomes, activeTiers, maxHorizonDays, onReady }: CausalGraphProps) {
  const layoutPositions = useMemo(() => computeRadialLayout(graph.nodes), [graph]);
  const nodeMap = useMemo(() => new Map(graph.nodes.map((n) => [n.id, n])), [graph]);

  const cleanEdges = useMemo(
    () =>
      graph.edges.filter(
        (e) => nodeMap.has(e.source_id) && nodeMap.has(e.target_id) && e.source_id !== e.target_id
      ),
    [graph, nodeMap]
  );

  // ── Session 4 addition: GraphControls filtering ──
  // null = no filter active → every node/edge from cleanEdges is shown, identical
  // to Session 3 behaviour when activeTiers/maxHorizonDays are both omitted.
  const visibleNodeIds = useMemo(() => {
    if (!activeTiers && maxHorizonDays == null) return null;

    const horizonOk = (days: number) => maxHorizonDays == null || days <= maxHorizonDays;
    const allowedEdges = cleanEdges.filter((e) => horizonOk(e.time_horizon_days));

    const adjacency = new Map<string, string[]>();
    for (const e of allowedEdges) {
      if (!adjacency.has(e.source_id)) adjacency.set(e.source_id, []);
      adjacency.get(e.source_id)!.push(e.target_id);
    }

    // Reachability from decision_root over horizon-allowed edges only — a node
    // whose only path in exceeds the horizon cap disappears along with that edge.
    const reachable = new Set<string>(["decision_root"]);
    const queue = ["decision_root"];
    while (queue.length) {
      const cur = queue.shift()!;
      for (const next of adjacency.get(cur) ?? []) {
        if (!reachable.has(next)) {
          reachable.add(next);
          queue.push(next);
        }
      }
    }

    const tierOk = (n: CausalNodeType) =>
      !activeTiers || n.id === "decision_root" || activeTiers.includes(n.confidence_tier);

    const ids = new Set<string>();
    for (const n of graph.nodes) {
      if (reachable.has(n.id) && tierOk(n)) ids.add(n.id);
    }
    return ids;
  }, [graph, cleanEdges, activeTiers, maxHorizonDays]);

  const visibleEdges = useMemo(() => {
    if (!visibleNodeIds) return cleanEdges;
    return cleanEdges.filter((e) => visibleNodeIds.has(e.source_id) && visibleNodeIds.has(e.target_id));
  }, [cleanEdges, visibleNodeIds]);

  const [revealStage, setRevealStage] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<CausalFlowNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  // ── Session 4 addition: expose imperative zoom controls to GraphControls.tsx ──
  // Safe inside CausalGraphInner because it's always rendered under ReactFlowProvider
  // (see the default export below).
  const reactFlowApi = useReactFlow();
  useEffect(() => {
    if (!onReady) return;
    onReady({
      zoomIn: () => reactFlowApi.zoomIn({ duration: 200 }),
      zoomOut: () => reactFlowApi.zoomOut({ duration: 200 }),
      fitView: () => reactFlowApi.fitView({ padding: 0.25, duration: 300 }),
    });
  }, [onReady, reactFlowApi]);

  // Layer-by-layer reveal: center, then ring 1, ring 2, ring 3 — 220ms apart.
  // Respects prefers-reduced-motion by skipping straight to fully revealed.
  useEffect(() => {
    setRevealStage(0);
    setSelectedId(null);
    const reduced =
      typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) {
      setRevealStage(3);
      return;
    }
    const timers = [1, 2, 3].map((stage, i) => setTimeout(() => setRevealStage(stage), (i + 1) * 220));
    return () => timers.forEach(clearTimeout);
  }, [graph]);

  // Rebuild node data on outcome/reveal/filter changes, but preserve any position
  // the user has dragged to — positions only reset when the graph itself changes.
  useEffect(() => {
    setNodes((prev) => {
      const prevPositions = new Map(prev.map((n) => [n.id, n.position]));
      const visible = visibleNodeIds ? graph.nodes.filter((n) => visibleNodeIds.has(n.id)) : graph.nodes;
      return visible.map((node) => {
        const outcome = outcomes[node.id] ?? { mean: 0, p10: 0, p90: 0, std: 0 };
        return {
          id: node.id,
          type: "causal" as const,
          position: prevPositions.get(node.id) ?? layoutPositions.get(node.id) ?? { x: 0, y: 0 },
          data: { node: { ...node, outcome }, revealed: revealStage >= revealStageFor(node) },
          draggable: true,
          connectable: false,
        };
      });
    });
  }, [graph, outcomes, revealStage, layoutPositions, visibleNodeIds, setNodes]);

  useEffect(() => {
    setEdges(
      visibleEdges.map((e) => {
        const targetNode = nodeMap.get(e.target_id)!;
        const revealed = revealStage >= revealStageFor(targetNode);
        return {
          id: `${e.source_id}->${e.target_id}`,
          source: e.source_id,
          target: e.target_id,
          animated: revealed,
          style: {
            stroke: polarityColor(e.polarity),
            strokeWidth: 1 + e.magnitude_estimate * 2.5,
            opacity: revealed ? 0.8 : 0,
            transition: "opacity 520ms ease-out",
          },
        };
      })
    );
  }, [visibleEdges, nodeMap, revealStage, setEdges]);

  const handleNodeClick: NodeMouseHandler = (_, node) => {
    setSelectedId((current) => (current === node.id ? null : node.id));
  };

  const selectedNode = selectedId ? nodeMap.get(selectedId) ?? null : null;
  const selectedOutcome = selectedId ? outcomes[selectedId] : undefined;
  const incomingEdges = selectedId ? visibleEdges.filter((e) => e.target_id === selectedId) : [];

  return (
    <div
      className="relative w-full overflow-hidden rounded-2xl border"
      style={{ height: "min(680px, 72vh)", borderColor: "var(--color-border)", background: "var(--color-bg)" }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        onNodeClick={handleNodeClick}
        onPaneClick={() => setSelectedId(null)}
        nodesConnectable={false}
        fitView
        fitViewOptions={{ padding: 0.25 }}
        minZoom={0.3}
        maxZoom={1.6}
      >
        <Background color="var(--color-border-subtle)" gap={28} size={1} />
        <Controls showInteractive={false} />
      </ReactFlow>

      {selectedNode && (
        <aside
          className="absolute right-4 top-4 z-10 w-72 max-w-[80vw] rounded-xl border p-4 font-sans text-sm shadow-2xl"
          style={{ background: "var(--color-bg-card)", borderColor: "var(--color-border)" }}
        >
          <div className="flex items-start justify-between gap-2">
            <div>
              <div className="font-sans text-[10px] uppercase tracking-wider" style={{ color: "var(--color-text-muted)" }}>
                {selectedNode.domain}
                {selectedNode.id !== "decision_root" && ` · ${TIER_LABEL[selectedNode.confidence_tier]}`}
              </div>
              <div className="mt-0.5 font-display text-base font-medium" style={{ color: "var(--color-text-primary)" }}>
                {selectedNode.label}
              </div>
            </div>
            <button
              type="button"
              onClick={() => setSelectedId(null)}
              aria-label="Close detail panel"
              className="shrink-0 rounded-md px-1.5 font-mono text-base leading-none"
              style={{ color: "var(--color-text-muted)" }}
            >
              ×
            </button>
          </div>

          <p className="mt-2 leading-relaxed" style={{ color: "var(--color-text-secondary)" }}>
            {selectedNode.description}
          </p>

          {selectedOutcome && selectedNode.id !== "decision_root" && (
            <div
              className="mt-3 rounded-lg border px-3 py-2 font-mono text-xs"
              style={{ borderColor: "var(--color-border-subtle)", color: "var(--color-text-secondary)" }}
            >
              mean {formatSigned(selectedOutcome.mean)} · simulated range{" "}
              {formatSigned(selectedOutcome.p10)} to {formatSigned(selectedOutcome.p90)}
            </div>
          )}

          {incomingEdges.length > 0 && (
            <div className="mt-3 space-y-2">
              <div className="font-sans text-[10px] uppercase tracking-wider" style={{ color: "var(--color-text-muted)" }}>
                Why
              </div>
              {incomingEdges.map((e) => (
                <div
                  key={`${e.source_id}-${e.target_id}`}
                  className="flex gap-1.5 text-xs leading-relaxed"
                  style={{ color: "var(--color-text-secondary)" }}
                >
                  <span style={{ color: polarityColor(e.polarity) }}>{e.polarity === 1 ? "↑" : "↓"}</span>
                  <span>{e.rationale_citation}</span>
                </div>
              ))}
            </div>
          )}
        </aside>
      )}
    </div>
  );
}

export default function CausalGraph(props: CausalGraphProps) {
  return (
    <ReactFlowProvider>
      <CausalGraphInner {...props} />
    </ReactFlowProvider>
  );
}
