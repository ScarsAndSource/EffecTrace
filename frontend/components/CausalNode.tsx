"use client";

import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import type { RenderableNode } from "@/lib/types";
import { COLOR, TIER_COLOR, domainHue } from "@/lib/visualTokens";
import { formatSigned } from "@/lib/format";

export interface CausalNodeData extends Record<string, unknown> {
  node: RenderableNode;
  /** Drives the layer-by-layer "bloom" animation — false until this node's ring is staged in. */
  revealed: boolean;
}

export type CausalFlowNode = Node<CausalNodeData, "causal">;

export default function CausalNode({ data, selected }: NodeProps<CausalFlowNode>) {
  const { node, revealed } = data;
  const isRoot = node.id === "decision_root";
  const ringColor = isRoot ? COLOR.decisionRoot : TIER_COLOR[node.confidence_tier];

  return (
    <div
      style={{
        opacity: revealed ? 1 : 0,
        transform: revealed ? "scale(1)" : "scale(0.55)",
        transition: "opacity 460ms cubic-bezier(0.16, 1, 0.3, 1), transform 460ms cubic-bezier(0.16, 1, 0.3, 1)",
        pointerEvents: revealed ? "auto" : "none",
      }}
    >
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />

      <div
        className="cursor-pointer rounded-xl border-2 px-3 py-2.5 transition-[filter] hover:brightness-110"
        style={{
          background: isRoot
            ? "rgba(139,143,245,0.10)"
            : `hsl(${domainHue(node.domain)}, 55%, 11%)`,
          borderColor: ringColor,
          boxShadow: selected
            ? `0 0 0 2px ${ringColor}, 0 0 28px ${ringColor}55`
            : `0 0 14px ${ringColor}22`,
          minWidth: isRoot ? 170 : 164,
          maxWidth: 208,
        }}
      >
        <div
          className="font-sans text-[10px] font-medium uppercase tracking-wider"
          style={{ color: "var(--color-text-muted)" }}
        >
          {node.domain}
        </div>
        <div
          className="mt-0.5 font-display text-[13.5px] font-medium leading-snug"
          style={{ color: "var(--color-text-primary)" }}
        >
          {node.label}
        </div>
        {!isRoot && (
          <div className="mt-2" title={`${formatSigned(node.outcome.p10)} to ${formatSigned(node.outcome.p90)}`}>
            <div
              className="relative h-1.5 w-full overflow-hidden rounded-full"
              style={{ background: "var(--color-border-subtle)" }}
            >
              <div
                className="absolute h-full rounded-full"
                style={{
                  left: `${((node.outcome.p10 + 1) / 2) * 100}%`,
                  width: `${((node.outcome.p90 - node.outcome.p10) / 2) * 100}%`,
                  background:
                    node.direction === "positive"
                      ? "var(--color-polarity-amplify)"
                      : node.direction === "negative"
                      ? "var(--color-polarity-dampen)"
                      : "var(--color-text-muted)",
                  opacity: 0.85,
                }}
              />
            </div>
            <div
              className="mt-0.5 flex justify-between font-mono text-[9px]"
              style={{ color: "var(--color-text-muted)" }}
            >
              <span>{formatSigned(node.outcome.p10)}</span>
              <span>{formatSigned(node.outcome.p90)}</span>
            </div>
          </div>
        )}
      </div>

      <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} />
    </div>
  );
}