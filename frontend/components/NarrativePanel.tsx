"use client";

import { useState } from "react";
import { TIER_COLOR } from "@/lib/visualTokens";

interface NarrativePanelProps {
  narrative: string | null;
  loading: boolean;
  error: string | null;
  isOpen: boolean;
  decisionSummary?: string;
  onGenerate: () => void;
  onClose: () => void;
}

/** Exact inline flags Session 2's narration prompt is instructed to emit —
 * confirmed against backend/test_narrate.py's CONFIDENCE_FLAGS list. Any other
 * bracketed text in the narrative is left untouched, not styled. */
const TAG_COLOR: Record<string, string> = {
  "[Data-Grounded]": TIER_COLOR.data_grounded,
  "[Historically-Precedented]": TIER_COLOR.historically_precedented,
  "[Speculative]": TIER_COLOR.speculative,
};

const TAG_PATTERN = /(\[Data-Grounded\]|\[Historically-Precedented\]|\[Speculative\])/g;

function renderNarrative(text: string) {
  return text.split("\n\n").map((para, pIdx) => (
    <p key={pIdx} className="mb-3 leading-relaxed last:mb-0">
      {para.split(TAG_PATTERN).map((chunk, i) => {
        const color = TAG_COLOR[chunk];
        if (!color) return <span key={i}>{chunk}</span>;
        return (
          <span
            key={i}
            className="mx-0.5 rounded px-1.5 py-0.5 font-mono text-[10px] font-medium uppercase tracking-wide"
            style={{ color, background: `${color}1a`, border: `1px solid ${color}40` }}
          >
            {chunk.replace(/[[\]]/g, "")}
          </span>
        );
      })}
    </p>
  ));
}

export default function NarrativePanel({
  narrative,
  loading,
  error,
  isOpen,
  decisionSummary,
  onGenerate,
  onClose,
}: NarrativePanelProps) {
  const [copied, setCopied] = useState(false);
  const [exporting, setExporting] = useState(false);

  async function handleCopyMarkdown() {
    if (!narrative) return;
    const md = `# Board Memo${decisionSummary ? `\n\n_${decisionSummary}_` : ""}\n\n${narrative}`;
    await navigator.clipboard.writeText(md);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }

  async function handleDownloadPdf() {
    if (!narrative) return;
    setExporting(true);
    try {
      // Dynamic import keeps jsPDF out of the server bundle entirely — this
      // export is explicitly client-side-only, no backend dependency (Section 11.1).
      const { jsPDF } = await import("jspdf");
      const doc = new jsPDF({ unit: "pt", format: "letter" });
      const marginX = 56;
      const maxWidth = doc.internal.pageSize.getWidth() - marginX * 2;
      const pageHeight = doc.internal.pageSize.getHeight();
      let y = 72;

      doc.setFont("helvetica", "bold");
      doc.setFontSize(16);
      doc.text("Board Memo — Butterfly Effect", marginX, y);
      y += 22;

      if (decisionSummary) {
        doc.setFont("helvetica", "italic");
        doc.setFontSize(10);
        const summaryLines = doc.splitTextToSize(decisionSummary, maxWidth);
        doc.text(summaryLines, marginX, y);
        y += summaryLines.length * 13 + 14;
      }

      // Strip the inline [Tag] markers for the PDF body — they're a screen
      // affordance; the printed artifact reads cleaner as plain prose.
      const plain = narrative.replace(TAG_PATTERN, "").replace(/ {2,}/g, " ");
      doc.setFont("helvetica", "normal");
      doc.setFontSize(11);
      for (const para of plain.split("\n\n")) {
        const lines = doc.splitTextToSize(para.trim(), maxWidth);
        for (const line of lines) {
          if (y > pageHeight - 56) {
            doc.addPage();
            y = 72;
          }
          doc.text(line, marginX, y);
          y += 15;
        }
        y += 9;
      }

      doc.save("butterfly-effect-board-memo.pdf");
    } finally {
      setExporting(false);
    }
  }

  if (!isOpen) return null;

  return (
    <aside
      className="fixed right-0 top-0 z-20 flex h-full w-full max-w-md flex-col border-l shadow-2xl"
      style={{
        background: "var(--color-bg-card)",
        borderColor: "var(--color-border)",
        animation: "narrative-slide-in 320ms cubic-bezier(0.16,1,0.3,1)",
      }}
    >
      <style>{`
        @keyframes narrative-slide-in {
          from { transform: translateX(100%); opacity: 0.4; }
          to { transform: translateX(0); opacity: 1; }
        }
      `}</style>

      <div className="flex items-start justify-between gap-3 border-b px-5 py-4" style={{ borderColor: "var(--color-border)" }}>
        <div className="min-w-0">
          <div className="font-display text-base font-medium" style={{ color: "var(--color-text-primary)" }}>
            Board Memo
          </div>
          {decisionSummary && (
            <p
              className="mt-0.5 font-sans text-[11px] leading-snug"
              style={{ color: "var(--color-text-muted)" }}
            >
              {decisionSummary.length > 60 ? decisionSummary.slice(0, 58) + "…" : decisionSummary}
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close board memo"
          className="shrink-0 rounded-md px-1.5 font-mono text-lg leading-none"
          style={{ color: "var(--color-text-muted)" }}
        >
          ×
        </button>
      </div>

      <div
        className="flex-1 overflow-y-auto px-5 py-4 font-sans text-[13.5px]"
        style={{ color: "var(--color-text-secondary)" }}
      >
        {loading && (
          <div className="flex flex-col gap-4 pt-2">
            <div className="flex items-center gap-2">
              <div
                className="h-2 w-2 animate-pulse rounded-full"
                style={{ background: "var(--color-decision-root)" }}
              />
              <span className="font-mono text-[11px]" style={{ color: "var(--color-text-muted)" }}>
                Synthesizing narrative…
              </span>
            </div>
            <div className="space-y-2.5">
              {[0.9, 0.75, 0.92, 0.6, 0.82].map((w, i) => (
                <div
                  key={i}
                  className="h-2.5 animate-pulse rounded"
                  style={{ width: `${w * 100}%`, background: "var(--color-border-subtle)" }}
                />
              ))}
            </div>
          </div>
        )}

        {!loading && error && (
          <div
            className="rounded-lg border px-3 py-2.5 text-xs"
            style={{ borderColor: "var(--color-tier-speculative)", color: "var(--color-text-secondary)" }}
          >
            Couldn't generate the memo: {error}
            <button
              type="button"
              onClick={onGenerate}
              className="ml-2 underline underline-offset-2"
              style={{ color: "var(--color-decision-root)" }}
            >
              Retry
            </button>
          </div>
        )}

        {!loading && !error && narrative && renderNarrative(narrative)}

        {!loading && !error && !narrative && (
          <button
            type="button"
            onClick={onGenerate}
            className="rounded-lg border px-3 py-2 text-xs"
            style={{ borderColor: "var(--color-border)", color: "var(--color-text-secondary)" }}
          >
            Generate board memo
          </button>
        )}
      </div>

      {narrative && !loading && !error && (
        <div className="flex items-center gap-2 border-t px-5 py-3" style={{ borderColor: "var(--color-border)" }}>
          <button
            type="button"
            onClick={handleCopyMarkdown}
            className="rounded-lg border px-3 py-1.5 font-sans text-xs transition-opacity"
            style={{ borderColor: "var(--color-border)", color: "var(--color-text-secondary)" }}
          >
            {copied ? "Copied" : "Copy as Markdown"}
          </button>
          <button
            type="button"
            onClick={handleDownloadPdf}
            disabled={exporting}
            className="rounded-lg px-3 py-1.5 font-sans text-xs font-medium transition-opacity disabled:opacity-40"
            style={{ background: "var(--color-decision-root)", color: "#0a0b0f" }}
          >
            {exporting ? "Exporting…" : "Download PDF"}
          </button>
        </div>
      )}
    </aside>
  );
}

/** Standalone trigger for the page to place near the graph — kept separate from
 * the panel so the page controls layout/positioning, not this component. */
export function GenerateMemoButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="rounded-xl border px-4 py-2.5 font-sans text-sm font-medium transition-opacity hover:brightness-110"
      style={{ borderColor: "var(--color-decision-root)", color: "var(--color-decision-root)" }}
    >
      Generate board memo
    </button>
  );
}