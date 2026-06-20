"use client";

import { useState, type KeyboardEvent } from "react";

interface DecisionInputProps {
  onSubmit: (decisionText: string) => void;
  /** Reserved for Session 5 live wiring (disable while a real generate call is in flight).
   * Unused in Session 3 — demo mode never has a pending state. */
  loading?: boolean;
  /** Controlled mode — lets parents (e.g. example decision chips on the
   * landing page) pre-fill the input. Falls back to local state if omitted. */
  value?: string;
  onChange?: (v: string) => void;
}

export default function DecisionInput({
  onSubmit,
  loading = false,
  value,
  onChange,
}: DecisionInputProps) {
  const [localValue, setLocalValue] = useState("");
  const displayValue = value ?? localValue;

  function handleChange(v: string) {
    if (onChange) {
      onChange(v);
    } else {
      setLocalValue(v);
    }
  }

  function handleSubmit() {
    const trimmed = displayValue.trim();
    if (!trimmed || loading) return;
    onSubmit(trimmed);
  }

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") handleSubmit();
  }

  return (
    <div className="w-full max-w-2xl">
      <div
        className="relative flex items-center gap-3 rounded-2xl border px-5 py-4 transition-colors focus-within:border-[var(--color-decision-root)]"
        style={{ background: "var(--color-bg-card)", borderColor: "var(--color-border)" }}
      >
        <input
          type="text"
          value={displayValue}
          onChange={(e) => handleChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="A decision your team is debating right now."
          disabled={loading}
          className="flex-1 bg-transparent font-sans text-base outline-none placeholder:text-[var(--color-text-muted)]"
          style={{ color: "var(--color-text-primary)" }}
          autoFocus
        />
        {displayValue.length > 80 && (
          <span
            className="absolute bottom-1.5 right-[148px] font-mono text-[10px]"
            style={{
              color:
                displayValue.length > 200
                  ? "var(--color-tier-speculative)"
                  : "var(--color-text-muted)",
            }}
          >
            {displayValue.length}
          </span>
        )}
        <button
          type="button"
          onClick={handleSubmit}
          disabled={loading || !displayValue.trim()}
          className="shrink-0 rounded-xl px-4 py-2.5 font-sans text-sm font-medium transition-opacity disabled:opacity-40"
          style={{ background: "var(--color-decision-root)", color: "#0a0b0f" }}
        >
          Map the consequences
        </button>
      </div>
    </div>
  );
}