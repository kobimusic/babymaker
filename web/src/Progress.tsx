import type { CSSProperties } from "react";
import { cx } from "./cx.ts";
import styles from "./Progress.module.css";

const R = 26;                        // arc radius, in the 64-unit viewBox
const C = 2 * Math.PI * R;
const SWEEP = 0.28;                  // how much of the ring the indeterminate arc covers

export type ProgressSize = "sm" | "md" | "lg";

/**
 * A ring that fills clockwise from twelve o'clock.
 *
 * Omit `value` and it sweeps instead of filling — for work whose size is
 * not known until it is done. It has no state and no timers, so it renders
 * on the server and costs nothing on a page that only shows it; a caller
 * measuring real work is the one that should keep `value` from going
 * backwards, since only the caller knows what its phases are worth.
 */
export function Progress({
  value, label, size = "md", showValue, showLabel = true, immediate, driven, ring, className,
}: {
  /** 0..1. Omit for an indeterminate sweep. */
  value?: number;
  label?: string;
  size?: ProgressSize;
  /** Defaults to on, except at `sm`, where there is no room for a numeral. */
  showValue?: boolean;
  /**
   * Whether the label is drawn as well as said. Off where the words are
   * already on the screen beside it — a tree's row names the file it is
   * working on, and drawing "working on song.mid" there both repeats it and
   * takes the room the name needs. The label is still the accessible name,
   * which is the half that is not already said.
   */
  showLabel?: boolean;
  /** Skip the anti-flash delay — for a load already known to be slow. */
  immediate?: boolean;
  /**
   * `value` is already updated every frame (a requestAnimationFrame loop, a
   * held pointer), not in the occasional jump the arc's own transition is
   * smoothing for. Left on, that transition chases a target that moves
   * again before it arrives: over a one-second hold it never gets past
   * about a third of the way around. This skips it, so the arc is exactly
   * where `value` says.
   */
  driven?: boolean;
  /** An exact diameter, when none of the three sizes fits — e.g. a ring
   *  drawn around an existing control. Any CSS length. */
  ring?: string;
  className?: string;
}) {
  const known = typeof value === "number" && Number.isFinite(value);
  const pct = known ? Math.round(Math.min(1, Math.max(0, value)) * 100) : 0;
  const numeral = showValue ?? (known && size !== "sm");

  return (
    <div
      className={cx(styles.progress, styles[size], !immediate && styles.delayed, driven && styles.driven, className)}
      // inline so it beats the size class whatever order the sheets load in
      style={ring ? ({ "--ring": ring } as CSSProperties) : undefined}
      role="progressbar"
      aria-label={label ?? "loading"}
      // An indeterminate bar is one with no aria-valuenow; assistive
      // technology says "busy" rather than inventing a number.
      aria-valuemin={known ? 0 : undefined}
      aria-valuemax={known ? 100 : undefined}
      aria-valuenow={known ? pct : undefined}
      aria-valuetext={known && label ? `${label}, ${pct}%` : undefined}
    >
      <svg className={styles.ring} viewBox="0 0 64 64" aria-hidden="true">
        <circle className={styles.track} cx="32" cy="32" r={R} />
        {known ? (
          // At zero there is no arc to draw: a round cap would otherwise
          // leave a dot sitting at twelve o'clock.
          pct > 0 && (
            <circle className={styles.arc} cx="32" cy="32" r={R}
              strokeDasharray={C} strokeDashoffset={C * (1 - pct / 100)} />
          )
        ) : (
          <circle className={cx(styles.arc, styles.sweep)} cx="32" cy="32" r={R}
            strokeDasharray={`${C * SWEEP} ${C}`} />
        )}
      </svg>
      {numeral && (
        <span className={styles.value} aria-hidden="true">
          {pct}<span className={styles.sign}>%</span>
        </span>
      )}
      {label && showLabel && <span className={styles.label} aria-hidden="true">{label}</span>}
    </div>
  );
}
