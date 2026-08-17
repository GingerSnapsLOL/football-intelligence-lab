import type { MouseEvent } from "react";

export const PITCH_LENGTH = 120;
export const PITCH_WIDTH = 80;

export type PitchMarker = {
  id: string;
  x: number;
  y: number;
  goal: boolean;
  xg?: number | null;
  title?: string;
};

type PitchProps = {
  shots?: PitchMarker[];
  cursor?: { x: number; y: number } | null;
  highlightedId?: string | null;
  interactive?: boolean;
  onPitchClick?: (x: number, y: number) => void;
  onShotHover?: (id: string | null) => void;
  onShotClick?: (id: string) => void;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function radiusForXg(xg: number | null | undefined): number {
  if (xg === null || xg === undefined) {
    return 1.35;
  }
  return 1.05 + 4.2 * Math.min(1, Math.max(0, xg));
}

function toSvgY(y: number): number {
  return PITCH_WIDTH - y;
}

function eventToPitch(event: MouseEvent<SVGSVGElement>): { x: number; y: number } | null {
  const svg = event.currentTarget;
  const ctm = svg.getScreenCTM();
  if (!ctm) {
    return null;
  }
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  const loc = point.matrixTransform(ctm.inverse());
  return {
    x: clamp(loc.x, 0, PITCH_LENGTH),
    y: clamp(PITCH_WIDTH - loc.y, 0, PITCH_WIDTH),
  };
}

export function Pitch({
  shots = [],
  cursor = null,
  highlightedId = null,
  interactive = false,
  onPitchClick,
  onShotHover,
  onShotClick,
}: PitchProps) {
  const handleClick = (event: MouseEvent<SVGSVGElement>) => {
    if (!onPitchClick) {
      return;
    }
    const loc = eventToPitch(event);
    if (loc) {
      onPitchClick(Number(loc.x.toFixed(1)), Number(loc.y.toFixed(1)));
    }
  };

  return (
    <svg
      className={`pitch${interactive ? " pitch--interactive" : ""}`}
      viewBox={`-2 -2 ${PITCH_LENGTH + 4} ${PITCH_WIDTH + 4}`}
      role="img"
      aria-label="Football pitch, attacking toward the right-hand goal"
      onClick={handleClick}
    >
      <rect
        x={-2}
        y={-2}
        width={PITCH_LENGTH + 4}
        height={PITCH_WIDTH + 4}
        className="pitch__grass-outer"
      />
      <rect x={0} y={0} width={PITCH_LENGTH} height={PITCH_WIDTH} className="pitch__grass" />
      <rect
        x={0}
        y={0}
        width={PITCH_LENGTH}
        height={PITCH_WIDTH}
        className="pitch__line"
        fill="none"
      />
      <line x1={60} y1={0} x2={60} y2={80} className="pitch__line" />
      <circle cx={60} cy={40} r={10} className="pitch__line" fill="none" />
      <circle cx={60} cy={40} r={0.45} className="pitch__spot" />
      {/* Penalty areas */}
      <rect x={0} y={18} width={18} height={44} className="pitch__line" fill="none" />
      <rect x={102} y={18} width={18} height={44} className="pitch__line" fill="none" />
      <rect x={0} y={30} width={6} height={20} className="pitch__line" fill="none" />
      <rect x={114} y={30} width={6} height={20} className="pitch__line" fill="none" />
      <circle cx={12} cy={40} r={0.45} className="pitch__spot" />
      <circle cx={108} cy={40} r={0.45} className="pitch__spot" />
      {/* Penalty arcs */}
      <path d="M 18 32.3 A 10 10 0 0 0 18 47.7" className="pitch__line" fill="none" />
      <path d="M 102 32.3 A 10 10 0 0 1 102 47.7" className="pitch__line" fill="none" />
      {/* Goals */}
      <rect x={-1.4} y={36} width={1.4} height={8} className="pitch__goal" />
      <rect x={120} y={36} width={1.4} height={8} className="pitch__goal" />

      {shots.map((shot) => {
        const highlighted = shot.id === highlightedId;
        return (
          <circle
            key={shot.id}
            cx={shot.x}
            cy={toSvgY(shot.y)}
            r={radiusForXg(shot.xg) * (highlighted ? 1.25 : 1)}
            className={`pitch__shot${shot.goal ? " pitch__shot--goal" : " pitch__shot--miss"}${
              highlighted ? " pitch__shot--hot" : ""
            }`}
            onMouseEnter={() => onShotHover?.(shot.id)}
            onMouseLeave={() => onShotHover?.(null)}
            onClick={(event) => {
              event.stopPropagation();
              onShotClick?.(shot.id);
            }}
          >
            <title>{shot.title ?? `${shot.goal ? "Goal" : "Miss"} (${shot.x.toFixed(1)}, ${shot.y.toFixed(1)})`}</title>
          </circle>
        );
      })}

      {cursor ? (
        <g>
          <circle cx={cursor.x} cy={toSvgY(cursor.y)} r={2.2} className="pitch__cursor" />
          <line
            x1={cursor.x}
            y1={toSvgY(cursor.y)}
            x2={120}
            y2={40}
            className="pitch__sight"
          />
        </g>
      ) : null}
    </svg>
  );
}
