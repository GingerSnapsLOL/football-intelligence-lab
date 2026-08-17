import { useEffect, useMemo, useState } from "react";
import { getAllMatches, getMatchShots, getModels } from "../api/client";
import { displayModelName, formatXg } from "../api/format";
import type { MatchShotsResponse, MatchSummary, ModelInfo } from "../api/types";
import { Pitch } from "../components/Pitch";
import { Status } from "../components/Status";

function matchLabel(match: MatchSummary): string {
  return `${match.match_date} · ${match.home_team} ${match.home_score}–${match.away_score} ${match.away_team}`;
}

export function MatchesPage() {
  const [matches, setMatches] = useState<MatchSummary[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [matchId, setMatchId] = useState<number | null>(null);
  const [modelName, setModelName] = useState<string>("");
  const [detail, setDetail] = useState<MatchShotsResponse | null>(null);
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingShots, setLoadingShots] = useState(false);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    let cancelled = false;
    Promise.all([getAllMatches(), getModels()])
      .then(([nextMatches, nextModels]) => {
        if (cancelled) {
          return;
        }
        setMatches(nextMatches);
        setModels(nextModels.models);
        if (nextMatches[0]) {
          setMatchId(nextMatches[0].match_id);
        }
        if (nextModels.models[0]) {
          setModelName(nextModels.models[0].name);
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : String(caught));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (matchId === null) {
      return;
    }
    let cancelled = false;
    setLoadingShots(true);
    getMatchShots(matchId, modelName || undefined)
      .then((payload) => {
        if (!cancelled) {
          setDetail(payload);
          setHoverId(null);
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : String(caught));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingShots(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [matchId, modelName]);

  const filtered = useMemo(() => {
    const query = filter.trim().toLowerCase();
    if (!query) {
      return matches;
    }
    return matches.filter((match) => matchLabel(match).toLowerCase().includes(query));
  }, [filter, matches]);

  const hovered = detail?.shots.find((shot) => shot.shot_id === hoverId) ?? null;

  return (
    <section>
      <header className="page-head">
        <h1>Match explorer</h1>
        <p>Real shot locations on a StatsBomb pitch. Marker size is model xG when the API provides it.</p>
      </header>
      <Status loading={loading} error={error} />

      <div className="toolbar">
        <label>
          Filter
          <input
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            placeholder="team or date"
          />
        </label>
        <label>
          Match
          <select
            value={matchId ?? ""}
            onChange={(event) => setMatchId(Number(event.target.value))}
          >
            {filtered.map((match) => (
              <option key={match.match_id} value={match.match_id}>
                {matchLabel(match)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Model
          <select value={modelName} onChange={(event) => setModelName(event.target.value)}>
            {models.map((model) => (
              <option key={model.name} value={model.name}>
                {displayModelName(model.name)}
              </option>
            ))}
          </select>
        </label>
      </div>

      {detail ? (
        <>
          <div className="match-hero">
            <div>
              <div className="match-hero__comp">
                {detail.match_date} · {matches.find((m) => m.match_id === detail.match_id)?.competition_name}
              </div>
              <h2>
                {detail.home_team} {detail.home_score}–{detail.away_score} {detail.away_team}
              </h2>
              <p className="muted">
                {detail.shots.length} shots · {detail.shots.filter((shot) => shot.goal).length} goals
                {detail.model ? ` · scored with ${displayModelName(detail.model)}` : ""}
              </p>
            </div>
            <div className="legend">
              <span>
                <i className="swatch swatch--goal" /> Goal
              </span>
              <span>
                <i className="swatch swatch--miss" /> Miss
              </span>
              <span>Larger marker = higher xG</span>
            </div>
          </div>

          <div className="split">
            <div className="pitch-frame">
              {loadingShots ? <Status loading /> : null}
              <Pitch
                shots={detail.shots.map((shot) => ({
                  id: shot.shot_id,
                  x: shot.x,
                  y: shot.y,
                  goal: shot.goal,
                  xg: shot.predicted_xg ?? shot.statsbomb_xg,
                  title: `${shot.player} · ${shot.minute}' · ${shot.outcome ?? (shot.goal ? "Goal" : "Miss")} · xG ${formatXg(shot.predicted_xg)}`,
                }))}
                highlightedId={hoverId}
                onShotHover={setHoverId}
              />
              {hovered ? (
                <div className="tooltip-card">
                  <strong>{hovered.player}</strong>
                  <span>
                    {hovered.team} · {hovered.minute}&apos; · {hovered.outcome ?? (hovered.goal ? "Goal" : "Miss")}
                  </span>
                  <span>
                    xG {formatXg(hovered.predicted_xg)}
                    {hovered.statsbomb_xg !== null ? ` · StatsBomb ${formatXg(hovered.statsbomb_xg)}` : ""}
                  </span>
                </div>
              ) : null}
            </div>
            <div className="table-wrap table-wrap--compact">
              <table>
                <thead>
                  <tr>
                    <th>Min</th>
                    <th>Player</th>
                    <th>Outcome</th>
                    <th>xG</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.shots.map((shot) => (
                    <tr
                      key={shot.shot_id}
                      className={shot.shot_id === hoverId ? "is-hot" : ""}
                      onMouseEnter={() => setHoverId(shot.shot_id)}
                      onMouseLeave={() => setHoverId(null)}
                    >
                      <td>{shot.minute}</td>
                      <td>
                        {shot.player}
                        <div className="cell-sub">{shot.team}</div>
                      </td>
                      <td>{shot.outcome ?? (shot.goal ? "Goal" : "Miss")}</td>
                      <td>{formatXg(shot.predicted_xg)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : null}
    </section>
  );
}
