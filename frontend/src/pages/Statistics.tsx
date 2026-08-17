import { useEffect, useState } from "react";
import { getStatistics } from "../api/client";
import { formatInt, formatMetric, formatPValue, formatPct, interpretFinding } from "../api/format";
import type { StatisticsSummary } from "../api/types";
import { Status } from "../components/Status";

export function StatisticsPage() {
  const [data, setData] = useState<StatisticsSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    getStatistics()
      .then((payload) => {
        if (!cancelled) {
          setData(payload);
          setError(null);
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

  return (
    <section>
      <header className="page-head">
        <h1>Statistical lab</h1>
        <p>
          Tests are computed on the processed shot dataset by the API. p-values are not treated as
          proof that an effect matters.
        </p>
      </header>
      <Status loading={loading} error={error} />

      {data ? (
        <>
          <div className="kpi-grid">
            <article className="kpi">
              <div className="kpi__label">Shots</div>
              <div className="kpi__value">{formatInt(data.shots)}</div>
            </article>
            <article className="kpi">
              <div className="kpi__label">Goals</div>
              <div className="kpi__value">{formatInt(data.goals)}</div>
            </article>
            <article className="kpi">
              <div className="kpi__label">Goal rate</div>
              <div className="kpi__value">{formatPct(data.goal_rate)}</div>
            </article>
          </div>

          <div className="split">
            <article className="panel">
              <h2>Shot distance</h2>
              <NumericTable summary={data.shot_distance} unit="yd" />
            </article>
            <article className="panel">
              <h2>Shot angle</h2>
              <NumericTable summary={data.shot_angle_degrees} unit="°" />
            </article>
          </div>

          <h2 className="section-title">Conversion by body part</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Body part</th>
                  <th>Shots</th>
                  <th>Goals</th>
                  <th>Conversion</th>
                </tr>
              </thead>
              <tbody>
                {data.conversion_by_body_part.map((row) => (
                  <tr key={row.group}>
                    <td>{row.group}</td>
                    <td>{formatInt(row.shots)}</td>
                    <td>{formatInt(row.goals)}</td>
                    <td>{formatPct(row.conversion_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h2 className="section-title">Hypothesis tests</h2>
          <div className="findings">
            {data.findings.length === 0 ? (
              <Status empty="The API returned no tests for this dataset." />
            ) : (
              data.findings.map((finding) => (
                <article key={finding.name} className="panel">
                  <div className="panel__kicker">{finding.test_name}</div>
                  <h3>{finding.question}</h3>
                  <dl className="metric-row">
                    <div>
                      <dt>Statistic</dt>
                      <dd>{formatMetric(finding.statistic)}</dd>
                    </div>
                    <div>
                      <dt>p-value</dt>
                      <dd>{formatPValue(finding.p_value)}</dd>
                    </div>
                    <div>
                      <dt>n</dt>
                      <dd>{formatInt(finding.n_total)}</dd>
                    </div>
                    {finding.estimate !== null ? (
                      <div>
                        <dt>{finding.estimate_name ?? "Effect"}</dt>
                        <dd>{formatMetric(finding.estimate)}</dd>
                      </div>
                    ) : null}
                  </dl>
                  {finding.confidence_interval ? (
                    <p className="muted">
                      95% CI {finding.confidence_interval[0].toFixed(3)} to{" "}
                      {finding.confidence_interval[1].toFixed(3)}
                    </p>
                  ) : null}
                  <p className="interpretation">{interpretFinding(finding)}</p>
                </article>
              ))
            )}
          </div>
        </>
      ) : null}
    </section>
  );
}

function NumericTable({
  summary,
  unit,
}: {
  summary: StatisticsSummary["shot_distance"];
  unit: string;
}) {
  const rows = [
    ["n", formatInt(summary.n), ""],
    ["Mean", formatMetric(summary.mean, 2), unit],
    ["Median", formatMetric(summary.median, 2), unit],
    ["SD", formatMetric(summary.std, 2), unit],
    ["Q1", formatMetric(summary.q1, 2), unit],
    ["Q3", formatMetric(summary.q3, 2), unit],
    ["IQR", formatMetric(summary.iqr, 2), unit],
    ["Skewness", formatMetric(summary.skewness, 2), ""],
  ];
  return (
    <dl className="stat-list">
      {rows.map(([label, value, suffix]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>
            {value}
            {suffix ? ` ${suffix}` : ""}
          </dd>
        </div>
      ))}
    </dl>
  );
}
