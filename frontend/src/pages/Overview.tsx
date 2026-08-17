import { useEffect, useState } from "react";
import { getModels, getSummary } from "../api/client";
import { displayModelName, formatInt, formatMetric, formatPct, modelFamily } from "../api/format";
import type { DatasetSummary, ModelInfo } from "../api/types";
import { Status } from "../components/Status";

export function OverviewPage() {
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    Promise.all([getSummary(), getModels()])
      .then(([nextSummary, nextModels]) => {
        if (cancelled) {
          return;
        }
        setSummary(nextSummary);
        setModels(nextModels.models);
        setError(null);
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
        <h1>Overview</h1>
        <p>Development subset: FIFA World Cup 2018 and UEFA Euro 2020, queried live from processed Parquet.</p>
      </header>
      <Status loading={loading} error={error} />
      {summary ? (
        <div className="kpi-grid">
          <Kpi label="Matches" value={formatInt(summary.matches)} />
          <Kpi label="Players" value={formatInt(summary.players)} />
          <Kpi label="Shots" value={formatInt(summary.shots)} hint="in-play, shootouts excluded" />
          <Kpi label="Goals" value={formatInt(summary.goals)} />
          <Kpi label="Goal rate" value={formatPct(summary.goal_rate)} hint="baseline prevalence" />
        </div>
      ) : null}

      <h2 className="section-title">Available models</h2>
      {models.length === 0 && !loading && !error ? (
        <Status empty="No trained artifacts. Run `make train` then restart the API." />
      ) : (
        <div className="model-list">
          {models.map((model) => (
            <article key={model.name} className="panel">
              <div className="panel__kicker">{modelFamily(model.name)}</div>
              <h3>{displayModelName(model.name)}</h3>
              <p className="muted">
                {model.validation_strategy.replaceAll("_", " ")} · {formatInt(model.training_samples)}{" "}
                train / {formatInt(model.test_samples)} test
              </p>
              <dl className="metric-row">
                <div>
                  <dt>ROC-AUC</dt>
                  <dd>{formatMetric(model.metrics.roc_auc)}</dd>
                </div>
                <div>
                  <dt>PR-AUC</dt>
                  <dd>{formatMetric(model.metrics.pr_auc)}</dd>
                </div>
                <div>
                  <dt>Log loss</dt>
                  <dd>{formatMetric(model.metrics.log_loss)}</dd>
                </div>
                <div>
                  <dt>Brier</dt>
                  <dd>{formatMetric(model.metrics.brier_score)}</dd>
                </div>
              </dl>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function Kpi({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <article className="kpi">
      <div className="kpi__label">{label}</div>
      <div className="kpi__value">{value}</div>
      {hint ? <div className="kpi__hint">{hint}</div> : null}
    </article>
  );
}
