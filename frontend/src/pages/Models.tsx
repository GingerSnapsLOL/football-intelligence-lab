import { useEffect, useState } from "react";
import { getModels } from "../api/client";
import {
  MODEL_FAMILIES,
  displayModelName,
  formatInt,
  formatMetric,
  formatPct,
  modelFamily,
  modelsInFamily,
} from "../api/format";
import type { ModelInfo } from "../api/types";
import { Status } from "../components/Status";

export function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    getModels()
      .then((payload) => {
        if (!cancelled) {
          setModels(payload.models);
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
        <h1>Model comparison</h1>
        <p>
          Metrics are held-out scores from persisted artifacts. Missing families have not been
          trained yet — they are not filled with placeholder numbers.
        </p>
      </header>
      <Status loading={loading} error={error} />

      <div className="family-grid">
        {MODEL_FAMILIES.map((family) => {
          const members = modelsInFamily(models, family);
          return (
            <article key={family} className="panel">
              <h2>{family}</h2>
              {members.length === 0 ? (
                <p className="muted">Not trained. No metrics to show.</p>
              ) : (
                members.map((model) => (
                  <div key={model.name} className="family-model">
                    <div className="family-model__name">{displayModelName(model.name)}</div>
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
                  </div>
                ))
              )}
            </article>
          );
        })}
      </div>

      {models.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Family</th>
                <th>n</th>
                <th>Prevalence</th>
                <th>ROC-AUC</th>
                <th>PR-AUC</th>
                <th>Log loss</th>
                <th>Brier</th>
                <th>Brier skill</th>
                <th>Validation</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model) => (
                <tr key={model.name}>
                  <td>{displayModelName(model.name)}</td>
                  <td>{modelFamily(model.name)}</td>
                  <td>{formatInt(model.metrics.n)}</td>
                  <td>{formatPct(model.metrics.prevalence)}</td>
                  <td>{formatMetric(model.metrics.roc_auc)}</td>
                  <td>{formatMetric(model.metrics.pr_auc)}</td>
                  <td>{formatMetric(model.metrics.log_loss)}</td>
                  <td>{formatMetric(model.metrics.brier_score)}</td>
                  <td>{formatMetric(model.metrics.brier_skill)}</td>
                  <td>{model.validation_strategy.replaceAll("_", " ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}
