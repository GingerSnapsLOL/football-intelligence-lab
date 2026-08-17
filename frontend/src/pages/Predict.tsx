import { useEffect, useMemo, useState } from "react";
import { predictShot, getModels } from "../api/client";
import { displayModelName, formatMetric, formatXg } from "../api/format";
import type { ModelInfo, ShotPredictionResponse } from "../api/types";
import { Pitch } from "../components/Pitch";
import { Status } from "../components/Status";

const BODY_PARTS = ["Right Foot", "Left Foot", "Head", "Other"];
const SHOT_TYPES = ["Open Play", "Free Kick", "Penalty", "Corner"];
const TECHNIQUES = ["Normal", "Volley", "Half Volley", "Lob", "Backheel", "Diving Header"];

export function PredictPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelName, setModelName] = useState("");
  const [x, setX] = useState(108);
  const [y, setY] = useState(40);
  const [bodyPart, setBodyPart] = useState("Right Foot");
  const [shotType, setShotType] = useState("Open Play");
  const [technique, setTechnique] = useState("Normal");
  const [underPressure, setUnderPressure] = useState(false);
  const [firstTime, setFirstTime] = useState(false);
  const [result, setResult] = useState<ShotPredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getModels()
      .then((payload) => {
        setModels(payload.models);
        if (payload.models[0]) {
          setModelName(payload.models[0].name);
        }
      })
      .catch((caught: unknown) => {
        setError(caught instanceof Error ? caught.message : String(caught));
      });
  }, []);

  const selected = models.find((model) => model.name === modelName);
  const needsContext = useMemo(() => {
    const features = selected?.features ?? [];
    return {
      body_part: features.includes("body_part"),
      shot_type: features.includes("shot_type"),
      technique: features.includes("technique"),
      under_pressure: features.includes("under_pressure"),
      first_time: features.includes("first_time"),
    };
  }, [selected]);

  const submit = () => {
    setLoading(true);
    setError(null);
    predictShot({
      model: modelName || null,
      x,
      y,
      body_part: needsContext.body_part ? bodyPart : null,
      shot_type: needsContext.shot_type ? shotType : null,
      technique: needsContext.technique ? technique : null,
      under_pressure: needsContext.under_pressure ? underPressure : false,
      first_time: needsContext.first_time ? firstTime : false,
    })
      .then(setResult)
      .catch((caught: unknown) => {
        setResult(null);
        setError(caught instanceof Error ? caught.message : String(caught));
      })
      .finally(() => setLoading(false));
  };

  return (
    <section>
      <header className="page-head">
        <h1>Shot predictor</h1>
        <p>Click the pitch or edit coordinates. Distance and angle are derived by the API from x and y.</p>
      </header>

      <div className="split">
        <div className="pitch-frame">
          <Pitch
            interactive
            cursor={{ x, y }}
            onPitchClick={(nextX, nextY) => {
              setX(nextX);
              setY(nextY);
            }}
            shots={
              result
                ? [
                    {
                      id: "prediction",
                      x: result.features.x,
                      y: result.features.y,
                      goal: false,
                      xg: result.predicted_xg,
                      title: `Predicted xG ${formatXg(result.predicted_xg)}`,
                    },
                  ]
                : []
            }
          />
          <p className="muted">Attacking toward the right-hand goal. Click to place the shot.</p>
        </div>

        <form
          className="form"
          onSubmit={(event) => {
            event.preventDefault();
            submit();
          }}
        >
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
          <div className="form__row">
            <label>
              x (yards)
              <input
                type="number"
                min={0}
                max={120}
                step={0.1}
                value={x}
                onChange={(event) => setX(Number(event.target.value))}
              />
            </label>
            <label>
              y (yards)
              <input
                type="number"
                min={0}
                max={80}
                step={0.1}
                value={y}
                onChange={(event) => setY(Number(event.target.value))}
              />
            </label>
          </div>
          {needsContext.body_part ? (
            <label>
              Body part
              <select value={bodyPart} onChange={(event) => setBodyPart(event.target.value)}>
                {BODY_PARTS.map((option) => (
                  <option key={option}>{option}</option>
                ))}
              </select>
            </label>
          ) : null}
          {needsContext.shot_type ? (
            <label>
              Shot type
              <select value={shotType} onChange={(event) => setShotType(event.target.value)}>
                {SHOT_TYPES.map((option) => (
                  <option key={option}>{option}</option>
                ))}
              </select>
            </label>
          ) : null}
          {needsContext.technique ? (
            <label>
              Technique
              <select value={technique} onChange={(event) => setTechnique(event.target.value)}>
                {TECHNIQUES.map((option) => (
                  <option key={option}>{option}</option>
                ))}
              </select>
            </label>
          ) : null}
          {needsContext.under_pressure ? (
            <label className="check">
              <input
                type="checkbox"
                checked={underPressure}
                onChange={(event) => setUnderPressure(event.target.checked)}
              />
              Under pressure
            </label>
          ) : null}
          {needsContext.first_time ? (
            <label className="check">
              <input
                type="checkbox"
                checked={firstTime}
                onChange={(event) => setFirstTime(event.target.checked)}
              />
              First time
            </label>
          ) : null}

          <button type="submit" disabled={loading || models.length === 0}>
            {loading ? "Scoring…" : "Predict xG"}
          </button>
          <Status error={error} />

          {result ? (
            <div className="xg-result">
              <div className="xg-result__label">{displayModelName(result.model)}</div>
              <div className="xg-result__value">{formatXg(result.predicted_xg)}</div>
              <dl className="metric-row">
                <div>
                  <dt>Distance</dt>
                  <dd>{formatMetric(result.features.shot_distance, 1)} yd</dd>
                </div>
                <div>
                  <dt>Angle</dt>
                  <dd>{formatMetric(result.features.shot_angle_degrees, 1)}°</dd>
                </div>
              </dl>
            </div>
          ) : null}
        </form>
      </div>
    </section>
  );
}
