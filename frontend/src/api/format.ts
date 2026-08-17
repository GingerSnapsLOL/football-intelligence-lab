import type { ModelInfo, StatisticalFinding } from "./types";

export function formatInt(value: number): string {
  return new Intl.NumberFormat("en-GB").format(value);
}

export function formatPct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatMetric(value: number, digits = 3): string {
  return value.toFixed(digits);
}

export function formatPValue(value: number): string {
  if (value < 0.001) {
    return value.toExponential(2);
  }
  return value.toFixed(3);
}

export function formatXg(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "—";
  }
  return value.toFixed(3);
}

export function modelFamily(name: string): string {
  const key = name.toLowerCase();
  if (key.includes("logistic")) {
    return "Logistic Regression";
  }
  if (key.includes("xgboost") || key.includes("xgb")) {
    return "XGBoost";
  }
  if (key.includes("lightgbm") || key.includes("lgbm")) {
    return "LightGBM";
  }
  if (key.includes("catboost")) {
    return "CatBoost";
  }
  if (key.includes("gam")) {
    return "GAM";
  }
  return name;
}

export function displayModelName(name: string): string {
  return name.replaceAll("_", " ");
}

export function interpretFinding(finding: StatisticalFinding): string {
  const effect =
    finding.estimate !== null && finding.estimate_name
      ? ` ${finding.estimate_name} = ${finding.estimate.toFixed(3)}${
          finding.confidence_interval
            ? ` (95% CI ${finding.confidence_interval[0].toFixed(3)} to ${finding.confidence_interval[1].toFixed(3)})`
            : ""
        }.`
      : "";
  const caveat =
    finding.warnings[0] ??
    finding.notes[0] ??
    "A p-value is the probability of data at least this extreme if the null were true; it is not the probability the null is true, and it does not say whether the effect is large enough to matter.";
  return `${finding.question} ${finding.test_name}: statistic = ${finding.statistic.toFixed(3)}, p = ${formatPValue(finding.p_value)}.${effect} ${caveat}`;
}

export const MODEL_FAMILIES = [
  "Logistic Regression",
  "XGBoost",
  "LightGBM",
  "CatBoost",
] as const;

export function modelsInFamily(models: ModelInfo[], family: string): ModelInfo[] {
  return models.filter((model) => modelFamily(model.name) === family);
}
