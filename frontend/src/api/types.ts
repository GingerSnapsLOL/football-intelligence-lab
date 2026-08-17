/** Types matching the FastAPI Pydantic schemas. */

export type DatasetSummary = {
  matches: number;
  players: number;
  shots: number;
  goals: number;
  goal_rate: number;
};

export type ModelMetrics = {
  n: number;
  positives: number;
  prevalence: number;
  roc_auc: number;
  pr_auc: number;
  log_loss: number;
  log_loss_skill: number;
  brier_score: number;
  brier_skill: number;
  mean_prediction: number;
  calibration_in_the_large: number;
};

export type ModelInfo = {
  name: string;
  task: string;
  features: string[];
  training_samples: number;
  test_samples: number;
  validation_strategy: string;
  metrics: ModelMetrics;
};

export type ModelListResponse = {
  models: ModelInfo[];
};

export type ShotPredictionRequest = {
  model?: string | null;
  x: number;
  y: number;
  body_part?: string | null;
  shot_type?: string | null;
  technique?: string | null;
  under_pressure?: boolean;
  first_time?: boolean;
};

export type DerivedShotFeatures = {
  x: number;
  y: number;
  shot_distance: number;
  shot_angle: number;
  shot_angle_degrees: number;
  body_part: string | null;
  shot_type: string | null;
  technique: string | null;
  under_pressure: boolean;
  first_time: boolean;
};

export type ShotPredictionResponse = {
  model: string;
  predicted_xg: number;
  features: DerivedShotFeatures;
};

export type NumericSummary = {
  name: string;
  n: number;
  mean: number;
  median: number;
  std: number;
  q1: number;
  q3: number;
  iqr: number;
  skewness: number;
};

export type ConversionRow = {
  group: string;
  shots: number;
  goals: number;
  conversion_rate: number;
};

export type StatisticalFinding = {
  name: string;
  question: string;
  test_name: string;
  statistic: number;
  p_value: number;
  n_total: number;
  n_a: number | null;
  n_b: number | null;
  estimate: number | null;
  estimate_name: string | null;
  confidence_interval: [number, number] | null;
  warnings: string[];
  notes: string[];
};

export type StatisticsSummary = {
  shots: number;
  goals: number;
  goal_rate: number;
  shot_distance: NumericSummary;
  shot_angle_degrees: NumericSummary;
  conversion_by_body_part: ConversionRow[];
  findings: StatisticalFinding[];
};

export type MatchSummary = {
  match_id: number;
  match_date: string;
  competition_name: string;
  season_name: string;
  stage: string | null;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
};

export type MatchListResponse = {
  total: number;
  limit: number;
  offset: number;
  matches: MatchSummary[];
};

export type MatchShot = {
  shot_id: string;
  period: number;
  minute: number;
  team: string;
  player: string;
  x: number;
  y: number;
  shot_distance: number;
  shot_angle: number;
  goal: boolean;
  outcome: string | null;
  statsbomb_xg: number | null;
  predicted_xg: number | null;
};

export type MatchShotsResponse = {
  match_id: number;
  match_date: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  model: string | null;
  shots: MatchShot[];
};
