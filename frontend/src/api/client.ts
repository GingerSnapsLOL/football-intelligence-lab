import type {
  DatasetSummary,
  MatchListResponse,
  MatchShotsResponse,
  MatchSummary,
  ModelListResponse,
  ShotPredictionRequest,
  ShotPredictionResponse,
  StatisticsSummary,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function readDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (typeof body === "object" && body !== null && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string") {
        return detail;
      }
      if (Array.isArray(detail)) {
        return detail
          .map((item) => {
            if (typeof item === "object" && item !== null && "msg" in item) {
              return String((item as { msg: unknown }).msg);
            }
            return JSON.stringify(item);
          })
          .join("; ");
      }
    }
  } catch {
    /* fall through */
  }
  return `${response.status} ${response.statusText}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    throw new ApiError(response.status, await readDetail(response));
  }
  return (await response.json()) as T;
}

export function getHealth(): Promise<{ status: string }> {
  return request("/health");
}

export function getSummary(): Promise<DatasetSummary> {
  return request("/api/summary");
}

export function getModels(): Promise<ModelListResponse> {
  return request("/api/models");
}

export function getStatistics(): Promise<StatisticsSummary> {
  return request("/api/statistics/summary");
}

export function getMatches(limit = 100, offset = 0): Promise<MatchListResponse> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  return request(`/api/matches?${params.toString()}`);
}

export async function getAllMatches(): Promise<MatchSummary[]> {
  const pageSize = 100;
  const first = await getMatches(pageSize, 0);
  if (first.total <= first.matches.length) {
    return first.matches;
  }
  const rest = await getMatches(pageSize, first.matches.length);
  return [...first.matches, ...rest.matches];
}

export function getMatchShots(
  matchId: number,
  model?: string,
): Promise<MatchShotsResponse> {
  const params = new URLSearchParams();
  if (model) {
    params.set("model", model);
  }
  const query = params.toString();
  return request(`/api/matches/${matchId}/shots${query ? `?${query}` : ""}`);
}

export function predictShot(payload: ShotPredictionRequest): Promise<ShotPredictionResponse> {
  return request("/api/predict/shot", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
