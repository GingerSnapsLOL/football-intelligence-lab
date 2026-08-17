import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { MatchesPage } from "./pages/Matches";
import { ModelsPage } from "./pages/Models";
import { OverviewPage } from "./pages/Overview";
import { PredictPage } from "./pages/Predict";
import { StatisticsPage } from "./pages/Statistics";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<OverviewPage />} />
          <Route path="models" element={<ModelsPage />} />
          <Route path="matches" element={<MatchesPage />} />
          <Route path="predict" element={<PredictPage />} />
          <Route path="statistics" element={<StatisticsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
