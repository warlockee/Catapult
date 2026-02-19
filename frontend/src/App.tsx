import { lazy, Suspense } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { ErrorBoundary } from './components/ErrorBoundary';

// Lazy load heavy components for code splitting
const Dashboard = lazy(() => import('./components/Dashboard').then(m => ({ default: m.Dashboard })));
const ImageList = lazy(() => import('./components/ImageList').then(m => ({ default: m.ImageList })));
const ModelDetail = lazy(() => import('./components/ModelDetail').then(m => ({ default: m.ModelDetail })));
const ReleaseList = lazy(() => import('./components/ReleaseList').then(m => ({ default: m.ReleaseList })));
const ReleaseDetail = lazy(() => import('./components/ReleaseDetail').then(m => ({ default: m.ReleaseDetail })));
const Deployments = lazy(() => import('./components/Deployments').then(m => ({ default: m.Deployments })));
const DeploymentDetail = lazy(() => import('./components/DeploymentDetail').then(m => ({ default: m.DeploymentDetail })));
const ApiKeyManagement = lazy(() => import('./components/ApiKeyManagement').then(m => ({ default: m.ApiKeyManagement })));
const ArtifactManagement = lazy(() => import('./components/ArtifactManagement').then(m => ({ default: m.ArtifactManagement })));
const Settings = lazy(() => import('./components/Settings').then(m => ({ default: m.Settings })));
const Help = lazy(() => import('./components/Help').then(m => ({ default: m.Help })));
const ModelCard = lazy(() => import('./components/ModelCard').then(m => ({ default: m.ModelCard })));

// Loading fallback component
const PageLoader = () => (
  <div className="flex items-center justify-center h-64">
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
  </div>
);

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50 flex">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <Header />
        <main className="flex-1 p-8 min-w-0">
          <ErrorBoundary>
            <Suspense fallback={<PageLoader />}>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/dashboard" element={<Navigate to="/" replace />} />
                <Route path="/models" element={<ImageList />} />
                <Route path="/models/:modelId" element={<ModelDetail />} />
                <Route path="/models/:modelId/card" element={<ModelCard />} />
                {/* Version routes - canonical */}
                <Route path="/models/:modelId/versions/:releaseId" element={<ReleaseDetail />} />
                <Route path="/versions" element={<ReleaseList />} />
                <Route path="/versions/:releaseId" element={<ReleaseDetail />} />
                {/* Release routes - kept for backward compatibility and UI consistency */}
                <Route path="/models/:modelId/releases/:releaseId" element={<ReleaseDetail />} />
                <Route path="/releases" element={<ReleaseList />} />
                <Route path="/releases/:releaseId" element={<ReleaseDetail />} />
                <Route path="/deployments" element={<Deployments />} />
                <Route path="/deployments/:deploymentId" element={<DeploymentDetail />} />
                <Route path="/artifacts" element={<ArtifactManagement />} />
                <Route path="/api-keys" element={<ApiKeyManagement />} />
                <Route path="/settings" element={<Settings />} />
                <Route path="/help" element={<Help />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </Suspense>
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}
