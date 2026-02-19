// Route paths for the application
export const ROUTES = {
  DASHBOARD: '/',
  MODELS: '/models',
  MODEL_DETAIL: '/models/:modelId',
  MODEL_VERSION: '/models/:modelId/versions/:versionId',
  VERSIONS: '/versions',
  VERSION_DETAIL: '/versions/:versionId',
  // Keep releases routes for backward compatibility (redirects to versions)
  RELEASES: '/releases',
  RELEASE_DETAIL: '/releases/:releaseId',
  MODEL_RELEASE: '/models/:modelId/releases/:releaseId',
  DEPLOYMENTS: '/deployments',
  ARTIFACTS: '/artifacts',
  API_KEYS: '/api-keys',
  SETTINGS: '/settings',
  HELP: '/help',
} as const;

// Helper type for route params
export type ModelParams = { modelId: string };
export type VersionParams = { versionId: string };
export type ModelVersionParams = { modelId: string; versionId: string };
// Backward compatibility aliases
export type ReleaseParams = { releaseId: string };
export type ModelReleaseParams = { modelId: string; releaseId: string };
