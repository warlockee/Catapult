/**
 * API client for Docker Release Registry
 */

// Vite environment variable types
declare global {
  interface ImportMetaEnv {
    readonly VITE_API_URL?: string;
    readonly VITE_DEFAULT_API_KEY?: string;
    readonly VITE_USE_SESSION_STORAGE?: string;
  }
}

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

// System key for internal tool mode (baked in at build time)
const SYSTEM_KEY = import.meta.env.VITE_DEFAULT_API_KEY;

// Use sessionStorage instead of localStorage for better security
// sessionStorage is cleared when the browser tab closes, reducing exposure window
// Can be overridden via env var VITE_USE_SESSION_STORAGE=false for persistent storage
const USE_SESSION_STORAGE = import.meta.env.VITE_USE_SESSION_STORAGE !== 'false';

const API_KEY_STORAGE_KEY = 'model_registry_api_key';

// Validate API key format (basic validation to prevent injection)
const isValidApiKeyFormat = (key: string): boolean => {
  // API keys should be alphanumeric with possible dots, underscores, hyphens
  // Typical format: prefix.secret or just a base64-like string
  const validPattern = /^[a-zA-Z0-9._-]{10,200}$/;
  return validPattern.test(key);
};

// Get the appropriate storage (session or local)
const getStorage = (): Storage => {
  return USE_SESSION_STORAGE ? sessionStorage : localStorage;
};

// Get API key from storage, or use system key if configured
const getApiKey = (): string | null => {
  // If a system key is baked in, use it universally (Internal Tool Mode)
  // This is the most secure option for internal deployments
  if (SYSTEM_KEY) {
    return SYSTEM_KEY;
  }

  const storage = getStorage();
  const apiKey = storage.getItem(API_KEY_STORAGE_KEY);

  // Validate stored key format
  if (apiKey && !isValidApiKeyFormat(apiKey)) {
    // Invalid key format detected, clear it
    console.warn('Invalid API key format detected in storage, clearing');
    storage.removeItem(API_KEY_STORAGE_KEY);
    return null;
  }

  return apiKey;
};

// Set API key in storage
export const setApiKey = (key: string): void => {
  if (!isValidApiKeyFormat(key)) {
    throw new Error('Invalid API key format');
  }
  getStorage().setItem(API_KEY_STORAGE_KEY, key);
};

// Clear API key from storage
export const clearApiKey = (): void => {
  getStorage().removeItem(API_KEY_STORAGE_KEY);
};

// Check if using system key (for UI indication)
export const isUsingSystemKey = (): boolean => {
  return !!SYSTEM_KEY;
};

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Build URL query params from an object, filtering out null/undefined values
 */
function buildQueryString(params: Record<string, string | number | boolean | undefined | null>): string {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      query.set(key, String(value));
    }
  });
  return query.toString();
}

/**
 * Create a streaming reader with standardized error handling
 */
async function createStreamReader(
  endpoint: string,
  signal?: AbortSignal
): Promise<ReadableStreamDefaultReader<Uint8Array>> {
  const apiKey = getApiKey();

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    headers: {
      ...(apiKey && { 'X-API-Key': apiKey }),
    },
    signal,
  });

  if (!response.ok) {
    let errorMessage = response.statusText;
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || errorMessage;
    } catch {
      // Response might not be JSON
    }
    throw new ApiError(response.status, `Stream error: ${errorMessage}`);
  }

  if (!response.body) {
    throw new Error('No response body - streaming not supported by server');
  }

  return response.body.getReader();
}

async function fetchApi<T>(
  endpoint: string,
  options: RequestInit = {},
  signal?: AbortSignal
): Promise<T> {
  const apiKey = getApiKey();
  const url = `${API_BASE_URL}${endpoint}`;

  // Build headers - only include API key if user has set one
  // nginx injects default "admin" key for requests without X-API-Key
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(apiKey && { 'X-API-Key': apiKey }),
    ...options.headers,
  };

  const response = await fetch(url, {
    ...options,
    headers,
    // Support request cancellation via AbortController
    signal: signal || options.signal,
  });

  if (!response.ok) {
    let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
    try {
      const errorData = await response.json();
      if (errorData.detail) {
        errorMessage = errorData.detail;
      }
    } catch {
      // Ignore JSON parse errors
    }
    throw new ApiError(response.status, errorMessage);
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return {} as T;
  }

  return response.json();
}

// Types matching backend schemas
// Server types for model deployment - must match backend ServerType in schemas/model.py
export type ServerType =
  | 'vllm'       // LLMs - vLLM OpenAI-compatible server
  | 'audio'      // Generic audio models
  | 'whisper'    // Whisper ASR models
  | 'tts'        // Text-to-speech models
  | 'stt'        // Speech-to-text models
  | 'codec'      // Audio codec models
  | 'embedding'  // Embedding models
  | 'multimodal' // Vision-language models (LLaVA, Qwen-VL)
  | 'onnx'       // ONNX runtime models
  | 'triton'     // Triton inference server
  | 'generic'    // Generic FastAPI server
  | 'custom';    // Custom Dockerfile required

export interface Image {
  id: string;
  name: string;
  storage_path: string;
  repository?: string;
  company?: string;
  base_model?: string;
  parameter_count?: string;
  description?: string;
  tags: string[];
  metadata: Record<string, any>;
  requires_gpu: boolean;
  server_type?: ServerType;
  created_at: string;
  updated_at: string;
  version_count?: number;
  // Backward compatibility alias
  release_count?: number;
}

// Slim types for dropdowns - reduces data transfer
export interface ImageOption {
  id: string;
  name: string;
}

export interface VersionOption {
  id: string;
  version: string;
  tag: string;
  model_name?: string;
  // Backward compatibility
  image_name?: string;
}

// Backward compatibility alias
export type ReleaseOption = VersionOption;

export interface Version {
  id: string;
  image_id: string;
  model_name?: string;
  model_repository?: string;
  // Backward compatibility aliases
  model_id?: string;
  image_name?: string;
  image_repository?: string;
  version: string;
  tag: string;
  digest: string;
  quantization?: string;
  size_bytes?: number;
  platform: string;
  architecture: string;
  os: string;
  release_notes?: string;
  metadata: Record<string, any>;
  ceph_path?: string;
  mlflow_url?: string;
  status: string;
  created_at: string;
  is_release: boolean;
}

// Backward compatibility alias
export type Release = Version;

export interface MlflowModelVersionInfo {
  version?: string;
  current_stage?: string;
  status?: string;
  source?: string;
  run_id?: string;
}

export interface MlflowMetadata {
  resource_type: 'run' | 'experiment' | 'registered_model';
  url: string;
  fetched_at?: string;
  tags?: Record<string, string>;
  // Run fields
  run_id?: string;
  experiment_id?: string;
  run_name?: string;
  status?: string;
  start_time?: number;
  end_time?: number;
  artifact_uri?: string;
  params?: Record<string, string>;
  metrics?: Record<string, number>;
  // Experiment fields
  experiment_name?: string;
  artifact_location?: string;
  lifecycle_stage?: string;
  // Registered model fields
  model_name?: string;
  description?: string;
  creation_timestamp?: number;
  last_updated_timestamp?: number;
  latest_versions?: MlflowModelVersionInfo[];
  requested_version?: string;
}

export type DeploymentType = 'metadata' | 'local' | 'k8s';
export type DeploymentStatus = 'pending' | 'deploying' | 'running' | 'stopping' | 'stopped' | 'failed' | 'success';
export type HealthStatus = 'unknown' | 'healthy' | 'unhealthy';

export interface Deployment {
  id: string;
  release_id: string;
  release_version?: string;
  image_name?: string;
  environment: string;
  deployed_by?: string;
  deployed_at: string;
  terminated_at?: string;
  status: DeploymentStatus | string;
  metadata: Record<string, any>;
  // Execution fields
  container_id?: string;
  host_port?: number;
  deployment_type: DeploymentType;
  health_status: HealthStatus;
  started_at?: string;
  stopped_at?: string;
  gpu_enabled: boolean;
  endpoint_url?: string;
  image_tag?: string;
}

export interface ContainerStatus {
  running: boolean;
  healthy: boolean;
  exit_code?: number;
  started_at?: string;
  error?: string;
}

export interface DeploymentLogs {
  deployment_id: string;
  logs: string;
  truncated: boolean;
}

export interface ReleaseConfig {
  machine: string;
  port: number;
  template_name: string;
  model_name: string;
  model_path: string;
  gpu_ids: number[];
  tensor_parallel: number;
  max_model_len?: number | null;
  extra_config: Record<string, any>;
  server_type?: string;
  version?: string;
  endpoint_url?: string;
  gpu_count?: number;
}

export interface ReleaseConfigCreate {
  machine: string;
  port: number;
  template_name: string;
  model_name: string;
  model_path: string;
  gpu_ids: number[];
  tensor_parallel: number;
  max_model_len?: number | null;
  extra_config?: Record<string, string>;
}

export interface PRResponse {
  pr_url: string;
  status: string;
}

export interface ApiKey {
  id: string;
  name: string;
  key?: string; // Only present when creating
  is_active: boolean;
  created_at: string;
  last_used_at?: string;
  expires_at?: string;
}

export interface AuditLog {
  id: string;
  api_key_name?: string;
  action: string;
  resource_type: string;
  resource_id?: string;
  details: Record<string, any>;
  ip_address?: string;
  created_at: string;
}

export interface Artifact {
  id: string;
  release_id: string;
  model_id: string | null;
  name: string;
  artifact_type: string;
  file_path: string;
  size_bytes: number;
  checksum: string;
  checksum_type: string;
  platform: string;
  python_version: string | null;
  metadata: Record<string, any>;
  created_at: string;
  uploaded_by: string;
  image_name?: string;
  release_version?: string;
}

// Artifact Source types (for read-only filesystem browsing)
export interface ArtifactSource {
  id: string;
  name: string;
  description: string;
  path: string;
  available: boolean;
  readonly: boolean;
}

export interface ArtifactSourceFolder {
  name: string;
  path: string;
  item_count: number;
}

export interface ArtifactSourceFile {
  name: string;
  path: string;
  size_bytes: number;
  modified_at: number;
  file_type: string;
}

export interface ArtifactSourceBreadcrumb {
  name: string;
  path: string;
}

export interface ArtifactSourceBrowseResult {
  source_id: string;
  source_name: string;
  current_path: string;
  breadcrumbs: ArtifactSourceBreadcrumb[];
  folders: ArtifactSourceFolder[];
  files: ArtifactSourceFile[];
  readonly: boolean;
}

export interface ArtifactSourceFilesResult {
  source_id: string;
  source_name: string;
  files: ArtifactSourceFile[];
  total: number;
  readonly: boolean;
}

export interface DockerBuild {
  id: string;
  release_id: string;
  artifact_id: string;
  status: 'pending' | 'building' | 'success' | 'failed' | 'cancelled';
  image_tag: string;
  build_type: 'organic' | 'azure' | 'test' | 'optimized' | 'asr-vllm' | 'asr-allinone' | 'asr-azure-allinone';
  log_path?: string;
  created_at: string;
  completed_at?: string;
  error_message?: string;
  // GC tracking fields
  superseded_at?: string;
  cleaned_at?: string;
  cleanup_scheduled_at?: string;
  days_until_cleanup?: number;
  is_current: boolean;
  is_cleaned: boolean;
}

export interface DockerDiskUsage {
  images: { count: number; size_bytes: number; size_human: string };
  build_cache: { count: number; size_bytes: number; size_human: string };
  containers: { count: number; size_bytes: number; size_human: string };
  volumes: { count: number; size_bytes: number; size_human: string };
  total_docker_bytes: number;
  total_docker_human: string;
  disk_available_bytes: number;
  disk_available_human: string;
  disk_total_bytes: number;
  disk_total_human: string;
}

export type BenchmarkStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface BenchmarkStageResult {
  stage: string;
  success: boolean;
  model_id?: string;
  model_type?: string;
  message?: string;
  endpoint?: string;
  ttft_avg_ms?: number;
  tokens_per_second_avg?: number;
  total_tokens?: number;
  requests_per_second?: number;
  successful?: number;
  failed?: number;
}

export interface Benchmark {
  id: string;
  deployment_id?: string;
  endpoint_url?: string;
  production_endpoint_id?: number;
  endpoint_path: string;
  method: string;
  concurrent_requests: number;
  total_requests: number;
  timeout_seconds: number;
  status: BenchmarkStatus;
  error_message?: string;
  // Progress tracking
  current_stage?: string;
  stage_progress?: string;  // e.g., "3/5"
  stages_completed: BenchmarkStageResult[];
  // Latency metrics (ms)
  latency_avg_ms?: number;
  latency_min_ms?: number;
  latency_max_ms?: number;
  latency_p50_ms?: number;
  latency_p90_ms?: number;
  latency_p95_ms?: number;
  latency_p99_ms?: number;
  // Throughput metrics
  requests_per_second?: number;
  total_requests_sent?: number;
  successful_requests?: number;
  failed_requests?: number;
  error_rate?: number;
  // Timing
  duration_seconds?: number;
  started_at?: string;
  completed_at?: string;
  created_at: string;
  metadata: Record<string, any>;
}

export interface BenchmarkSummary {
  has_data: boolean;
  last_run_at?: string;
  status?: string;
  model_id?: string;
  model_type?: string;  // text, audio, multimodal
  benchmark_endpoint?: string;  // The API endpoint that was benchmarked
  latency_avg_ms?: number;
  latency_p50_ms?: number;
  latency_p95_ms?: number;
  latency_p99_ms?: number;
  requests_per_second?: number;
  total_requests?: number;
  error_rate?: number;
  // Inference metrics (TTFT/TPS)
  ttft_avg_ms?: number;
  ttft_p50_ms?: number;
  ttft_p95_ms?: number;
  tokens_per_second_avg?: number;
  total_tokens_generated?: number;
}

export interface ApiEndpoint {
  method: string;
  path: string;
  summary: string;
  description: string;
  tags: string[];
  status?: number;  // HTTP status from probing
  response?: Record<string, any> | null;  // Actual API response data
  requires_file_upload?: boolean;  // True if endpoint requires file upload (multipart/form-data)
}

export interface ApiDiscoveryResult {
  api_type: 'openai' | 'audio' | 'fastapi' | 'generic' | 'unknown';
  openapi_spec: Record<string, any> | null;
  endpoints: ApiEndpoint[];
  detected_endpoints: string[];
  recommended_benchmark_endpoint: string | null;
}

export interface ApiSpec {
  api_type: 'openai' | 'fastapi' | 'audio' | 'generic' | 'unknown';
  openapi_spec?: Record<string, any>;
  endpoints: ApiEndpoint[];
  detected_endpoints: string[];
  error?: string;
}

export interface BenchmarkCreate {
  deployment_id?: string;
  endpoint_url?: string;  // For production endpoints
  production_endpoint_id?: number;  // EID from production endpoints
  endpoint_path?: string;
  method?: string;
  concurrent_requests?: number;
  total_requests?: number;
  timeout_seconds?: number;
  request_body?: Record<string, any>;
  headers?: Record<string, string>;
}

// Evaluation types (separate from benchmarks)
export interface Evaluation {
  id: string;
  deployment_id?: string;
  production_endpoint_id?: number;
  endpoint_url?: string;
  evaluation_type: string;
  evaluator_name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  error_message?: string;
  current_stage?: string;
  stage_progress?: string;
  primary_metric?: number;
  primary_metric_name?: string;
  secondary_metric?: number;
  secondary_metric_name?: string;
  wer?: number;
  cer?: number;
  samples_total?: number;
  samples_evaluated?: number;
  samples_with_errors?: number;
  no_speech_count?: number;
  dataset_path?: string;
  dataset_name?: string;
  config: Record<string, any>;
  results: Record<string, any>;
  duration_seconds?: number;
  started_at?: string;
  completed_at?: string;
  created_at: string;
}

export interface EvaluationSummary {
  has_data: boolean;
  evaluation_type?: string;
  evaluator_name?: string;
  status?: string;
  primary_metric?: number;
  primary_metric_name?: string;
  secondary_metric?: number;
  secondary_metric_name?: string;
  wer?: number;
  cer?: number;
  samples_evaluated?: number;
  no_speech_count?: number;
  dataset_path?: string;
}

export interface EvaluationCreate {
  endpoint_url: string;
  model_name: string;
  model_type?: string;
  deployment_id?: string;
  production_endpoint_id?: number;
  dataset_path?: string;
  limit?: number;
  language?: string;
}

// Pagination
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

// API Client
export const api = {
  // Images
  async listImages(params?: { search?: string; page?: number; size?: number; signal?: AbortSignal }): Promise<PaginatedResponse<Image>> {
    const query = buildQueryString({ search: params?.search, page: params?.page, size: params?.size });
    return fetchApi<PaginatedResponse<Image>>(`/v1/models${query ? `?${query}` : ''}`, {}, params?.signal);
  },

  async getImage(id: string, signal?: AbortSignal): Promise<Image> {
    return fetchApi<Image>(`/v1/models/${id}`, {}, signal);
  },

  async listImageOptions(signal?: AbortSignal): Promise<ImageOption[]> {
    return fetchApi<ImageOption[]>('/v1/models/options', {}, signal);
  },

  async createImage(data: {
    name: string;
    storage_path: string;
    company?: string;
    base_model?: string;
    parameter_count?: string;
    description?: string;
    tags?: string[];
    metadata?: Record<string, any>;
  }): Promise<Image> {
    return fetchApi<Image>('/v1/models', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  async updateImage(id: string, data: {
    storage_path?: string;
    repository?: string;
    company?: string;
    base_model?: string;
    parameter_count?: string;
    description?: string;
    tags?: string[];
    metadata?: Record<string, any>;
  }): Promise<Image> {
    return fetchApi<Image>(`/v1/models/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  async deleteImage(id: string): Promise<void> {
    return fetchApi<void>(`/v1/models/${id}`, {
      method: 'DELETE',
    });
  },

  async getModelVersions(modelId: string, params?: { page?: number; size?: number }): Promise<PaginatedResponse<Version>> {
    const query = buildQueryString({ page: params?.page, size: params?.size });
    return fetchApi<PaginatedResponse<Version>>(`/v1/models/${modelId}/versions${query ? `?${query}` : ''}`);
  },

  // Backward compatibility alias
  async getImageReleases(imageId: string, params?: { page?: number; size?: number }): Promise<PaginatedResponse<Version>> {
    return this.getModelVersions(imageId, params);
  },

  // Versions (formerly Releases)
  async listVersions(params?: {
    model_name?: string;
    version?: string;
    environment?: string;
    is_release?: boolean;
    status?: string;
    page?: number;
    size?: number;
    signal?: AbortSignal;
  }): Promise<PaginatedResponse<Version>> {
    const query = buildQueryString({
      model_name: params?.model_name,
      version: params?.version,
      environment: params?.environment,
      is_release: params?.is_release,
      status: params?.status,
      page: params?.page,
      size: params?.size,
    });
    return fetchApi<PaginatedResponse<Version>>(`/v1/versions${query ? `?${query}` : ''}`, {}, params?.signal);
  },

  // Backward compatibility alias
  async listReleases(params?: {
    model_name?: string;
    version?: string;
    environment?: string;
    is_release?: boolean;
    status?: string;
    page?: number;
    size?: number;
    signal?: AbortSignal;
  }): Promise<PaginatedResponse<Version>> {
    return this.listVersions(params);
  },

  async getVersion(id: string, signal?: AbortSignal): Promise<Version> {
    return fetchApi<Version>(`/v1/versions/${id}`, {}, signal);
  },

  // Backward compatibility alias
  async getRelease(id: string, signal?: AbortSignal): Promise<Version> {
    return this.getVersion(id, signal);
  },

  async listVersionOptions(signal?: AbortSignal): Promise<VersionOption[]> {
    return fetchApi<VersionOption[]>('/v1/versions/options', {}, signal);
  },

  // Backward compatibility alias
  async listReleaseOptions(signal?: AbortSignal): Promise<VersionOption[]> {
    return this.listVersionOptions(signal);
  },

  async getLatestVersion(modelName: string, environment?: string): Promise<Version | null> {
    const queryParams = new URLSearchParams({ model_name: modelName });
    if (environment) queryParams.set('environment', environment);

    try {
      return await fetchApi<Version>(`/v1/versions/latest?${queryParams.toString()}`);
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        return null;
      }
      throw error;
    }
  },

  // Backward compatibility alias
  async getLatestRelease(imageName: string, environment?: string): Promise<Version | null> {
    return this.getLatestVersion(imageName, environment);
  },

  async createVersion(data: {
    model_id: string;
    version: string;
    tag: string;
    digest: string;
    size_bytes?: number;
    platform?: string;
    architecture?: string;
    os?: string;
    metadata?: Record<string, any>;
    ceph_path?: string;
    mlflow_url?: string;
    auto_build?: boolean;
    build_config?: Record<string, any>;
  }): Promise<Version> {
    // Parse platform to get os and architecture if not provided
    let os = data.os;
    let architecture = data.architecture;

    if (data.platform && (!os || !architecture)) {
      const parts = data.platform.split('/');
      if (parts.length === 2) {
        if (!os) os = parts[0];
        if (!architecture) architecture = parts[1];
      }
    }

    const payload = {
      ...data,
      os: os || 'linux', // Default to linux if not found
      architecture: architecture || 'amd64', // Default to amd64 if not found
    };

    return fetchApi<Version>('/v1/versions', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  // Backward compatibility alias
  async createRelease(data: {
    image_id: string;
    version: string;
    tag: string;
    digest: string;
    size_bytes?: number;
    platform?: string;
    architecture?: string;
    os?: string;
    metadata?: Record<string, any>;
    ceph_path?: string;
    mlflow_url?: string;
    auto_build?: boolean;
    build_config?: Record<string, any>;
  }): Promise<Version> {
    return this.createVersion({ ...data, model_id: data.image_id });
  },

  async updateVersion(
    id: string,
    data: {
      metadata?: Record<string, any>;
      status?: string;
      ceph_path?: string;
      is_release?: boolean;
    }
  ): Promise<Version> {
    return fetchApi<Version>(`/v1/versions/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  // Backward compatibility alias
  async updateRelease(
    id: string,
    data: {
      metadata?: Record<string, any>;
      status?: string;
      ceph_path?: string;
    }
  ): Promise<Version> {
    return this.updateVersion(id, data);
  },

  async deleteVersion(id: string): Promise<void> {
    return fetchApi<void>(`/v1/versions/${id}`, {
      method: 'DELETE',
    });
  },

  // Backward compatibility alias
  async deleteRelease(id: string): Promise<void> {
    return this.deleteVersion(id);
  },

  async promoteVersion(id: string, isRelease: boolean): Promise<Version> {
    return fetchApi<Version>(`/v1/versions/${id}`, {
      method: 'PUT',
      body: JSON.stringify({ is_release: isRelease }),
    });
  },

  // Backward compatibility alias
  async promoteRelease(id: string, isRelease: boolean): Promise<Version> {
    return this.promoteVersion(id, isRelease);
  },

  async getVersionDeployments(versionId: string): Promise<Deployment[]> {
    return fetchApi<Deployment[]>(`/v1/versions/${versionId}/deployments`);
  },

  // Backward compatibility alias
  async getReleaseDeployments(releaseId: string): Promise<Deployment[]> {
    return this.getVersionDeployments(releaseId);
  },

  // MLflow Metadata
  async getMlflowMetadata(versionId: string, signal?: AbortSignal): Promise<MlflowMetadata> {
    return fetchApi<MlflowMetadata>(`/v1/versions/${versionId}/mlflow-metadata`, {}, signal);
  },

  async syncMlflowMetadata(versionId: string): Promise<MlflowMetadata> {
    return fetchApi<MlflowMetadata>(`/v1/versions/${versionId}/mlflow-metadata/sync`, {
      method: 'POST',
    });
  },

  // Deployments
  async listDeployments(params?: {
    environment?: string;
    status?: string;
    release_id?: string;
    page?: number;
    size?: number;
    signal?: AbortSignal;
  }): Promise<PaginatedResponse<Deployment>> {
    const query = buildQueryString({
      environment: params?.environment,
      status: params?.status,
      release_id: params?.release_id,
      page: params?.page,
      size: params?.size,
    });
    return fetchApi<PaginatedResponse<Deployment>>(`/v1/deployments${query ? `?${query}` : ''}`, {}, params?.signal);
  },

  async listReleaseConfigTemplates(signal?: AbortSignal): Promise<string[]> {
    return fetchApi<string[]>('/v1/release-configs/templates', {}, signal);
  },

  async getReleaseConfigTemplateVariables(templateName: string, signal?: AbortSignal): Promise<string[]> {
    return fetchApi<string[]>(`/v1/release-configs/templates/${templateName}/variables`, {}, signal);
  },

  async listAvailableBackends(signal?: AbortSignal): Promise<string[]> {
    return fetchApi<string[]>('/v1/release-configs/backends', {}, signal);
  },

  async proposeReleaseConfig(config: ReleaseConfigCreate, description?: string): Promise<PRResponse> {
    const query = buildQueryString({ description });
    return fetchApi<PRResponse>(`/v1/release-configs/propose${query ? `?${query}` : ''}`, {
      method: 'POST',
      body: JSON.stringify(config),
    });
  },

  async getDeployment(id: string, signal?: AbortSignal): Promise<Deployment> {
    return fetchApi<Deployment>(`/v1/deployments/${id}`, {}, signal);
  },

  async createDeployment(data: {
    release_id: string;
    environment: string;
    metadata?: Record<string, any>;
    status?: string;
  }): Promise<Deployment> {
    return fetchApi<Deployment>('/v1/deployments', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  // Deployment Execution
  async executeDeployment(data: {
    release_id: string;
    environment: string;
    deployment_type: DeploymentType;
    docker_build_id?: string;  // Specific build to deploy
    image_tag?: string;        // Or specific image tag
    gpu_enabled?: boolean;
    environment_vars?: Record<string, string>;
    volume_mounts?: Record<string, string>;
    memory_limit?: string;
    cpu_limit?: number;
    metadata?: Record<string, any>;
  }): Promise<Deployment> {
    return fetchApi<Deployment>('/v1/deployments/execute', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  async startDeployment(id: string): Promise<Deployment> {
    return fetchApi<Deployment>(`/v1/deployments/${id}/start`, {
      method: 'POST',
    });
  },

  async stopDeployment(id: string): Promise<Deployment> {
    return fetchApi<Deployment>(`/v1/deployments/${id}/stop`, {
      method: 'POST',
    });
  },

  async restartDeployment(id: string): Promise<Deployment> {
    return fetchApi<Deployment>(`/v1/deployments/${id}/restart`, {
      method: 'POST',
    });
  },

  async getDeploymentStatus(id: string): Promise<ContainerStatus> {
    return fetchApi<ContainerStatus>(`/v1/deployments/${id}/status`);
  },

  async getDeploymentLogs(id: string, tail?: number): Promise<DeploymentLogs> {
    const queryParams = new URLSearchParams();
    if (tail) queryParams.set('tail', String(tail));
    const query = queryParams.toString();
    return fetchApi<DeploymentLogs>(`/v1/deployments/${id}/logs${query ? `?${query}` : ''}`);
  },

  async getDeploymentApiSpec(id: string): Promise<ApiSpec> {
    return fetchApi<ApiSpec>(`/v1/deployments/${id}/api-spec`);
  },

  async getAvailableBuildsForDeployment(releaseId: string): Promise<{
    has_builds: boolean;
    builds: Array<{
      id: string;
      image_tag: string;
      server_type: string;
      build_type: string;
      created_at: string | null;
      completed_at: string | null;
      is_current: boolean;
    }>;
    model_server_type: string | null;
  }> {
    // Use existing listDockerBuilds API and filter for successful, non-cleaned builds
    const response = await this.listDockerBuilds({ release_id: releaseId, size: 50 });
    const successfulBuilds = response.items
      .filter(b => b.status === 'success' && !b.is_cleaned)
      .map(b => ({
        id: b.id,
        image_tag: b.image_tag,
        server_type: b.build_type, // Use build_type as server_type for now
        build_type: b.build_type,
        created_at: b.created_at,
        completed_at: b.completed_at || null,
        is_current: b.is_current,
      }));

    return {
      has_builds: successfulBuilds.length > 0,
      builds: successfulBuilds,
      model_server_type: null, // Not available from this endpoint
    };
  },

  async streamDeploymentLogs(
    id: string,
    signal?: AbortSignal
  ): Promise<ReadableStreamDefaultReader<Uint8Array>> {
    return createStreamReader(`/v1/deployments/${id}/logs/stream`, signal);
  },


  // Audit Logs
  async listAuditLogs(params?: {
    action?: string;
    resource_type?: string;
    api_key_name?: string;
    limit?: number;
    offset?: number;
  }): Promise<AuditLog[]> {
    const query = buildQueryString({
      action: params?.action,
      resource_type: params?.resource_type,
      api_key_name: params?.api_key_name,
      limit: params?.limit,
      offset: params?.offset,
    });
    return fetchApi<AuditLog[]>(`/v1/audit-logs${query ? `?${query}` : ''}`);
  },


  // Artifacts
  async listArtifacts(params?: {
    release_id?: string;
    artifact_type?: string;
    platform?: string;
    page?: number;
    size?: number;
  }): Promise<PaginatedResponse<Artifact>> {
    const query = buildQueryString({
      release_id: params?.release_id,
      artifact_type: params?.artifact_type,
      platform: params?.platform,
      page: params?.page,
      size: params?.size,
    });
    return fetchApi<PaginatedResponse<Artifact>>(`/v1/artifacts${query ? `?${query}` : ''}`);
  },

  async getArtifact(id: string): Promise<Artifact> {
    return fetchApi<Artifact>(`/v1/artifacts/${id}`);
  },

  async uploadArtifact(formData: FormData): Promise<Artifact> {
    const apiKey = getApiKey();

    const response = await fetch(`${API_BASE_URL}/v1/artifacts/upload`, {
      method: 'POST',
      headers: {
        ...(apiKey && { 'X-API-Key': apiKey }),
      },
      body: formData,
    });

    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          errorMessage = errorData.detail;
        }
      } catch {
        // Ignore JSON parse errors
      }
      throw new ApiError(response.status, errorMessage);
    }

    return response.json();
  },

  async registerArtifact(data: {
    release_id?: string;
    model_id?: string;
    name?: string;
    artifact_type?: string;
    file_path: string;
    platform?: string;
    python_version?: string;
    metadata?: Record<string, any>;
  }): Promise<Artifact> {
    return fetchApi<Artifact>('/v1/artifacts/register', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  // Docker Builds
  async createDockerBuild(data: {
    release_id: string;
    artifact_id?: string;
    artifact_ids?: string[];
    image_tag: string;
    build_type: 'organic' | 'azure' | 'test' | 'optimized' | 'asr-vllm' | 'asr-allinone' | 'asr-azure-allinone';
    dockerfile_content?: string;
  }): Promise<DockerBuild> {
    return fetchApi<DockerBuild>('/v1/docker/builds', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  async getDockerfileTemplate(type: string, releaseId?: string): Promise<{ content: string; template_type?: string }> {
    const params = new URLSearchParams();
    if (releaseId) params.set('release_id', releaseId);
    const query = params.toString();
    return fetchApi<{ content: string; template_type?: string }>(`/v1/docker/templates/${type}${query ? `?${query}` : ''}`);
  },

  async listDockerBuilds(params?: { release_id?: string; page?: number; size?: number }): Promise<PaginatedResponse<DockerBuild>> {
    const queryParams = new URLSearchParams();
    if (params?.release_id) queryParams.append('release_id', params.release_id);
    if (params?.page) queryParams.append('page', String(params.page));
    if (params?.size) queryParams.append('size', String(params.size));
    return fetchApi<PaginatedResponse<DockerBuild>>(`/v1/docker/builds?${queryParams.toString()}`);
  },

  async getDockerBuild(buildId: string): Promise<DockerBuild> {
    return fetchApi<DockerBuild>(`/v1/docker/builds/${buildId}`);
  },

  async getDockerBuildLogs(buildId: string): Promise<{ logs: string }> {
    return fetchApi<{ logs: string }>(`/v1/docker/builds/${buildId}/logs`);
  },

  async streamBuildLogs(
    buildId: string,
    signal?: AbortSignal
  ): Promise<ReadableStreamDefaultReader<Uint8Array>> {
    return createStreamReader(`/v1/docker/builds/${buildId}/logs/stream`, signal);
  },

  async getDockerDiskUsage(): Promise<DockerDiskUsage> {
    return fetchApi<DockerDiskUsage>('/v1/docker/disk-usage');
  },

  async cancelDockerBuild(buildId: string): Promise<DockerBuild> {
    return fetchApi<DockerBuild>(`/v1/docker/builds/${buildId}/cancel`, {
      method: 'POST',
    });
  },

  async downloadArtifact(id: string): Promise<Blob> {
    const apiKey = getApiKey();

    const response = await fetch(`${API_BASE_URL}/v1/artifacts/${id}/download`, {
      headers: {
        ...(apiKey && { 'X-API-Key': apiKey }),
      },
    });

    if (!response.ok) {
      throw new ApiError(response.status, response.statusText);
    }

    return response.blob();
  },

  async deleteArtifact(id: string): Promise<void> {
    return fetchApi<void>(`/v1/artifacts/${id}`, {
      method: 'DELETE',
    });
  },

  // Artifact Sources (Read-only filesystem browsing)
  async listArtifactSources(): Promise<ArtifactSource[]> {
    return fetchApi<ArtifactSource[]>('/v1/artifacts/sources/list');
  },

  async browseArtifactSource(sourceId: string, path: string = ''): Promise<ArtifactSourceBrowseResult> {
    const query = buildQueryString({ path });
    return fetchApi<ArtifactSourceBrowseResult>(`/v1/artifacts/sources/${sourceId}/browse${query ? `?${query}` : ''}`);
  },

  async downloadFromArtifactSource(sourceId: string, path: string): Promise<Blob> {
    const apiKey = getApiKey();
    const query = buildQueryString({ path });

    const response = await fetch(`${API_BASE_URL}/v1/artifacts/sources/${sourceId}/download?${query}`, {
      headers: {
        ...(apiKey && { 'X-API-Key': apiKey }),
      },
    });

    if (!response.ok) {
      throw new ApiError(response.status, response.statusText);
    }

    return response.blob();
  },

  async listArtifactSourceFiles(sourceId: string): Promise<ArtifactSourceFilesResult> {
    return fetchApi<ArtifactSourceFilesResult>(`/v1/artifacts/sources/${sourceId}/files`);
  },

  // System
  async getSystemStorage(): Promise<{ total: number; used: number; free: number }> {
    return fetchApi<{ total: number; used: number; free: number }>('/v1/system/storage');
  },

  async getGpuInfo(): Promise<{ available: boolean; count: number }> {
    return fetchApi<{ available: boolean; count: number }>('/v1/system/gpu');
  },

  async listSystemFiles(path: string = '/'): Promise<{ name: string; is_directory: boolean; size_bytes: number; modified_at: string }[]> {
    const params = new URLSearchParams({ path });
    return fetchApi(`/v1/system/files?${params.toString()}`);
  },

  // Benchmarks
  async runBenchmark(data: BenchmarkCreate): Promise<Benchmark> {
    return fetchApi<Benchmark>('/v1/benchmarks', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  async runBenchmarkAsync(data: BenchmarkCreate): Promise<Benchmark> {
    return fetchApi<Benchmark>('/v1/benchmarks/async', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  async listBenchmarks(params?: { limit?: number; status?: string }): Promise<Benchmark[]> {
    const searchParams = new URLSearchParams();
    if (params?.limit) searchParams.set('limit', params.limit.toString());
    if (params?.status) searchParams.set('status', params.status);
    const query = searchParams.toString();
    return fetchApi<Benchmark[]>(`/v1/benchmarks${query ? `?${query}` : ''}`);
  },

  async getBenchmark(id: string): Promise<Benchmark> {
    return fetchApi<Benchmark>(`/v1/benchmarks/${id}`);
  },

  async cancelBenchmark(id: string): Promise<Benchmark> {
    return fetchApi<Benchmark>(`/v1/benchmarks/${id}/cancel`, {
      method: 'POST',
    });
  },

  async getDeploymentBenchmarks(deploymentId: string, limit: number = 10): Promise<Benchmark[]> {
    return fetchApi<Benchmark[]>(`/v1/benchmarks/deployment/${deploymentId}?limit=${limit}`);
  },

  async getDeploymentBenchmarkSummary(deploymentId: string): Promise<BenchmarkSummary> {
    return fetchApi<BenchmarkSummary>(`/v1/benchmarks/deployment/${deploymentId}/summary`);
  },


  // Evaluations (WER/CER quality metrics - separate from benchmarks)
  async runEvaluation(data: EvaluationCreate): Promise<Evaluation> {
    return fetchApi<Evaluation>('/v1/evaluations', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  async getEvaluation(id: string): Promise<Evaluation> {
    return fetchApi<Evaluation>(`/v1/evaluations/${id}`);
  },

  async cancelEvaluation(id: string): Promise<Evaluation> {
    return fetchApi<Evaluation>(`/v1/evaluations/${id}/cancel`, {
      method: 'POST',
    });
  },

  async getDeploymentEvaluations(deploymentId: string, limit: number = 10): Promise<Evaluation[]> {
    return fetchApi<Evaluation[]>(`/v1/evaluations/deployment/${deploymentId}?limit=${limit}`);
  },

  async getDeploymentEvaluationSummary(deploymentId: string): Promise<EvaluationSummary> {
    return fetchApi<EvaluationSummary>(`/v1/evaluations/deployment/${deploymentId}/summary`);
  },


  // Health Check
  async health(): Promise<{ status: string; components: Record<string, any> }> {
    return fetchApi('/health');
  },

  // API Keys
  async getApiKeys(signal?: AbortSignal): Promise<ApiKey[]> {
    return fetchApi<ApiKey[]>('/v1/api-keys', {}, signal);
  },

  async createApiKey(data: { name: string; role?: string; expires_at?: string }): Promise<ApiKey> {
    return fetchApi<ApiKey>('/v1/api-keys', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  async deleteApiKey(id: string): Promise<void> {
    return fetchApi<void>(`/v1/api-keys/${id}`, {
      method: 'DELETE',
    });
  },

};

export { ApiError };
