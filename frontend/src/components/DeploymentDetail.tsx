import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  ArrowLeft, Play, Square, RotateCw, ExternalLink, Copy,
  Server, Cpu, Activity, CheckCircle, XCircle, AlertCircle,
  Loader2, Globe, Gauge, Terminal, FileJson, Package
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { api } from '../lib/api';
import {
  deploymentStatusColors,
  healthStatusColors,
  formatDate,
  canStartDeployment,
  canStopDeployment,
  canRestartDeployment,
  getEndpointUrl,
  needsStatusPolling,
} from '../lib/deployment-utils';
import {
  BenchmarkStatsCard,
  DiscoveredAPIsSection,
  ASREvaluationCard,
  getRequestBodyForEndpoint,
  copyToClipboard,
} from './shared';

export function DeploymentDetail() {
  const { deploymentId } = useParams<{ deploymentId: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [selectedBenchmarkEndpoint, setSelectedBenchmarkEndpoint] = useState<string | null>(null);

  // Benchmark state (fully separate from evaluation)
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);
  const [activeBenchmarkId, setActiveBenchmarkId] = useState<string | null>(null);

  // Evaluation state (fully separate from benchmark)
  const [evaluationRunning, setEvaluationRunning] = useState(false);
  const [activeEvaluationId, setActiveEvaluationId] = useState<string | null>(null);

  // ASR evaluation options
  const [asrEvalLimit, setAsrEvalLimit] = useState(500);  // Default to 500 samples
  const logsContainerRef = useRef<HTMLPreElement>(null);
  const userScrolledUp = useRef(false);

  const { data: deployment, isLoading, isError, refetch } = useQuery({
    queryKey: ['deployment', deploymentId],
    queryFn: () => api.getDeployment(deploymentId!),
    enabled: !!deploymentId,
    staleTime: 5000, // Cache for 5s to reduce duplicate requests on page load
    retry: 2,
    refetchInterval: (query) => {
      // Auto-refresh while status could change (running or transitional)
      const data = query.state.data;
      if (data && needsStatusPolling(data.status)) {
        return 3000;
      }
      return false;
    },
  });

  // Benchmark summary query (for latency/throughput data)
  const { data: benchmarkSummary, refetch: refetchBenchmark } = useQuery({
    queryKey: ['benchmarkSummary', deploymentId],
    queryFn: () => api.getDeploymentBenchmarkSummary(deploymentId!),
    enabled: !!deploymentId && !benchmarkRunning,
    staleTime: 60000,
    retry: 1,
  });

  // Evaluation summary query (for WER/CER data - separate from benchmarks)
  const { data: evaluationSummary, refetch: refetchEvaluation } = useQuery({
    queryKey: ['evaluationSummary', deploymentId],
    queryFn: () => api.getDeploymentEvaluationSummary(deploymentId!),
    enabled: !!deploymentId && !evaluationRunning,
    staleTime: 60000,
    retry: 1,
  });

  // Check for running benchmarks/evaluations on page load (one-time check)
  const [hasRestoredState, setHasRestoredState] = useState(false);
  const { data: recentBenchmarks } = useQuery({
    queryKey: ['recentBenchmarks', deploymentId],
    queryFn: () => api.getDeploymentBenchmarks(deploymentId!, 5),
    enabled: !!deploymentId && !hasRestoredState, // Only fetch once on load
    staleTime: Infinity, // Never refetch automatically
    retry: 1,
  });

  // Restore running benchmark state (one-time on load)
  useEffect(() => {
    if (recentBenchmarks && !hasRestoredState) {
      setHasRestoredState(true);
      const runningBenchmarks = recentBenchmarks.filter(b => b.status === 'running');
      for (const job of runningBenchmarks) {
        if (!benchmarkRunning) {
          setBenchmarkRunning(true);
          setActiveBenchmarkId(job.id);
          break; // Only restore first running benchmark
        }
      }
    }
  }, [recentBenchmarks, hasRestoredState, benchmarkRunning, evaluationRunning]);

  // Active benchmark progress query - polls while benchmark is running
  const { data: activeBenchmark } = useQuery({
    queryKey: ['benchmark', activeBenchmarkId],
    queryFn: () => api.getBenchmark(activeBenchmarkId!),
    enabled: !!activeBenchmarkId && benchmarkRunning,
    refetchInterval: 3000, // Poll every 3s for progress
    retry: 1,
    staleTime: 2000,
  });

  // Active evaluation progress query - polls while evaluation is running
  const { data: activeEvaluation } = useQuery({
    queryKey: ['evaluation', activeEvaluationId],
    queryFn: () => api.getEvaluation(activeEvaluationId!),
    enabled: !!activeEvaluationId && evaluationRunning,
    refetchInterval: 3000, // Poll every 3s for progress
    retry: 1,
    staleTime: 2000,
  });

  // Cancel benchmark mutation
  const cancelBenchmarkMutation = useMutation({
    mutationFn: (id: string) => api.cancelBenchmark(id),
    onSuccess: () => {
      setBenchmarkRunning(false);
      setActiveBenchmarkId(null);
      refetchBenchmark();
    },
  });

  // Cancel evaluation mutation
  const cancelEvaluationMutation = useMutation({
    mutationFn: (id: string) => api.cancelEvaluation(id),
    onSuccess: () => {
      setEvaluationRunning(false);
      setActiveEvaluationId(null);
      refetchEvaluation();
    },
  });

  // Detect benchmark completion
  useEffect(() => {
    if (activeBenchmark && benchmarkRunning) {
      if (activeBenchmark.status === 'completed' || activeBenchmark.status === 'failed' || activeBenchmark.status === 'cancelled') {
        setBenchmarkRunning(false);
        setActiveBenchmarkId(null);
        refetchBenchmark();
      }
    }
  }, [activeBenchmark?.status, benchmarkRunning, refetchBenchmark]);

  // Detect evaluation completion
  useEffect(() => {
    if (activeEvaluation && evaluationRunning) {
      if (activeEvaluation.status === 'completed' || activeEvaluation.status === 'failed') {
        setEvaluationRunning(false);
        setActiveEvaluationId(null);
        refetchEvaluation();
      }
    }
  }, [activeEvaluation?.status, evaluationRunning, refetchEvaluation]);

  // Timeout: stop benchmark polling after 5 minutes
  useEffect(() => {
    if (!benchmarkRunning) return;
    const timeout = setTimeout(() => {
      setBenchmarkRunning(false);
      setActiveBenchmarkId(null);
    }, 300000); // 5 minutes
    return () => clearTimeout(timeout);
  }, [benchmarkRunning]);

  // Timeout: stop evaluation polling after 10 minutes (evaluations can take longer)
  useEffect(() => {
    if (!evaluationRunning) return;
    const timeout = setTimeout(() => {
      setEvaluationRunning(false);
      setActiveEvaluationId(null);
    }, 600000); // 10 minutes
    return () => clearTimeout(timeout);
  }, [evaluationRunning]);

  // API spec discovery - auto-fetch when deployment is ready
  // Local deployments: wait for health check to pass
  // External deployments: trigger when running (no health check available)
  const isLocalDeployment = deployment?.deployment_type === 'local';
  const isReadyForApiDiscovery = deployment?.status === 'running' &&
    (!isLocalDeployment || deployment?.health_status === 'healthy');

  const { data: apiSpec, isLoading: apiSpecLoading, refetch: refetchApiSpec } = useQuery({
    queryKey: ['apiSpec', deploymentId, isReadyForApiDiscovery],
    queryFn: () => api.getDeploymentApiSpec(deploymentId!),
    enabled: !!deploymentId && isReadyForApiDiscovery,
    staleTime: 30000, // Cache for 30 seconds
    retry: 1, // Limit retries - API discovery is best-effort
    refetchInterval: isReadyForApiDiscovery ? 15000 : false, // Refresh every 15s when ready (reduced from 10s)
  });

  // Auto-select first valid benchmark endpoint when API spec loads
  useEffect(() => {
    if (apiSpec?.endpoints && !selectedBenchmarkEndpoint) {
      // Filter to POST endpoints that are benchmarkable (not health/metrics/batch)
      // Note: File upload endpoints like /transcribe are now supported by the benchmarker
      // Batch endpoints are excluded due to compatibility issues
      const benchmarkableEndpoints = apiSpec.endpoints.filter(ep =>
        ep.method === 'POST' &&
        !ep.path.includes('/health') &&
        !ep.path.includes('/metrics') &&
        !ep.path.includes('/batch')
      );
      // Prefer recommended endpoint, otherwise first benchmarkable one
      const defaultEndpoint = apiSpec.recommended_benchmark_endpoint ||
        benchmarkableEndpoints[0]?.path;
      if (defaultEndpoint) {
        setSelectedBenchmarkEndpoint(defaultEndpoint);
      }
    }
  }, [apiSpec, selectedBenchmarkEndpoint]);

  // Container logs query with auto-refresh when running
  // Only fetch logs for local deployments with a container
  const isDeploymentRunning = deployment?.status === 'running';
  const hasContainer = !!deployment?.container_id && deployment?.deployment_type === 'local';
  const { data: logsData, isLoading: logsLoading, refetch: refetchLogs } = useQuery({
    queryKey: ['deploymentLogs', deploymentId],
    queryFn: () => api.getDeploymentLogs(deploymentId!, 200),
    enabled: !!deploymentId && hasContainer,
    staleTime: 2000, // Cache briefly to prevent duplicate requests
    refetchInterval: isDeploymentRunning && hasContainer ? 5000 : false, // Reduced from 3s to 5s
    retry: false, // Don't retry on 404
  });

  // Benchmark mutation - runs inference benchmark with TTFT/TPS (no ASR evaluation)
  const runBenchmarkMutation = useMutation({
    mutationFn: () => {
      const endpointPath = selectedBenchmarkEndpoint || undefined;
      const requestBody = endpointPath ? getRequestBodyForEndpoint(endpointPath, deployment?.image_name || 'model') : undefined;
      return api.runBenchmarkAsync({
        deployment_id: deploymentId!,
        endpoint_path: endpointPath,
        request_body: requestBody,
        method: 'POST',
        concurrent_requests: 5,
        total_requests: 20,
      });
    },
    onMutate: () => {
      setBenchmarkRunning(true);
    },
    onSuccess: (benchmark) => {
      setActiveBenchmarkId(benchmark.id);
    },
    onError: () => {
      setBenchmarkRunning(false);
      setActiveBenchmarkId(null);
    },
  });

  // Evaluation mutation - runs ASR WER evaluation (separate from benchmarks)
  const runEvaluationMutation = useMutation({
    mutationFn: () => {
      const endpointUrl = deployment?.endpoint_url || `http://localhost:${deployment?.port}`;
      const modelName = deployment?.release?.model?.name || 'default';
      return api.runEvaluation({
        deployment_id: deploymentId!,
        endpoint_url: endpointUrl,
        model_name: modelName,
        model_type: 'asr',
        limit: asrEvalLimit,
      });
    },
    onMutate: () => {
      setEvaluationRunning(true);
    },
    onSuccess: (result) => {
      setActiveEvaluationId(result.id);
    },
    onError: () => {
      setEvaluationRunning(false);
      setActiveEvaluationId(null);
    },
  });

  const logs = logsData?.logs || '';

  // Smart auto-scroll: only scroll to bottom if user hasn't scrolled up
  useEffect(() => {
    const container = logsContainerRef.current;
    if (container && !userScrolledUp.current) {
      container.scrollTop = container.scrollHeight;
    }
  }, [logs]);

  // Track if user has scrolled up (to disable auto-scroll)
  const handleLogsScroll = () => {
    const container = logsContainerRef.current;
    if (container) {
      const isAtBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 50;
      userScrolledUp.current = !isAtBottom;
    }
  };

  // Invalidate all deployment-related queries to update all views
  const invalidateDeploymentQueries = () => {
    queryClient.invalidateQueries({ queryKey: ['deployment'] });
    queryClient.invalidateQueries({ queryKey: ['deployments'] });
  };

  const handleStart = async () => {
    if (!deploymentId) return;
    setActionLoading('start');
    try {
      await api.startDeployment(deploymentId);
      invalidateDeploymentQueries();
    } catch (error) {
      console.error('Failed to start deployment:', error);
    } finally {
      setActionLoading(null);
    }
  };

  const handleStop = async () => {
    if (!deploymentId) return;
    setActionLoading('stop');
    try {
      await api.stopDeployment(deploymentId);
      invalidateDeploymentQueries();
    } catch (error) {
      console.error('Failed to stop deployment:', error);
    } finally {
      setActionLoading(null);
    }
  };

  const handleRestart = async () => {
    if (!deploymentId) return;
    setActionLoading('restart');
    try {
      await api.restartDeployment(deploymentId);
      invalidateDeploymentQueries();
    } catch (error) {
      console.error('Failed to restart deployment:', error);
    } finally {
      setActionLoading(null);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  if (isLoading) {
    return (
      <div className="text-center py-12">
        <Loader2 className="size-8 animate-spin mx-auto text-blue-600" />
        <p className="text-gray-500 mt-4">Loading deployment...</p>
      </div>
    );
  }

  if (isError || !deployment) {
    return (
      <div className="text-center py-12">
        <AlertCircle className="size-12 text-red-500 mx-auto mb-4" />
        <p className="text-gray-500">Deployment not found</p>
        <Button onClick={() => navigate('/deployments')} className="mt-4">
          Go Back
        </Button>
      </div>
    );
  }

  const isRunning = deployment.status === 'running';
  const canStart = canStartDeployment(deployment.status, deployment.deployment_type);
  const canStop = canStopDeployment(deployment.status, deployment.deployment_type);
  const canRestart = canRestartDeployment(deployment.status, deployment.deployment_type);

  // Generate endpoint URL
  const endpointUrl = getEndpointUrl(deployment);

  const modelName = deployment?.image_name || 'model';

  return (
    <div className="space-y-8 pb-12 w-full max-w-full overflow-x-hidden">
      {/* Header */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-center gap-4 max-w-full overflow-hidden">
          <Button variant="ghost" onClick={() => navigate('/deployments')} className="shrink-0">
            <ArrowLeft className="size-4 mr-2" />
            Back
          </Button>
          <Button variant="outline" onClick={() => navigate(`/versions/${deployment.release_id}`)} className="shrink-0">
            <Package className="size-4 mr-2" />
            View Version
          </Button>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-3 flex-wrap">
              <h1 className="text-2xl font-bold text-gray-900 truncate">{deployment.image_name || 'Deployment'}</h1>
              {deployment.release_version && (
                <Badge variant="outline" className="text-base shrink-0">v{deployment.release_version}</Badge>
              )}
              <Badge className={`${deploymentStatusColors[deployment.status] || 'bg-gray-100'} shrink-0`}>
                {deployment.status}
              </Badge>
              {isLocalDeployment && deployment.health_status && deployment.health_status !== 'unknown' && (
                <Badge className={`${healthStatusColors[deployment.health_status]} shrink-0`}>
                  {deployment.health_status}
                </Badge>
              )}
            </div>
            <p className="text-gray-500 mt-1 truncate">
              Deployed to {deployment.environment}
            </p>
          </div>
        </div>
        {isLocalDeployment && (
          <div className="flex items-center gap-2 shrink-0">
            {canStart && (
              <Button
                variant="outline"
                onClick={handleStart}
                disabled={actionLoading !== null}
                className="text-green-600 hover:text-green-700"
              >
                {actionLoading === 'start' ? (
                  <Loader2 className="size-4 mr-2 animate-spin" />
                ) : (
                  <Play className="size-4 mr-2" />
                )}
                Start
              </Button>
            )}
            {canStop && (
              <Button
                variant="outline"
                onClick={handleStop}
                disabled={actionLoading !== null}
                className="text-red-600 hover:text-red-700"
              >
                {actionLoading === 'stop' ? (
                  <Loader2 className="size-4 mr-2 animate-spin" />
                ) : (
                  <Square className="size-4 mr-2" />
                )}
                Stop
              </Button>
            )}
            {canRestart && (
              <Button
                variant="outline"
                onClick={handleRestart}
                disabled={actionLoading !== null}
              >
                {actionLoading === 'restart' ? (
                  <Loader2 className="size-4 mr-2 animate-spin" />
                ) : (
                  <RotateCw className="size-4 mr-2" />
                )}
                Restart
              </Button>
            )}
          </div>
        )}
      </div>

      {/* 1. Deployment Metadata Section */}
      <section className="space-y-6">
        <h2 className="text-xl font-semibold flex items-center gap-2 text-gray-800 border-b pb-2">
          <Server className="size-5 text-blue-600" />
          Deployment Metadata
        </h2>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className={`size-10 rounded-lg flex items-center justify-center ${isRunning ? 'bg-green-100' : 'bg-gray-100'
                  }`}>
                  <Activity className={`size-5 ${isRunning ? 'text-green-600' : 'text-gray-600'}`} />
                </div>
                <div>
                  <div className="text-xs text-gray-500">Status</div>
                  <div className="text-sm font-medium">{deployment.status}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="size-10 bg-blue-100 rounded-lg flex items-center justify-center">
                  <Globe className="size-5 text-blue-600" />
                </div>
                <div>
                  <div className="text-xs text-gray-500">Port</div>
                  <div className="text-sm font-medium">{deployment.host_port || 'N/A'}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="size-10 bg-purple-100 rounded-lg flex items-center justify-center">
                  <Server className="size-5 text-purple-600" />
                </div>
                <div>
                  <div className="text-xs text-gray-500">Type</div>
                  <div className="text-sm font-medium">{deployment.deployment_type}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className={`size-10 rounded-lg flex items-center justify-center ${deployment.gpu_enabled ? 'bg-orange-100' : 'bg-gray-100'
                  }`}>
                  <Cpu className={`size-5 ${deployment.gpu_enabled ? 'text-orange-600' : 'text-gray-600'}`} />
                </div>
                <div>
                  <div className="text-xs text-gray-500">GPU</div>
                  <div className="text-sm font-medium">{deployment.gpu_enabled ? 'Enabled' : 'Disabled'}</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full max-w-full">
          {/* Endpoint Information */}
          <Card className="min-w-0">
            <CardHeader>
              <CardTitle>Endpoint Information</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {endpointUrl && (
                <div>
                  <div className="text-sm text-gray-500 mb-1">Endpoint URL</div>
                  <div className="flex items-center gap-2 w-full max-w-full">
                    <a
                      href={endpointUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:underline flex items-center gap-1 break-all min-w-0"
                    >
                      {endpointUrl}
                      <ExternalLink className="size-3 shrink-0" />
                    </a>
                    <Button variant="ghost" size="sm" onClick={() => copyToClipboard(endpointUrl)} className="shrink-0">
                      <Copy className="size-3" />
                    </Button>
                  </div>
                </div>
              )}
              <div>
                <div className="text-sm text-gray-500 mb-1">Environment</div>
                <Badge variant={deployment.environment === 'production' ? 'default' : 'secondary'}>
                  {deployment.environment}
                </Badge>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-gray-500 mb-1">Deployed At</div>
                  <div>{formatDate(deployment.deployed_at)}</div>
                </div>
                {deployment.started_at && (
                  <div>
                    <div className="text-sm text-gray-500 mb-1">Started At</div>
                    <div>{formatDate(deployment.started_at)}</div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Container Information */}
          <Card className="min-w-0">
            <CardHeader>
              <CardTitle>Container Details</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="text-sm text-gray-500 mb-1">Container ID</div>
                {deployment.container_id ? (
                  <div className="flex items-center gap-2">
                    <code className="px-2 py-1 bg-gray-100 rounded text-xs">
                      {deployment.container_id.substring(0, 12)}
                    </code>
                    <Button variant="ghost" size="sm" onClick={() => copyToClipboard(deployment.container_id!)}>
                      <Copy className="size-3" />
                    </Button>
                  </div>
                ) : (
                  <span className="text-gray-400">Not available</span>
                )}
              </div>
              <div>
                <div className="text-sm text-gray-500 mb-1">Image</div>
                <code className="px-2 py-1 bg-gray-100 rounded text-sm break-all">
                  {deployment.image_tag
                    ? deployment.image_tag
                    : deployment.image_name && deployment.release_version
                      ? `${deployment.image_name}:${deployment.release_version}`
                      : deployment.image_name || deployment.release_version || 'N/A'}
                </code>
              </div>
              <div>
                <div className="text-sm text-gray-500 mb-1">Health Status</div>
                <div className="flex items-center gap-2">
                  {deployment.health_status === 'healthy' ? (
                    <CheckCircle className="size-4 text-green-600" />
                  ) : deployment.health_status === 'unhealthy' ? (
                    <XCircle className="size-4 text-red-600" />
                  ) : (
                    <AlertCircle className="size-4 text-gray-400" />
                  )}
                  <span className={
                    deployment.health_status === 'healthy' ? 'text-green-600' :
                      deployment.health_status === 'unhealthy' ? 'text-red-600' : 'text-gray-500'
                  }>
                    {deployment.health_status || 'Unknown'}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>


      {/* 2. Container Logs (Optional but recommended) */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b pb-2">
          <h2 className="text-xl font-semibold flex items-center gap-2 text-gray-800">
            <Terminal className="size-5 text-gray-600" />
            Container Logs
          </h2>
          <Button variant="outline" size="sm" onClick={() => refetchLogs()} disabled={logsLoading}>
            {logsLoading ? <Loader2 className="size-3 animate-spin mr-1" /> : <RotateCw className="size-3 mr-1" />}
            Refresh Logs
          </Button>
        </div>

        <Card className="min-w-0">
          <CardContent className="p-0">
            {!deployment.container_id ? (
              <div className="p-8 text-center text-gray-500">
                No container running. Start the deployment to view logs.
              </div>
            ) : (
              <div className="bg-gray-900 rounded-lg overflow-hidden w-full min-w-0">
                <div className="p-2 bg-gray-800 text-gray-400 text-xs flex justify-between px-4">
                  <span>stderr/stdout</span>
                  <span>{deployment.container_id.substring(0, 12)}</span>
                </div>
                <pre
                  ref={logsContainerRef}
                  onScroll={handleLogsScroll}
                  className="text-gray-100 p-4 overflow-auto max-h-[300px] text-xs font-mono whitespace-pre-wrap w-full min-w-0 break-all"
                >
                  {logs || 'No logs available'}
                </pre>
              </div>
            )}
          </CardContent>
        </Card>
      </section>

      {/* 3. Benchmark Stats */}
      <section className="space-y-6">
        <h2 className="text-xl font-semibold flex items-center gap-2 text-gray-800 border-b pb-2">
          <Gauge className="size-5 text-purple-600" />
          Benchmark Stats
        </h2>

        <BenchmarkStatsCard
          benchmarkRunning={benchmarkRunning}
          activeBenchmark={activeBenchmark}
          activeBenchmarkId={activeBenchmarkId}
          benchmarkSummary={benchmarkSummary}
          apiSpec={apiSpec}
          apiSpecLoading={apiSpecLoading}
          selectedEndpoint={selectedBenchmarkEndpoint}
          onSelectEndpoint={setSelectedBenchmarkEndpoint}
          onRunBenchmark={() => runBenchmarkMutation.mutate()}
          runBenchmarkPending={runBenchmarkMutation.isPending}
          onCancel={activeBenchmarkId ? () => cancelBenchmarkMutation.mutate(activeBenchmarkId) : undefined}
          cancelPending={cancelBenchmarkMutation.isPending}
          formatDate={formatDate}
        />

        {/* ASR Evaluation Card */}
        <ASREvaluationCard
          evaluationRunning={evaluationRunning}
          activeEvaluation={activeEvaluation}
          evaluationSummary={evaluationSummary}
          isRunning={deployment.status === 'running'}
          asrEvalLimit={asrEvalLimit}
          onLimitChange={setAsrEvalLimit}
          onRunEvaluation={() => runEvaluationMutation.mutate()}
          onCancelEvaluation={activeEvaluationId ? () => cancelEvaluationMutation.mutate(activeEvaluationId) : undefined}
        />
      </section>

      {/* 4. Discovered APIs */}
      <DiscoveredAPIsSection
        apiSpec={apiSpec}
        apiSpecLoading={apiSpecLoading}
        endpointUrl={endpointUrl}
        modelName={modelName}
        isRunning={deployment.status === 'running'}
        onRefresh={() => refetchApiSpec()}
      />

      {/* 5. Additional Metadata */}
      {deployment.metadata && Object.keys(deployment.metadata).length > 0 && (
        <section className="space-y-6">
          <h2 className="text-xl font-semibold flex items-center gap-2 text-gray-800 border-b pb-2">
            <FileJson className="size-5 text-gray-600" />
            Additional Metadata
          </h2>
          <Card className="min-w-0">
            <CardContent className="p-6">
              <pre className="bg-gray-50 p-4 rounded-lg overflow-x-auto text-sm w-full">
                {JSON.stringify(deployment.metadata, null, 2)}
              </pre>
            </CardContent>
          </Card>
        </section>
      )}
    </div>
  );
}
