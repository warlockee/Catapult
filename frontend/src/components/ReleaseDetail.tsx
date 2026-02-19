import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { ArrowLeft, GitBranch, Calendar, HardDrive, Cpu, CheckCircle, Copy, Loader2, Play, Square, RotateCw, ExternalLink, AlertCircle, ChevronDown, ChevronRight, Upload } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Input } from './ui/input';
import { Switch } from './ui/switch';
import { api } from '../lib/api';
import { MlflowMetadataCard } from './MlflowMetadataCard';
import { formatBytes } from '../lib/utils';
import {
  deploymentStatusColors,
  healthStatusColors,
  formatDate as formatDateTime,
  canStartDeployment,
  canStopDeployment,
  canRestartDeployment,
  getEndpointUrl,
} from '../lib/deployment-utils';
import { PublishDialog } from './PublishDialog';
import { BuildDockerDialog } from './BuildDockerDialog';
import { BuildDockerButton, useBuildState } from './BuildDockerButton';
import { BuildMatrix } from './BuildMatrix';
import type { Release, Deployment, DockerBuild, Image } from '../lib/api';

// Module-level formatDate function for use in all components
const formatDate = (dateString: string) => {
  return new Date(dateString).toLocaleString();
};

type DeployMode = 'local';

interface DeploymentCardProps {
  deployment: Deployment;
  onRefresh: () => void;
}

function DeploymentCard({ deployment, onRefresh }: DeploymentCardProps) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  // Invalidate all deployment-related queries to update all views
  const invalidateAndRefresh = () => {
    queryClient.invalidateQueries({ queryKey: ['deployment'] });
    queryClient.invalidateQueries({ queryKey: ['deployments'] });
    onRefresh();
  };

  const handleStart = async () => {
    setActionLoading('start');
    try {
      await api.startDeployment(deployment.id);
      invalidateAndRefresh();
    } catch (error) {
      console.error('Failed to start deployment:', error);
    } finally {
      setActionLoading(null);
    }
  };

  const handleStop = async () => {
    setActionLoading('stop');
    try {
      await api.stopDeployment(deployment.id);
      invalidateAndRefresh();
    } catch (error) {
      console.error('Failed to stop deployment:', error);
    } finally {
      setActionLoading(null);
    }
  };

  const handleRestart = async () => {
    setActionLoading('restart');
    try {
      await api.restartDeployment(deployment.id);
      invalidateAndRefresh();
    } catch (error) {
      console.error('Failed to restart deployment:', error);
    } finally {
      setActionLoading(null);
    }
  };

  const isLocalDeployment = deployment.deployment_type === 'local';
  const canStart = canStartDeployment(deployment.status, deployment.deployment_type);
  const canStop = canStopDeployment(deployment.status, deployment.deployment_type);
  const canRestart = canRestartDeployment(deployment.status, deployment.deployment_type);
  const endpointUrl = getEndpointUrl(deployment);

  const handleCardClick = (e: React.MouseEvent) => {
    // Only navigate if the click wasn't on a button
    if (!(e.target as HTMLElement).closest('button')) {
      navigate(`/deployments/${deployment.id}`);
    }
  };

  return (
    <div
      className="border rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
      onClick={handleCardClick}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-4">
          {/* Status Icon */}
          <div className={`
            size-10 rounded-full flex items-center justify-center
            ${deployment.status === 'running' ? 'bg-green-100' :
              deployment.status === 'stopped' ? 'bg-gray-100' :
                deployment.status === 'failed' ? 'bg-red-100' :
                  deployment.status === 'deploying' ? 'bg-blue-100' : 'bg-yellow-100'}
          `}>
            {deployment.status === 'running' ? (
              <Play className="size-5 text-green-600" />
            ) : deployment.status === 'stopped' ? (
              <Square className="size-5 text-gray-600" />
            ) : deployment.status === 'failed' ? (
              <AlertCircle className="size-5 text-red-600" />
            ) : deployment.status === 'deploying' ? (
              <Loader2 className="size-5 text-blue-600 animate-spin" />
            ) : (
              <CheckCircle className="size-5 text-yellow-600" />
            )}
          </div>

          {/* Deployment Info */}
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1 flex-wrap">
              <Badge variant={deployment.environment === 'production' ? 'default' : 'secondary'}>
                {deployment.environment}
              </Badge>
              <Badge className={deploymentStatusColors[deployment.status] || 'bg-gray-100'}>
                {deployment.status}
              </Badge>
              {isLocalDeployment && (
                <Badge variant="outline">{deployment.deployment_type}</Badge>
              )}
              {isLocalDeployment && deployment.health_status !== 'unknown' && (
                <Badge className={healthStatusColors[deployment.health_status]}>
                  {deployment.health_status}
                </Badge>
              )}
              {deployment.gpu_enabled && (
                <Badge variant="outline" className="text-purple-700 border-purple-300">
                  <Cpu className="size-3 mr-1" />GPU
                </Badge>
              )}
            </div>
            <div className="text-sm text-gray-500">
              {formatDate(deployment.deployed_at)}
            </div>

            {/* Container Info for local deployments */}
            {isLocalDeployment && (
              <div className="mt-2 space-y-1 text-sm">
                {deployment.endpoint_url && (
                  <div className="flex items-center gap-2">
                    <span className="text-gray-500">Endpoint:</span>
                    <a
                      href={deployment.endpoint_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:underline flex items-center gap-1"
                    >
                      {deployment.endpoint_url}
                      <ExternalLink className="size-3" />
                    </a>
                  </div>
                )}
                {deployment.host_port && (
                  <div className="flex items-center gap-2">
                    <span className="text-gray-500">Port:</span>
                    <code className="text-gray-700">{deployment.host_port}</code>
                  </div>
                )}
                {deployment.container_id && (
                  <div className="flex items-center gap-2">
                    <span className="text-gray-500">Container:</span>
                    <code className="text-gray-700 text-xs">{deployment.container_id.substring(0, 12)}</code>
                  </div>
                )}
              </div>
            )}

            {/* Metadata deployment info */}
            {!isLocalDeployment && deployment.metadata?.kubernetes_namespace && (
              <div className="mt-2 flex gap-4 text-sm">
                <span className="text-gray-500">
                  Namespace: <code className="text-gray-700">{deployment.metadata.kubernetes_namespace}</code>
                </span>
                {deployment.metadata.replicas && (
                  <span className="text-gray-500">
                    Replicas: {deployment.metadata.replicas}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Actions for local deployments */}
        {isLocalDeployment && (
          <div className="flex items-center gap-2">
            {canStart && (
              <Button
                size="sm"
                variant="outline"
                onClick={handleStart}
                disabled={actionLoading !== null}
                className="text-green-600 hover:text-green-700"
              >
                {actionLoading === 'start' ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Play className="size-4" />
                )}
              </Button>
            )}
            {canStop && (
              <Button
                size="sm"
                variant="outline"
                onClick={handleStop}
                disabled={actionLoading !== null}
                className="text-red-600 hover:text-red-700"
              >
                {actionLoading === 'stop' ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Square className="size-4" />
                )}
              </Button>
            )}
            {canRestart && (
              <Button
                size="sm"
                variant="outline"
                onClick={handleRestart}
                disabled={actionLoading !== null}
              >
                {actionLoading === 'restart' ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <RotateCw className="size-4" />
                )}
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export function ReleaseDetail() {
  const { releaseId, modelId } = useParams<{ releaseId: string; modelId?: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [release, setRelease] = useState<Release | null>(null);
  const [model, setModel] = useState<Image | null>(null);
  const [gpuInfo, setGpuInfo] = useState<{ available: boolean; count: number } | null>(null);
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [dockerBuilds, setDockerBuilds] = useState<DockerBuild[]>([]);
  const [loading, setLoading] = useState(true);
  const [showDeployDialog, setShowDeployDialog] = useState(false);
  const [showBuildDialog, setShowBuildDialog] = useState(false);
  const [showPublishDialog, setShowPublishDialog] = useState(false);
  const [isPromoting, setIsPromoting] = useState(false);
  const [deployFormData, setDeployFormData] = useState({
    deployment_type: 'local' as DeployMode,
    environment: 'development',
    gpu_enabled: false,  // Will be auto-set based on GPU detection when dialog opens
  });
  const [deploying, setDeploying] = useState(false);
  const [availableBuilds, setAvailableBuilds] = useState<{
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
  } | null>(null);
  const [loadingBuilds, setLoadingBuilds] = useState(false);
  const [selectedBuildId, setSelectedBuildId] = useState<string | null>(null);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);

  // Shared build state between button and dialog
  const buildState = useBuildState();

  // Get successful docker builds for local deployment
  const successfulBuilds = dockerBuilds.filter(b => b.status === 'success' && b.is_current);
  const hasDockerBuild = successfulBuilds.length > 0;

  // Load GPU info and sync state when dialog opens
  useEffect(() => {
    if (showDeployDialog) {
      loadGpuInfo();
    }
  }, [showDeployDialog]);

  // Sync GPU enabled state when GPU info becomes available
  useEffect(() => {
    if (showDeployDialog && gpuInfo) {
      const shouldEnableGpu = (model?.requires_gpu ?? true) && gpuInfo.available;
      setDeployFormData(prev => ({ ...prev, gpu_enabled: shouldEnableGpu }));
    }
  }, [showDeployDialog, gpuInfo, model?.requires_gpu]);

  // Fetch available builds when deploy dialog opens for local deployments
  useEffect(() => {
    if (showDeployDialog && releaseId && deployFormData.deployment_type === 'local') {
      setLoadingBuilds(true);
      api.getAvailableBuildsForDeployment(releaseId)
        .then(result => {
          setAvailableBuilds(result);
          // Auto-select the current/latest build if available
          const currentBuild = result.builds.find(b => b.is_current);
          if (currentBuild) {
            setSelectedBuildId(currentBuild.id);
          } else if (result.builds.length > 0) {
            setSelectedBuildId(result.builds[0].id);
          }
        })
        .catch(error => {
          console.error('Failed to fetch available builds:', error);
          setAvailableBuilds({ has_builds: false, builds: [], model_server_type: null });
        })
        .finally(() => {
          setLoadingBuilds(false);
        });
    } else if (!showDeployDialog) {
      // Reset when dialog closes
      setAvailableBuilds(null);
      setSelectedBuildId(null);
    }
  }, [showDeployDialog, releaseId, deployFormData.deployment_type]);

  useEffect(() => {
    loadReleaseData();
  }, [releaseId]);

  const loadReleaseData = async () => {
    try {
      if (!release) setLoading(true);
      const [releaseData, deploymentsData, buildsData] = await Promise.all([
        api.getRelease(releaseId),
        api.listDeployments({ release_id: releaseId }),
        api.listDockerBuilds({ release_id: releaseId, size: 50 }),
      ]);
      setRelease(releaseData);
      setDeployments(deploymentsData.items || []);
      setDockerBuilds(buildsData.items || []);

      // End loading state immediately - don't block UI on model fetch or GPU detection
      if (!release) setLoading(false);

      // Fetch model data in background (non-blocking) for requires_gpu setting
      if (releaseData.image_id) {
        api.getImage(releaseData.image_id)
          .then(modelData => setModel(modelData))
          .catch(error => console.error('Failed to load model data:', error));
      }
    } catch (error) {
      console.error('Failed to load release data:', error);
      if (!release) setLoading(false);
    }
  };

  // Lazy load GPU info only when deploy dialog opens
  const loadGpuInfo = async () => {
    if (gpuInfo) return; // Already loaded
    try {
      const data = await api.getGpuInfo();
      setGpuInfo(data);
    } catch {
      setGpuInfo({ available: false, count: 0 });
    }
  };

  const handleDeploy = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!deployFormData.environment) return;

    try {
      setDeploying(true);

      // Determine which build to use
      // If only one build, use it; otherwise use the selected one
      const buildId = availableBuilds?.builds.length === 1
        ? availableBuilds.builds[0].id
        : selectedBuildId;

      // Execute local deployment - starts actual Docker container
      const deployment = await api.executeDeployment({
        release_id: releaseId!,
        environment: deployFormData.environment,
        deployment_type: 'local',
        docker_build_id: buildId || undefined,
        gpu_enabled: deployFormData.gpu_enabled,
      });

      setShowDeployDialog(false);
      setDeployFormData({
        deployment_type: 'local',
        environment: 'development',
        gpu_enabled: false,  // Reset - will be auto-set when dialog opens again
      });

      // Invalidate all deployment queries to update other views
      queryClient.invalidateQueries({ queryKey: ['deployment'] });
      queryClient.invalidateQueries({ queryKey: ['deployments'] });

      // Navigate to deployment detail page
      navigate(`/deployments/${deployment.id}`);
    } catch (error) {
      console.error('Failed to create deployment:', error);
    } finally {
      setDeploying(false);
    }
  };

  if (loading) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">Loading model...</p>
      </div>
    );
  }

  if (!release) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">Release not found</p>
        <Button onClick={() => navigate(-1)} className="mt-4">
          Go Back
        </Button>
      </div>
    );
  }

  const copyToClipboard = (text: string) => {
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text);
    } else {
      // Fallback for non-secure contexts
      const textarea = document.createElement('textarea');
      textarea.value = text;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
    }
  };

  const handlePromote = async () => {
    if (!release) return;
    setIsPromoting(true);
    try {
      await api.promoteRelease(release.id, true);
      await loadReleaseData();
      // Invalidate releases list so it shows updated is_release status
      queryClient.invalidateQueries({ queryKey: ['releases'] });
    } catch (error) {
      console.error('Failed to promote release:', error);
    } finally {
      setIsPromoting(false);
    }
  };

  const handleDemote = async () => {
    if (!release) return;
    setIsPromoting(true);
    try {
      await api.promoteRelease(release.id, false);
      await loadReleaseData();
      // Invalidate releases list so it shows updated is_release status
      queryClient.invalidateQueries({ queryKey: ['releases'] });
    } catch (error) {
      console.error('Failed to demote release:', error);
    } finally {
      setIsPromoting(false);
    }
  };

  const deploySubmitDisabled = deploying ||
    loadingBuilds ||
    (deployFormData.deployment_type === 'local' && !availableBuilds?.has_builds) ||
    (deployFormData.deployment_type === 'local' &&
      availableBuilds &&
      availableBuilds.builds.length > 1 &&
      !selectedBuildId);



  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="space-y-3 overflow-hidden">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-4">
            <Button variant="ghost" onClick={() => navigate(-1)} className="shrink-0">
              <ArrowLeft className="size-4 mr-2" />
              Back
            </Button>
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <h1>{release.image_name}</h1>
                {release.is_release ? (
                  <Badge className="bg-blue-600">Official Release</Badge>
                ) : (
                  <Badge variant="secondary">Version</Badge>
                )}
                <Badge variant={release.status === 'active' ? 'default' : 'secondary'}>
                  {release.status}
                </Badge>
              </div>
              <p className="text-gray-500 mt-1">
                {release.is_release ? 'Release Details' : 'Version Details'}
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2 shrink-0">
            <BuildDockerButton
              releaseId={releaseId!}
              onClick={() => setShowBuildDialog(true)}
              activeBuild={showBuildDialog ? buildState.activeBuild : undefined}
              buildProgress={showBuildDialog ? buildState.buildProgress : undefined}
              onBuildComplete={(completedBuild) => {
                // Update dockerBuilds when a build completes (detected via button polling)
                setDockerBuilds(prev => {
                  const exists = prev.find(b => b.id === completedBuild.id);
                  if (exists) {
                    return prev.map(b => b.id === completedBuild.id ? completedBuild : b);
                  }
                  return [completedBuild, ...prev];
                });
              }}
            />
            <Button className="bg-blue-600 hover:bg-blue-700" onClick={() => setShowDeployDialog(true)}>
              <Play className="size-4 mr-2" />
              Deploy
            </Button>
            <Button variant="outline" onClick={() => setShowPublishDialog(true)}>
              <Upload className="size-4 mr-2" />
              Publish
            </Button>
            {release.is_release ? (
              <Button variant="outline" onClick={handleDemote} disabled={isPromoting}>
                {isPromoting ? (
                  <Loader2 className="size-4 mr-2 animate-spin" />
                ) : (
                  <GitBranch className="size-4 mr-2" />
                )}
                Demote
              </Button>
            ) : (
              <Button className="bg-green-600 hover:bg-green-700 text-white" onClick={handlePromote} disabled={isPromoting}>
                {isPromoting ? (
                  <Loader2 className="size-4 mr-2 animate-spin" />
                ) : (
                  <GitBranch className="size-4 mr-2" />
                )}
                Promote to Release
              </Button>
            )}
          </div>
        </div>
        <code className="block px-3 py-2 bg-gray-100 rounded-lg text-sm overflow-x-auto whitespace-nowrap">
          v{release.version}{release.tag !== release.version ? ` · ${release.tag}` : ''}
        </code>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="size-10 bg-blue-100 rounded-lg flex items-center justify-center">
                <Calendar className="size-5 text-blue-600" />
              </div>
              <div>
                <div className="text-xs text-gray-500">Created</div>
                <div className="text-sm">{new Date(release.created_at).toLocaleDateString()}</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="size-10 bg-green-100 rounded-lg flex items-center justify-center">
                <HardDrive className="size-5 text-green-600" />
              </div>
              <div>
                <div className="text-xs text-gray-500">Size</div>
                <div className="text-sm">{release.size_bytes ? formatBytes(release.size_bytes) : 'N/A'}</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="size-10 bg-purple-100 rounded-lg flex items-center justify-center">
                <Cpu className="size-5 text-purple-600" />
              </div>
              <div>
                <div className="text-xs text-gray-500">Platform</div>
                <div className="text-sm">{release.platform}</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="size-10 bg-green-100 rounded-lg flex items-center justify-center">
                <CheckCircle className="size-5 text-green-600" />
              </div>
              <div>
                <div className="text-xs text-gray-500">Deployments</div>
                <div className="text-sm">{deployments.length} times</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="metadata">Metadata</TabsTrigger>
          <TabsTrigger value="deployments">Deployments</TabsTrigger>
          {release.is_release && <TabsTrigger value="pull">Pull Command</TabsTrigger>}
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Deployment Tracking Card */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-lg">Deployment Status</CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  const tabsTrigger = document.querySelector('[data-state="inactive"][value="deployments"]') as HTMLElement;
                  if (tabsTrigger) tabsTrigger.click();
                }}
                className="text-blue-600 hover:text-blue-700"
              >
                View All
              </Button>
            </CardHeader>
            <CardContent>
              {deployments.length > 0 ? (
                <div className="space-y-3">
                  {deployments.slice(0, 3).map((deployment) => {
                    const isRunning = deployment.status === 'running';
                    const isStopped = deployment.status === 'stopped';
                    const isFailed = deployment.status === 'failed';
                    return (
                      <div
                        key={deployment.id}
                        className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 cursor-pointer transition-colors"
                        onClick={() => navigate(`/deployments/${deployment.id}`)}
                      >
                        <div className="flex items-center gap-3">
                          <div className={`size-3 rounded-full ${isRunning ? 'bg-green-500' :
                              isFailed ? 'bg-red-500' :
                                isStopped ? 'bg-gray-400' : 'bg-yellow-500'
                            }`} />
                          <div>
                            <div className="flex items-center gap-2">
                              <Badge variant={deployment.environment === 'production' ? 'default' : 'secondary'} className="text-xs">
                                {deployment.environment}
                              </Badge>
                              <Badge className={`text-xs ${deploymentStatusColors[deployment.status] || 'bg-gray-100'}`}>
                                {deployment.status}
                              </Badge>
                            </div>
                            <p className="text-xs text-gray-500 mt-1">
                              {formatDateTime(deployment.deployed_at)}
                            </p>
                          </div>
                        </div>
                        <ChevronRight className="size-4 text-gray-400" />
                      </div>
                    );
                  })}
                  {deployments.length > 3 && (
                    <p className="text-xs text-gray-500 text-center pt-2">
                      +{deployments.length - 3} more deployment{deployments.length - 3 > 1 ? 's' : ''}
                    </p>
                  )}
                </div>
              ) : (
                <div className="text-center py-6 text-gray-500">
                  <p className="text-sm">No deployments yet</p>
                  <Button
                    variant="outline"
                    size="sm"
                    className="mt-3"
                    onClick={() => setShowDeployDialog(true)}
                  >
                    <Play className="size-3 mr-2" />
                    Deploy Now
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Version Information */}
            <Card className="overflow-hidden">
              <CardHeader>
                <CardTitle>Version Information</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 min-w-0 overflow-hidden">
                <div>
                  <div className="text-sm text-gray-500 mb-1">Version</div>
                  <code className="px-2 py-1 bg-gray-100 rounded text-sm block overflow-x-auto whitespace-nowrap">{release.version}</code>
                </div>
                <div>
                  <div className="text-sm text-gray-500 mb-1">Tag</div>
                  <div className="flex items-center gap-2">
                    <code className="px-2 py-1 bg-gray-100 rounded text-sm overflow-x-auto whitespace-nowrap">{release.tag}</code>
                    <Button variant="ghost" size="sm" onClick={() => copyToClipboard(release.tag)} className="shrink-0">
                      <Copy className="size-3" />
                    </Button>
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-500 mb-1">Digest</div>
                  <div className="flex items-center gap-2">
                    <code className="min-w-0 px-2 py-1 bg-gray-100 rounded text-xs overflow-x-auto whitespace-nowrap">
                      {release.digest}
                    </code>
                    <Button variant="ghost" size="sm" onClick={() => copyToClipboard(release.digest)} className="shrink-0">
                      <Copy className="size-3" />
                    </Button>
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-500 mb-1">Created At</div>
                  <div>{formatDate(release.created_at)}</div>
                </div>
                {release.ceph_path && (
                  <div>
                    <div className="text-sm text-gray-500 mb-1">Storage Path</div>
                    <div className="flex items-center gap-2">
                      <code className="px-2 py-1 bg-gray-100 rounded text-sm overflow-x-auto whitespace-nowrap">
                        {release.ceph_path}
                      </code>
                      <Button variant="ghost" size="sm" onClick={() => copyToClipboard(release.ceph_path!)} className="shrink-0">
                        <Copy className="size-3" />
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Platform Details */}
            <Card className="overflow-hidden">
              <CardHeader>
                <CardTitle>Platform & Build</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 min-w-0">
                <div>
                  <div className="text-sm text-gray-500 mb-1">Operating System</div>
                  <div>{release.os || 'N/A'}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-500 mb-1">Architecture</div>
                  <div>{release.architecture || 'N/A'}</div>
                </div>
                {release.metadata.git_commit && (
                  <div>
                    <div className="text-sm text-gray-500 mb-1">Git Commit</div>
                    <div className="flex items-center gap-2">
                      <code className="px-2 py-1 bg-gray-100 rounded text-sm overflow-x-auto whitespace-nowrap">
                        {release.metadata.git_commit}
                      </code>
                      <Badge variant="outline" className="shrink-0">{release.metadata.git_branch}</Badge>
                    </div>
                  </div>
                )}
                {release.metadata.build_timestamp && (
                  <div>
                    <div className="text-sm text-gray-500 mb-1">Build Time</div>
                    <div>{formatDate(release.metadata.build_timestamp)}</div>
                  </div>
                )}
                {release.metadata.ci_pipeline && (
                  <div>
                    <div className="text-sm text-gray-500 mb-1">CI Pipeline</div>
                    <code className="px-2 py-1 bg-gray-100 rounded text-sm block overflow-x-auto whitespace-nowrap">
                      {release.metadata.ci_pipeline}
                    </code>
                  </div>
                )}

                {/* Docker Build Matrix */}
                <div className="pt-4 border-t">
                  <div className="text-sm text-gray-500 mb-2">Docker Builds</div>
                  <BuildMatrix builds={dockerBuilds} maxCells={28} columns={7} />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* MLflow Metadata */}
          {release.mlflow_url && (
            <MlflowMetadataCard
              versionId={release.id}
              mlflowUrl={release.mlflow_url}
            />
          )}

          {/* Model Performance */}
          {(release.metadata.accuracy || release.metadata.model_architecture) && (
            <Card>
              <CardHeader>
                <CardTitle>Model Performance</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                  {release.metadata.model_architecture && (
                    <div>
                      <div className="text-sm text-gray-500 mb-1">Architecture</div>
                      <div>{release.metadata.model_architecture}</div>
                    </div>
                  )}
                  {release.metadata.accuracy && (
                    <div>
                      <div className="text-sm text-gray-500 mb-1">Accuracy</div>
                      <div className="text-green-600">
                        {(release.metadata.accuracy * 100).toFixed(2)}%
                      </div>
                    </div>
                  )}
                  {release.metadata.loss && (
                    <div>
                      <div className="text-sm text-gray-500 mb-1">Loss</div>
                      <div>{release.metadata.loss.toFixed(4)}</div>
                    </div>
                  )}
                  {release.metadata.epochs && (
                    <div>
                      <div className="text-sm text-gray-500 mb-1">Epochs</div>
                      <div>{release.metadata.epochs}</div>
                    </div>
                  )}
                  {release.metadata.training_dataset && (
                    <div>
                      <div className="text-sm text-gray-500 mb-1">Training Dataset</div>
                      <div>{release.metadata.training_dataset}</div>
                    </div>
                  )}
                  {release.metadata.training_duration && (
                    <div>
                      <div className="text-sm text-gray-500 mb-1">Training Duration</div>
                      <div>{release.metadata.training_duration}</div>
                    </div>
                  )}
                  {release.metadata.pytorch_version && (
                    <div>
                      <div className="text-sm text-gray-500 mb-1">PyTorch Version</div>
                      <div>{release.metadata.pytorch_version}</div>
                    </div>
                  )}
                  {release.metadata.cuda_version && (
                    <div>
                      <div className="text-sm text-gray-500 mb-1">CUDA Version</div>
                      <div>{release.metadata.cuda_version}</div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="metadata">
          <Card>
            <CardHeader>
              <CardTitle>Full Metadata</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="bg-gray-50 p-4 rounded-lg overflow-x-auto text-sm">
                {JSON.stringify(release.metadata, null, 2)}
              </pre>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="deployments">
          <Card>
            <CardHeader>
              <CardTitle>Deployment History</CardTitle>
            </CardHeader>
            <CardContent>
              {deployments.length > 0 ? (
                <div className="space-y-4">
                  {deployments.map((deployment) => (
                    <DeploymentCard
                      key={deployment.id}
                      deployment={deployment}
                      onRefresh={loadReleaseData}
                    />
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  No deployments yet
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {release.is_release && (
          <>
            <TabsContent value="pull">
              <Card>
                <CardHeader>
                  <CardTitle>Pull Command</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="text-gray-600">
                    Use the following command to pull this release:
                  </p>
                  <div className="relative">
                    <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                      <code>docker pull {release.image_name}:{release.tag}</code>
                    </pre>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="absolute top-2 right-2 text-gray-400 hover:text-white"
                      onClick={() => copyToClipboard(`docker pull ${release.image_name}:${release.tag}`)}
                    >
                      <Copy className="size-4" />
                    </Button>
                  </div>

                  <div className="pt-4 border-t">
                    <h4 className="mb-3">Python SDK</h4>
                    <div className="relative">
                      <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm">
                        {`from bso import Registry

registry = Registry(
    base_url="https://registry.example.com",
    api_key="your-api-key"
)

release = registry.get_release("${release.id}")
print(f"Version: {release.version}")
print(f"Accuracy: {release.metadata.get('accuracy')}")`}
                      </pre>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="absolute top-2 right-2 text-gray-400 hover:text-white"
                        onClick={() => copyToClipboard(`from bso import Registry\n\nregistry = Registry(\n    base_url="https://registry.example.com",\n    api_key="your-api-key"\n)\n\nrelease = registry.get_release("${release.id}")\nprint(f"Version: {release.version}")\nprint(f"Accuracy: {release.metadata.get('accuracy')}")`)}
                      >
                        <Copy className="size-4" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </>
        )}
      </Tabs>

      {/* Deploy Dialog */}
      <Dialog open={showDeployDialog} onOpenChange={(open) => {
        setShowDeployDialog(open);
        if (!open) {
          setShowAdvancedOptions(false);  // Reset when closing
        }
      }}>
        <DialogContent className="max-w-lg max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Deploy Release</DialogTitle>
            <DialogDescription>
              {`Deploy ${release.version} to an environment`}
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={handleDeploy}>
            <div className="space-y-4 py-4">
              {/* Environment */}
              <div className="space-y-2">
                <Label htmlFor="environment">Environment</Label>
                <Select
                  value={deployFormData.environment}
                  onValueChange={(value: string) => setDeployFormData({ ...deployFormData, environment: value })}
                >
                  <SelectTrigger id="environment">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="development">Development</SelectItem>
                    <SelectItem value="staging">Staging</SelectItem>
                    <SelectItem value="production">Production</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Local deployment - check for docker builds */}
              {deployFormData.deployment_type === 'local' && (
                <>
                  {/* Loading state */}
                  {loadingBuilds && (
                    <div className="flex items-center gap-2 p-3 bg-gray-50 border border-gray-200 rounded-lg text-gray-600">
                      <Loader2 className="size-4 animate-spin" />
                      <span className="text-sm">Checking Docker images...</span>
                    </div>
                  )}

                  {/* No builds available */}
                  {!loadingBuilds && availableBuilds && !availableBuilds.has_builds && (
                    <div className="flex items-start gap-2 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-yellow-800">
                      <AlertCircle className="size-5 mt-0.5 shrink-0" />
                      <div className="text-sm">
                        <p className="font-medium">No Docker image available</p>
                        <p className="text-yellow-700">Build a Docker image first before deploying locally.</p>
                      </div>
                    </div>
                  )}

                  {/* Summary when builds available */}
                  {!loadingBuilds && availableBuilds && availableBuilds.has_builds && !showAdvancedOptions && (
                    <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 text-sm text-gray-600">
                          <span>Image:</span>
                          <code className="font-mono text-gray-900">
                            {availableBuilds.builds.length === 1
                              ? availableBuilds.builds[0].image_tag
                              : availableBuilds.builds.find(b => b.id === selectedBuildId)?.image_tag || 'Select image'}
                          </code>
                          {deployFormData.gpu_enabled && (
                            <Badge variant="outline" className="text-purple-700 border-purple-300">
                              <Cpu className="size-3 mr-1" />GPU
                            </Badge>
                          )}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Advanced Options Toggle */}
                  {!loadingBuilds && availableBuilds && availableBuilds.has_builds && (
                    <button
                      type="button"
                      className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
                      onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
                    >
                      {showAdvancedOptions ? (
                        <ChevronDown className="size-4" />
                      ) : (
                        <ChevronRight className="size-4" />
                      )}
                      Advanced Options
                    </button>
                  )}

                  {/* Advanced Options Content */}
                  {showAdvancedOptions && !loadingBuilds && availableBuilds && availableBuilds.has_builds && (
                    <div className="space-y-4 pl-6 border-l-2 border-gray-200">
                      {/* Docker Image Selection */}
                      {availableBuilds.builds.length > 1 && (
                        <div className="space-y-2">
                          <Label>Docker Image</Label>
                          <Select
                            value={selectedBuildId || ''}
                            onValueChange={(value: string) => setSelectedBuildId(value)}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Select Docker image" />
                            </SelectTrigger>
                            <SelectContent>
                              {availableBuilds.builds.map((build) => (
                                <SelectItem key={build.id} value={build.id}>
                                  <div className="flex items-center gap-2">
                                    <code className="text-sm">{build.image_tag}</code>
                                    {build.server_type && (
                                      <Badge variant="outline" className="text-xs">
                                        {build.server_type}
                                      </Badge>
                                    )}
                                    {build.is_current && (
                                      <Badge variant="secondary" className="text-xs">latest</Badge>
                                    )}
                                  </div>
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <p className="text-xs text-gray-500">
                            {availableBuilds.builds.length} Docker images available
                          </p>
                        </div>
                      )}

                      {/* Single build info */}
                      {availableBuilds.builds.length === 1 && (
                        <div className="space-y-2">
                          <Label className="text-gray-500">Docker Image</Label>
                          <div className="flex items-center gap-2">
                            <code className="text-sm font-mono">{availableBuilds.builds[0].image_tag}</code>
                            {availableBuilds.builds[0].server_type && (
                              <Badge variant="outline" className="text-xs">
                                {availableBuilds.builds[0].server_type}
                              </Badge>
                            )}
                          </div>
                        </div>
                      )}

                      {/* GPU Toggle */}
                      <div className="flex items-center justify-between">
                        <div>
                          <Label htmlFor="gpu_enabled">Enable GPU</Label>
                          <p className="text-xs text-gray-500">
                            {gpuInfo?.available
                              ? `${gpuInfo.count} GPU(s) detected`
                              : 'No GPUs detected'}
                          </p>
                        </div>
                        <Switch
                          id="gpu_enabled"
                          checked={deployFormData.gpu_enabled}
                          onCheckedChange={(checked) => setDeployFormData({ ...deployFormData, gpu_enabled: checked })}
                          disabled={!gpuInfo?.available}
                        />
                      </div>
                      {deployFormData.gpu_enabled && !gpuInfo?.available && (
                        <p className="text-xs text-red-500">
                          Warning: GPU enabled but no GPUs detected. Deployment will fail.
                        </p>
                      )}
                    </div>
                  )}
                </>
              )}

            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setShowDeployDialog(false)}>
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={deploySubmitDisabled}
              >
                {deploying ? (
                  <>
                    <Loader2 className="size-4 mr-2 animate-spin" />
                    Deploying...
                  </>
                ) : (
                  <>
                    <Play className="size-4 mr-2" />
                    Deploy
                  </>
                )}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>


      {/* Publish Dialog */}
      <PublishDialog
        versionId={releaseId!}
        modelId={release.image_id}
        open={showPublishDialog}
        onOpenChange={setShowPublishDialog}
      />

      {/* Build Docker Dialog */}
      {release && (
        <BuildDockerDialog
          release={release}
          open={showBuildDialog}
          onOpenChange={setShowBuildDialog}
          onSuccess={(completedBuild) => {
            // Update dockerBuilds with the completed build
            // This ensures BuildMatrix shows the green success square immediately
            setDockerBuilds(prev => {
              const exists = prev.find(b => b.id === completedBuild.id);
              if (exists) {
                return prev.map(b => b.id === completedBuild.id ? completedBuild : b);
              }
              return [completedBuild, ...prev];
            });
            // Delay refresh to avoid race condition where API returns stale "building" status
            // that overwrites our correct "success" state
            setTimeout(() => loadReleaseData(), 2000);
          }}
          onBuildStateChange={(build, progress) => {
            // Update button state
            buildState.updateBuildState(build, progress);
            // Also update dockerBuilds so BuildMatrix shows the building indicator
            if (build) {
              setDockerBuilds(prev => {
                const exists = prev.find(b => b.id === build.id);
                if (exists) {
                  // Update existing build with new status
                  return prev.map(b => b.id === build.id ? build : b);
                }
                // Add new build to the list
                return [build, ...prev];
              });
            }
          }}
        />
      )}
    </div>
  );
}
