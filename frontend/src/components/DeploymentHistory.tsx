import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { GitBranch, CheckCircle, XCircle, Clock, Filter, Calendar, Play, Square, AlertCircle, Globe, RefreshCw } from 'lucide-react';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog';
import { Label } from './ui/label';
import { Input } from './ui/input';
import { ErrorState } from './ErrorState';
import { LoadingState } from './LoadingState';
import { EmptyState } from './EmptyState';
import { Pagination } from './Pagination';
import { api } from '../lib/api';
import {
  deploymentStatusColors,
  deploymentTypeColors,
  getRelativeTime,
  formatDate,
  needsStatusPolling,
} from '../lib/deployment-utils';
import type { Deployment } from '../lib/api';

export function DeploymentHistory({ embedded = false }: { embedded?: boolean }) {
  const navigate = useNavigate();
  const [environmentFilter, setEnvironmentFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [page, setPage] = useState(1);
  const pageSize = 10;

  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [formData, setFormData] = useState({
    release_id: '',
    environment: 'development',
    kubernetes_namespace: '',
    replicas: '1',
    gpu_type: '',
  });
  const queryClient = useQueryClient();

  // Reset page when filters change
  const handleEnvFilter = (val: string) => {
    setEnvironmentFilter(val);
    setPage(1);
  };
  const handleStatusFilter = (val: string) => {
    setStatusFilter(val);
    setPage(1);
  };

  const { data: deploymentData, isLoading: deploymentsLoading, isError: deploymentsError, refetch, isFetching } = useQuery({
    queryKey: ['deployments', page, environmentFilter, statusFilter],
    queryFn: ({ signal }) => api.listDeployments({
      page,
      size: pageSize,
      environment: environmentFilter === 'all' ? undefined : environmentFilter,
      status: statusFilter === 'all' ? undefined : statusFilter,
      signal, // Enable request cancellation
    }),
    refetchInterval: (query) => {
      // Auto-refresh every 5s if any deployment needs status polling (running or transitional)
      const data = query.state.data;
      if (data?.items?.some((d: Deployment) => needsStatusPolling(d.status))) {
        return 5000;
      }
      // No active deployments - no need to poll frequently
      return false;
    },
  });

  const deployments = deploymentData?.items || [];
  const totalPages = deploymentData?.pages || 0;

  // Use slim endpoint to reduce data transfer for dropdown
  const { data: releases = [], isLoading: releasesLoading, isError: releasesError } = useQuery({
    queryKey: ['releases', 'options'],
    queryFn: ({ signal }) => api.listReleaseOptions(signal),
  });

  const isError = deploymentsError || releasesError;

  const loading = deploymentsLoading || releasesLoading;

  const createMutation = useMutation({
    mutationFn: (data: Parameters<typeof api.createDeployment>[0]) => api.createDeployment(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['deployments'] });
      setShowCreateDialog(false);
      setFormData({
        release_id: '',
        environment: 'development',
        kubernetes_namespace: '',
        replicas: '1',
        gpu_type: '',
      });
    },
    onError: (error) => {
      console.error('Failed to create deployment:', error);
    },
  });

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.release_id || !formData.environment) return;

    createMutation.mutate({
      release_id: formData.release_id,
      environment: formData.environment,
      status: 'success',
      metadata: {
        kubernetes_namespace: formData.kubernetes_namespace || undefined,
        replicas: formData.replicas ? parseInt(formData.replicas) : undefined,
        gpu_type: formData.gpu_type || undefined,
      },
    });
  };

  // Client-side filtering is no longer needed/possible since we filter on backend page
  // But we still need filteredDeployments variable if used below?
  // Let's check usages.
  // Used for: deploymentsByDate reducer.
  // And deploymentsByDate is used for rendering.
  // So we just use `deployments` (which is already filtered by backend)
  const filteredDeployments = deployments;


  // Get environments from actual fetched data + hardcoded defaults if empty?
  // Or just rely on what we have.
  // The filter dropdown uses `environments` variable.
  // Previously: const environments = Array.from(new Set(deployments.map(d => d.environment)));
  // With pagination, we only see environments on current page. 
  // Ideally backend provides aggregation. For now, we can keep this or hardcode common envs.
  // In `SelectContent` below (line 233 in original), it maps `environments`.
  // Let's hardcode 'development', 'staging', 'production' as they are standard, plus any others found.
  const environments = Array.from(new Set(['development', 'staging', 'production', ...deployments.map(d => d.environment)]));

  const deploymentsByDate = filteredDeployments.reduce((acc, deployment) => {
    const date = new Date(deployment.deployed_at).toLocaleDateString();
    if (!acc[date]) acc[date] = [];
    acc[date].push(deployment);
    return acc;
  }, {} as Record<string, Deployment[]>);

  const submitDisabled = createMutation.isPending;

  if (loading) {
    return <LoadingState title="Dev Deployments" />;
  }

  if (isError) {
    return (
      <div className="space-y-6">
        <div>
          {!embedded && <h1>Dev Deployments</h1>}
          <ErrorState
            title="Failed to load deployments"
            onRetry={() => {
              queryClient.invalidateQueries({ queryKey: ['deployments'] });
              queryClient.invalidateQueries({ queryKey: ['releases'] });
            }}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {!embedded && (
        <div className="flex items-center justify-between">
          <div>
            <h1>Dev Deployments</h1>
            <p className="text-gray-500 mt-1">Track model deployments in development environments</p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => refetch()} disabled={isFetching}>
              <RefreshCw className={`size-4 mr-2 ${isFetching ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
            <Button className="bg-blue-600 hover:bg-blue-700" onClick={() => setShowCreateDialog(true)}>
              <GitBranch className="size-4 mr-2" />
              New Deployment
            </Button>
          </div>
        </div>
      )}
      {embedded && (
        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={() => refetch()} disabled={isFetching}>
            <RefreshCw className={`size-4 mr-2 ${isFetching ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button className="bg-blue-600 hover:bg-blue-700" onClick={() => setShowCreateDialog(true)}>
            <GitBranch className="size-4 mr-2" />
            New Deployment
          </Button>
        </div>
      )}

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-500 mb-1">Total Deployments</div>
                <div className="text-3xl">{deploymentData?.total || 0}</div>
              </div>
              <GitBranch className="size-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-500 mb-1">Running</div>
                <div className="text-3xl text-green-600">
                  {Array.isArray(deployments) ? deployments.filter(d => d.status === 'running').length : 0}
                </div>
              </div>
              <Play className="size-8 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-500 mb-1">Failed</div>
                <div className="text-3xl text-red-600">
                  {Array.isArray(deployments) ? deployments.filter(d => d.status === 'failed').length : 0}
                </div>
              </div>
              <XCircle className="size-8 text-red-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-500 mb-1">Today</div>
                <div className="text-3xl">
                  {Array.isArray(deployments) ? deployments.filter(d =>
                    new Date(d.deployed_at).toDateString() === new Date().toDateString()
                  ).length : 0}
                </div>
              </div>
              <Calendar className="size-8 text-purple-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex gap-4">
            <div className="flex items-center gap-2">
              <Filter className="size-4 text-gray-500" />
              <span className="text-sm text-gray-600">Filters:</span>
            </div>
            <Select value={environmentFilter} onValueChange={handleEnvFilter}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Environment" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Environments</SelectItem>
                {environments.map(env => (
                  <SelectItem key={env} value={env}>{env}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={statusFilter} onValueChange={handleStatusFilter}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="running">Running</SelectItem>
                <SelectItem value="stopped">Stopped</SelectItem>
                <SelectItem value="success">Success</SelectItem>
                <SelectItem value="failed">Failed</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="deploying">Deploying</SelectItem>
                <SelectItem value="stopping">Stopping</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Timeline */}
      <div className="space-y-8">
        {Object.entries(deploymentsByDate).map(([date, deployments]) => (
          <div key={date}>
            <div className="flex items-center gap-3 mb-4">
              <Calendar className="size-5 text-gray-400" />
              <h2>{date}</h2>
              <div className="h-px flex-1 bg-gray-200" />
            </div>
            <div className="space-y-4">
              {deployments.map((deployment, index) => (
                <Card
                  key={deployment.id}
                  className="hover:shadow-md transition-shadow cursor-pointer"
                  onClick={() => navigate(`/deployments/${deployment.id}`)}
                >
                  <CardContent className="p-6">
                    <div className="flex items-start gap-4">
                      {/* Status Icon */}
                      <div className="relative">
                        <div className={`
                          size-12 rounded-full flex items-center justify-center
                          ${deployment.status === 'success' || deployment.status === 'running' ? 'bg-green-100' :
                            deployment.status === 'failed' ? 'bg-red-100' :
                            deployment.status === 'stopped' ? 'bg-gray-100' : 'bg-yellow-100'}
                        `}>
                          {deployment.status === 'success' ? (
                            <CheckCircle className="size-6 text-green-600" />
                          ) : deployment.status === 'running' ? (
                            <Play className="size-6 text-green-600" />
                          ) : deployment.status === 'stopped' ? (
                            <Square className="size-6 text-gray-600" />
                          ) : deployment.status === 'failed' ? (
                            <AlertCircle className="size-6 text-red-600" />
                          ) : (
                            <Clock className="size-6 text-yellow-600" />
                          )}
                        </div>
                        {index < deployments.length - 1 && (
                          <div className="absolute top-12 left-1/2 -translate-x-1/2 w-0.5 h-8 bg-gray-200" />
                        )}
                      </div>

                      {/* Deployment Info */}
                      <div className="flex-1">
                        <div className="flex items-start justify-between mb-2">
                          <div>
                            <div className="flex items-center gap-2 mb-1 flex-wrap">
                              <h3>{deployment.image_name}</h3>
                              <Badge variant="outline">v{deployment.release_version}</Badge>
                              <Badge variant={
                                deployment.environment === 'production' ? 'default' :
                                  deployment.environment === 'staging' ? 'secondary' : 'outline'
                              }>
                                {deployment.environment}
                              </Badge>
                              <Badge className={deploymentTypeColors[deployment.deployment_type] || 'bg-gray-100'}>
                                {deployment.deployment_type}
                              </Badge>
                              {deployment.host_port && (
                                <Badge variant="outline" className="text-blue-600 border-blue-300">
                                  <Globe className="size-3 mr-1" />
                                  :{deployment.host_port}
                                </Badge>
                              )}
                            </div>
                            <div className="flex items-center gap-4 text-sm text-gray-500">
                              <span className="flex items-center gap-1">
                                <Clock className="size-3" />
                                {getRelativeTime(deployment.deployed_at)}
                              </span>
                              <span>Deployed by {deployment.deployed_by}</span>
                              <span>{formatDate(deployment.deployed_at)}</span>
                            </div>
                          </div>
                          <Badge className={deploymentStatusColors[deployment.status] || 'bg-gray-100'}>
                            {deployment.status}
                          </Badge>
                        </div>

                        {/* Metadata */}
                        {deployment.metadata && Object.keys(deployment.metadata).length > 0 && (
                          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                            <div className="grid grid-cols-3 gap-4 text-sm">
                              {deployment.metadata.kubernetes_namespace && (
                                <div>
                                  <div className="text-gray-500 mb-1">Namespace</div>
                                  <code className="text-gray-700">
                                    {deployment.metadata.kubernetes_namespace}
                                  </code>
                                </div>
                              )}
                              {deployment.metadata.replicas && (
                                <div>
                                  <div className="text-gray-500 mb-1">Replicas</div>
                                  <div>{deployment.metadata.replicas}</div>
                                </div>
                              )}
                              {deployment.metadata.gpu_type && (
                                <div>
                                  <div className="text-gray-500 mb-1">GPU Type</div>
                                  <div>{deployment.metadata.gpu_type}</div>
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        ))}
      </div>

      {filteredDeployments.length === 0 && (
        <EmptyState
          icon={GitBranch}
          title="No deployments found"
          description="Try adjusting your filters or deploy a new release"
        />
      )}

      <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />

      {/* Create Deployment Dialog */}
      <Dialog
        open={showCreateDialog}
        onOpenChange={(open) => {
          setShowCreateDialog(open);
          if (!open) {
            setFormData({
              release_id: '',
              environment: 'development',
              kubernetes_namespace: '',
              replicas: '1',
              gpu_type: '',
            });
          }
        }}
      >
        <DialogContent className="max-h-[85vh] overflow-y-auto overflow-x-hidden sm:max-w-3xl">
          <DialogHeader>
            <DialogTitle>New Deployment</DialogTitle>
            <DialogDescription>
              Deploy a release to an environment
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={handleCreate}>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="release">Release *</Label>
                <Select
                  value={formData.release_id}
                  onValueChange={(value: string) => {
                    setFormData({ ...formData, release_id: value });
                  }}
                >
                  <SelectTrigger id="release">
                    <SelectValue placeholder="Select a release" />
                  </SelectTrigger>
                  <SelectContent>
                    {releases.map((release) => (
                      <SelectItem key={release.id} value={release.id}>
                        {release.model_name || release.image_name || 'Unknown'} v{release.version} ({release.tag})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="environment">Environment *</Label>
                <Select
                  value={formData.environment}
                  onValueChange={(value: string) => {
                    setFormData({ ...formData, environment: value });
                  }}
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
              <div className="space-y-2">
                    <Label htmlFor="kubernetes_namespace">Kubernetes Namespace</Label>
                    <Input
                      id="kubernetes_namespace"
                      placeholder="e.g., ml-models"
                      value={formData.kubernetes_namespace}
                      onChange={(e) => setFormData({ ...formData, kubernetes_namespace: e.target.value })}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="replicas">Replicas</Label>
                    <Input
                      id="replicas"
                      type="number"
                      min="1"
                      placeholder="1"
                      value={formData.replicas}
                      onChange={(e) => setFormData({ ...formData, replicas: e.target.value })}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="gpu_type">GPU Type</Label>
                    <Input
                      id="gpu_type"
                      placeholder="e.g., nvidia-t4"
                      value={formData.gpu_type}
                      onChange={(e) => setFormData({ ...formData, gpu_type: e.target.value })}
                    />
                  </div>
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setShowCreateDialog(false)}>
                Cancel
              </Button>
              <Button type="submit" disabled={submitDisabled}>
                {createMutation.isPending ? 'Deploying...' : 'Deploy'}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}
