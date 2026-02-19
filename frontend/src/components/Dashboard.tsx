import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { TrendingUp, Package, GitBranch, Activity, ArrowRight, HardDrive } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Skeleton } from './ui/skeleton';
import { ErrorState } from './ErrorState';
import { api } from '../lib/api';
import { formatBytes } from '../lib/utils';
import { needsStatusPolling } from '../lib/deployment-utils';
import type { Image, Release, Deployment, DockerBuild } from '../lib/api';

export function Dashboard() {
  const navigate = useNavigate();

  // Reduced fetch sizes: we only display top 5 items in lists/charts
  // The paginated response still returns the total count for stats
  const { data: imagesData, isLoading: imagesLoading, isError: imagesError } = useQuery({
    queryKey: ['images', 'dashboard'],
    queryFn: () => api.listImages({ size: 20 }),
    refetchInterval: 30000, // Auto-refresh every 30 seconds
  });

  const images = imagesData?.items || [];

  const { data: versionsData, isLoading: versionsLoading, isError: versionsError } = useQuery({
    queryKey: ['versions', 'dashboard'],
    queryFn: () => api.listVersions({ size: 20 }),
    refetchInterval: 30000, // Auto-refresh every 30 seconds
  });
  const versions = versionsData?.items || [];
  // Backward compatibility alias
  const releases = versions;
  const releasesData = versionsData;
  const releasesLoading = versionsLoading;
  const releasesError = versionsError;

  const { data: deploymentsData, isLoading: deploymentsLoading, isError: deploymentsError } = useQuery({
    queryKey: ['deployments', 'dashboard'],
    queryFn: () => api.listDeployments({ size: 20 }),
    refetchInterval: (query) => {
      // Poll faster when any deployment needs status polling, otherwise regular interval
      const items = query.state.data?.items;
      if (items?.some((d: Deployment) => needsStatusPolling(d.status))) {
        return 5000; // 5 seconds for active deployments
      }
      return 30000; // 30 seconds otherwise
    },
  });
  const deployments = deploymentsData?.items || [];

  // Dedicated query for Active Deployments count
  const { data: activeDeploymentsData, isLoading: activeLoading } = useQuery({
    queryKey: ['deployments', 'active-count'],
    queryFn: () => api.listDeployments({ status: 'running', size: 1 }),
    refetchInterval: (query) => {
      // Poll faster when dashboard deployments are actively changing, otherwise regular interval
      return deploymentsData?.items?.some((d: Deployment) => needsStatusPolling(d.status)) ? 5000 : 30000;
    },
  });

  const { data: storageStats, isLoading: storageLoading, isError: storageError } = useQuery({
    queryKey: ['storage'],
    queryFn: () => api.getSystemStorage(),
    refetchInterval: 60000, // Auto-refresh every 60 seconds
  });

  const { data: dockerDiskUsage, isLoading: dockerDiskLoading } = useQuery({
    queryKey: ['docker-disk-usage'],
    queryFn: () => api.getDockerDiskUsage(),
    refetchInterval: 60000, // Refresh every minute
  });

  const { data: dockerBuildsData, isLoading: dockerBuildsLoading } = useQuery({
    queryKey: ['docker-builds', 'dashboard'],
    queryFn: () => api.listDockerBuilds({ size: 10 }),
    refetchInterval: (query) => {
      // Poll faster when there are active builds, otherwise regular interval
      const items = query.state.data?.items;
      if (items?.some((b: DockerBuild) => ['pending', 'building'].includes(b.status))) {
        return 5000; // 5 seconds for active builds
      }
      return 30000; // 30 seconds otherwise
    },
  });
  const dockerBuilds = dockerBuildsData?.items || [];

  const loading = imagesLoading || releasesLoading || deploymentsLoading || storageLoading || activeLoading;

  // Calculate stats
  // Use independent total counters if possible, otherwise fallback to existing logic (which might be limited by page size)
  // For images, versions, deployments, the PaginatedResponse gives us the REAL total.
  const totalImages = imagesData?.total || 0;
  const totalVersions = versionsData?.total || 0;
  const totalDeployments = deploymentsData?.total || 0;
  const activeDeployments = activeDeploymentsData?.total || 0;
  // Backward compatibility alias
  const totalReleases = totalVersions;

  // Calculate trends (last 7 days) - memoized to avoid recalculation on every render
  const activityTrendData = useMemo(() => {
    const days = 7;
    const data = [];
    const now = new Date();

    for (let i = days - 1; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

      // Count deployments for this day
      const dayDeployments = Array.isArray(deployments) ? deployments.filter(d => {
        const dDate = new Date(d.deployed_at);
        return dDate.getDate() === date.getDate() && dDate.getMonth() === date.getMonth();
      }).length : 0;

      // Count releases for this day
      const dayReleases = Array.isArray(releases) ? releases.filter(r => {
        const rDate = new Date(r.created_at);
        return rDate.getDate() === date.getDate() && rDate.getMonth() === date.getMonth();
      }).length : 0;

      // Count docker builds for this day
      const dayBuilds = Array.isArray(dockerBuilds) ? dockerBuilds.filter(b => {
        const bDate = new Date(b.created_at);
        return bDate.getDate() === date.getDate() && bDate.getMonth() === date.getMonth();
      }).length : 0;

      data.push({
        date: dateStr,
        deployments: dayDeployments,
        releases: dayReleases,
        builds: dayBuilds,
      });
    }
    return data;
  }, [deployments, releases, dockerBuilds]);

  // Recent docker builds - parse model name and version from image_tag
  const recentDockerBuilds = useMemo(() => {
    return [...dockerBuilds]
      .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
      .slice(0, 5)
      .map(build => {
        // image_tag format is "model-name:version"
        const [model_name, version] = build.image_tag.split(':');
        return {
          ...build,
          model_name: model_name || 'Unknown',
          version: version || 'Unknown',
        };
      });
  }, [dockerBuilds]);

  // Get recent releases with image names - memoized
  const recentReleases = useMemo(() =>
    [...releases]
      .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
      .slice(0, 5)
      .map(release => {
        const image = images.find(i => i.id === release.image_id);
        return {
          ...release,
          image_name: release.image_name || image?.name || 'Unknown'
        };
      }),
    [releases, images]);

  // Get recent deployments with image names - memoized
  const recentDeployments = useMemo(() =>
    [...deployments]
      .sort((a, b) => new Date(b.deployed_at).getTime() - new Date(a.deployed_at).getTime())
      .slice(0, 5)
      .map(deployment => {
        const release = releases.find(r => r.id === deployment.release_id);
        const image = images.find(i => i.id === release?.image_id);
        return {
          ...deployment,
          image_name: deployment.image_name || image?.name || 'Unknown',
          release_version: deployment.release_version || release?.version || 'Unknown'
        };
      }),
    [deployments, releases, images]);

  // Top models by version count - memoized
  const topModels = useMemo(() =>
    [...images]
      .map(image => ({
        ...image,
        version_count: image.version_count || (Array.isArray(versions) ? versions.filter(v => v.image_id === image.id).length : 0),
        latest_version: Array.isArray(versions) ? versions
          .filter(v => v.image_id === image.id)
          .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())[0]?.version || 'N/A' : 'N/A'
      }))
      .sort((a, b) => (b.version_count || 0) - (a.version_count || 0))
      .slice(0, 3),
    [images, versions]);

  const isError = imagesError || releasesError || deploymentsError || storageError;

  if (isError) {
    return (
      <div className="space-y-6">
        <div>
          <h1>Dashboard</h1>
          <ErrorState
            title="Failed to load dashboard data"
            onRetry={() => window.location.reload()}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <h1>Dashboard</h1>
        <p className="text-gray-500 mt-1">Overview of your model registry</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm text-gray-500">Total Models</CardTitle>
            <Package className="size-4 text-gray-400" />
          </CardHeader>
          <CardContent>
            {imagesLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="flex items-baseline gap-2">
                <div className="text-3xl">{totalImages}</div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm text-gray-500">Total Versions</CardTitle>
            <GitBranch className="size-4 text-gray-400" />
          </CardHeader>
          <CardContent>
            {versionsLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="flex items-baseline gap-2">
                <div className="text-3xl">{totalVersions}</div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm text-gray-500">Active Deployments</CardTitle>
            <Activity className="size-4 text-gray-400" />
          </CardHeader>
          <CardContent>
            {deploymentsLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="flex items-baseline gap-2">
                <div className="text-3xl">{activeDeployments}</div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm text-gray-500">Storage Used</CardTitle>
            <Package className="size-4 text-gray-400" />
          </CardHeader>
          <CardContent>
            {storageLoading ? (
              <Skeleton className="h-8 w-32" />
            ) : (
              <div className="flex items-baseline gap-2">
                <div className="text-3xl">
                  {storageStats
                    ? formatBytes(storageStats.used)
                    : '0 Bytes'}
                </div>
                <span className="text-sm text-gray-500">
                  of {storageStats ? formatBytes(storageStats.total) : '0 Bytes'}
                </span>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm text-gray-500">Docker Disk</CardTitle>
            <HardDrive className="size-4 text-gray-400" />
          </CardHeader>
          <CardContent>
            {dockerDiskLoading ? (
              <Skeleton className="h-8 w-32" />
            ) : dockerDiskUsage ? (
              <div className="flex items-baseline gap-2">
                <div className="text-3xl">{dockerDiskUsage.total_docker_human}</div>
                <span className="text-sm text-gray-500">
                  ({dockerDiskUsage.disk_available_human} free)
                </span>
              </div>
            ) : (
              <div className="text-sm text-gray-500">N/A</div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Activity Trends (7 Days)</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <Skeleton className="h-[300px] w-full" />
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={activityTrendData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200" />
                  <XAxis dataKey="date" className="text-xs" />
                  <YAxis className="text-xs" allowDecimals={false} />
                  <Tooltip />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="deployments"
                    stackId="1"
                    stroke="#3b82f6"
                    fill="#3b82f6"
                    fillOpacity={0.6}
                    name="Deployments"
                  />
                  <Area
                    type="monotone"
                    dataKey="releases"
                    stackId="1"
                    stroke="#8b5cf6"
                    fill="#8b5cf6"
                    fillOpacity={0.6}
                    name="Releases"
                  />
                  <Area
                    type="monotone"
                    dataKey="builds"
                    stackId="1"
                    stroke="#10b981"
                    fill="#10b981"
                    fillOpacity={0.6}
                    name="Docker Builds"
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Deployments</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {deploymentsLoading ? (
                Array(5).fill(0).map((_, i) => (
                  <div key={i} className="flex items-center gap-4">
                    <Skeleton className="h-10 w-full" />
                  </div>
                ))
              ) : recentDeployments.length > 0 ? (
                recentDeployments.map((deployment) => (
                  <div
                    key={deployment.id}
                    className="flex items-center justify-between py-2 border-b last:border-0 cursor-pointer hover:bg-gray-50 -mx-2 px-2 rounded"
                    onClick={() => navigate(`/deployments/${deployment.id}`)}
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span>{deployment.image_name}</span>
                        <span className={`
                          px-2 py-0.5 text-xs rounded
                          ${deployment.environment === 'production'
                            ? 'bg-green-100 text-green-700'
                            : 'bg-yellow-100 text-yellow-700'
                          }
                        `}>
                          {deployment.environment}
                        </span>
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {new Date(deployment.deployed_at).toLocaleString()}
                      </div>
                    </div>
                    <div className={`
                      size-2 rounded-full
                      ${deployment.status === 'running' || deployment.status === 'success' ? 'bg-green-500' :
                        deployment.status === 'stopped' ? 'bg-gray-400' :
                        ['pending', 'deploying', 'stopping'].includes(deployment.status) ? 'bg-yellow-500' :
                        'bg-red-500'}
                    `} />
                  </div>
                ))
              ) : (
                <div className="h-[200px] flex items-center justify-center text-gray-500">
                  No deployments yet
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Recent Releases</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {releasesLoading ? (
                Array(3).fill(0).map((_, i) => (
                  <div key={i} className="flex items-center gap-4">
                    <Skeleton className="h-10 w-full" />
                  </div>
                ))
              ) : recentReleases.length > 0 ? (
                recentReleases.map((release) => (
                  <div
                    key={release.id}
                    className="flex items-center justify-between py-2 border-b last:border-0 cursor-pointer hover:bg-gray-50 -mx-2 px-2 rounded"
                    onClick={() => navigate(`/releases/${release.id}`)}
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span>{release.image_name}</span>
                        <span className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded">
                          {release.version}
                        </span>
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {new Date(release.created_at).toLocaleString()}
                      </div>
                    </div>
                    <div className="text-xs text-gray-500">
                      {release.metadata?.accuracy && `${(release.metadata.accuracy * 100).toFixed(1)}% acc`}
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-4 text-gray-500">No releases yet</div>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Docker Builds</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {dockerBuildsLoading ? (
                Array(3).fill(0).map((_, i) => (
                  <div key={i} className="flex items-center gap-4">
                    <Skeleton className="h-10 w-full" />
                  </div>
                ))
              ) : recentDockerBuilds.length > 0 ? (
                recentDockerBuilds.map((build) => (
                  <div
                    key={build.id}
                    className="flex items-center justify-between py-2 border-b last:border-0 cursor-pointer hover:bg-gray-50 -mx-2 px-2 rounded"
                    onClick={() => navigate(`/releases/${build.release_id}`)}
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span>{build.model_name}</span>
                        <span className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded">
                          {build.version}
                        </span>
                        <span className={`
                          px-2 py-0.5 text-xs rounded
                          ${build.build_type === 'organic' ? 'bg-green-100 text-green-700' :
                            build.build_type === 'optimized' ? 'bg-purple-100 text-purple-700' :
                            'bg-gray-100 text-gray-700'}
                        `}>
                          {build.build_type}
                        </span>
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {new Date(build.created_at).toLocaleString()}
                      </div>
                    </div>
                    <div className={`
                      px-2 py-0.5 text-xs rounded
                      ${build.status === 'success' ? 'bg-green-100 text-green-700' :
                        build.status === 'building' ? 'bg-yellow-100 text-yellow-700' :
                        build.status === 'pending' ? 'bg-gray-100 text-gray-700' :
                        build.status === 'failed' ? 'bg-red-100 text-red-700' :
                        'bg-gray-100 text-gray-700'}
                    `}>
                      {build.status}
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-4 text-gray-500">No docker builds yet</div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Top Models</CardTitle>
        </CardHeader>
        <CardContent>
          {imagesLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {Array(3).fill(0).map((_, i) => (
                <Skeleton key={i} className="h-24" />
              ))}
            </div>
          ) : topModels.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {topModels.map((image) => (
                <div
                  key={image.id}
                  className="p-4 border rounded-lg hover:border-blue-500 hover:shadow-md transition-all cursor-pointer"
                  onClick={() => navigate(`/models/${image.id}`)}
                >
                  <div className="flex items-start justify-between mb-2">
                    <Package className="size-8 text-blue-600" />
                    <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                      {image.version_count} versions
                    </span>
                  </div>
                  <h3 className="mb-1">{image.name}</h3>
                  <p className="text-sm text-gray-500 mb-3 line-clamp-2">{image.description || 'No description'}</p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">v{image.latest_version}</span>
                    <ArrowRight className="size-4 text-blue-600" />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-4 text-gray-500">No models yet</div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
