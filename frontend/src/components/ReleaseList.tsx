import { useState, useMemo } from 'react';
import { useQuery, useQueries, useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { GitBranch, Search, Clock, Tag, Package, TrendingUp, Loader2, Trash2 } from 'lucide-react';
import { Card, CardContent } from './ui/card';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { api } from '../lib/api';
import { formatBytes, formatRelativeDate } from '../lib/utils';
import { CreateReleaseDialog } from './CreateReleaseDialog';
import { ErrorState } from './ErrorState';
import { LoadingState } from './LoadingState';
import { EmptyState } from './EmptyState';
import { Pagination } from './Pagination';
import { useDebouncedState } from '../hooks/useDebouncedState';
import type { Release, DockerBuild } from '../lib/api';

// Get cleanup status summary for a release's Docker builds
function getBuildCleanupSummary(builds: DockerBuild[]): { text: string; color: string } | null {
  if (!builds || builds.length === 0) return null;

  const successBuilds = builds.filter(b => b.status === 'success');
  if (successBuilds.length === 0) return null;

  // Check for builds pending cleanup
  const pendingCleanup = successBuilds.filter(b =>
    !b.is_current && !b.is_cleaned && b.superseded_at
  );

  if (pendingCleanup.length > 0) {
    // Find the one closest to cleanup
    const daysLeft = pendingCleanup
      .filter(b => b.days_until_cleanup !== null && b.days_until_cleanup !== undefined)
      .map(b => b.days_until_cleanup!);

    if (daysLeft.length > 0) {
      const minDays = Math.min(...daysLeft);
      if (minDays <= 0) {
        return { text: 'Cleanup pending', color: 'bg-orange-100 text-orange-700' };
      }
      return { text: `Cleanup in ${minDays}d`, color: 'bg-yellow-100 text-yellow-700' };
    }
  }

  return null;
}

export function ReleaseList() {
  const navigate = useNavigate();
  const [inputValue, searchQuery, setInputValue] = useDebouncedState('', 300);
  const [page, setPage] = useState(1);
  const pageSize = 10;

  const [selectedFilter, setSelectedFilter] = useState<'all' | 'active' | 'recent' | 'releases' | 'versions'>('releases');
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [promotingId, setPromotingId] = useState<string | null>(null);

  const queryClient = useQueryClient();

  // Handle search input and reset page
  const handleSearch = (val: string) => {
    setInputValue(val);
    setPage(1);
  };

  const handleFilterChange = (filter: 'all' | 'active' | 'recent' | 'releases' | 'versions') => {
    setSelectedFilter(filter);
    setPage(1);
  };

  // Calculate backend filters
  let isRelease: boolean | undefined = undefined;
  if (selectedFilter === 'releases') isRelease = true;
  if (selectedFilter === 'versions') isRelease = false;

  let statusFilter: string | undefined = undefined;
  if (selectedFilter === 'active') statusFilter = 'active';

  const { data, isLoading: loading, isError } = useQuery({
    queryKey: ['releases', page, searchQuery, selectedFilter],
    queryFn: () => api.listReleases({
      page,
      size: pageSize,
      is_release: isRelease,
      status: statusFilter,
      version: searchQuery || undefined,
    }),
  });

  const releases = data?.items || [];
  const totalPages = data?.pages || 0;

  // Fetch Docker builds for each release to show cleanup summary
  const buildQueries = useQueries({
    queries: releases.map(release => ({
      queryKey: ['dockerBuilds', release.id],
      queryFn: () => api.listDockerBuilds({ release_id: release.id, size: 20 }),
      staleTime: 30000, // Cache for 30 seconds to avoid excessive requests
    })),
  });

  // Map release IDs to their builds
  const buildsByReleaseId = useMemo(() => {
    const map: Record<string, DockerBuild[]> = {};
    releases.forEach((release, idx) => {
      const query = buildQueries[idx];
      if (query?.data?.items) {
        map[release.id] = query.data.items;
      }
    });
    return map;
  }, [releases, buildQueries]);

  const promoteMutation = useMutation({
    mutationFn: ({ id, isRelease }: { id: string; isRelease: boolean }) =>
      api.promoteRelease(id, isRelease),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['releases'] });
    },
  });

  const loadReleases = () => queryClient.invalidateQueries({ queryKey: ['releases'] });

  const handlePromote = async (release: Release) => {
    try {
      setPromotingId(release.id);
      await promoteMutation.mutateAsync({ id: release.id, isRelease: true });
    } catch (error) {
      console.error('Failed to promote release:', error);
    } finally {
      setPromotingId(null);
    }
  };

  const handleDemote = async (release: Release) => {
    try {
      setPromotingId(release.id);
      await promoteMutation.mutateAsync({ id: release.id, isRelease: false });
    } catch (error) {
      console.error('Failed to demote release:', error);
    } finally {
      setPromotingId(null);
    }
  };

  if (loading) {
    return <LoadingState title="Releases" />;
  }

  if (isError) {
    return (
      <div className="space-y-6">
        <div>
          <h1>Releases</h1>
          <ErrorState
            title="Failed to load releases"
            onRetry={() => queryClient.invalidateQueries({ queryKey: ['releases'] })}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1>Releases</h1>
          <p className="text-gray-500 mt-1">View all model releases and versions</p>
        </div>
        <Button className="bg-blue-600 hover:bg-blue-700" onClick={() => setShowCreateDialog(true)}>
          <GitBranch className="size-4 mr-2" />
          Create Release
        </Button>
      </div>

      {/* Search and Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-gray-400" />
              <Input
                type="text"
                placeholder="Search releases by version, tag, or model name..."
                value={inputValue}
                onChange={(e) => handleSearch(e.target.value)}
                className="pl-10"
                autoComplete="off"
              />
            </div>
            <div className="flex gap-2">
              <Button
                variant={selectedFilter === 'all' ? 'default' : 'outline'}
                onClick={() => handleFilterChange('all')}
              >
                All
              </Button>
              <Button
                variant={selectedFilter === 'releases' ? 'default' : 'outline'}
                onClick={() => handleFilterChange('releases')}
              >
                <GitBranch className="size-4 mr-2" />
                Releases
              </Button>
              <Button
                variant={selectedFilter === 'versions' ? 'default' : 'outline'}
                onClick={() => handleFilterChange('versions')}
              >
                <Clock className="size-4 mr-2" />
                Versions
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Releases List */}
      <div className="space-y-4">
        {releases.map((release) => (
          <Card
            key={release.id}
            className="hover:shadow-lg transition-shadow cursor-pointer overflow-hidden"
            onClick={() => navigate(`/releases/${release.id}`)}
          >
            <CardContent className="p-6">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-3 mb-2">
                    <div className={`size-10 rounded-lg flex items-center justify-center shrink-0 ${release.is_release ? 'bg-purple-100' : 'bg-gray-100'}`}>
                      <GitBranch className={`size-5 ${release.is_release ? 'text-purple-600' : 'text-gray-600'}`} />
                    </div>
                    <div className="min-w-0 flex-1 overflow-hidden">
                      <div className="flex items-center gap-2 flex-wrap">
                        <Badge variant="secondary" className={release.is_release ? "bg-purple-100 text-purple-700" : "bg-gray-100 text-gray-700"}>
                          {release.tag}
                        </Badge>
                        {release.is_release && (
                          <Badge className="bg-blue-100 text-blue-700 hover:bg-blue-200">
                            Official Release
                          </Badge>
                        )}
                        {release.status === 'active' && (
                          <Badge className="bg-green-100 text-green-700">
                            Active
                          </Badge>
                        )}
                        {(() => {
                          const builds = buildsByReleaseId[release.id];
                          const summary = getBuildCleanupSummary(builds || []);
                          return summary && (
                            <Badge className={summary.color}>
                              <Trash2 className="size-3 mr-1" />
                              {summary.text}
                            </Badge>
                          );
                        })()}
                      </div>
                      <code className="mt-1.5 block px-2 py-1 bg-gray-100 rounded text-sm font-semibold overflow-x-auto whitespace-nowrap max-h-[3.25rem] leading-relaxed">
                        {release.version}
                      </code>
                      <div className="flex items-center gap-2 text-sm text-gray-500 mt-1.5 flex-wrap">
                        <Package className="size-3 shrink-0" />
                        <span className="break-all">{release.image_name || 'Unknown Model'}</span>
                        {release.image_repository && (
                          <>
                            <span className="text-gray-300 shrink-0">•</span>
                            <Tag className="size-3 shrink-0" />
                            <span className="text-xs break-all">{release.image_repository}</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-4 gap-4 mt-4">
                    <div>
                      <p className="text-xs text-gray-500 mb-1">Platform</p>
                      <p className="text-sm font-medium">{release.platform}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500 mb-1">Size</p>
                      <p className="text-sm font-medium">{release.size_bytes ? formatBytes(release.size_bytes) : 'N/A'}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500 mb-1">Created</p>
                      <p className="text-sm font-medium">{formatRelativeDate(release.created_at)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500 mb-1">Digest</p>
                      <p className="text-sm font-mono text-gray-600">
                        {release.digest.substring(7, 19)}...
                      </p>
                    </div>
                  </div>

                  {release.metadata && Object.keys(release.metadata).length > 0 && (
                    <div className="mt-4 pt-4 border-t">
                      <p className="text-xs text-gray-500 mb-2">Metadata</p>
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(release.metadata)
                          .filter(([, value]) => value !== null && typeof value !== 'object')
                          .map(([key, value]) => {
                            const display = typeof value === 'number' && key === 'accuracy'
                              ? `${(value * 100).toFixed(1)}%`
                              : String(value);
                            return (
                              <div key={key} className="flex items-center gap-1 px-2 py-1 bg-gray-100 rounded text-xs">
                                {key === 'accuracy' && <TrendingUp className="size-3 text-green-600 shrink-0" />}
                                <span className="text-gray-600 shrink-0">{key}:</span>
                                <span className="font-medium break-all">
                                  {display}
                                </span>
                              </div>
                            );
                          })}
                      </div>
                    </div>
                  )}
                </div>
                <div className="shrink-0 flex flex-col gap-2">
                  {!release.is_release ? (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e: React.MouseEvent) => {
                        e.stopPropagation();
                        handlePromote(release);
                      }}
                    >
                      {promotingId === release.id && <Loader2 className="size-3 mr-1 animate-spin" />}
                      Promote to Release
                    </Button>
                  ) : (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e: React.MouseEvent) => {
                        e.stopPropagation();
                        handleDemote(release);
                      }}
                    >
                      {promotingId === release.id && <Loader2 className="size-3 mr-1 animate-spin" />}
                      Demote
                    </Button>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {releases.length === 0 && (
        <EmptyState
          icon={GitBranch}
          title="No releases found"
          description={searchQuery ? 'Try adjusting your search or filters' : 'No releases have been created yet'}
        />
      )}

      <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />

      {/* Create Release Dialog */}
      <CreateReleaseDialog
        open={showCreateDialog}
        onOpenChange={setShowCreateDialog}
        onSuccess={loadReleases}
      />
    </div>
  );
}
