import { useState, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Download, Trash2, Plus, Search, FileArchive, Lock } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog';
import { api } from '../lib/api';
import { formatBytes } from '../lib/utils';
import type { Artifact } from '../lib/api';
import { UploadArtifactForm } from './UploadArtifactForm';
import { ErrorState } from './ErrorState';

// Unified artifact item for display
interface UnifiedArtifact {
  id: string;
  name: string;
  path: string;
  artifact_type: string;
  size_bytes: number;
  platform?: string;
  python_version?: string;
  created_at?: string;
  modified_at?: number;
  uploaded_by?: string;
  image_name?: string;
  release_version?: string;
  source: 'uploaded' | 'vllm_wheels';
  readonly: boolean;
  // Original data for operations
  original?: Artifact;
}

export function ArtifactManagement() {
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [selectedArtifact, setSelectedArtifact] = useState<UnifiedArtifact | null>(null);
  const [page, setPage] = useState(1);
  const pageSize = 20;

  const queryClient = useQueryClient();

  // Query for uploaded artifacts
  const { data: artifactsData, isLoading: artifactsLoading, isError: artifactsError } = useQuery({
    queryKey: ['artifacts', typeFilter !== 'all' ? typeFilter : undefined, page],
    queryFn: () => {
      const params: any = { page, size: pageSize };
      if (typeFilter !== 'all') params.artifact_type = typeFilter;
      return api.listArtifacts(params);
    },
  });

  // Query for vLLM wheels (flat list)
  const { data: vllmData, isLoading: vllmLoading, isError: vllmError } = useQuery({
    queryKey: ['artifact-source-files', 'vllm_wheels'],
    queryFn: () => api.listArtifactSourceFiles('vllm_wheels'),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.deleteArtifact(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['artifacts'] });
      setShowDeleteDialog(false);
      setSelectedArtifact(null);
    },
  });

  const refreshArtifacts = () => queryClient.invalidateQueries({ queryKey: ['artifacts'] });

  // Convert uploaded artifacts to unified format
  const uploadedArtifacts: UnifiedArtifact[] = useMemo(() => {
    if (!artifactsData?.items) return [];
    return artifactsData.items.map((artifact): UnifiedArtifact => ({
      id: artifact.id,
      name: artifact.name,
      path: artifact.file_path,
      artifact_type: artifact.artifact_type,
      size_bytes: artifact.size_bytes,
      platform: artifact.platform,
      python_version: artifact.python_version || undefined,
      created_at: artifact.created_at,
      uploaded_by: artifact.uploaded_by,
      image_name: artifact.image_name,
      release_version: artifact.release_version,
      source: 'uploaded',
      readonly: false,
      original: artifact,
    }));
  }, [artifactsData?.items]);

  // Convert vLLM wheels to unified format
  const vllmArtifacts: UnifiedArtifact[] = useMemo(() => {
    if (!vllmData?.files) return [];
    return vllmData.files
      .filter(file => typeFilter === 'all' || file.file_type === typeFilter)
      .map((file): UnifiedArtifact => ({
        id: `vllm_${file.path}`,
        name: file.name,
        path: file.path,
        artifact_type: file.file_type,
        size_bytes: file.size_bytes,
        modified_at: file.modified_at,
        source: 'vllm_wheels',
        readonly: true,
      }));
  }, [vllmData?.files, typeFilter]);

  // Combine all artifacts
  const allArtifacts = useMemo(() => {
    return [...uploadedArtifacts, ...vllmArtifacts];
  }, [uploadedArtifacts, vllmArtifacts]);

  // Filter by search term
  const filteredArtifacts = useMemo(() => {
    if (!searchTerm.trim()) return allArtifacts;
    const term = searchTerm.toLowerCase();
    return allArtifacts.filter(artifact =>
      artifact.name.toLowerCase().includes(term) ||
      artifact.path.toLowerCase().includes(term) ||
      (artifact.image_name || '').toLowerCase().includes(term) ||
      (artifact.release_version || '').toLowerCase().includes(term)
    );
  }, [allArtifacts, searchTerm]);

  const handleDownload = async (artifact: UnifiedArtifact) => {
    try {
      let blob: Blob;
      if (artifact.source === 'uploaded' && artifact.original) {
        blob = await api.downloadArtifact(artifact.original.id);
      } else {
        blob = await api.downloadFromArtifactSource('vllm_wheels', artifact.path);
      }
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = artifact.name;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Failed to download artifact:', error);
    }
  };

  const handleDelete = async () => {
    if (!selectedArtifact || !selectedArtifact.original) return;
    try {
      await deleteMutation.mutateAsync(selectedArtifact.original.id);
    } catch (error) {
      console.error('Failed to delete artifact:', error);
    }
  };

  const getArtifactIcon = (type: string) => {
    switch (type) {
      case 'wheel': return '🎡';
      case 'sdist': return '📦';
      case 'tarball': return '🗜️';
      case 'binary': return '⚙️';
      default: return '📄';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'wheel': return 'bg-blue-100 text-blue-800';
      case 'sdist': return 'bg-green-100 text-green-800';
      case 'tarball': return 'bg-purple-100 text-purple-800';
      case 'binary': return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const formatDate = (artifact: UnifiedArtifact) => {
    if (artifact.created_at) {
      return new Date(artifact.created_at).toLocaleDateString();
    }
    if (artifact.modified_at) {
      return new Date(artifact.modified_at * 1000).toLocaleDateString();
    }
    return 'Unknown';
  };

  const artifactTypes = ['all', 'wheel', 'sdist', 'tarball', 'binary'];

  const isLoading = artifactsLoading || vllmLoading;
  const isError = artifactsError || vllmError;

  // Calculate stats
  const totalUploaded = artifactsData?.total || 0;
  const totalVllm = vllmData?.total || 0;
  const totalSize = allArtifacts.reduce((sum, a) => sum + a.size_bytes, 0);
  const wheelCount = allArtifacts.filter(a => a.artifact_type === 'wheel').length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Artifact Management</h2>
          <p className="text-gray-500 mt-1">
            Manage prebuilt wheels, source distributions, and other build artifacts
          </p>
        </div>
        <Button onClick={() => setShowCreateDialog(true)}>
          <Plus className="mr-2 size-4" />
          Add Artifact
        </Button>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between flex-wrap gap-3">
            {/* Search */}
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 size-4 text-gray-400" />
              <Input
                type="text"
                placeholder="Search artifacts by name or path..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
                autoComplete="off"
              />
            </div>

            {/* Type filter */}
            <div className="flex gap-2">
              <select
                value={typeFilter}
                onChange={(e) => {
                  setTypeFilter(e.target.value);
                  setPage(1);
                }}
                className="px-3 py-2 border rounded-lg text-sm"
              >
                {artifactTypes.map(type => (
                  <option key={type} value={type}>
                    {type === 'all' ? 'All Types' : type.charAt(0).toUpperCase() + type.slice(1)}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          {isLoading ? (
            <div className="text-center py-12 text-gray-500">Loading...</div>
          ) : isError ? (
            <ErrorState
              title="Failed to load artifacts"
              onRetry={() => {
                queryClient.invalidateQueries({ queryKey: ['artifacts'] });
                queryClient.invalidateQueries({ queryKey: ['artifact-source-files'] });
              }}
            />
          ) : filteredArtifacts.length === 0 ? (
            <div className="text-center py-12">
              <FileArchive className="mx-auto size-12 text-gray-400 mb-4" />
              <p className="text-gray-500">No artifacts found</p>
              <Button
                variant="outline"
                className="mt-4"
                onClick={() => setShowCreateDialog(true)}
              >
                Upload your first artifact
              </Button>
            </div>
          ) : (
            <div className="space-y-2">
              {filteredArtifacts.map((artifact) => (
                <div
                  key={artifact.id}
                  className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-start gap-3 flex-1 min-w-0">
                    <div className="text-2xl flex-shrink-0">{getArtifactIcon(artifact.artifact_type)}</div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1 flex-wrap">
                        <h3 className="font-semibold truncate">{artifact.name}</h3>
                        <Badge className={getTypeColor(artifact.artifact_type)}>
                          {artifact.artifact_type}
                        </Badge>
                        {artifact.readonly && (
                          <Badge variant="secondary" className="flex items-center gap-1">
                            <Lock className="size-3" />
                            RO
                          </Badge>
                        )}
                        {artifact.platform && (
                          <Badge variant="outline">{artifact.platform}</Badge>
                        )}
                        {artifact.python_version && (
                          <Badge variant="outline">py{artifact.python_version}</Badge>
                        )}
                      </div>
                      <div className="text-sm text-gray-500 space-y-0.5">
                        {artifact.readonly && (
                          <div className="font-mono text-xs text-gray-400 truncate" title={artifact.path}>
                            {artifact.path}
                          </div>
                        )}
                        <div className="flex items-center gap-3 flex-wrap">
                          <span>{formatBytes(artifact.size_bytes)}</span>
                          <span>•</span>
                          <span>{formatDate(artifact)}</span>
                          {artifact.image_name && artifact.release_version && (
                            <>
                              <span>•</span>
                              <span>{artifact.image_name} @ {artifact.release_version}</span>
                            </>
                          )}
                          {artifact.uploaded_by && (
                            <>
                              <span>•</span>
                              <span>by {artifact.uploaded_by}</span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="flex gap-2 flex-shrink-0 ml-2">
                    <Button variant="outline" size="sm" onClick={() => handleDownload(artifact)}>
                      <Download className="size-4" />
                    </Button>
                    {!artifact.readonly && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          setSelectedArtifact(artifact);
                          setShowDeleteDialog(true);
                        }}
                      >
                        <Trash2 className="size-4 text-red-600" />
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Summary */}
          {!isLoading && !isError && filteredArtifacts.length > 0 && (
            <div className="mt-4 pt-4 border-t text-sm text-gray-500">
              Showing {filteredArtifacts.length} artifacts
              {searchTerm && ` matching "${searchTerm}"`}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Statistics */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardDescription>Uploaded</CardDescription>
            <CardTitle className="text-3xl">{totalUploaded}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardDescription>vLLM Wheels</CardDescription>
            <CardTitle className="text-3xl flex items-center gap-2">
              {totalVllm}
              <Lock className="size-4 text-gray-400" />
            </CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardDescription>Total Wheels</CardDescription>
            <CardTitle className="text-3xl">{wheelCount}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardDescription>Total Size</CardDescription>
            <CardTitle className="text-3xl">{formatBytes(totalSize)}</CardTitle>
          </CardHeader>
        </Card>
      </div>

      {/* Delete Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Artifact</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete <strong>{selectedArtifact?.name}</strong>?
              This will move the artifact to the trash. It will be permanently deleted after 30 days.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowDeleteDialog(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Upload Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Upload Artifact</DialogTitle>
            <DialogDescription>
              Upload a new build artifact (wheel, source distribution, etc.)
            </DialogDescription>
          </DialogHeader>
          <UploadArtifactForm
            onSuccess={() => {
              refreshArtifacts();
              setShowCreateDialog(false);
            }}
            onCancel={() => setShowCreateDialog(false)}
          />
        </DialogContent>
      </Dialog>
    </div>
  );
}
