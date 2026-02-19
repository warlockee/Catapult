import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Search, FileArchive, Lock, Check, Loader2 } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from './ui/dialog';
import { ScrollArea } from './ui/scroll-area';
import { api } from '../lib/api';
import { formatBytes } from '../lib/utils';
import type { Artifact } from '../lib/api';

// Unified artifact item for selection
interface SelectableArtifact {
  id: string;
  name: string;
  path: string;
  artifact_type: string;
  size_bytes: number;
  source: 'uploaded' | 'vllm_wheels';
  readonly: boolean;
  // For uploaded artifacts, store the original
  original?: Artifact;
}

interface ArtifactPickerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelect: (artifact: SelectableArtifact) => void;
  selectedIds?: string[];
  releaseId?: string;
  modelId?: string;
}

export function ArtifactPicker({
  open,
  onOpenChange,
  onSelect,
  selectedIds = [],
  releaseId,
  modelId,
}: ArtifactPickerProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState<string>('all');

  // Query for uploaded artifacts
  const { data: artifactsData, isLoading: artifactsLoading, error: artifactsError } = useQuery({
    queryKey: ['artifacts-picker', releaseId],
    queryFn: () => api.listArtifacts({ size: 100 }),
    enabled: open,
  });

  // Query for vLLM wheels (flat list)
  const { data: vllmData, isLoading: vllmLoading, error: vllmError } = useQuery({
    queryKey: ['artifact-source-files', 'vllm_wheels'],
    queryFn: () => api.listArtifactSourceFiles('vllm_wheels'),
    enabled: open,
  });

  // Convert uploaded artifacts to unified format
  const uploadedArtifacts: SelectableArtifact[] = useMemo(() => {
    if (!artifactsData?.items) return [];
    // Filter for relevant artifacts (same release or model, or orphaned)
    return artifactsData.items
      .filter(a =>
        !releaseId ||
        a.release_id === releaseId ||
        (modelId && a.model_id === modelId) ||
        !a.model_id
      )
      .map((artifact): SelectableArtifact => ({
        id: artifact.id,
        name: artifact.name,
        path: artifact.file_path,
        artifact_type: artifact.artifact_type,
        size_bytes: artifact.size_bytes,
        source: 'uploaded',
        readonly: false,
        original: artifact,
      }));
  }, [artifactsData?.items, releaseId, modelId]);

  // Convert vLLM wheels to unified format
  const vllmArtifacts: SelectableArtifact[] = useMemo(() => {
    if (!vllmData?.files) return [];
    return vllmData.files.map((file): SelectableArtifact => ({
      id: `vllm_${file.path}`,
      name: file.name,
      path: `/fsx/vllm_wheels_prebuilt/${file.path}`,
      artifact_type: file.file_type,
      size_bytes: file.size_bytes,
      source: 'vllm_wheels',
      readonly: true,
    }));
  }, [vllmData?.files]);

  // Combine and filter
  const allArtifacts = useMemo(() => {
    let combined = [...uploadedArtifacts, ...vllmArtifacts];

    // Filter by type
    if (typeFilter !== 'all') {
      combined = combined.filter(a => a.artifact_type === typeFilter);
    }

    // Filter by search
    if (searchTerm.trim()) {
      const term = searchTerm.toLowerCase();
      combined = combined.filter(a =>
        a.name.toLowerCase().includes(term) ||
        a.path.toLowerCase().includes(term)
      );
    }

    return combined;
  }, [uploadedArtifacts, vllmArtifacts, typeFilter, searchTerm]);

  const isLoading = artifactsLoading || vllmLoading;

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'wheel': return 'bg-blue-100 text-blue-800';
      case 'sdist': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>Select Artifact</DialogTitle>
          <DialogDescription>
            Choose an artifact for your Docker build. Includes uploaded artifacts and vLLM prebuilt wheels.
          </DialogDescription>
        </DialogHeader>

        {/* Search and filters */}
        <div className="flex items-center gap-3 py-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 size-4 text-gray-400" />
            <Input
              type="text"
              placeholder="Search by name or path..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
              autoComplete="off"
            />
          </div>
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className="px-3 py-2 border rounded-lg text-sm"
          >
            <option value="all">All Types</option>
            <option value="wheel">Wheels</option>
            <option value="sdist">Source Dists</option>
          </select>
        </div>

        {/* Artifact list */}
        <ScrollArea className="flex-1 -mx-6 px-6">
          {(artifactsError || vllmError) ? (
            <div className="text-center py-12 text-red-500">
              <p>Error loading artifacts</p>
              <p className="text-sm mt-2">{String(artifactsError || vllmError)}</p>
            </div>
          ) : isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="size-6 animate-spin text-gray-400" />
            </div>
          ) : allArtifacts.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <FileArchive className="mx-auto size-10 text-gray-300 mb-3" />
              <p>No artifacts found</p>
              <p className="text-xs mt-2 text-gray-400">
                Uploaded: {artifactsData?.items?.length || 0}, vLLM: {vllmData?.files?.length || 0}
              </p>
            </div>
          ) : (
            <div className="space-y-1 py-2">
              {allArtifacts.map((artifact) => {
                const isSelected = selectedIds.includes(artifact.id);
                return (
                  <div
                    key={artifact.id}
                    className={`
                      flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors
                      ${isSelected ? 'bg-blue-50 border-blue-200 border' : 'hover:bg-gray-50 border border-transparent'}
                    `}
                    onClick={() => {
                      onSelect(artifact);
                      onOpenChange(false);
                    }}
                  >
                    <div className="flex items-start gap-3 flex-1 min-w-0">
                      <div className="text-xl flex-shrink-0">
                        {artifact.artifact_type === 'wheel' ? '🎡' : '📦'}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-0.5 flex-wrap">
                          <span className="font-medium truncate">{artifact.name}</span>
                          <Badge className={`text-xs ${getTypeColor(artifact.artifact_type)}`}>
                            {artifact.artifact_type}
                          </Badge>
                          {artifact.readonly && (
                            <Badge variant="secondary" className="flex items-center gap-1 text-xs">
                              <Lock className="size-3" />
                              vLLM
                            </Badge>
                          )}
                        </div>
                        <div className="font-mono text-xs text-gray-400 truncate" title={artifact.path}>
                          {artifact.path}
                        </div>
                        <div className="text-xs text-gray-500 mt-0.5">
                          {formatBytes(artifact.size_bytes)}
                        </div>
                      </div>
                    </div>
                    {isSelected && (
                      <Check className="size-5 text-blue-600 flex-shrink-0 ml-2" />
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </ScrollArea>

        <DialogFooter className="pt-2 border-t">
          <div className="flex items-center justify-between w-full">
            <span className="text-sm text-gray-500">
              {allArtifacts.length} artifact{allArtifacts.length !== 1 ? 's' : ''} available
            </span>
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
