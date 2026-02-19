import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { Package, Search, Clock, Tag } from 'lucide-react';
import { Card, CardContent } from './ui/card';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog';
import { Label } from './ui/label';
import { ErrorState } from './ErrorState';
import { LoadingState } from './LoadingState';
import { EmptyState } from './EmptyState';
import { Pagination } from './Pagination';
import { api } from '../lib/api';
import { formatRelativeDate } from '../lib/utils';
import { useDebouncedState } from '../hooks/useDebouncedState';
import type { Image } from '../lib/api';


export function ImageList() {
  const navigate = useNavigate();
  const [inputValue, searchQuery, setInputValue] = useDebouncedState('', 300);
  const [page, setPage] = useState(1);
  const pageSize = 12; // Grid layout 3x4

  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [formData, setFormData] = useState({ name: '', storage_path: '', description: '' });

  const queryClient = useQueryClient();

  const { data, isLoading: loading, isError } = useQuery({
    queryKey: ['images', page, searchQuery],
    queryFn: ({ signal }) => api.listImages({
      page,
      size: pageSize,
      search: searchQuery || undefined,
      signal, // Enable request cancellation
    }),
  });

  const images = data?.items || [];
  const total = data?.total || 0;
  const totalPages = data?.pages || 0;

  const createMutation = useMutation({
    mutationFn: (data: typeof formData) => api.createImage(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['images'] });
      setShowCreateDialog(false);
      setFormData({ name: '', storage_path: '', description: '' });
    },
    onError: (error) => {
      console.error('Failed to create image:', error);
    },
  });

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.name || !formData.storage_path) return;
    createMutation.mutate(formData);
  };

  // Handle search input and reset page
  const handleSearch = (val: string) => {
    setInputValue(val);
    setPage(1);
  };

  if (loading) {
    return <LoadingState title="Models" />;
  }

  if (isError) {
    return (
      <div className="space-y-6">
        <div>
          <h1>Models</h1>
          <ErrorState
            title="Failed to load models"
            onRetry={() => queryClient.invalidateQueries({ queryKey: ['images'] })}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1>Models</h1>
          <p className="text-gray-500 mt-1">Browse and manage your ML models</p>
        </div>
        <Button className="bg-blue-600 hover:bg-blue-700" onClick={() => setShowCreateDialog(true)}>
          <Package className="size-4 mr-2" />
          Register New Model
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
                placeholder="Search models by name or description..."
                value={inputValue}
                onChange={(e) => handleSearch(e.target.value)}
                className="pl-10"
                autoComplete="off"
              />
            </div>
            {/* Filter buttons logic removed for simplicity in pagination refactor or needs update? 
                Actually 'active' filter was clientside. Backend doesn't support status filter on models yet.
                For now, let's keep search. 
            */}
          </div>
        </CardContent>
      </Card>

      {/* Models Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {images.map((image) => (
          <Card
            key={image.id}
            className="hover:shadow-lg transition-shadow cursor-pointer"
            onClick={() => navigate(`/models/${image.id}`)}
          >
            <CardContent className="p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="size-12 bg-blue-100 rounded-lg flex items-center justify-center">
                  <Package className="size-6 text-blue-600" />
                </div>
                <Badge variant="secondary">
                  {image.version_count || 0} versions
                </Badge>
              </div>

              <h3 className="mb-2">{image.name}</h3>
              <p className="text-sm text-gray-500 mb-4 line-clamp-2 min-h-[40px]">
                {image.description || 'No description'}
              </p>

              <div className="pt-4 border-t flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <Clock className="size-3" />
                  Updated {formatRelativeDate(image.updated_at)}
                </div>
                <div className="flex items-center gap-1">
                  <Tag className="size-3 text-gray-400" />
                  <span className="text-xs text-gray-500 truncate max-w-[150px]">{image.storage_path}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />

      {images.length === 0 && (
        <EmptyState
          icon={Package}
          title="No models found"
          description={searchQuery ? 'Try adjusting your search' : 'Register your first model to get started'}
        />
      )}

      {/* Create Model Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Register New Model</DialogTitle>
            <DialogDescription>
              Add a new ML model to the registry
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={handleCreate}>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="name">Model Name *</Label>
                <Input
                  id="name"
                  placeholder="e.g., my-classifier"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="storage_path">Storage Path *</Label>
                <Input
                  id="storage_path"
                  placeholder="e.g., s3://ml-models/my-classifier/ or /mnt/ceph/models/my-classifier/"
                  value={formData.storage_path}
                  onChange={(e) => setFormData({ ...formData, storage_path: e.target.value })}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="description">Description</Label>
                <Input
                  id="description"
                  placeholder="Brief description of the model"
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                />
              </div>
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setShowCreateDialog(false)}>
                Cancel
              </Button>
              <Button type="submit" disabled={createMutation.isPending}>
                {createMutation.isPending ? 'Creating...' : 'Register Model'}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}
