import { useState, useEffect } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Button } from './ui/button';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog';
import { Label } from './ui/label';
import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Checkbox } from './ui/checkbox';
import { api } from '../lib/api';

interface CreateReleaseDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    onSuccess: () => void;
    preselectedModelId?: string;
}

export function CreateReleaseDialog({ open, onOpenChange, onSuccess, preselectedModelId }: CreateReleaseDialogProps) {
    const [formData, setFormData] = useState({
        image_id: '',
        version: '',
        tag: '',
        digest: '',
        platform: 'linux/amd64',
    });
    const [autoBuild, setAutoBuild] = useState(true);

    // Use slim endpoint to reduce data transfer for dropdown
    const { data: images = [] } = useQuery({
        queryKey: ['models', 'options'],
        queryFn: ({ signal }) => api.listImageOptions(signal),
        enabled: open,
    });

    useEffect(() => {
        if (open) {
            // Reset form when opening, but keep preselected model if available
            setFormData(prev => ({
                ...prev,
                image_id: preselectedModelId || prev.image_id || '',
                version: '',
                tag: '',
                digest: '',
                platform: 'linux/amd64',
            }));
        }
    }, [open, preselectedModelId]);

    const createMutation = useMutation({
        mutationFn: (data: Parameters<typeof api.createRelease>[0]) => api.createRelease(data),
        onSuccess: () => {
            onOpenChange(false);
            onSuccess();
        },
        onError: (error) => {
            console.error('Failed to create release:', error);
        },
    });

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!formData.image_id || !formData.version || !formData.tag || !formData.digest) return;

        createMutation.mutate({
            ...formData,
            metadata: {},
            auto_build: autoBuild,
        });
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent>
                <DialogHeader>
                    <DialogTitle>Create Release</DialogTitle>
                    <DialogDescription>
                        Create a new release for a model
                    </DialogDescription>
                </DialogHeader>
                <form onSubmit={handleSubmit}>
                    <div className="space-y-4 py-4">
                        <div className="space-y-2">
                            <Label htmlFor="image">Model *</Label>
                            <Select
                                value={formData.image_id}
                                onValueChange={(value: string) => setFormData({ ...formData, image_id: value })}
                                disabled={!!preselectedModelId}
                            >
                                <SelectTrigger id="image">
                                    <SelectValue placeholder="Select a model" />
                                </SelectTrigger>
                                <SelectContent>
                                    {images.map((image) => (
                                        <SelectItem key={image.id} value={image.id}>
                                            {image.name}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="version">Version *</Label>
                            <Input
                                id="version"
                                placeholder="e.g., 1.0.0"
                                value={formData.version}
                                onChange={(e) => setFormData({ ...formData, version: e.target.value })}
                                required
                            />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="tag">Tag *</Label>
                            <Input
                                id="tag"
                                placeholder="e.g., latest, stable, v1.0.0"
                                value={formData.tag}
                                onChange={(e) => setFormData({ ...formData, tag: e.target.value })}
                                required
                            />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="digest">Digest *</Label>
                            <Input
                                id="digest"
                                placeholder="e.g., sha256:abc123..."
                                value={formData.digest}
                                onChange={(e) => setFormData({ ...formData, digest: e.target.value })}
                                required
                            />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="platform">Platform</Label>
                            <Select value={formData.platform} onValueChange={(value: string) => setFormData({ ...formData, platform: value })}>
                                <SelectTrigger id="platform">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="linux/amd64">linux/amd64</SelectItem>
                                    <SelectItem value="linux/arm64">linux/arm64</SelectItem>
                                    <SelectItem value="darwin/amd64">darwin/amd64</SelectItem>
                                    <SelectItem value="darwin/arm64">darwin/arm64</SelectItem>
                                    <SelectItem value="windows/amd64">windows/amd64</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                    </div>
                    <div className="flex items-center space-x-2 pt-2">
                        <Checkbox
                            id="autoBuild"
                            checked={autoBuild}
                            onCheckedChange={(checked: boolean | 'indeterminate') => setAutoBuild(checked === true)}
                        />
                        <Label htmlFor="autoBuild" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Auto-build Docker Image
                        </Label>
                    </div>
                    <DialogFooter>
                        <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
                            Cancel
                        </Button>
                        <Button type="submit" disabled={createMutation.isPending}>
                            {createMutation.isPending ? 'Creating...' : 'Create Release'}
                        </Button>
                    </DialogFooter>
                </form>
            </DialogContent>
        </Dialog >
    );
}
