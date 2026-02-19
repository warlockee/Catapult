import { useState, useEffect } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Button } from './ui/button';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog';
import { Label } from './ui/label';
import { Input } from './ui/input';
import { api } from '../lib/api';

interface CreateVersionDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    onSuccess: () => void;
    preselectedModelId: string;
}

export function CreateVersionDialog({ open, onOpenChange, onSuccess, preselectedModelId }: CreateVersionDialogProps) {
    const [formData, setFormData] = useState({
        version: '',
        path: '',
        mlflow_url: '',
    });

    useEffect(() => {
        if (open) {
            setFormData({
                version: '',
                path: '',
                mlflow_url: '',
            });
        }
    }, [open]);

    const createMutation = useMutation({
        mutationFn: (data: Parameters<typeof api.createRelease>[0]) => api.createRelease(data),
        onSuccess: () => {
            onOpenChange(false);
            onSuccess();
        },
        onError: (error) => {
            console.error('Failed to create version:', error);
        },
    });

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!formData.version || !formData.path) return;

        // Auto-generate tag and digest as they are required by the API but not asked from the user
        const tag = formData.version;
        // Generate a placeholder digest since it's required but not provided
        // A valid sha256 is 64 hex characters.
        const placeholderDigest = `sha256:${Array(64).fill('0').join('')}`;

        createMutation.mutate({
            image_id: preselectedModelId,
            version: formData.version,
            tag: tag,
            digest: placeholderDigest,
            ceph_path: formData.path,
            mlflow_url: formData.mlflow_url || undefined,
            platform: 'linux/amd64', // Default
            metadata: {},
        });
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent>
                <DialogHeader>
                    <DialogTitle>Create Version</DialogTitle>
                    <DialogDescription>
                        Create a new version for this model
                    </DialogDescription>
                </DialogHeader>
                <form onSubmit={handleSubmit}>
                    <div className="space-y-4 py-4">
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
                            <Label htmlFor="path">Path *</Label>
                            <Input
                                id="path"
                                placeholder="e.g., /mnt/ceph/models/v1"
                                value={formData.path}
                                onChange={(e) => setFormData({ ...formData, path: e.target.value })}
                                required
                            />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="mlflow_url">MLflow URL</Label>
                            <Input
                                id="mlflow_url"
                                placeholder="e.g., http://10.1.1.205:5000/#/experiments/64/runs/c9666dc8c3814c3a9ebb85ad50394c46"
                                value={formData.mlflow_url}
                                onChange={(e) => setFormData({ ...formData, mlflow_url: e.target.value })}
                            />
                        </div>
                    </div>
                    <DialogFooter>
                        <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
                            Cancel
                        </Button>
                        <Button type="submit" disabled={createMutation.isPending}>
                            {createMutation.isPending ? 'Creating...' : 'Create Version'}
                        </Button>
                    </DialogFooter>
                </form>
            </DialogContent>
        </Dialog>
    );
}
