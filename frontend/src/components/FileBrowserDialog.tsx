import React, { useState, useEffect } from 'react';
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
} from './ui/dialog';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import { api } from '../lib/api';
import { Folder, File, ArrowUp, ChevronRight, Loader2 } from 'lucide-react';

interface FileItem {
    name: string;
    is_directory: boolean;
    size_bytes: number;
    modified_at: string;
}

interface FileBrowserDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    onSelect: (path: string, item: FileItem) => void;
    initialPath?: string;
    title?: string;
}

export function FileBrowserDialog({
    open,
    onOpenChange,
    onSelect,
    initialPath = 'artifacts',
    title = 'Browse Files',
}: FileBrowserDialogProps) {
    const [currentPath, setCurrentPath] = useState(initialPath);
    const [items, setItems] = useState<FileItem[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (open) {
            loadFiles(currentPath);
        }
    }, [currentPath, open]);

    const loadFiles = async (path: string) => {
        setLoading(true);
        setError(null);
        try {
            const data = await api.listSystemFiles(path);
            setItems(data);
        } catch (err) {
            setError('Failed to load files');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const formatSize = (bytes: number) => {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    };

    const handleNavigate = (folderName: string) => {
        const newPath = currentPath.endsWith('/')
            ? `${currentPath}${folderName}`
            : `${currentPath}/${folderName}`;
        setCurrentPath(newPath);
    };

    const handleUp = () => {
        if (currentPath === '' || currentPath === '/') return;
        const parts = currentPath.split('/').filter(Boolean);
        parts.pop();
        setCurrentPath(parts.join('/') || '/');
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-[600px] h-[500px] flex flex-col">
                <DialogHeader>
                    <DialogTitle>{title}</DialogTitle>
                    <DialogDescription className="hidden">
                        Browse and select files from the server filesystem.
                    </DialogDescription>
                </DialogHeader>

                <div className="flex items-center space-x-2 py-2 border-b bg-muted/20 px-4 -mx-6">
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={handleUp}
                        disabled={!currentPath || currentPath === '/' || currentPath === 'artifacts'} // prevent going above artifacts root logic if desired
                        className="h-8 w-8"
                    >
                        <ArrowUp className="h-4 w-4" />
                    </Button>
                    <div className="flex-1 font-mono text-sm truncate flex items-center text-muted-foreground">
                        /fsx/{currentPath}
                    </div>
                </div>

                {error && (
                    <div className="p-4 text-center text-red-500 text-sm">{error}</div>
                )}

                <ScrollArea className="flex-1 -mx-6 px-6">
                    <div className="space-y-1 py-2">
                        {loading ? (
                            <div className="flex justify-center py-8">
                                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                            </div>
                        ) : items.length === 0 ? (
                            <div className="py-8 text-center text-sm text-muted-foreground">
                                No files found
                            </div>
                        ) : (
                            items.map((item) => (
                                <div
                                    key={item.name}
                                    className="flex items-center justify-between p-2 rounded hover:bg-muted cursor-pointer group"
                                    onClick={() => {
                                        if (item.is_directory) {
                                            handleNavigate(item.name);
                                        } else {
                                            // On select
                                            const fullPath = currentPath.endsWith('/')
                                                ? `${currentPath}${item.name}`
                                                : `${currentPath}/${item.name}`;
                                            onSelect(fullPath, item);
                                            onOpenChange(false);
                                        }
                                    }}
                                >
                                    <div className="flex items-center space-x-3 truncate">
                                        {item.is_directory ? (
                                            <Folder className="h-4 w-4 text-blue-500 fill-blue-500/20" />
                                        ) : (
                                            <File className="h-4 w-4 text-gray-500" />
                                        )}
                                        <span className="text-sm font-medium group-hover:text-primary transition-colors">
                                            {item.name}
                                        </span>
                                    </div>

                                    <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                                        {!item.is_directory && <span>{formatSize(item.size_bytes)}</span>}
                                        {item.is_directory && <ChevronRight className="h-4 w-4 opacity-50" />}
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </ScrollArea>
            </DialogContent>
        </Dialog>
    );
}
