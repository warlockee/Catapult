import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Package, GitBranch, Clock, Tag, TrendingUp, CheckCircle, Copy, Check, FileText } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { api } from '../lib/api';
import { formatBytes } from '../lib/utils';
import type { Image, Release } from '../lib/api';
import { CreateVersionDialog } from './CreateVersionDialog';

export function ModelDetail() {
    const { modelId: imageId } = useParams<{ modelId: string }>();
    const navigate = useNavigate();
    const [image, setImage] = useState<Image | null>(null);
    const [releases, setReleases] = useState<Release[]>([]);
    const [totalPages, setTotalPages] = useState(0);
    const [page, setPage] = useState(1);
    const pageSize = 10;

    const [loading, setLoading] = useState(true);
    const [showCreateDialog, setShowCreateDialog] = useState(false);
    const [copied, setCopied] = useState(false);

    const copyToClipboard = async (text: string) => {
        try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
        }
    };

    useEffect(() => {
        loadModelData();
    }, [imageId, page]);

    const loadModelData = async () => {
        try {
            setLoading(true);
            // Fetch model details only once or if changed? Actually it's fine to refetch.
            // But optimal would be separate calls. For simplicity, keeping together but careful with errors.
            const [imageData, releasesData] = await Promise.all([
                api.getImage(imageId),
                api.getImageReleases(imageId, { page, size: pageSize })
            ]);
            setImage(imageData);
            setReleases(releasesData.items);
            setTotalPages(releasesData.pages);
        } catch (error) {
            console.error('Failed to load model data:', error);
        } finally {
            setLoading(false);
        }
    };

    const formatDate = (dateString: string) => {
        return new Date(dateString).toLocaleString();
    };


    if (loading && !image) {
        return (
            <div className="text-center py-12">
                <p className="text-gray-500">Loading model...</p>
            </div>
        );
    }

    if (!image) {
        return (
            <div className="text-center py-12">
                <p className="text-gray-500">Model not found</p>
                <Button onClick={() => navigate('/models')} className="mt-4">
                    Go Back
                </Button>
            </div>
        );
    }

    // We can't rely on 'releases' array for total count for header badge anymore since it's paginated.
    // We might need total count from backend response. Ideally we'd store it.
    // For now assuming user accepts the badge might show "versions on this page" OR we use the 'total' from response if we stored it.
    // Actually, let's use a simpler label or omit count if inconsistent.
    // Or better: The model object usually has a release_count if we fetched it via list. But get model doesn't always have it?
    // Let's just say "Versions".

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="space-y-3 overflow-hidden">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Button variant="ghost" onClick={() => navigate('/models')}>
                            <ArrowLeft className="size-4 mr-2" />
                            Back
                        </Button>
                        <div>
                            <h1>Model Detail</h1>
                            <p className="text-gray-500 mt-1">{image.description || 'No description'}</p>
                        </div>
                    </div>
                    <div className="flex gap-2 shrink-0">
                        <Button variant="outline" onClick={() => navigate(`/models/${imageId}/card`)}>
                            <FileText className="size-4 mr-2" />
                            Model Card
                        </Button>
                        <Button className="bg-blue-600 hover:bg-blue-700" onClick={() => setShowCreateDialog(true)}>
                            <GitBranch className="size-4 mr-2" />
                            Create Version
                        </Button>
                    </div>
                </div>
                <code className="block px-3 py-2 bg-gray-100 rounded-lg text-sm truncate max-w-full" title={image.name}>
                    {image.name}
                </code>
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card>
                    <CardContent className="p-4">
                        <div className="flex items-center gap-3">
                            <div className="size-10 bg-blue-100 rounded-lg flex items-center justify-center shrink-0">
                                <Package className="size-5 text-blue-600" />
                            </div>
                            <div className="min-w-0 flex-1 overflow-hidden w-0">
                                <div className="text-xs text-gray-500">Storage Path</div>
                                <div className="flex items-center gap-2">
                                    <code className="text-sm font-mono bg-gray-100 px-2 py-1 rounded overflow-x-auto whitespace-nowrap block max-w-full">
                                        {image.storage_path || 'N/A'}
                                    </code>
                                    {image.storage_path && (
                                        <button
                                            onClick={() => copyToClipboard(image.storage_path)}
                                            className="p-1 hover:bg-gray-100 rounded transition-colors shrink-0"
                                            title="Copy to clipboard"
                                        >
                                            {copied ? (
                                                <Check className="size-4 text-green-600" />
                                            ) : (
                                                <Copy className="size-4 text-gray-500" />
                                            )}
                                        </button>
                                    )}
                                </div>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                <Card>
                    <CardContent className="p-4">
                        <div className="flex items-center gap-3">
                            <div className="size-10 bg-purple-100 rounded-lg flex items-center justify-center">
                                <GitBranch className="size-5 text-purple-600" />
                            </div>
                            <div>
                                <div className="text-xs text-gray-500">Latest Version</div>
                                {/* Correct logic: we only know latest of current page. Ideally backend provides this. */}
                                <div className="text-sm">View list below</div>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                <Card>
                    <CardContent className="p-4">
                        <div className="flex items-center gap-3">
                            <div className="size-10 bg-green-100 rounded-lg flex items-center justify-center">
                                <Clock className="size-5 text-green-600" />
                            </div>
                            <div>
                                <div className="text-xs text-gray-500">Last Updated</div>
                                <div className="text-sm">{formatDate(image.updated_at)}</div>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Main Content */}
            <Tabs defaultValue="versions" className="space-y-6">
                <TabsList>
                    <TabsTrigger value="versions">Versions</TabsTrigger>
                    <TabsTrigger value="overview">Overview</TabsTrigger>
                </TabsList>

                <TabsContent value="versions" className="space-y-4">
                    {releases.length > 0 ? (
                        <>
                            {releases.map((release) => (
                                <Card
                                    key={release.id}
                                    className="hover:shadow-md transition-shadow cursor-pointer overflow-hidden"
                                    onClick={() => navigate(`/models/${imageId}/releases/${release.id}`)}
                                >
                                    <CardContent className="p-4 min-w-0 overflow-hidden">
                                        <div className="grid grid-cols-[auto_minmax(0,1fr)_auto] gap-4 items-start">
                                            <div className={`size-10 rounded-lg flex items-center justify-center ${release.is_release ? 'bg-purple-100' : 'bg-gray-100'}`}>
                                                <GitBranch className={`size-5 ${release.is_release ? 'text-purple-600' : 'text-gray-600'}`} />
                                            </div>
                                            <div className="min-w-0 overflow-hidden">
                                                <div className="flex items-center gap-2 flex-wrap">
                                                    <Badge variant="secondary" className={release.is_release ? "bg-purple-100 text-purple-700" : "bg-gray-100 text-gray-700"}>
                                                        {release.tag}
                                                    </Badge>
                                                    {release.is_release && (
                                                        <Badge className="bg-blue-100 text-blue-700">Official Release</Badge>
                                                    )}
                                                </div>
                                                <code className="mt-1.5 block px-2 py-1 bg-gray-100 rounded text-sm font-semibold overflow-x-auto whitespace-nowrap max-h-[3.25rem] leading-relaxed max-w-full">
                                                    {release.version}
                                                </code>
                                                <div className="flex items-center gap-4 text-xs text-gray-500 mt-1.5">
                                                    <span>{formatDate(release.created_at)}</span>
                                                    <span>{release.size_bytes ? formatBytes(release.size_bytes) : 'N/A'}</span>
                                                    <span>{release.platform}</span>
                                                </div>
                                            </div>
                                            <div className="flex items-center gap-4">
                                                {release.metadata?.accuracy && (
                                                    <div className="text-right">
                                                        <div className="text-xs text-gray-500">Accuracy</div>
                                                        <div className="font-medium text-green-600">{(release.metadata.accuracy * 100).toFixed(1)}%</div>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </CardContent>
                                </Card>
                            ))}

                            {/* Pagination Controls */}
                            {totalPages > 1 && (
                                <div className="flex items-center justify-center gap-4 py-4">
                                    <Button
                                        variant="outline"
                                        onClick={() => setPage(p => Math.max(1, p - 1))}
                                        disabled={page === 1}
                                    >
                                        Previous
                                    </Button>
                                    <span className="text-sm text-gray-500">
                                        Page {page} of {totalPages}
                                    </span>
                                    <Button
                                        variant="outline"
                                        onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                                        disabled={page === totalPages}
                                    >
                                        Next
                                    </Button>
                                </div>
                            )}
                        </>
                    ) : (
                        <div className="text-center py-8 text-gray-500">
                            No versions found for this model
                        </div>
                    )}
                </TabsContent>

                <TabsContent value="overview">
                    <Card>
                        <CardHeader>
                            <CardTitle>Model Metadata</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-4">
                                <div>
                                    <div className="text-sm text-gray-500 mb-1">Repository</div>
                                    <div>{image.repository || 'N/A'}</div>
                                </div>
                                <div>
                                    <div className="text-sm text-gray-500 mb-1">Company</div>
                                    <div>{image.company || 'N/A'}</div>
                                </div>
                                <div>
                                    <div className="text-sm text-gray-500 mb-1">Base Model</div>
                                    <div>{image.base_model || 'N/A'}</div>
                                </div>
                                <div>
                                    <div className="text-sm text-gray-500 mb-1">Parameters</div>
                                    <div>{image.parameter_count || 'N/A'}</div>
                                </div>
                                {image.tags && image.tags.length > 0 && (
                                    <div>
                                        <div className="text-sm text-gray-500 mb-2">Tags</div>
                                        <div className="flex flex-wrap gap-2">
                                            {image.tags.map(tag => (
                                                <Badge key={tag} variant="outline">{tag}</Badge>
                                            ))}
                                        </div>
                                    </div>
                                )}
                                {image.metadata && Object.keys(image.metadata).length > 0 && (
                                    <div>
                                        <div className="text-sm text-gray-500 mb-2">Additional Metadata</div>
                                        <pre className="bg-gray-50 p-4 rounded-lg overflow-x-auto text-sm">
                                            {JSON.stringify(image.metadata, null, 2)}
                                        </pre>
                                    </div>
                                )}
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>

            <CreateVersionDialog
                open={showCreateDialog}
                onOpenChange={setShowCreateDialog}
                onSuccess={loadModelData}
                preselectedModelId={imageId}
            />
        </div>
    );
}
