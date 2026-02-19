import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Terminal, Loader2, CheckCircle, XCircle, AlertTriangle, Copy, Clipboard as ClipboardIcon, Zap, Plus, RotateCcw, History, Bell, Square, Volume2, VolumeX, StopCircle, Cpu } from 'lucide-react';
import { Button } from './ui/button';
import { LogViewer } from './LogViewer';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog';
import { Label } from './ui/label';
import { Input } from './ui/input';
import { RadioGroup, RadioGroupItem } from './ui/radio-group';
import { Progress } from './ui/progress';
import { api } from '../lib/api';
import type { Release, Artifact, DockerBuild } from '../lib/api';
import { ArtifactPicker } from './ArtifactPicker';

// Browser notification helper
async function sendNotification(title: string, body: string, success: boolean) {
    if (!('Notification' in window)) return;

    if (Notification.permission === 'default') {
        await Notification.requestPermission();
    }

    if (Notification.permission === 'granted') {
        new Notification(title, {
            body,
            icon: success ? '/favicon.ico' : undefined,
            tag: 'docker-build',
        });
    }
}

// Docker tag validation - matches backend pattern
function isValidDockerTag(tag: string): boolean {
    if (!tag || tag.length > 256) return false;
    // Check for dangerous characters
    if (/[$`|;&><\\'"()]/.test(tag)) return false;
    // Basic docker tag pattern: optional registry/repo:tag (allows uppercase for model names)
    const pattern = /^[a-zA-Z0-9]([a-zA-Z0-9._/-]*[a-zA-Z0-9])?(:[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127})?$/;
    return pattern.test(tag);
}

// Normalize docker tag - lowercase the repository part (Docker requirement)
function normalizeDockerTag(tag: string): string {
    if (!tag.includes(':')) return tag.toLowerCase();
    const [repo, version] = tag.split(':');
    return `${repo.toLowerCase()}:${version}`;
}

// Extract vLLM version from wheel filename (e.g., "vllm-0.10.2-cp312-cp312-linux_x86_64.whl" -> "0.10.2")
function extractVllmVersion(filename: string): string | null {
    const match = filename.match(/^vllm[_-](\d+\.\d+(?:\.\d+)?(?:\.post\d+)?)/i);
    return match ? match[1] : null;
}

// Sound notification
const NOTIFICATION_SOUND_SUCCESS = 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdH2Onq6wn4x1YVJYaHiIl6e3wbOdf2FNQkpbcImdq7jCs6N/XUc7QVNpgJaou8THt5x1VDw0P1Rri6O3xszDq4hfPSwsPVJxkay/ysnArIVbNyYlNFBxlbHFzMm+pXtPKxweNFZzlrPHzce7oXJIJBMbL1JzmLfKz8i3m2c8FAwXL1Z0m7vO0se1lV4wCgcaL1l5oMHT1Me0kFYnAwMVLFl7pcfX18S0i04dAAEOJFV5pcrZ2MSyhUUTAAUNIlF1o8rZ2cSxfz4MAAcNIE9xoMjZ28SxfjoJAAoRIk9wo8nb28OuezcHAA0TI09xps3d28KrdDMEABEWJ1BzqNDe3MGoby8BABQZKlJ2q9Pg3L+kaikAABccLlV5sNbi3LyhaicAABkdMFh9s9nk3LqeZiQAABogM1t/t9zl3bmbYyEAAB0jNl6CuuDn3beYYB4AACAmOWGFveLp3rWVXBsAAiMoO2SIwOTq3rOSWhkABC';
const NOTIFICATION_SOUND_FAIL = 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJGpp6WMdmFSUF1qeYeMj4V3ZlhTWWRweIKIiYN4bmVfY2tze4KFhH93bWVhZm5zd3t9e3ZxbGhnbHF1eHp6d3NvbGxvcnV4enp4dHFubW9ydXh6enl2cm9tb3J1eHp6eXZyb21vcnV4enp5dnJvbW9ydXh6enl2cm9tb3J1eHp6eXZ';

function playNotificationSound(success: boolean) {
    try {
        const audio = new Audio(success ? NOTIFICATION_SOUND_SUCCESS : NOTIFICATION_SOUND_FAIL);
        audio.volume = 0.3;
        audio.play().catch(() => {}); // Ignore errors if autoplay blocked
    } catch (e) {
        // Audio not supported
    }
}

interface BuildDockerDialogProps {
    release: Release;
    open: boolean;
    onOpenChange: (open: boolean) => void;
    onSuccess?: (build: DockerBuild) => void;
    onBuildStateChange?: (build: DockerBuild | null, progress: { current: number; total: number }) => void;
}

export function BuildDockerDialog({ release, open, onOpenChange, onSuccess, onBuildStateChange }: BuildDockerDialogProps) {
    const [selectedArtifacts, setSelectedArtifacts] = useState<Array<{id: string; name: string; path: string; source: 'uploaded' | 'vllm_wheels'}>>([]);
    const [dockerTag, setDockerTag] = useState<string>('');
    const [buildType, setBuildType] = useState<'organic' | 'azure' | 'test' | 'optimized' | 'asr-vllm' | 'asr-allinone' | 'asr-azure-allinone'>('organic');
    const [detectedAsrVllm, setDetectedAsrVllm] = useState(false);
    const [building, setBuilding] = useState(false);
    const [currentBuild, setCurrentBuild] = useState<DockerBuild | null>(null);
    const [logs, setLogs] = useState<string>('');
    const [error, setError] = useState<string | null>(null);
    const logIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

    const [dockerfileContent, setDockerfileContent] = useState<string>('');
    const [dockerfileEdited, setDockerfileEdited] = useState(false);  // Track if user edited the Dockerfile
    const [showEditor, setShowEditor] = useState(false);
    const [loadingTemplate, setLoadingTemplate] = useState(false);
    const [showArtifactPicker, setShowArtifactPicker] = useState(false);
    const [buildProgress, setBuildProgress] = useState<{ current: number; total: number }>({ current: 0, total: 0 });
    const [buildHistory, setBuildHistory] = useState<DockerBuild[]>([]);
    const [showHistory, setShowHistory] = useState(false);
    const [notificationsEnabled, setNotificationsEnabled] = useState(() => {
        // Load from localStorage
        if (typeof window !== 'undefined') {
            return localStorage.getItem('docker-build-notifications') === 'true';
        }
        return false;
    });
    const [soundEnabled, setSoundEnabled] = useState(() => {
        if (typeof window !== 'undefined') {
            return localStorage.getItem('docker-build-sound') !== 'false'; // Default on
        }
        return true;
    });
    const [isViewingHistory, setIsViewingHistory] = useState(false);
    const [cancelling, setCancelling] = useState(false);

    // Detect vLLM version from selected artifacts
    const selectedVllmVersion = useMemo(() => {
        for (const artifact of selectedArtifacts) {
            const version = extractVllmVersion(artifact.name);
            if (version) return version;
        }
        return null;
    }, [selectedArtifacts]);

    // Update docker tag when vLLM version is detected or build type changes
    useEffect(() => {
        if (selectedVllmVersion && !currentBuild) {
            // Update tag based on build type (e.g., asr-model:vllm-0.10.2, asr-model:allinone-0.10.2)
            const tagSuffix = buildType === 'asr-vllm' ? 'vllm' :
                              buildType === 'asr-allinone' ? 'allinone' :
                              buildType === 'asr-azure-allinone' ? 'azure' : 'vllm';
            setDockerTag(`${release.image_name}:${tagSuffix}-${selectedVllmVersion}`);
        }
    }, [selectedVllmVersion, release.image_name, currentBuild, buildType]);

    // Auto-Resume and Streaming Logic
    useEffect(() => {
        if (open) {
            checkActiveBuilds();
        }
        return () => stopStreaming();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [open]);

    // Report build state changes to parent (for button sync)
    useEffect(() => {
        if (open) {
            onBuildStateChange?.(currentBuild, buildProgress);
        }
    }, [open, currentBuild, buildProgress, onBuildStateChange]);

    // Keyboard shortcuts
    useEffect(() => {
        if (!open) return;

        const handleKeyDown = (e: KeyboardEvent) => {
            // Don't trigger shortcuts when typing in inputs
            if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
                return;
            }

            // Ctrl/Cmd + Enter to start build
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !currentBuild && dockerTag && isValidDockerTag(dockerTag) && !building) {
                e.preventDefault();
                startBuild();
            }

            // 'r' to retry failed build
            if (e.key === 'r' && currentBuild?.status === 'failed' && !isViewingHistory && !building) {
                e.preventDefault();
                retryBuild();
            }

            // 'h' to toggle history
            if (e.key === 'h' && buildHistory.length > 0) {
                e.preventDefault();
                setShowHistory(!showHistory);
            }

            // 'n' for new build when viewing history
            if (e.key === 'n' && isViewingHistory) {
                e.preventDefault();
                startNewBuild();
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [open, currentBuild, dockerTag, building, isViewingHistory, showHistory, buildHistory.length]);

    const checkActiveBuilds = async () => {
        try {
            const response = await api.listDockerBuilds({ release_id: release.id, page: 1, size: 20 });
            const builds = response.items;

            // Store build history (completed builds)
            const completedBuilds = builds.filter(b => ['success', 'failed', 'cancelled'].includes(b.status));
            setBuildHistory(completedBuilds);

            // Sort by created_at desc if API doesn't guarantee it (backend does)
            const activeBuild = builds.find(b => ['pending', 'building'].includes(b.status));

            if (activeBuild) {
                console.log('Resuming active build:', activeBuild.id);
                setBuilding(true);
                setCurrentBuild(activeBuild);
                // Load the Dockerfile used for this build
                if (activeBuild.dockerfile_content) {
                    setDockerfileContent(activeBuild.dockerfile_content);
                }
                startStreaming(activeBuild.id);
            } else {
                // No active build - check if there's a recently completed build we should report
                // This handles the case where dialog was closed while build was in progress
                const mostRecentBuild = completedBuilds[0]; // Already sorted by created_at desc
                if (mostRecentBuild) {
                    // Notify parent of the completed build to update BuildMatrix
                    if (mostRecentBuild.status === 'success') {
                        onSuccess?.(mostRecentBuild);
                    }
                    // Also report via onBuildStateChange so BuildMatrix updates immediately
                    onBuildStateChange?.(mostRecentBuild, { current: 0, total: 0 });
                }

                // Normal init - reset dialog state
                setSelectedArtifacts([]);
                setDockerTag(`${release.image_name}:${release.version}`);
                setCurrentBuild(null);
                setLogs('');
                setError(null);
                setBuilding(false);
                setDockerfileContent('');
                setDockerfileEdited(false);  // Reset edit flag
                setShowEditor(false);
                loadTemplate(buildType);
            }
        } catch (err) {
            console.error('Failed to check active builds:', err);
        }
    };

    const abortControllerRef = useRef<AbortController | null>(null);

    const stopStreaming = () => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
        }
        if (logIntervalRef.current) {
            clearInterval(logIntervalRef.current);
            logIntervalRef.current = null;
        }
    };

    const startStreaming = async (buildId: string, retryCount = 0) => {
        const MAX_RETRIES = 3;
        const RETRY_DELAY = 2000;

        stopStreaming();
        abortControllerRef.current = new AbortController();
        const signal = abortControllerRef.current.signal;

        try {
            const reader = await api.streamBuildLogs(buildId, signal);
            const decoder = new TextDecoder();

            // Don't clear logs - append to existing progress messages

            // Background polling for status updates (since logs might just be streaming)
            // Ideally backend sends event for status change, but here mixed approach works well.
            // Poll status every 3s to know when to stop if logs are quiet.
            const statusInterval = setInterval(async () => {
                try {
                    const b = await api.getDockerBuild(buildId);
                    setCurrentBuild(b);
                    if (b.status === 'success' || b.status === 'failed' || b.status === 'cancelled') {
                        clearInterval(statusInterval);
                        setBuilding(false);
                        stopStreaming(); // Stop stream reader

                        // Send browser notification
                        if (notificationsEnabled) {
                            if (b.status === 'success') {
                                sendNotification(
                                    'Docker Build Complete',
                                    `Build for ${release.image_name}:${release.version} succeeded!`,
                                    true
                                );
                            } else if (b.status === 'cancelled') {
                                sendNotification(
                                    'Docker Build Cancelled',
                                    `Build for ${release.image_name}:${release.version} was cancelled.`,
                                    false
                                );
                            } else {
                                sendNotification(
                                    'Docker Build Failed',
                                    `Build for ${release.image_name}:${release.version} failed.`,
                                    false
                                );
                            }
                        }

                        // Play sound notification
                        if (soundEnabled) {
                            playNotificationSound(b.status === 'success');
                        }

                        if (b.status === 'success') onSuccess?.(b);
                        else setError(b.error_message || (b.status === 'cancelled' ? 'Build cancelled' : 'Build failed'));
                    }
                } catch (e) { console.error(e); }
            }, 3000);
            logIntervalRef.current = statusInterval; // Reuse ref for cleanup

            // Buffer for incoming logs to prevent React render thrashing
            let logBuffer = '';
            let lastUpdate = Date.now();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true });
                logBuffer += chunk;

                const now = Date.now();
                if (now - lastUpdate > 100) { // Update every 100ms
                    setLogs(prev => prev + logBuffer);
                    logBuffer = '';
                    lastUpdate = now;
                }
            }

            // Final flush
            if (logBuffer) {
                setLogs(prev => prev + logBuffer);
            }

        } catch (err: any) {
            // Handle different error types appropriately
            if (err.name === 'AbortError') {
                // User cancelled - don't show error
                return;
            }

            // Network errors may be transient - retry
            const isNetworkError = err.message?.includes('network') ||
                err.message?.includes('fetch') ||
                err.name === 'TypeError'; // fetch throws TypeError on network failure

            if (isNetworkError && retryCount < MAX_RETRIES) {
                console.warn(`Stream connection lost, retrying (${retryCount + 1}/${MAX_RETRIES})...`);
                setLogs(prev => prev + `\n[Connection lost, reconnecting...]\n`);
                setTimeout(() => startStreaming(buildId, retryCount + 1), RETRY_DELAY);
                return;
            }

            // Non-retryable or max retries exceeded
            console.error('Streaming failed:', err);
            setError(err.message || 'Failed to stream build logs');

            // Fall back to polling for logs if streaming fails completely
            setLogs(prev => prev + `\n[Streaming unavailable, switching to polling...]\n`);
            pollLogsAndStatus(buildId);
        }
    };

    // Fallback polling when streaming is not available
    const pollLogsAndStatus = async (buildId: string) => {
        const pollInterval = setInterval(async () => {
            try {
                const [build, logsData] = await Promise.all([
                    api.getDockerBuild(buildId),
                    api.getDockerBuildLogs(buildId)
                ]);

                setCurrentBuild(build);
                setLogs(logsData.logs || '');

                if (build.status === 'success' || build.status === 'failed' || build.status === 'cancelled') {
                    clearInterval(pollInterval);
                    setBuilding(false);
                    if (build.status === 'success') onSuccess?.(build);
                    else setError(build.error_message || (build.status === 'cancelled' ? 'Build cancelled' : 'Build failed'));
                }
            } catch (e) {
                console.error('Polling failed:', e);
            }
        }, 3000);

        logIntervalRef.current = pollInterval;
    };

    const logsEndRef = useRef<HTMLDivElement>(null);

    // Auto-scroll logs
    // Throttle scrolling too? 
    // useEffect on [logs] is fine if logs only update every 100ms.
    useEffect(() => {
        if (logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [logs]);

    const loadTemplate = async (type: string) => {
        console.log('loadTemplate called with:', type);
        try {
            setLoadingTemplate(true);
            // Pass release.id so backend can select correct template based on model's server_type
            const data = await api.getDockerfileTemplate(type, release.id);
            console.log('Template loaded, template_type:', data.template_type);
            setDockerfileContent(data.content);

            // If backend detected a different template type (e.g., asr-vllm for audio models)
            // Only auto-switch if user requested a generic type (organic/default)
            const genericTypes = ['organic', 'default'];
            if (data.template_type && data.template_type !== type && genericTypes.includes(type)) {
                console.log('Backend detected different type:', data.template_type, 'for generic request:', type);
                if (data.template_type === 'asr-vllm' || data.template_type === 'asr-allinone' || data.template_type === 'asr-azure-allinone') {
                    console.log('Setting buildType to:', data.template_type);
                    setBuildType(data.template_type as 'asr-vllm' | 'asr-allinone' | 'asr-azure-allinone');
                    setDetectedAsrVllm(true);
                }
            }
            // Ensure detectedAsrVllm stays true when switching between ASR types
            if (data.template_type === 'asr-vllm' || data.template_type === 'asr-allinone' || data.template_type === 'asr-azure-allinone') {
                setDetectedAsrVllm(true);
            }
        } catch (err) {
            console.error('Failed to load template:', err);
        } finally {
            setLoadingTemplate(false);
        }
    };

    const handleBuildTypeChange = (type: 'organic' | 'azure' | 'test' | 'optimized' | 'asr-vllm' | 'asr-allinone' | 'asr-azure-allinone') => {
        console.log('handleBuildTypeChange called with:', type, 'current buildType:', buildType);
        setBuildType(type);
        console.log('setBuildType called, loading template...');
        setDockerfileEdited(false);  // Reset edit flag when changing build type
        loadTemplate(type);

        // Update docker tag based on build type
        if (selectedVllmVersion) {
            const tagSuffix = type === 'asr-vllm' ? 'vllm' :
                              type === 'asr-allinone' ? 'allinone' :
                              type === 'asr-azure-allinone' ? 'azure' : type;
            setDockerTag(`${release.image_name}:${tagSuffix}-${selectedVllmVersion}`);
        }
    };

    const handleArtifactSelect = async (artifact: { id: string; name: string; path: string; source: 'uploaded' | 'vllm_wheels'; original?: Artifact }) => {
        try {
            // Check if already selected
            if (selectedArtifacts.some(a => a.path === artifact.path)) {
                return;
            }

            // For vLLM wheels, we need to register them as artifacts first
            if (artifact.source === 'vllm_wheels') {
                // Register the vLLM wheel as an artifact
                const type = artifact.name.endsWith('.whl') ? 'wheel' : 'file';

                const newArtifact = await api.registerArtifact({
                    release_id: release.id,
                    name: artifact.name,
                    file_path: artifact.path,
                    artifact_type: type,
                    metadata: { source: 'vllm_wheels' }
                });

                setSelectedArtifacts(prev => [...prev, {
                    id: newArtifact.id,
                    name: newArtifact.name,
                    path: newArtifact.file_path,
                    source: 'vllm_wheels'
                }]);
            } else {
                // For uploaded artifacts, use the existing ID
                setSelectedArtifacts(prev => [...prev, {
                    id: artifact.original?.id || artifact.id,
                    name: artifact.name,
                    path: artifact.path,
                    source: 'uploaded'
                }]);
            }
        } catch (err) {
            console.error('Failed to add artifact:', err);
            setError('Failed to add artifact: ' + String(err));
        }
    };

    const startBuild = async () => {
        if (!dockerTag) return;

        try {
            setBuilding(true);
            setError(null);
            // Immediate feedback before API call
            setLogs('🚀 Initiating build request...\n');

            // Normalize tag (lowercase repo) - Docker requirement
            const normalizedTag = normalizeDockerTag(dockerTag);

            const buildPayload: any = {
                release_id: release.id,
                image_tag: normalizedTag,
                build_type: buildType,
            };

            // Only send dockerfile_content if user explicitly edited it
            // Otherwise, let backend select template based on model's server_type
            if (dockerfileEdited && dockerfileContent) {
                buildPayload.dockerfile_content = dockerfileContent;
            }

            if (selectedArtifacts.length > 0) {
                const artifactIds = selectedArtifacts.map(a => a.id);
                buildPayload.artifact_ids = artifactIds;
                buildPayload.artifact_id = artifactIds[0]; // legacy fallback
            }

            const build = await api.createDockerBuild(buildPayload);

            setCurrentBuild(build);
            setBuildProgress({ current: 0, total: 0 });
            setLogs(prev => prev + `✓ Build queued (ID: ${build.id.substring(0, 8)}...)\n`);
            startStreaming(build.id);
        } catch (err) {
            console.error('Failed to start build:', err);
            setError('Failed to start build');
            setBuilding(false);
        }
    };

    const retryBuild = async () => {
        // Reset state and start a new build
        setCurrentBuild(null);
        setError(null);
        setBuildProgress({ current: 0, total: 0 });
        setBuilding(true);
        // Immediate feedback
        setLogs('🔄 Retrying build...\n');

        try {
            // Normalize tag (lowercase repo) - Docker requirement
            const normalizedTag = normalizeDockerTag(dockerTag);

            const buildPayload: any = {
                release_id: release.id,
                image_tag: normalizedTag,
                build_type: buildType,
            };

            // Only send dockerfile_content if user explicitly edited it
            if (dockerfileEdited && dockerfileContent) {
                buildPayload.dockerfile_content = dockerfileContent;
            }

            if (selectedArtifacts.length > 0) {
                const artifactIds = selectedArtifacts.map(a => a.id);
                buildPayload.artifact_ids = artifactIds;
                buildPayload.artifact_id = artifactIds[0];
            }

            const build = await api.createDockerBuild(buildPayload);
            setCurrentBuild(build);
            setLogs(prev => prev + `✓ Build queued (ID: ${build.id.substring(0, 8)}...)\n`);
            startStreaming(build.id);
        } catch (err) {
            console.error('Failed to retry build:', err);
            setError('Failed to start build');
            setBuilding(false);
        }
    };

    const viewHistoryBuild = async (build: DockerBuild) => {
        setCurrentBuild(build);
        setShowHistory(false);
        setBuilding(false);
        setIsViewingHistory(true);
        setError(build.status === 'failed' ? (build.error_message || 'Build failed') : null);

        // Load the Dockerfile that was used for this build
        if (build.dockerfile_content) {
            setDockerfileContent(build.dockerfile_content);
        }

        // Fetch logs for this build
        try {
            const logsData = await api.getDockerBuildLogs(build.id);
            setLogs(logsData.logs || '');
        } catch (err) {
            console.error('Failed to fetch build logs:', err);
            setLogs('Failed to load logs for this build.');
        }
    };

    const startNewBuild = () => {
        setCurrentBuild(null);
        setLogs('');
        setError(null);
        setIsViewingHistory(false);
        setBuildProgress({ current: 0, total: 0 });
    };

    const cancelBuild = async () => {
        if (!currentBuild || cancelling) return;

        try {
            setCancelling(true);
            setLogs(prev => prev + '\n⚠️ Cancelling build...\n');

            await api.cancelDockerBuild(currentBuild.id);

            // Stop streaming
            stopStreaming();

            // Update local state
            setCurrentBuild(prev => prev ? { ...prev, status: 'cancelled', error_message: 'Build cancelled by user' } : null);
            setBuilding(false);
            setError('Build cancelled by user');
            setLogs(prev => prev + '✓ Build cancelled successfully\n');

        } catch (err: any) {
            console.error('Failed to cancel build:', err);
            setLogs(prev => prev + `\n❌ Failed to cancel build: ${err.message || 'Unknown error'}\n`);
        } finally {
            setCancelling(false);
        }
    };

    const toggleNotifications = async () => {
        if (notificationsEnabled) {
            setNotificationsEnabled(false);
            localStorage.setItem('docker-build-notifications', 'false');
        } else {
            if ('Notification' in window) {
                const permission = await Notification.requestPermission();
                const enabled = permission === 'granted';
                setNotificationsEnabled(enabled);
                localStorage.setItem('docker-build-notifications', String(enabled));
            }
        }
    };

    return (
        <>
            <ArtifactPicker
                open={showArtifactPicker}
                onOpenChange={setShowArtifactPicker}
                onSelect={handleArtifactSelect}
                selectedIds={selectedArtifacts.map(a => a.id)}
                releaseId={release.id}
                modelId={release.image_id}
            />

            <Dialog open={showEditor} onOpenChange={setShowEditor}>
                <DialogContent className="sm:max-w-7xl max-h-[95vh] flex flex-col">
                    <DialogHeader>
                        <DialogTitle>Edit Dockerfile</DialogTitle>
                        <DialogDescription>
                            Customize the Dockerfile for this build.
                        </DialogDescription>
                    </DialogHeader>

                    <div className="flex justify-end gap-2 px-1">
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                                navigator.clipboard.writeText(dockerfileContent);
                            }}
                        >
                            <Copy className="size-3 mr-2" />
                            Copy
                        </Button>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={async () => {
                                try {
                                    const text = await navigator.clipboard.readText();
                                    setDockerfileContent(text);
                                } catch (err) {
                                    console.error('Failed to read clipboard:', err);
                                }
                            }}
                        >
                            <ClipboardIcon className="size-3 mr-2" />
                            Paste
                        </Button>
                    </div>

                    <div className="flex-1 overflow-hidden py-2">
                        <div className="flex h-[700px] bg-gray-900 rounded-md overflow-hidden border border-gray-700">
                            {/* Line numbers - synced scroll */}
                            <div
                                className="flex-shrink-0 bg-gray-800 text-gray-500 text-right font-mono text-sm select-none overflow-hidden w-12"
                                style={{ paddingTop: '16px', paddingBottom: '16px' }}
                            >
                                <div id="dockerfile-line-numbers" className="pr-2">
                                    {dockerfileContent.split('\n').map((_, i) => (
                                        <div key={i} className="leading-6 h-6">{i + 1}</div>
                                    ))}
                                </div>
                            </div>
                            {/* Simple textarea with good contrast */}
                            <textarea
                                className="flex-1 p-4 pl-3 font-mono text-sm bg-gray-900 text-gray-100 resize-none focus:outline-none focus:ring-1 focus:ring-blue-500 leading-6 overflow-auto"
                                value={dockerfileContent}
                                onChange={(e) => {
                                    setDockerfileContent(e.target.value);
                                    setDockerfileEdited(true);  // Mark as edited when user types
                                }}
                                onScroll={(e) => {
                                    const lineNumbers = document.getElementById('dockerfile-line-numbers');
                                    if (lineNumbers) {
                                        lineNumbers.style.transform = `translateY(-${e.currentTarget.scrollTop}px)`;
                                    }
                                }}
                                spellCheck={false}
                                wrap="off"
                                placeholder="# Dockerfile content..."
                            />
                        </div>
                    </div>
                    <DialogFooter>
                        <Button onClick={() => setShowEditor(false)}>Close & Save</Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            {/* Build History Dialog */}
            <Dialog open={showHistory} onOpenChange={setShowHistory}>
                <DialogContent className="sm:max-w-md">
                    <DialogHeader>
                        <DialogTitle>Build History</DialogTitle>
                        <DialogDescription>
                            Previous builds for this release
                        </DialogDescription>
                    </DialogHeader>
                    <div className="max-h-[400px] overflow-y-auto space-y-2">
                        {buildHistory.length === 0 ? (
                            <p className="text-sm text-muted-foreground py-4 text-center">No previous builds</p>
                        ) : (
                            buildHistory.map((build) => (
                                <div
                                    key={build.id}
                                    className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/50 cursor-pointer"
                                    onClick={() => viewHistoryBuild(build)}
                                >
                                    <div className="flex items-center gap-2">
                                        {build.status === 'success' ? (
                                            <CheckCircle className="h-4 w-4 text-green-500" />
                                        ) : build.status === 'cancelled' ? (
                                            <StopCircle className="h-4 w-4 text-orange-500" />
                                        ) : (
                                            <XCircle className="h-4 w-4 text-red-500" />
                                        )}
                                        <div>
                                            <div className="flex items-center gap-2">
                                                <p className="text-sm font-medium">{build.image_tag}</p>
                                                <span className="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                                                    {build.build_type}
                                                </span>
                                            </div>
                                            <p className="text-xs text-muted-foreground">
                                                {new Date(build.created_at).toLocaleString()}
                                                {build.completed_at && (
                                                    <span className="ml-2">
                                                        ({Math.round((new Date(build.completed_at).getTime() - new Date(build.created_at).getTime()) / 1000)}s)
                                                    </span>
                                                )}
                                            </p>
                                        </div>
                                    </div>
                                    <span className={`text-xs px-2 py-1 rounded ${
                                        build.status === 'success' ? 'bg-green-100 text-green-800' :
                                        build.status === 'cancelled' ? 'bg-orange-100 text-orange-800' :
                                        'bg-red-100 text-red-800'
                                    }`}>
                                        {build.status}
                                    </span>
                                </div>
                            ))
                        )}
                    </div>
                </DialogContent>
            </Dialog>

            <Dialog open={open} onOpenChange={onOpenChange}>
                <DialogContent className="sm:max-w-lg sm:max-h-[85vh] flex flex-col">
                    <DialogHeader>
                        <div className="flex items-center justify-between">
                            <div>
                                <DialogTitle>Build Docker Image</DialogTitle>
                                <DialogDescription>
                                    Create a Docker image for {release.image_name} v{release.version}
                                </DialogDescription>
                            </div>
                            <div className="flex items-center gap-1">
                                {buildHistory.length > 0 && (
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-8 w-8"
                                        onClick={() => setShowHistory(true)}
                                        title="View build history"
                                    >
                                        <History className="h-4 w-4" />
                                    </Button>
                                )}
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className={`h-8 w-8 ${soundEnabled ? 'text-primary' : ''}`}
                                    onClick={() => {
                                        const newVal = !soundEnabled;
                                        setSoundEnabled(newVal);
                                        localStorage.setItem('docker-build-sound', String(newVal));
                                    }}
                                    title={soundEnabled ? 'Sound enabled (click to disable)' : 'Enable sound'}
                                >
                                    {soundEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
                                </Button>
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className={`h-8 w-8 ${notificationsEnabled ? 'text-primary' : ''}`}
                                    onClick={toggleNotifications}
                                    title={notificationsEnabled ? 'Notifications enabled (click to disable)' : 'Enable notifications'}
                                >
                                    <Bell className={`h-4 w-4 ${notificationsEnabled ? 'fill-current' : ''}`} />
                                </Button>
                            </div>
                        </div>
                    </DialogHeader>

                    <div className="flex-1 overflow-y-auto py-4 space-y-6">
                        {isViewingHistory && currentBuild && (
                            <div className="bg-blue-50 border border-blue-200 text-blue-800 px-4 py-2 rounded-lg text-sm flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <History className="size-4" />
                                    <span>Viewing build from {new Date(currentBuild.created_at).toLocaleString()}</span>
                                </div>
                                <Button variant="ghost" size="sm" className="h-6 text-blue-800 hover:text-blue-900" onClick={startNewBuild}>
                                    Start New Build
                                </Button>
                            </div>
                        )}

                        {error && (
                            <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg text-sm flex items-center gap-2">
                                <AlertTriangle className="size-4" />
                                {error}
                            </div>
                        )}

                        {!currentBuild ? (
                            // Configuration Form
                            <div className="space-y-4">

                                <>
                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <Label>Model</Label>
                                            <Input value={release.image_name} disabled className="mt-1.5" />
                                        </div>
                                        <div>
                                            <Label>Version</Label>
                                            <Input value={release.version} disabled className="mt-1.5" />
                                        </div>
                                    </div>

                                    <div>
                                        <div className="flex items-center justify-between mb-2">
                                            <Label>Artifacts (Optional)</Label>

                                        </div>

                                        {/* Selected Artifacts List */}
                                        <div className="space-y-2 mb-3">
                                            {selectedArtifacts.map(artifact => (
                                                <div key={artifact.id} className="flex items-center justify-between p-2 bg-slate-100 rounded text-sm border">
                                                    <div className="flex flex-col overflow-hidden">
                                                        <div className="flex items-center gap-2">
                                                            <span className="font-medium truncate" title={artifact.name}>{artifact.name}</span>
                                                            {artifact.source === 'vllm_wheels' && (
                                                                <span className="text-xs bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded">vLLM</span>
                                                            )}
                                                        </div>
                                                        <span className="text-xs text-slate-500 truncate" title={artifact.path}>
                                                            {artifact.path}
                                                        </span>
                                                    </div>
                                                    <Button
                                                        variant="ghost"
                                                        size="sm"
                                                        className="h-6 w-6 p-0 hover:bg-red-100 hover:text-red-600 shrink-0 ml-2"
                                                        onClick={() => setSelectedArtifacts(prev => prev.filter(a => a.id !== artifact.id))}
                                                    >
                                                        <XCircle className="size-4" />
                                                    </Button>
                                                </div>
                                            ))}
                                        </div>

                                        <Button
                                            variant="outline"
                                            className="w-full dashed border-dashed"
                                            onClick={() => setShowArtifactPicker(true)}
                                        >
                                            <Plus className="mr-2 h-4 w-4" />
                                            Add Artifact
                                        </Button>

                                    </div>

                                    <div>
                                        <Label htmlFor="tag">Docker Tag</Label>
                                        <Input
                                            id="tag"
                                            value={dockerTag}
                                            onChange={(e) => setDockerTag(e.target.value)}
                                            className={`mt-1.5 ${dockerTag && !isValidDockerTag(dockerTag) ? 'border-red-500 focus-visible:ring-red-500' : ''}`}
                                            placeholder="repository/image:tag"
                                        />
                                        {dockerTag && !isValidDockerTag(dockerTag) && (
                                            <p className="text-xs text-red-500 mt-1">
                                                Invalid tag format. Use: name:tag or registry/name:tag
                                            </p>
                                        )}
                                    </div>

                                    <div>
                                        <Label className="mb-2 block">Build Type</Label>
                                        {detectedAsrVllm ? (
                                            <div className="space-y-3">
                                                <div className="text-xs text-muted-foreground bg-blue-50 border border-blue-200 px-3 py-2 rounded-md flex items-center gap-2">
                                                    <Volume2 className="size-4 text-blue-600" />
                                                    <span>Audio model detected. Choose your preferred API style:</span>
                                                </div>
                                                <div className="grid grid-cols-3 gap-3">
                                                    <div
                                                        onClick={() => { console.log('ASR vLLM clicked!'); handleBuildTypeChange('asr-vllm'); }}
                                                        className={`
                                                            cursor-pointer rounded-md border-2 p-3 flex flex-col gap-2 transition-all
                                                            ${buildType === 'asr-vllm' ? 'border-primary bg-primary/5' : 'border-muted hover:border-primary/50'}
                                                        `}
                                                    >
                                                        <div className="flex items-center gap-2">
                                                            <Terminal className={`size-4 ${buildType === 'asr-vllm' ? 'text-primary' : 'text-muted-foreground'}`} />
                                                            <span className={`text-sm font-medium ${buildType === 'asr-vllm' ? 'text-primary' : 'text-foreground'}`}>vLLM</span>
                                                        </div>
                                                        <p className="text-xs text-muted-foreground">
                                                            Raw API. Client handles VAD.
                                                        </p>
                                                    </div>
                                                    <div
                                                        onClick={() => { console.log('All-in-One clicked!'); handleBuildTypeChange('asr-allinone'); }}
                                                        className={`
                                                            cursor-pointer rounded-md border-2 p-3 flex flex-col gap-2 transition-all
                                                            ${buildType === 'asr-allinone' ? 'border-primary bg-primary/5' : 'border-muted hover:border-primary/50'}
                                                        `}
                                                    >
                                                        <div className="flex items-center gap-2">
                                                            <Zap className={`size-4 ${buildType === 'asr-allinone' ? 'text-primary' : 'text-muted-foreground'}`} />
                                                            <span className={`text-sm font-medium ${buildType === 'asr-allinone' ? 'text-primary' : 'text-foreground'}`}>All-in-One</span>
                                                        </div>
                                                        <p className="text-xs text-muted-foreground">
                                                            VAD + segmentation built-in.
                                                        </p>
                                                    </div>
                                                    <div
                                                        onClick={() => { console.log('Azure All-in-One clicked!'); handleBuildTypeChange('asr-azure-allinone'); }}
                                                        className={`
                                                            cursor-pointer rounded-md border-2 p-3 flex flex-col gap-2 transition-all
                                                            ${buildType === 'asr-azure-allinone' ? 'border-primary bg-primary/5' : 'border-muted hover:border-primary/50'}
                                                        `}
                                                    >
                                                        <div className="flex items-center gap-2">
                                                            <svg
                                                                className={`size-4 ${buildType === 'asr-azure-allinone' ? 'text-primary' : 'text-muted-foreground'}`}
                                                                viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                                                            >
                                                                <path d="M3 7l9-4 9 4v10l-9 4-9-4V7z" />
                                                                <path d="M3 7l9 4 9-4" />
                                                                <path d="M12 11v10" />
                                                            </svg>
                                                            <span className={`text-sm font-medium ${buildType === 'asr-azure-allinone' ? 'text-primary' : 'text-foreground'}`}>Azure</span>
                                                        </div>
                                                        <p className="text-xs text-muted-foreground">
                                                            Azure ML base + VAD.
                                                        </p>
                                                    </div>
                                                </div>
                                                {selectedVllmVersion && (
                                                    <div className="text-xs text-muted-foreground text-right">
                                                        vLLM version: <span className="font-mono bg-muted px-1.5 py-0.5 rounded">{selectedVllmVersion}</span>
                                                    </div>
                                                )}
                                            </div>
                                        ) : (
                                            <div className="grid grid-cols-2 gap-4">
                                                <div
                                                    onClick={() => handleBuildTypeChange('organic')}
                                                    className={`
                                                        cursor-pointer rounded-md border-2 p-4 flex flex-col items-center justify-center gap-2 transition-all
                                                        ${buildType === 'organic' ? 'border-primary bg-primary/5' : 'border-muted hover:border-primary/50'}
                                                    `}
                                                >
                                                    {selectedVllmVersion ? (
                                                        <>
                                                            <Cpu className={`size-6 ${buildType === 'organic' ? 'text-primary' : 'text-muted-foreground'}`} />
                                                            <span className={`font-medium ${buildType === 'organic' ? 'text-primary' : 'text-foreground'}`}>vLLM {selectedVllmVersion}</span>
                                                        </>
                                                    ) : (
                                                        <>
                                                            <Terminal className={`size-6 ${buildType === 'organic' ? 'text-primary' : 'text-muted-foreground'}`} />
                                                            <span className={`font-medium ${buildType === 'organic' ? 'text-primary' : 'text-foreground'}`}>Organic</span>
                                                        </>
                                                    )}
                                                </div>
                                                <div
                                                    onClick={() => handleBuildTypeChange('azure')}
                                                    className={`
                                                        cursor-pointer rounded-md border-2 p-4 flex flex-col items-center justify-center gap-2 transition-all
                                                        ${buildType === 'azure' ? 'border-primary bg-primary/5' : 'border-muted hover:border-primary/50'}
                                                    `}
                                                >
                                                    <svg
                                                        className={`size-6 ${buildType === 'azure' ? 'text-primary' : 'text-muted-foreground'}`}
                                                        xmlns="http://www.w3.org/2000/svg"
                                                        viewBox="0 0 24 24"
                                                        fill="none"
                                                        stroke="currentColor"
                                                        strokeWidth="2"
                                                        strokeLinecap="round"
                                                        strokeLinejoin="round"
                                                    >
                                                        <path d="M3 7l9-4 9 4v10l-9 4-9-4V7z" />
                                                        <path d="M3 7l9 4 9-4" />
                                                        <path d="M12 11v10" />
                                                    </svg>
                                                    <span className={`font-medium ${buildType === 'azure' ? 'text-primary' : 'text-foreground'}`}>Azure (MaaP)</span>
                                                </div>
                                                <div
                                                    onClick={() => handleBuildTypeChange('optimized')}
                                                    className={`
                                                        cursor-pointer rounded-md border-2 p-4 flex flex-col items-center justify-center gap-2 transition-all
                                                        ${buildType === 'optimized' ? 'border-primary bg-primary/5' : 'border-muted hover:border-primary/50'}
                                                    `}
                                                >
                                                    <Zap className={`size-6 ${buildType === 'optimized' ? 'text-primary' : 'text-muted-foreground'}`} />
                                                    <span className={`font-medium ${buildType === 'optimized' ? 'text-primary' : 'text-foreground'}`}>Optimized</span>
                                                </div>
                                                <div
                                                    onClick={() => handleBuildTypeChange('test')}
                                                    className={`
                                                        cursor-pointer rounded-md border-2 p-4 flex flex-col items-center justify-center gap-2 transition-all
                                                        ${buildType === 'test' ? 'border-primary bg-primary/5' : 'border-muted hover:border-primary/50'}
                                                    `}
                                                >
                                                    <CheckCircle className={`size-6 ${buildType === 'test' ? 'text-primary' : 'text-muted-foreground'}`} />
                                                    <span className={`font-medium ${buildType === 'test' ? 'text-primary' : 'text-foreground'}`}>Test / Diagnostic</span>
                                                </div>
                                            </div>
                                        )}
                                    </div>

                                    {/* Command Preview */}
                                    <div className="bg-slate-900 rounded-md p-3 font-mono text-xs text-slate-300 overflow-x-auto">
                                        <div className="text-slate-500 mb-1"># Command that will be executed:</div>
                                        {(buildType === 'asr-vllm' || buildType === 'asr-allinone' || buildType === 'asr-azure-allinone') ? (
                                            <>
                                                <div className="text-slate-500 mb-1"># Build with precompiled wheel{buildType === 'asr-azure-allinone' ? ' (Azure ML base)' : ''}</div>
                                                <div className="whitespace-nowrap">
                                                    DOCKER_BUILDKIT=1 docker build \
                                                </div>
                                                <div className="whitespace-nowrap pl-4">
                                                    --build-arg SETUPTOOLS_SCM_PRETEND_VERSION="{selectedVllmVersion || '0.10.2'}" \
                                                </div>
                                                <div className="whitespace-nowrap pl-4">
                                                    --build-arg VLLM_USE_PRECOMPILED=1 \
                                                </div>
                                                <div className="whitespace-nowrap pl-4">
                                                    --network=host -t {dockerTag || '<tag>'} \
                                                </div>
                                                <div className="whitespace-nowrap pl-4">
                                                    -f docker/Dockerfile .
                                                </div>
                                                {!selectedVllmVersion && (
                                                    <div className="text-amber-400 mt-2">
                                                        Add a vLLM wheel artifact (required for ASR builds)
                                                    </div>
                                                )}
                                            </>
                                        ) : (
                                            <>
                                                <div className="whitespace-nowrap">
                                                    docker build --progress=plain -t {dockerTag || '<tag>'} -f Dockerfile.{buildType} .
                                                </div>
                                                {selectedArtifacts.length > 0 && (
                                                    <div className="text-slate-500 mt-1">
                                                        # With {selectedArtifacts.length} artifact{selectedArtifacts.length > 1 ? 's' : ''} staged
                                                        {selectedVllmVersion && ` (vLLM ${selectedVllmVersion})`}
                                                    </div>
                                                )}
                                            </>
                                        )}
                                    </div>

                                </>

                            </div>
                        ) : (
                            // Build Progress
                            <div className="space-y-6">
                                <div className="flex items-center gap-4 p-4 border rounded-lg bg-gray-50">
                                    <div className="flex-1">
                                        <div className="flex items-center justify-between mb-2">
                                            <span className="font-medium">Status</span>
                                            <div className="flex items-center gap-2">
                                                <span className={`
                      px-2 py-1 rounded text-xs font-medium uppercase
                      ${currentBuild.status === 'success' ? 'bg-green-100 text-green-800' : ''}
                      ${currentBuild.status === 'failed' ? 'bg-red-100 text-red-800' : ''}
                      ${currentBuild.status === 'cancelled' ? 'bg-orange-100 text-orange-800' : ''}
                      ${currentBuild.status === 'building' ? 'bg-blue-100 text-blue-800' : ''}
                      ${currentBuild.status === 'pending' ? 'bg-gray-100 text-gray-800' : ''}
                    `}>
                                                    {currentBuild.status}
                                                </span>
                                                {['pending', 'building'].includes(currentBuild.status) && !isViewingHistory && (
                                                    <Button
                                                        variant="destructive"
                                                        size="sm"
                                                        onClick={cancelBuild}
                                                        disabled={cancelling}
                                                        className="h-7 px-2"
                                                        title="Kill build"
                                                    >
                                                        {cancelling ? (
                                                            <Loader2 className="size-3.5 animate-spin" />
                                                        ) : (
                                                            <StopCircle className="size-3.5" />
                                                        )}
                                                        <span className="ml-1.5">Kill</span>
                                                    </Button>
                                                )}
                                            </div>
                                        </div>
                                        {currentBuild.status === 'building' && (
                                            <div className="space-y-1">
                                                <Progress
                                                    value={buildProgress.total > 0 ? (buildProgress.current / buildProgress.total) * 100 : undefined}
                                                    className="h-2"
                                                />
                                                {buildProgress.total > 0 && (
                                                    <p className="text-xs text-muted-foreground text-right">
                                                        Step {buildProgress.current} of {buildProgress.total}
                                                    </p>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <Label>Build Logs</Label>
                                    <div className="h-[400px] border rounded-md shadow-inner bg-slate-950">
                                        <LogViewer
                                            logs={logs}
                                            className="h-full w-full"
                                            buildId={currentBuild.id}
                                            buildStatus={currentBuild.status}
                                            onProgressChange={(current, total) => setBuildProgress({ current, total })}
                                        />
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    <DialogFooter>
                        {!currentBuild ? (
                            <div className="flex gap-2">
                                <Button variant="outline" onClick={() => setShowEditor(true)} disabled={loadingTemplate}>
                                    {loadingTemplate ? <Loader2 className="size-4 animate-spin mr-2" /> : null}
                                    Edit Dockerfile
                                </Button>
                                <Button
                                    onClick={startBuild}
                                    disabled={!dockerTag || !isValidDockerTag(dockerTag) || building || ((buildType === 'asr-vllm' || buildType === 'asr-allinone' || buildType === 'asr-azure-allinone') && !selectedVllmVersion)}
                                    title={(buildType === 'asr-vllm' || buildType === 'asr-allinone' || buildType === 'asr-azure-allinone') && !selectedVllmVersion ? 'Add a vLLM wheel artifact first' : undefined}
                                >
                                    {building ? (
                                        <>
                                            <Loader2 className="mr-2 size-4 animate-spin" />
                                            Starting...
                                        </>
                                    ) : (
                                        'Start Build'
                                    )}
                                </Button>
                            </div>
                        ) : (
                            <div className="flex gap-2">
                                {isViewingHistory && (
                                    <Button
                                        variant="outline"
                                        onClick={startNewBuild}
                                    >
                                        <Plus className="mr-2 size-4" />
                                        New Build
                                    </Button>
                                )}
                                {currentBuild.status === 'failed' && (
                                    <>
                                        <Button
                                            variant="outline"
                                            onClick={() => setShowEditor(true)}
                                        >
                                            Edit Dockerfile
                                        </Button>
                                        {!isViewingHistory && (
                                            <Button
                                                variant="outline"
                                                onClick={retryBuild}
                                                disabled={building}
                                            >
                                                <RotateCcw className="mr-2 size-4" />
                                                Retry
                                            </Button>
                                        )}
                                    </>
                                )}
                                <Button
                                    onClick={() => onOpenChange(false)}
                                >
                                    {currentBuild.status === 'success' ? 'Done' : 'Close'}
                                </Button>
                            </div>
                        )}
                    </DialogFooter>
                </DialogContent>
            </Dialog >
        </>
    );
}
