import * as React from 'react';
import { useState, useEffect, useCallback, useRef } from 'react';
import { Terminal, Loader2, CheckCircle, XCircle, Clock, Package } from 'lucide-react';
import { Button } from './ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { cn } from './ui/utils';
import { api } from '../lib/api';
import type { DockerBuild } from '../lib/api';

/**
 * BuildDockerButton - A progress-aware button that transforms into a status indicator
 *
 * Visual States:
 * 1. IDLE        - Docker icon with subtle styling, shows "Build Docker"
 * 2. QUEUED      - Animated amber stripes, pulsing clock icon
 * 3. BUILDING    - Progress bar fills button, spinning loader, step counter
 * 4. SUCCESS     - Green flash animation, checkmark
 * 5. FAILED      - Red styling with X icon
 */

type BuildStatus = 'idle' | 'pending' | 'building' | 'success' | 'failed';

interface BuildDockerButtonProps {
  releaseId: string;
  onClick: () => void;
  className?: string;
  // Shared state from parent (when dialog is managing the build)
  activeBuild?: DockerBuild | null;
  buildProgress?: { current: number; total: number };
  // Callback when a build completes (detected via polling when dialog is closed)
  onBuildComplete?: (build: DockerBuild) => void;
}

// Parse Docker build step info from log content
function parseProgressFromLogs(logs: string): { current: number; total: number } {
  let maxStep = 0;
  let totalSteps = 0;

  // Pattern: #N [stage X/Y] or similar BuildKit output
  const stepPattern = /#(\d+)\s+\[([^\]]+)\s+(\d+)\/(\d+)\]/g;
  let match;
  while ((match = stepPattern.exec(logs)) !== null) {
    const current = parseInt(match[3], 10);
    const total = parseInt(match[4], 10);
    if (current > maxStep) maxStep = current;
    if (total > totalSteps) totalSteps = total;
  }

  return { current: maxStep, total: totalSteps };
}

export function BuildDockerButton({
  releaseId,
  onClick,
  className,
  activeBuild: externalBuild,
  buildProgress: externalProgress,
  onBuildComplete,
}: BuildDockerButtonProps) {
  // Local state for when component manages its own polling (when dialog is closed)
  const [localBuild, setLocalBuild] = useState<DockerBuild | null>(null);
  const [localProgress, setLocalProgress] = useState({ current: 0, total: 0 });
  const [showSuccessFlash, setShowSuccessFlash] = useState(false);
  const previousStatusRef = useRef<string | null>(null);
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const logPollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Use external state if provided (dialog is open), otherwise use local polling
  const hasExternalControl = externalBuild !== undefined;
  const activeBuild = hasExternalControl ? externalBuild : localBuild;
  const buildProgress = hasExternalControl && externalProgress ? externalProgress : localProgress;

  // Derive status from build state
  const status: BuildStatus = React.useMemo(() => {
    if (showSuccessFlash) return 'success';
    if (!activeBuild) return 'idle';
    switch (activeBuild.status) {
      case 'pending': return 'pending';
      case 'building': return 'building';
      case 'success': return 'success';
      case 'failed': return 'failed';
      default: return 'idle';
    }
  }, [activeBuild, showSuccessFlash]);

  // Calculate progress percentage
  const progressPercent = React.useMemo(() => {
    if (buildProgress.total > 0) {
      return Math.min(Math.round((buildProgress.current / buildProgress.total) * 100), 100);
    }
    return status === 'building' ? 5 : 0; // Show minimal progress when building starts
  }, [buildProgress, status]);

  // Poll for active builds when no external control
  useEffect(() => {
    if (hasExternalControl) {
      // Clear local polling when dialog takes over
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      return;
    }

    let mounted = true;
    let previousActiveBuildId: string | null = null;

    const checkBuilds = async () => {
      try {
        const response = await api.listDockerBuilds({ release_id: releaseId, size: 5 });
        const builds = response.items || [];
        const active = builds.find((b: DockerBuild) =>
          b.status === 'pending' || b.status === 'building'
        );

        if (mounted) {
          // Detect when a build completes: was active before, not active now
          if (previousActiveBuildId && !active) {
            // Find the completed build (most recent non-active)
            const completedBuild = builds.find((b: DockerBuild) =>
              b.id === previousActiveBuildId ||
              ['success', 'failed', 'cancelled'].includes(b.status)
            );
            if (completedBuild && onBuildComplete) {
              onBuildComplete(completedBuild);
            }
          }
          previousActiveBuildId = active?.id || null;
          setLocalBuild(active || null);
        }
      } catch {
        // Silently handle errors
      }
    };

    // Initial check
    checkBuilds();

    // Poll every 2 seconds for responsive updates
    pollIntervalRef.current = setInterval(checkBuilds, 2000);

    return () => {
      mounted = false;
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, [releaseId, hasExternalControl, onBuildComplete]);

  // Poll for logs to extract progress when building (only when we're managing state)
  useEffect(() => {
    if (hasExternalControl) {
      if (logPollIntervalRef.current) {
        clearInterval(logPollIntervalRef.current);
        logPollIntervalRef.current = null;
      }
      return;
    }

    if (!localBuild || localBuild.status !== 'building') {
      if (logPollIntervalRef.current) {
        clearInterval(logPollIntervalRef.current);
        logPollIntervalRef.current = null;
      }
      setLocalProgress({ current: 0, total: 0 });
      return;
    }

    let mounted = true;

    const pollLogs = async () => {
      try {
        const response = await api.getDockerBuildLogs(localBuild.id);
        if (mounted && response?.logs) {
          const progress = parseProgressFromLogs(response.logs);
          if (progress.total > 0) {
            setLocalProgress(progress);
          }
        }
      } catch {
        // Silently handle errors
      }
    };

    pollLogs();
    logPollIntervalRef.current = setInterval(pollLogs, 1500);

    return () => {
      mounted = false;
      if (logPollIntervalRef.current) {
        clearInterval(logPollIntervalRef.current);
        logPollIntervalRef.current = null;
      }
    };
  }, [localBuild?.id, localBuild?.status, hasExternalControl]);

  // Success flash animation
  useEffect(() => {
    const currentStatus = activeBuild?.status;
    if (currentStatus === 'success' && previousStatusRef.current === 'building') {
      setShowSuccessFlash(true);
      const timer = setTimeout(() => {
        setShowSuccessFlash(false);
        if (!hasExternalControl) {
          setLocalBuild(null);
          setLocalProgress({ current: 0, total: 0 });
        }
      }, 3000);
      return () => clearTimeout(timer);
    }
    previousStatusRef.current = currentStatus || null;
  }, [activeBuild?.status, hasExternalControl]);

  // Get display content based on status
  const getContent = useCallback(() => {
    switch (status) {
      case 'pending':
        return (
          <>
            <Clock className="size-4 mr-2 animate-pulse" />
            <span>Queued...</span>
          </>
        );
      case 'building':
        return (
          <>
            <Loader2 className="size-4 mr-2 animate-spin" />
            <span>
              {buildProgress.total > 0
                ? `${buildProgress.current}/${buildProgress.total}`
                : 'Building...'}
            </span>
          </>
        );
      case 'success':
        return (
          <>
            <CheckCircle className="size-4 mr-2" />
            <span>Built!</span>
          </>
        );
      case 'failed':
        return (
          <>
            <XCircle className="size-4 mr-2" />
            <span>Failed</span>
          </>
        );
      default:
        return (
          <>
            <Package className="size-4 mr-2" />
            <span>Build Docker</span>
          </>
        );
    }
  }, [status, buildProgress]);

  // Tooltip content
  const getTooltipContent = useCallback(() => {
    switch (status) {
      case 'pending':
        return 'Build is queued and waiting to start';
      case 'building':
        return buildProgress.total > 0
          ? `Building step ${buildProgress.current} of ${buildProgress.total} (${progressPercent}%)`
          : 'Docker build in progress...';
      case 'success':
        return 'Build completed successfully!';
      case 'failed':
        return 'Build failed. Click to view details and retry.';
      default:
        return 'Start a new Docker build';
    }
  }, [status, buildProgress, progressPercent]);

  return (
    <TooltipProvider delayDuration={300}>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="outline"
            onClick={onClick}
            className={cn(
              'relative overflow-hidden transition-all duration-300 min-w-[130px]',
              // Idle state - subtle docker-themed styling
              status === 'idle' && 'border-sky-200 hover:border-sky-400 hover:bg-sky-50/50 dark:border-sky-800 dark:hover:border-sky-600 dark:hover:bg-sky-950/30',
              // Pending - amber stripes
              status === 'pending' && 'border-amber-400 text-amber-700 dark:text-amber-400 shadow-sm shadow-amber-100 dark:shadow-amber-900/20',
              // Building - blue progress
              status === 'building' && 'border-blue-400 text-blue-700 dark:text-blue-400 shadow-sm shadow-blue-100 dark:shadow-blue-900/20',
              // Success - green celebration
              status === 'success' && 'border-green-500 bg-green-50 text-green-700 dark:bg-green-950/50 dark:text-green-400 shadow-sm shadow-green-100 dark:shadow-green-900/20',
              // Failed - red warning
              status === 'failed' && 'border-red-400 text-red-700 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/30',
              className,
            )}
          >
            {/* Progress fill background - smooth left-to-right fill */}
            {status === 'building' && (
              <div
                className="absolute inset-y-0 left-0 transition-all duration-500 ease-out rounded-l-md"
                style={{
                  width: `${progressPercent}%`,
                  background: 'linear-gradient(90deg, rgba(59,130,246,0.15) 0%, rgba(59,130,246,0.25) 100%)',
                }}
              />
            )}

            {/* Animated stripes for pending/queued state */}
            {status === 'pending' && (
              <div
                className="absolute inset-0 rounded-md"
                style={{
                  background: `repeating-linear-gradient(
                    -45deg,
                    transparent,
                    transparent 6px,
                    rgba(245,158,11,0.15) 6px,
                    rgba(245,158,11,0.15) 12px
                  )`,
                  animation: 'stripe-slide 0.8s linear infinite',
                }}
              />
            )}

            {/* Shimmer effect during building */}
            {status === 'building' && (
              <div
                className="absolute inset-0 pointer-events-none rounded-md"
                style={{
                  background: 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%)',
                  backgroundSize: '200% 100%',
                  animation: 'shimmer 2s ease-in-out infinite',
                }}
              />
            )}

            {/* Success flash */}
            {status === 'success' && showSuccessFlash && (
              <div
                className="absolute inset-0 bg-green-400/40 rounded-md"
                style={{ animation: 'success-flash 0.6s ease-out' }}
              />
            )}

            {/* Idle state subtle glow indicator */}
            {status === 'idle' && (
              <div className="absolute inset-0 rounded-md bg-gradient-to-r from-sky-500/5 to-blue-500/5" />
            )}

            {/* Content */}
            <span className="relative z-10 flex items-center whitespace-nowrap">
              {getContent()}
            </span>
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="max-w-xs">
          <p>{getTooltipContent()}</p>
          {(status === 'building' || status === 'pending') && (
            <div className="mt-1 text-xs text-muted-foreground">
              Click to view build logs
            </div>
          )}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// Hook for parent components to share build state
export function useBuildState() {
  const [activeBuild, setActiveBuild] = useState<DockerBuild | null>(null);
  const [buildProgress, setBuildProgress] = useState({ current: 0, total: 0 });

  const updateBuildState = useCallback((build: DockerBuild | null, progress?: { current: number; total: number }) => {
    setActiveBuild(build);
    if (progress) setBuildProgress(progress);
  }, []);

  const clearBuildState = useCallback(() => {
    setActiveBuild(null);
    setBuildProgress({ current: 0, total: 0 });
  }, []);

  return {
    activeBuild,
    buildProgress,
    updateBuildState,
    clearBuildState,
  };
}
