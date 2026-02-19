import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { ArrowDownCircle, PauseCircle, PlayCircle, Download, Maximize2, Minimize2, Search, X, ChevronUp, ChevronDown, AlertCircle, AlertTriangle, Layers, Copy, Clock, ChevronsUpDown, Regex } from "lucide-react";
import { cn } from "../lib/utils";

type FilterMode = 'all' | 'errors' | 'warnings' | 'steps';

interface LogViewerProps {
    logs: string;
    className?: string;
    autoScroll?: boolean;
    buildId?: string;
    onProgressChange?: (current: number, total: number) => void;
    buildStatus?: 'pending' | 'building' | 'success' | 'failed';
}

// ANSI color code to Tailwind CSS class mapping
const ANSI_COLORS: Record<number, string> = {
    30: 'text-gray-900',
    31: 'text-red-500',
    32: 'text-green-500',
    33: 'text-yellow-500',
    34: 'text-blue-500',
    35: 'text-purple-500',
    36: 'text-cyan-500',
    37: 'text-gray-300',
    90: 'text-gray-500',
    91: 'text-red-400',
    92: 'text-green-400',
    93: 'text-yellow-400',
    94: 'text-blue-400',
    95: 'text-purple-400',
    96: 'text-cyan-400',
    97: 'text-white',
};

const ANSI_BG_COLORS: Record<number, string> = {
    40: 'bg-gray-900',
    41: 'bg-red-900',
    42: 'bg-green-900',
    43: 'bg-yellow-900',
    44: 'bg-blue-900',
    45: 'bg-purple-900',
    46: 'bg-cyan-900',
    47: 'bg-gray-700',
};

// Parse ANSI escape codes and return styled segments
function parseAnsiLine(text: string): React.ReactNode[] {
    const segments: React.ReactNode[] = [];
    const ansiRegex = /\x1b\[([0-9;]*)m/g;

    let lastIndex = 0;
    let currentClasses: string[] = [];
    let match;

    while ((match = ansiRegex.exec(text)) !== null) {
        if (match.index > lastIndex) {
            const content = text.slice(lastIndex, match.index);
            if (content) {
                segments.push(
                    <span key={`${lastIndex}-text`} className={currentClasses.join(' ')}>
                        {content}
                    </span>
                );
            }
        }

        const codes = match[1].split(';').map(Number);
        for (const code of codes) {
            if (code === 0) {
                currentClasses = [];
            } else if (code === 1) {
                currentClasses.push('font-bold');
            } else if (code === 3) {
                currentClasses.push('italic');
            } else if (code === 4) {
                currentClasses.push('underline');
            } else if (ANSI_COLORS[code]) {
                currentClasses = currentClasses.filter(c => !c.startsWith('text-'));
                currentClasses.push(ANSI_COLORS[code]);
            } else if (ANSI_BG_COLORS[code]) {
                currentClasses = currentClasses.filter(c => !c.startsWith('bg-'));
                currentClasses.push(ANSI_BG_COLORS[code]);
            }
        }

        lastIndex = match.index + match[0].length;
    }

    if (lastIndex < text.length) {
        const content = text.slice(lastIndex);
        if (content) {
            segments.push(
                <span key={`${lastIndex}-text`} className={currentClasses.join(' ')}>
                    {content}
                </span>
            );
        }
    }

    return segments.length > 0 ? segments : [text];
}

// Check if a line contains error indicators
function isErrorLine(line: string): boolean {
    const lowerLine = line.toLowerCase();
    // Strip ANSI codes for matching
    const cleanLine = lowerLine.replace(/\x1b\[[0-9;]*m/g, '');
    return (
        cleanLine.includes('error:') ||
        cleanLine.includes('error[') ||
        /\berror\b/.test(cleanLine) && (cleanLine.includes('failed') || cleanLine.includes(':')) ||
        cleanLine.includes('failed') ||
        cleanLine.includes('fatal') ||
        cleanLine.includes('exception') ||
        cleanLine.includes('traceback') ||
        /^e:\s/.test(cleanLine) ||
        cleanLine.includes('command not found') ||
        cleanLine.includes('no such file') ||
        cleanLine.includes('permission denied') ||
        cleanLine.includes('cannot find') ||
        cleanLine.includes('not found') && !cleanLine.includes('warning')
    );
}

// Check if a line contains warning indicators
function isWarningLine(line: string): boolean {
    const lowerLine = line.toLowerCase();
    const cleanLine = lowerLine.replace(/\x1b\[[0-9;]*m/g, '');
    return (
        cleanLine.includes('warning') ||
        cleanLine.includes('warn:') ||
        cleanLine.includes('deprecated')
    );
}

// Check if line is a Docker build step and extract step info
function parseBuildStep(line: string): { isStep: boolean; current?: number; total?: number; stage?: string } {
    // Match patterns like "#5 [stage 2/3] RUN something" or "#12 [2/5] COPY"
    const buildkitMatch = line.match(/^#(\d+)\s+\[([^\]]+)\s+(\d+)\/(\d+)\]/);
    if (buildkitMatch) {
        return {
            isStep: true,
            current: parseInt(buildkitMatch[3]),
            total: parseInt(buildkitMatch[4]),
            stage: buildkitMatch[2]
        };
    }

    // Match simple "#N" pattern
    const simpleMatch = line.match(/^#(\d+)\s+/);
    if (simpleMatch) {
        return { isStep: true };
    }

    // Match "[1/10]" style
    const bracketMatch = line.match(/^\[(\d+)\/(\d+)\]/);
    if (bracketMatch) {
        return {
            isStep: true,
            current: parseInt(bracketMatch[1]),
            total: parseInt(bracketMatch[2])
        };
    }

    return { isStep: false };
}

// Parse timing from Docker output (e.g., "#5 DONE 45.2s" or "CACHED")
function parseStepTiming(line: string): { duration?: number; cached?: boolean } {
    const doneMatch = line.match(/^#\d+\s+DONE\s+(\d+\.?\d*)s/);
    if (doneMatch) {
        return { duration: parseFloat(doneMatch[1]) };
    }

    if (line.match(/^#\d+\s+CACHED/)) {
        return { cached: true };
    }

    return {};
}

// Format seconds to human readable
function formatDuration(seconds: number): string {
    if (seconds < 60) {
        return `${seconds.toFixed(1)}s`;
    }
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs.toFixed(0)}s`;
}

export function LogViewer({ logs, className, buildId, onProgressChange, buildStatus }: LogViewerProps) {
    const scrollRef = useRef<HTMLDivElement>(null);
    const [autoScroll, setAutoScroll] = useState(true);
    const [isAtBottom, setIsAtBottom] = useState(true);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [showSearch, setShowSearch] = useState(false);
    const [currentMatch, setCurrentMatch] = useState(0);
    const [matchCount, setMatchCount] = useState(0);
    const [filterMode, setFilterMode] = useState<FilterMode>('all');
    const [showErrorSummary, setShowErrorSummary] = useState(true);
    const [showStageJump, setShowStageJump] = useState(false);
    const [useRegex, setUseRegex] = useState(false);
    const [copiedLine, setCopiedLine] = useState<number | null>(null);
    const [hasAutoScrolledToError, setHasAutoScrolledToError] = useState(false);
    const matchRefs = useRef<(HTMLDivElement | null)[]>([]);
    const errorRefs = useRef<(HTMLDivElement | null)[]>([]);
    const stageRefs = useRef<Map<number, HTMLDivElement | null>>(new Map());

    // Parse logs into lines with metadata
    const { parsedLines, stats, progress, totalBuildTime, stages } = useMemo(() => {
        if (!logs) return { parsedLines: [], stats: { errors: 0, warnings: 0, steps: 0, cached: 0 }, progress: { current: 0, total: 0 }, totalBuildTime: 0, stages: [] };

        const lines = logs.split('\n');
        let matchIndex = 0;
        let errorIndex = 0;
        let maxProgress = { current: 0, total: 0 };
        const stats = { errors: 0, warnings: 0, steps: 0, cached: 0 };
        let totalTime = 0;
        const stagesMap = new Map<number, { lineNumber: number; description: string }>();

        const parsed = lines.map((line, idx) => {
            const isError = isErrorLine(line);
            const isWarning = isWarningLine(line);
            const stepInfo = parseBuildStep(line);
            const timing = parseStepTiming(line);
            let matchesSearch = false;
            if (searchQuery) {
                try {
                    if (useRegex) {
                        const regex = new RegExp(searchQuery, 'i');
                        matchesSearch = regex.test(line);
                    } else {
                        matchesSearch = line.toLowerCase().includes(searchQuery.toLowerCase());
                    }
                } catch (e) {
                    // Invalid regex, treat as literal
                    matchesSearch = line.toLowerCase().includes(searchQuery.toLowerCase());
                }
            }
            const thisMatchIndex = matchesSearch ? matchIndex++ : -1;
            const thisErrorIndex = isError ? errorIndex++ : -1;

            // Extract stage number for jump feature
            const stageMatch = line.match(/^#(\d+)\s+(.+)/);
            let stageId: number | undefined;
            if (stageMatch && !stagesMap.has(parseInt(stageMatch[1]))) {
                stageId = parseInt(stageMatch[1]);
                stagesMap.set(stageId, {
                    lineNumber: idx + 1,
                    description: stageMatch[2].slice(0, 50)
                });
            }

            if (isError) stats.errors++;
            if (isWarning) stats.warnings++;
            if (stepInfo.isStep) {
                stats.steps++;
                if (stepInfo.current && stepInfo.total) {
                    if (stepInfo.current > maxProgress.current || stepInfo.total > maxProgress.total) {
                        maxProgress = { current: stepInfo.current, total: stepInfo.total };
                    }
                }
            }
            if (timing.cached) stats.cached++;
            if (timing.duration) totalTime += timing.duration;

            return {
                content: line,
                lineNumber: idx + 1,
                isError,
                isWarning,
                isStep: stepInfo.isStep,
                stepInfo,
                timing,
                matchesSearch,
                matchIndex: thisMatchIndex,
                errorIndex: thisErrorIndex,
                stageId,
            };
        });

        const stages = Array.from(stagesMap.entries())
            .sort((a, b) => a[0] - b[0])
            .map(([id, info]) => ({ id, ...info }));

        return { parsedLines: parsed, stats, progress: maxProgress, totalBuildTime: totalTime, stages };
    }, [logs, searchQuery, useRegex]);

    // Report progress to parent
    useEffect(() => {
        if (onProgressChange && progress.total > 0) {
            onProgressChange(progress.current, progress.total);
        }
    }, [progress, onProgressChange]);

    // Get error lines for summary
    const errorLines = useMemo(() =>
        parsedLines.filter(l => l.isError).slice(0, 10), // Show max 10 errors in summary
        [parsedLines]
    );

    // Filter lines based on mode
    const filteredLines = useMemo(() => {
        if (filterMode === 'all') return parsedLines;
        return parsedLines.filter(line => {
            switch (filterMode) {
                case 'errors': return line.isError;
                case 'warnings': return line.isWarning;
                case 'steps': return line.isStep;
                default: return true;
            }
        });
    }, [parsedLines, filterMode]);

    // Update match count
    useEffect(() => {
        const count = parsedLines.filter(l => l.matchesSearch).length;
        setMatchCount(count);
        if (currentMatch >= count) {
            setCurrentMatch(Math.max(0, count - 1));
        }
    }, [parsedLines, currentMatch]);

    // Auto-scroll effect
    useEffect(() => {
        if (autoScroll && scrollRef.current && !showSearch) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs, autoScroll, showSearch]);

    const handleScroll = (event: React.UIEvent<HTMLDivElement>) => {
        const { scrollTop, scrollHeight, clientHeight } = event.currentTarget;
        const atBottom = scrollHeight - scrollTop - clientHeight < 50;
        setIsAtBottom(atBottom);
        if (!atBottom && autoScroll) {
            setAutoScroll(false);
        } else if (atBottom && !autoScroll) {
            setAutoScroll(true);
        }
    };

    const toggleAutoScroll = () => {
        setAutoScroll(!autoScroll);
        if (!autoScroll && scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
            setIsAtBottom(true);
        }
    };

    const handleDownload = useCallback(() => {
        const blob = new Blob([logs], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = buildId ? `docker-build-${buildId}.log` : 'docker-build.log';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, [logs, buildId]);

    const navigateMatch = useCallback((direction: 'next' | 'prev') => {
        if (matchCount === 0) return;
        let newMatch;
        if (direction === 'next') {
            newMatch = (currentMatch + 1) % matchCount;
        } else {
            newMatch = (currentMatch - 1 + matchCount) % matchCount;
        }
        setCurrentMatch(newMatch);
        const matchEl = matchRefs.current[newMatch];
        if (matchEl) {
            matchEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }, [currentMatch, matchCount]);

    const jumpToError = useCallback((errorIdx: number) => {
        const errorEl = errorRefs.current[errorIdx];
        if (errorEl) {
            setAutoScroll(false);
            errorEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }, []);

    const jumpToFirstError = useCallback(() => {
        if (errorRefs.current[0]) {
            setAutoScroll(false);
            errorRefs.current[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }, []);

    const jumpToStage = useCallback((stageId: number) => {
        const stageEl = stageRefs.current.get(stageId);
        if (stageEl) {
            setAutoScroll(false);
            setShowStageJump(false);
            stageEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }, []);

    const copyLine = useCallback((lineNumber: number, content: string) => {
        // Strip ANSI codes for clipboard
        const cleanContent = content.replace(/\x1b\[[0-9;]*m/g, '');
        navigator.clipboard.writeText(cleanContent);
        setCopiedLine(lineNumber);
        setTimeout(() => setCopiedLine(null), 1500);
    }, []);

    // Auto-scroll to first error when build fails
    useEffect(() => {
        if (buildStatus === 'failed' && !hasAutoScrolledToError && stats.errors > 0 && errorRefs.current[0]) {
            setHasAutoScrolledToError(true);
            setAutoScroll(false);
            setTimeout(() => {
                errorRefs.current[0]?.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 100);
        }
    }, [buildStatus, hasAutoScrolledToError, stats.errors]);

    const copyErrorsToClipboard = useCallback(() => {
        const errorText = errorLines
            .map(line => `Line ${line.lineNumber}: ${line.content}`)
            .join('\n');
        navigator.clipboard.writeText(errorText);
    }, [errorLines]);

    // Keyboard shortcuts
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'f' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                setShowSearch(true);
            } else if (e.key === 'Escape') {
                setShowSearch(false);
                setSearchQuery('');
                setShowStageJump(false);
            } else if (e.key === 'Enter' && showSearch) {
                e.preventDefault();
                navigateMatch(e.shiftKey ? 'prev' : 'next');
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [showSearch, navigateMatch]);

    // Close stage dropdown on click outside
    useEffect(() => {
        if (!showStageJump) return;
        const handleClick = (e: MouseEvent) => {
            const target = e.target as HTMLElement;
            if (!target.closest('[data-stage-dropdown]')) {
                setShowStageJump(false);
            }
        };
        document.addEventListener('click', handleClick);
        return () => document.removeEventListener('click', handleClick);
    }, [showStageJump]);

    const highlightMatch = (text: string, query: string): React.ReactNode => {
        if (!query) return parseAnsiLine(text);
        const parts = text.split(new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'));
        return parts.map((part, i) =>
            part.toLowerCase() === query.toLowerCase() ? (
                <mark key={i} className="bg-yellow-500 text-black px-0.5 rounded">{part}</mark>
            ) : (
                <React.Fragment key={i}>{parseAnsiLine(part)}</React.Fragment>
            )
        );
    };

    const containerClasses = isFullscreen
        ? "fixed inset-0 z-50 bg-black flex flex-col"
        : cn("relative border rounded-md bg-black flex flex-col", className);

    return (
        <div className={containerClasses}>
            {/* Error Summary Panel */}
            {showErrorSummary && stats.errors > 0 && (
                <div className="bg-red-950/80 border-b border-red-800 p-2 shrink-0">
                    <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-2 text-red-300 text-xs font-medium">
                            <AlertCircle className="h-3.5 w-3.5" />
                            <span>{stats.errors} error{stats.errors !== 1 ? 's' : ''} found</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <Button
                                variant="ghost"
                                size="sm"
                                className="h-5 px-2 text-xs text-red-300 hover:text-white hover:bg-red-900"
                                onClick={copyErrorsToClipboard}
                                title="Copy all errors to clipboard"
                            >
                                <Copy className="h-3 w-3 mr-1" />
                                Copy
                            </Button>
                            <Button
                                variant="ghost"
                                size="sm"
                                className="h-5 px-2 text-xs text-red-300 hover:text-white hover:bg-red-900"
                                onClick={jumpToFirstError}
                            >
                                Jump to first
                            </Button>
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-5 w-5 text-red-400 hover:text-white"
                                onClick={() => setShowErrorSummary(false)}
                            >
                                <X className="h-3 w-3" />
                            </Button>
                        </div>
                    </div>
                    <div className="space-y-0.5 max-h-20 overflow-auto">
                        {errorLines.map((line, idx) => (
                            <div
                                key={idx}
                                className="text-xs text-red-200 truncate cursor-pointer hover:bg-red-900/50 px-1 rounded"
                                onClick={() => jumpToError(line.errorIndex)}
                                title={`Line ${line.lineNumber}: ${line.content}`}
                            >
                                <span className="text-red-400">L{line.lineNumber}:</span> {line.content.slice(0, 100)}
                            </div>
                        ))}
                        {stats.errors > 10 && (
                            <div className="text-xs text-red-400 px-1">...and {stats.errors - 10} more</div>
                        )}
                    </div>
                </div>
            )}

            {/* Toolbar */}
            <div className="flex items-center justify-between px-2 py-1.5 bg-gray-900/80 border-b border-gray-800 shrink-0">
                {/* Left: Stats & Filters */}
                <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1 text-xs">
                        <button
                            onClick={() => setFilterMode('all')}
                            className={cn(
                                "px-2 py-0.5 rounded transition-colors",
                                filterMode === 'all' ? "bg-gray-700 text-white" : "text-gray-400 hover:text-white"
                            )}
                        >
                            All
                        </button>
                        <button
                            onClick={() => setFilterMode('errors')}
                            className={cn(
                                "px-2 py-0.5 rounded flex items-center gap-1 transition-colors",
                                filterMode === 'errors' ? "bg-red-900 text-red-200" : "text-gray-400 hover:text-red-300"
                            )}
                        >
                            <AlertCircle className="h-3 w-3" />
                            {stats.errors}
                        </button>
                        <button
                            onClick={() => setFilterMode('warnings')}
                            className={cn(
                                "px-2 py-0.5 rounded flex items-center gap-1 transition-colors",
                                filterMode === 'warnings' ? "bg-yellow-900 text-yellow-200" : "text-gray-400 hover:text-yellow-300"
                            )}
                        >
                            <AlertTriangle className="h-3 w-3" />
                            {stats.warnings}
                        </button>
                        <button
                            onClick={() => setFilterMode('steps')}
                            className={cn(
                                "px-2 py-0.5 rounded flex items-center gap-1 transition-colors",
                                filterMode === 'steps' ? "bg-blue-900 text-blue-200" : "text-gray-400 hover:text-blue-300"
                            )}
                        >
                            <Layers className="h-3 w-3" />
                            {stats.steps}
                        </button>
                    </div>

                    {progress.total > 0 && (
                        <div className="flex items-center gap-2 text-xs text-gray-400 border-l border-gray-700 pl-2 ml-1">
                            <span>Step {progress.current}/{progress.total}</span>
                            <div className="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-blue-500 transition-all duration-300"
                                    style={{ width: `${(progress.current / progress.total) * 100}%` }}
                                />
                            </div>
                        </div>
                    )}

                    {(totalBuildTime > 0 || stats.cached > 0) && (
                        <div className="flex items-center gap-2 text-xs text-gray-400 border-l border-gray-700 pl-2 ml-1">
                            {totalBuildTime > 0 && (
                                <span className="flex items-center gap-1" title="Total build time">
                                    <Clock className="h-3 w-3" />
                                    {formatDuration(totalBuildTime)}
                                </span>
                            )}
                            {stats.cached > 0 && (
                                <span className="text-green-400" title="Cached layers">
                                    {stats.cached} cached
                                </span>
                            )}
                        </div>
                    )}
                </div>

                {/* Right: Tools */}
                <div className="flex items-center gap-1">
                    {showSearch ? (
                        <div className="flex items-center gap-1 mr-1">
                            <Input
                                type="text"
                                placeholder={useRegex ? "Regex..." : "Search..."}
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className={cn("h-6 w-32 text-xs bg-gray-800 border-gray-700 text-white", useRegex && "font-mono")}
                                autoFocus
                            />
                            <Button
                                variant="ghost"
                                size="icon"
                                className={cn("h-6 w-6", useRegex ? "text-purple-400" : "text-gray-400 hover:text-white")}
                                onClick={() => setUseRegex(!useRegex)}
                                title="Toggle regex mode"
                            >
                                <Regex className="h-3 w-3" />
                            </Button>
                            {matchCount > 0 && (
                                <span className="text-xs text-gray-400 whitespace-nowrap">
                                    {currentMatch + 1}/{matchCount}
                                </span>
                            )}
                            <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-400 hover:text-white" onClick={() => navigateMatch('prev')} disabled={matchCount === 0}>
                                <ChevronUp className="h-3 w-3" />
                            </Button>
                            <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-400 hover:text-white" onClick={() => navigateMatch('next')} disabled={matchCount === 0}>
                                <ChevronDown className="h-3 w-3" />
                            </Button>
                            <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-400 hover:text-white" onClick={() => { setShowSearch(false); setSearchQuery(''); }}>
                                <X className="h-3 w-3" />
                            </Button>
                        </div>
                    ) : (
                        <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-400 hover:text-white" onClick={() => setShowSearch(true)} title="Search (Ctrl+F)">
                            <Search className="h-4 w-4" />
                        </Button>
                    )}

                    {stats.errors > 0 && !showErrorSummary && (
                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 text-red-400 hover:text-red-300"
                            onClick={() => setShowErrorSummary(true)}
                            title="Show error summary"
                        >
                            <AlertCircle className="h-4 w-4" />
                        </Button>
                    )}

                    <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-400 hover:text-white" onClick={handleDownload} title="Download logs" disabled={!logs}>
                        <Download className="h-4 w-4" />
                    </Button>

                    {/* Stage Jump Dropdown */}
                    {stages.length > 0 && (
                        <div className="relative" data-stage-dropdown>
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-6 w-6 text-gray-400 hover:text-white"
                                onClick={() => setShowStageJump(!showStageJump)}
                                title="Jump to build stage"
                            >
                                <ChevronsUpDown className="h-4 w-4" />
                            </Button>
                            {showStageJump && (
                                <div className="absolute right-0 top-full mt-1 z-50 bg-gray-800 border border-gray-700 rounded-md shadow-lg max-h-64 overflow-auto min-w-[200px]">
                                    <div className="px-2 py-1 text-xs text-gray-400 border-b border-gray-700 font-medium">
                                        Jump to Stage ({stages.length})
                                    </div>
                                    {stages.map((stage) => (
                                        <button
                                            key={stage.id}
                                            onClick={() => jumpToStage(stage.id)}
                                            className="w-full px-2 py-1.5 text-left text-xs text-gray-300 hover:bg-gray-700 hover:text-white flex items-center gap-2"
                                        >
                                            <span className="text-blue-400 font-mono shrink-0">#{stage.id}</span>
                                            <span className="truncate">{stage.description}</span>
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-400 hover:text-white" onClick={() => setIsFullscreen(!isFullscreen)} title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}>
                        {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
                    </Button>

                    <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-400 hover:text-white" onClick={toggleAutoScroll} title={autoScroll ? "Pause scrolling" : "Resume auto-scroll"}>
                        {autoScroll ? <PauseCircle className="h-4 w-4" /> : <PlayCircle className="h-4 w-4" />}
                    </Button>
                </div>
            </div>

            {/* Log Content */}
            <div
                ref={scrollRef}
                className="flex-1 overflow-auto"
                onScroll={handleScroll}
            >
                <div className="font-mono text-xs leading-relaxed">
                    {filteredLines.length === 0 ? (
                        <div className="p-4 text-gray-500">
                            {logs ? `No ${filterMode === 'all' ? 'logs' : filterMode} to display` : 'Waiting for logs...'}
                        </div>
                    ) : (
                        filteredLines.map((line, idx) => {
                            const lineClasses = cn(
                                "px-4 py-0.5 flex hover:bg-gray-900/50 group",
                                line.isError && "bg-red-950/50 border-l-2 border-red-500",
                                line.isWarning && !line.isError && "bg-yellow-950/30 border-l-2 border-yellow-500",
                                line.isStep && !line.isError && !line.isWarning && "bg-blue-950/30 border-l-2 border-blue-500 font-semibold",
                                line.matchesSearch && line.matchIndex === currentMatch && "bg-yellow-900/50 ring-1 ring-yellow-500"
                            );

                            return (
                                <div
                                    key={`${line.lineNumber}-${idx}`}
                                    ref={(el) => {
                                        if (line.matchesSearch) matchRefs.current[line.matchIndex] = el;
                                        if (line.isError) errorRefs.current[line.errorIndex] = el;
                                        if (line.stageId !== undefined) stageRefs.current.set(line.stageId, el);
                                    }}
                                    className={lineClasses}
                                >
                                    <span className="select-none text-gray-600 w-12 shrink-0 text-right pr-3 border-r border-gray-800 mr-3">
                                        {line.lineNumber}
                                    </span>
                                    <span className={cn(
                                        "text-gray-300 whitespace-pre-wrap break-all flex-1",
                                        line.isError && "text-red-300",
                                        line.isWarning && !line.isError && "text-yellow-300"
                                    )}>
                                        {searchQuery ? highlightMatch(line.content, searchQuery) : parseAnsiLine(line.content)}
                                    </span>
                                    {line.timing?.duration && (
                                        <span className="ml-2 shrink-0 text-xs text-cyan-400 font-mono">
                                            {formatDuration(line.timing.duration)}
                                        </span>
                                    )}
                                    {line.timing?.cached && (
                                        <span className="ml-2 shrink-0 text-xs text-green-400 font-mono">
                                            CACHED
                                        </span>
                                    )}
                                    {/* Copy button - visible on hover */}
                                    <button
                                        onClick={(e) => { e.stopPropagation(); copyLine(line.lineNumber, line.content); }}
                                        className="ml-1 shrink-0 opacity-0 group-hover:opacity-100 text-gray-500 hover:text-white transition-opacity"
                                        title="Copy line"
                                    >
                                        {copiedLine === line.lineNumber ? (
                                            <span className="text-xs text-green-400">Copied!</span>
                                        ) : (
                                            <Copy className="h-3 w-3" />
                                        )}
                                    </button>
                                </div>
                            );
                        })
                    )}
                </div>
            </div>

            {!isAtBottom && !autoScroll && (
                <Button
                    variant="secondary"
                    size="sm"
                    className="absolute bottom-4 right-4 shadow-lg opacity-90"
                    onClick={() => {
                        setAutoScroll(true);
                        if (scrollRef.current) {
                            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
                        }
                    }}
                >
                    <ArrowDownCircle className="mr-2 h-4 w-4" />
                    Scroll to Bottom
                </Button>
            )}
        </div>
    );
}
