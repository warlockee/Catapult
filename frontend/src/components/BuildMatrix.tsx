import { useState, useMemo } from 'react';
import { Copy, CheckCircle, Clock, Trash2 } from 'lucide-react';
import { Button } from './ui/button';
import type { DockerBuild } from '../lib/api';

function getCleanupBadge(build: DockerBuild) {
  if (build.is_cleaned) {
    return { text: 'Cleaned', color: 'bg-gray-500 text-white' };
  }
  if (build.is_current) {
    return { text: 'Active', color: 'bg-green-500 text-white' };
  }
  if (build.days_until_cleanup !== null && build.days_until_cleanup !== undefined) {
    if (build.days_until_cleanup <= 0) {
      return { text: 'Cleanup pending', color: 'bg-orange-500 text-white' };
    }
    return { text: `${build.days_until_cleanup}d left`, color: 'bg-yellow-500 text-black' };
  }
  return null;
}

interface BuildMatrixProps {
  builds: DockerBuild[];
  maxCells?: number;
  columns?: number;
}

export function BuildMatrix({ builds, maxCells = 28, columns = 7 }: BuildMatrixProps) {
  const [selectedBuild, setSelectedBuild] = useState<DockerBuild | null>(null);
  const [copied, setCopied] = useState(false);

  // Sort builds by created_at descending (most recent first) and take maxCells
  const sortedBuilds = useMemo(() => {
    return [...builds]
      .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
      .slice(0, maxCells);
  }, [builds, maxCells]);

  // Find most recent successful build
  const latestSuccessfulBuild = useMemo(() => {
    return sortedBuilds.find(b => b.status === 'success');
  }, [sortedBuilds]);

  // Create rows for the matrix
  const rows = useMemo(() => {
    const result: (DockerBuild | null)[][] = [];
    for (let i = 0; i < sortedBuilds.length; i += columns) {
      const row = sortedBuilds.slice(i, i + columns);
      // Pad row with nulls if not full
      while (row.length < columns) {
        row.push(null as unknown as DockerBuild);
      }
      result.push(row);
    }
    // Ensure at least 4 rows for visual consistency
    while (result.length < 4) {
      result.push(Array(columns).fill(null));
    }
    return result;
  }, [sortedBuilds, columns]);

  const getStatusColor = (build: DockerBuild | null) => {
    if (!build) return 'bg-muted';
    switch (build.status) {
      case 'success':
        return 'bg-green-500 hover:bg-green-400 cursor-pointer';
      case 'failed':
        return 'bg-red-500 hover:bg-red-400 cursor-pointer';
      case 'building':
        return 'bg-yellow-500 hover:bg-yellow-400 cursor-pointer animate-pulse';
      case 'pending':
        return 'bg-blue-500 hover:bg-blue-400 cursor-pointer';
      default:
        return 'bg-muted';
    }
  };

  const getStatusTooltip = (build: DockerBuild | null) => {
    if (!build) return 'No build';
    const date = new Date(build.created_at).toLocaleDateString();
    const time = new Date(build.created_at).toLocaleTimeString();
    let tooltip = `${build.status} - ${build.build_type} - ${date} ${time}`;

    // Add cleanup status
    if (build.is_cleaned) {
      tooltip += ' (Cleaned)';
    } else if (build.is_current) {
      tooltip += ' (Active)';
    } else if (build.days_until_cleanup !== null && build.days_until_cleanup !== undefined) {
      if (build.days_until_cleanup <= 0) {
        tooltip += ' (Cleanup pending)';
      } else {
        tooltip += ` (${build.days_until_cleanup} days until cleanup)`;
      }
    }
    return tooltip;
  };

  const handleCellClick = (build: DockerBuild | null) => {
    if (build && build.status === 'success') {
      setSelectedBuild(build);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (builds.length === 0) {
    return (
      <div className="text-sm text-muted-foreground">
        No builds yet
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Latest successful build - always shown above matrix */}
      {latestSuccessfulBuild && (
        <div className="flex items-center gap-2 text-sm flex-wrap">
          <CheckCircle className="size-4 text-green-500" />
          <span className="text-muted-foreground">Latest:</span>
          <code className="px-2 py-0.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded text-xs font-mono">
            {latestSuccessfulBuild.image_tag}
          </code>
          <Button
            variant="ghost"
            size="sm"
            className="h-6 w-6 p-0"
            onClick={() => copyToClipboard(latestSuccessfulBuild.image_tag)}
          >
            {copied ? <CheckCircle className="size-3 text-green-500" /> : <Copy className="size-3" />}
          </Button>
          {(() => {
            const badge = getCleanupBadge(latestSuccessfulBuild);
            return badge && (
              <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${badge.color}`}>
                {badge.text}
              </span>
            );
          })()}
        </div>
      )}

      {/* Build Matrix */}
      <div className="space-y-1">
        <div className="text-xs text-muted-foreground mb-2">Build History</div>
        <div className="flex flex-col gap-1">
          {rows.map((row, rowIndex) => (
            <div key={rowIndex} className="flex gap-1">
              {row.map((build, colIndex) => (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className={`size-4 rounded-sm transition-colors ${getStatusColor(build)}`}
                  title={getStatusTooltip(build)}
                  onClick={() => handleCellClick(build)}
                />
              ))}
            </div>
          ))}
        </div>
        <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <div className="size-3 rounded-sm bg-green-500" />
            <span>Success</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="size-3 rounded-sm bg-red-500" />
            <span>Failed</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="size-3 rounded-sm bg-yellow-500" />
            <span>Building</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="size-3 rounded-sm bg-muted" />
            <span>Empty</span>
          </div>
        </div>
      </div>

      {/* Selected build info - shown below matrix when clicked */}
      {selectedBuild && (
        <div className="mt-2 p-2 bg-muted/50 rounded-md">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span>{new Date(selectedBuild.created_at).toLocaleString()} - {selectedBuild.build_type}</span>
              {(() => {
                const badge = getCleanupBadge(selectedBuild);
                return badge && (
                  <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${badge.color}`}>
                    {badge.text}
                  </span>
                );
              })()}
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="h-5 text-xs"
              onClick={() => setSelectedBuild(null)}
            >
              Clear
            </Button>
          </div>
          <div className="flex items-center gap-2 mt-1">
            <code className="px-2 py-0.5 bg-background rounded text-xs font-mono">
              {selectedBuild.image_tag}
            </code>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0"
              onClick={() => copyToClipboard(selectedBuild.image_tag)}
            >
              <Copy className="size-3" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
