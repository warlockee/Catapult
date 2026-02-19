import { useState, useEffect } from 'react';
import { RefreshCw, ExternalLink, Loader2, Clock } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { api, MlflowMetadata } from '../lib/api';

interface MlflowMetadataCardProps {
  versionId: string;
  mlflowUrl: string;
}

function formatTimestamp(ms: number): string {
  return new Date(ms).toLocaleString();
}

function RunContent({ data }: { data: MlflowMetadata }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        {data.run_name && (
          <div className="overflow-hidden">
            <div className="text-sm text-gray-500 mb-1">Run Name</div>
            <div className="text-sm font-medium overflow-x-auto whitespace-nowrap">{data.run_name}</div>
          </div>
        )}
        {data.status && (
          <div>
            <div className="text-sm text-gray-500 mb-1">Status</div>
            <Badge variant={data.status === 'FINISHED' ? 'default' : 'secondary'}>
              {data.status}
            </Badge>
          </div>
        )}
        {data.start_time && (
          <div>
            <div className="text-sm text-gray-500 mb-1">Start Time</div>
            <div className="text-sm">{formatTimestamp(data.start_time)}</div>
          </div>
        )}
        {data.end_time && (
          <div>
            <div className="text-sm text-gray-500 mb-1">End Time</div>
            <div className="text-sm">{formatTimestamp(data.end_time)}</div>
          </div>
        )}
      </div>

      {data.metrics && Object.keys(data.metrics).length > 0 && (
        <div>
          <div className="text-sm text-gray-500 mb-2">Metrics</div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {Object.entries(data.metrics).map(([key, value]) => (
              <div key={key} className="bg-gray-50 p-2 rounded overflow-hidden">
                <div className="text-xs text-gray-500 overflow-x-auto whitespace-nowrap">{key}</div>
                <div className="text-sm font-mono overflow-x-auto whitespace-nowrap">
                  {typeof value === 'number' ? value.toFixed(4) : value}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.params && Object.keys(data.params).length > 0 && (
        <div>
          <div className="text-sm text-gray-500 mb-2">Parameters</div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {Object.entries(data.params).map(([key, value]) => (
              <div key={key} className="bg-gray-50 p-2 rounded overflow-hidden">
                <div className="text-xs text-gray-500 overflow-x-auto whitespace-nowrap">{key}</div>
                <div className="text-sm font-mono overflow-x-auto whitespace-nowrap">
                  {value}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.tags && Object.keys(data.tags).length > 0 && (
        <div>
          <div className="text-sm text-gray-500 mb-2">Tags</div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(data.tags).map(([key, value]) => (
              <Badge key={key} variant="outline" className="max-w-full overflow-hidden">
                <span className="overflow-x-auto whitespace-nowrap block">{key}: {value}</span>
              </Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ExperimentContent({ data }: { data: MlflowMetadata }) {
  return (
    <div className="grid grid-cols-2 gap-4">
      {data.experiment_name && (
        <div className="overflow-hidden">
          <div className="text-sm text-gray-500 mb-1">Experiment Name</div>
          <div className="text-sm font-medium overflow-x-auto whitespace-nowrap">{data.experiment_name}</div>
        </div>
      )}
      {data.experiment_id && (
        <div className="overflow-hidden">
          <div className="text-sm text-gray-500 mb-1">Experiment ID</div>
          <code className="text-sm block overflow-x-auto whitespace-nowrap">{data.experiment_id}</code>
        </div>
      )}
      {data.lifecycle_stage && (
        <div>
          <div className="text-sm text-gray-500 mb-1">Lifecycle Stage</div>
          <Badge>{data.lifecycle_stage}</Badge>
        </div>
      )}
      {data.artifact_location && (
        <div className="col-span-2 overflow-hidden">
          <div className="text-sm text-gray-500 mb-1">Artifact Location</div>
          <code className="text-xs block overflow-x-auto whitespace-nowrap">{data.artifact_location}</code>
        </div>
      )}
    </div>
  );
}

function RegisteredModelContent({ data }: { data: MlflowMetadata }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        {data.model_name && (
          <div className="overflow-hidden">
            <div className="text-sm text-gray-500 mb-1">Model Name</div>
            <div className="text-sm font-medium overflow-x-auto whitespace-nowrap">{data.model_name}</div>
          </div>
        )}
        {data.description && (
          <div className="col-span-2 overflow-hidden">
            <div className="text-sm text-gray-500 mb-1">Description</div>
            <div className="text-sm overflow-x-auto">{data.description}</div>
          </div>
        )}
      </div>

      {data.latest_versions && data.latest_versions.length > 0 && (
        <div>
          <div className="text-sm text-gray-500 mb-2">Latest Versions</div>
          <div className="space-y-2">
            {data.latest_versions.map((v, i) => (
              <div key={i} className="flex items-center gap-2 bg-gray-50 p-2 rounded overflow-x-auto">
                <Badge variant="outline" className="shrink-0">v{v.version}</Badge>
                {v.current_stage && <Badge variant="secondary" className="shrink-0">{v.current_stage}</Badge>}
                {v.status && <span className="text-xs text-gray-500 shrink-0">{v.status}</span>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

const RESOURCE_TYPE_LABELS: Record<string, string> = {
  run: 'Run',
  experiment: 'Experiment',
  registered_model: 'Registered Model',
};

export function MlflowMetadataCard({ versionId, mlflowUrl }: MlflowMetadataCardProps) {
  const [metadata, setMetadata] = useState<MlflowMetadata | null>(null);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const doSync = async () => {
    try {
      setSyncing(true);
      setError(null);
      const data = await api.syncMlflowMetadata(versionId);
      setMetadata(data);
    } catch (err: any) {
      setError(err.message || 'Failed to sync MLflow metadata');
    } finally {
      setSyncing(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await api.getMlflowMetadata(versionId);
        if (!cancelled) setMetadata(data);
      } catch (err: any) {
        if (!cancelled) {
          // 400 means no cached data yet — auto-sync instead of showing "Fetch" button
          if (err.status === 400) {
            setMetadata(null);
            setLoading(false);
            doSync();
            return;
          } else {
            setError(err.message || 'Failed to load MLflow metadata');
          }
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [versionId]);

  const handleSync = doSync;

  return (
    <Card className="overflow-hidden">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <div className="flex items-center gap-2">
          <CardTitle className="text-lg">MLflow Metadata</CardTitle>
          {metadata && (
            <Badge variant="outline">
              {RESOURCE_TYPE_LABELS[metadata.resource_type] || metadata.resource_type}
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          <a
            href={mlflowUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:text-blue-700"
            title="Open in MLflow"
          >
            <ExternalLink className="size-4" />
          </a>
          {(metadata || !syncing) && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleSync}
              disabled={syncing}
            >
              {syncing ? (
                <Loader2 className="size-4 animate-spin mr-1" />
              ) : (
                <RefreshCw className="size-4 mr-1" />
              )}
              {metadata ? 'Refresh' : 'Fetch'}
            </Button>
          )}
          {!metadata && syncing && (
            <Loader2 className="size-4 animate-spin text-gray-400" />
          )}
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="flex items-center gap-2 py-4 text-gray-500">
            <Loader2 className="size-4 animate-spin" />
            <span className="text-sm">Loading MLflow metadata...</span>
          </div>
        ) : error ? (
          <div className="py-4">
            <p className="text-sm text-red-600">{error}</p>
            <Button variant="outline" size="sm" onClick={handleSync} className="mt-2">
              <RefreshCw className="size-4 mr-1" />
              Try Again
            </Button>
          </div>
        ) : !metadata ? (
          syncing ? (
            <div className="flex items-center gap-2 py-4 text-gray-500">
              <Loader2 className="size-4 animate-spin" />
              <span className="text-sm">Fetching MLflow metadata...</span>
            </div>
          ) : (
            <div className="text-center py-6 text-gray-500">
              <p className="text-sm">No MLflow metadata cached yet.</p>
              <Button variant="outline" size="sm" onClick={handleSync} className="mt-3">
                <RefreshCw className="size-4 mr-1" />
                Fetch from MLflow
              </Button>
            </div>
          )
        ) : (
          <div className="space-y-4">
            {metadata.resource_type === 'run' && <RunContent data={metadata} />}
            {metadata.resource_type === 'experiment' && <ExperimentContent data={metadata} />}
            {metadata.resource_type === 'registered_model' && <RegisteredModelContent data={metadata} />}

            {metadata.fetched_at && (
              <div className="flex items-center gap-1 pt-2 border-t text-xs text-gray-400">
                <Clock className="size-3" />
                <span>Last synced: {new Date(metadata.fetched_at).toLocaleString()}</span>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
