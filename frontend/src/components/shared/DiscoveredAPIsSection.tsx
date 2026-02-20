import { Loader2, Globe, AlertCircle, CheckCircle, RefreshCw, Copy } from 'lucide-react';
import { Card, CardContent } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { ApiSpec } from '../../lib/api';
import { generateCurlExample, copyToClipboard } from './benchmark-utils';

interface DiscoveredAPIsSectionProps {
  apiSpec: ApiSpec | undefined;
  apiSpecLoading: boolean;
  endpointUrl: string | null;
  modelName: string;
  isRunning: boolean;
  onRefresh: () => void;
}

export function getApiTypeLabel(apiType: string): string {
  switch (apiType) {
    case 'openai': return 'OpenAI-Compatible API';
    case 'fastapi': return 'FastAPI (OpenAPI)';
    case 'audio': return 'Audio API';
    default: return 'Generic API';
  }
}

function getStatusBadge(status: number): { className: string; label: string } {
  if (status >= 200 && status < 300) {
    return { className: 'bg-green-100 text-green-700', label: String(status) };
  }
  if (status === 422 || status === 405) {
    return { className: 'bg-amber-100 text-amber-700', label: String(status) };
  }
  return { className: 'bg-red-100 text-red-700', label: String(status) };
}

export function DiscoveredAPIsSection({
  apiSpec,
  apiSpecLoading,
  endpointUrl,
  modelName,
  isRunning,
  onRefresh,
}: DiscoveredAPIsSectionProps) {
  return (
    <section className="space-y-6">
      <div className="flex items-center justify-between border-b pb-2">
        <h2 className="text-xl font-semibold flex items-center gap-2 text-gray-800">
          <Globe className="size-5 text-blue-600" />
          Discovered APIs
          {apiSpec && !apiSpec.error && (
            <Badge variant="outline" className="ml-2">
              {getApiTypeLabel(apiSpec.api_type)}
            </Badge>
          )}
        </h2>
        {isRunning && (
          <Button variant="outline" size="sm" onClick={onRefresh} disabled={apiSpecLoading}>
            {apiSpecLoading ? <Loader2 className="size-3 animate-spin mr-1" /> : <RefreshCw className="size-3 mr-1" />}
            Discover APIs
          </Button>
        )}
      </div>

      <Card className="min-w-0">
        <CardContent className="p-6">
          {!endpointUrl ? (
            <div className="text-center py-8 text-gray-500">
              No endpoint available. Start the deployment to discover API endpoints.
            </div>
          ) : !isRunning ? (
            <div className="text-center py-8 text-gray-500">
              Deployment is not running. Start the deployment to discover API endpoints.
            </div>
          ) : apiSpecLoading ? (
            <div className="flex flex-col items-center justify-center py-12">
              <Loader2 className="size-8 text-blue-600 animate-spin" />
              <p className="mt-4 text-sm text-gray-600">Discovering APIs from container...</p>
            </div>
          ) : apiSpec?.error ? (
            <div className="text-center py-8 text-gray-500">
              <AlertCircle className="size-8 text-amber-500 mx-auto mb-2" />
              <p>{apiSpec.error}</p>
            </div>
          ) : apiSpec && apiSpec.endpoints.length > 0 ? (
            <div className="space-y-6">
              {/* Show OpenAPI spec info if available */}
              {apiSpec.openapi_spec && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-4">
                  <div className="flex items-center gap-2 text-green-800">
                    <CheckCircle className="size-4" />
                    <span className="font-medium">OpenAPI Spec Detected</span>
                  </div>
                  <p className="text-sm text-green-700 mt-1">
                    {apiSpec.openapi_spec.info?.title || 'API'} - {apiSpec.openapi_spec.info?.version || 'v1'}
                  </p>
                </div>
              )}

              {apiSpec.endpoints.map((endpoint, index) => {
                const curlExample = generateCurlExample(endpointUrl || '', endpoint.method, endpoint.path, modelName, endpoint.sample_body, endpoint.requires_file_upload, endpoint.request_schema);
                const statusBadge = endpoint.status ? getStatusBadge(endpoint.status) : null;
                const isValidationError = endpoint.status === 422;
                return (
                  <div key={index} className="border rounded-lg p-5 hover:bg-gray-50 transition-colors w-full min-w-0">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 mb-3">
                      <div className="flex items-center gap-3 flex-wrap">
                        <Badge variant={endpoint.method === 'GET' ? 'secondary' : 'default'} className="font-mono shrink-0">
                          {endpoint.method}
                        </Badge>
                        <code className="text-sm font-semibold text-gray-800 break-all">{endpoint.path}</code>
                        {endpoint.tags && endpoint.tags.length > 0 && (
                          <span className="text-xs text-gray-400">{endpoint.tags.join(', ')}</span>
                        )}
                      </div>
                      <span className="text-sm font-medium text-gray-600 shrink-0">{endpoint.summary}</span>
                    </div>
                    {endpoint.description && endpoint.description !== endpoint.summary && (
                      <p className="text-sm text-gray-600 mb-4">{endpoint.description}</p>
                    )}

                    <div className="space-y-3 w-full min-w-0">
                      {/* Request (cURL) on top */}
                      <div>
                        <div className="text-xs text-gray-500 font-medium mb-1 uppercase tracking-wider">Request</div>
                        <div className="relative group w-full min-w-0">
                          <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-xs font-mono w-full min-w-0">
                            {curlExample}
                          </pre>
                          <Button
                            variant="secondary"
                            size="sm"
                            className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity h-7"
                            onClick={() => copyToClipboard(curlExample)}
                          >
                            <Copy className="size-3 mr-1" />
                            Copy
                          </Button>
                        </div>
                      </div>

                      {/* Response below */}
                      <div>
                        <div className="text-xs text-gray-500 font-medium mb-1 uppercase tracking-wider">
                          Response {statusBadge && (
                            <span className={`ml-2 px-1.5 py-0.5 rounded text-[10px] font-medium ${statusBadge.className}`}>
                              {statusBadge.label}
                            </span>
                          )}
                          {isValidationError && (
                            <span className="ml-2 text-[10px] text-amber-600">
                              Endpoint available — probe request had validation errors
                            </span>
                          )}
                        </div>
                        <div className="relative group w-full min-w-0">
                          {isValidationError && endpoint.requires_file_upload ? (
                            <div className="bg-amber-50 border border-amber-200 p-4 rounded-lg text-xs text-amber-700">
                              This endpoint requires a file upload (multipart/form-data) and cannot be probed with JSON.
                              Use the cURL example above to test it.
                            </div>
                          ) : endpoint.response ? (
                            <pre className="bg-gray-50 border p-4 rounded-lg overflow-x-auto text-xs font-mono w-full min-w-0 max-h-48">
                              {JSON.stringify(endpoint.response, null, 2)}
                            </pre>
                          ) : (
                            <div className="bg-gray-50 border p-4 rounded-lg text-xs text-gray-400">
                              No sample response available
                            </div>
                          )}
                          {endpoint.response && !(isValidationError && endpoint.requires_file_upload) && (
                            <Button
                              variant="secondary"
                              size="sm"
                              className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity h-7"
                              onClick={() => copyToClipboard(JSON.stringify(endpoint.response, null, 2))}
                            >
                              <Copy className="size-3 mr-1" />
                              Copy
                            </Button>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <Globe className="size-8 text-gray-400 mx-auto mb-2" />
              <p>No APIs discovered. Click "Discover APIs" to probe the container.</p>
            </div>
          )}
        </CardContent>
      </Card>
    </section>
  );
}
