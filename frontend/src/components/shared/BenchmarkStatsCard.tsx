import { Loader2, Gauge, Zap, RotateCw, CheckCircle } from 'lucide-react';
import { Card, CardContent } from '../ui/card';
import { Button } from '../ui/button';
import { ApiSpec, BenchmarkSummary, Benchmark } from '../../lib/api';
import { BenchmarkRunningState } from './BenchmarkRunningState';

interface BenchmarkStatsCardProps {
  benchmarkRunning: boolean;
  activeBenchmark: Benchmark | undefined;
  activeBenchmarkId: string | null;
  benchmarkSummary: BenchmarkSummary | undefined;
  apiSpec: ApiSpec | undefined;
  apiSpecLoading: boolean;
  selectedEndpoint: string | null;
  onSelectEndpoint: (endpoint: string) => void;
  onRunBenchmark: () => void;
  runBenchmarkPending: boolean;
  onCancel?: () => void;
  cancelPending?: boolean;
  formatDate: (date: string) => string;
}

export function BenchmarkStatsCard({
  benchmarkRunning,
  activeBenchmark,
  activeBenchmarkId,
  benchmarkSummary,
  apiSpec,
  apiSpecLoading,
  selectedEndpoint,
  onSelectEndpoint,
  onRunBenchmark,
  runBenchmarkPending,
  onCancel,
  cancelPending,
  formatDate,
}: BenchmarkStatsCardProps) {
  return (
    <Card className="min-w-0">
      <CardContent className="p-6">
        {/* State: Benchmark is running */}
        {benchmarkRunning ? (
          <BenchmarkRunningState
            benchmarkId={activeBenchmarkId}
            benchmark={activeBenchmark}
            onCancel={onCancel}
            cancelPending={cancelPending}
          />
        ) : benchmarkSummary?.has_data ? (
          /* State: Has benchmark data */
          <div className="space-y-6">
            {/* Available APIs & Benchmark Target - Selectable for re-run */}
            <div className="bg-gray-50 rounded-lg p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium text-gray-700">Select API to Re-benchmark</div>
                {benchmarkSummary.last_run_at && (
                  <div className="text-xs text-gray-400">
                    Last benchmark: {new Date(benchmarkSummary.last_run_at).toLocaleString()}
                  </div>
                )}
              </div>

              {/* API Endpoints List - Selectable */}
              {/* Filter out endpoints that require file uploads (can't benchmark with JSON) */}
              <div className="flex flex-wrap gap-2">
                {apiSpec?.endpoints?.filter(ep =>
                  ep.method === 'POST' && !ep.path.includes('/health') && !ep.path.includes('/metrics') && !ep.path.includes('/batch')
                ).slice(0, 8).map((ep, i) => {
                  const isBenchmarked = benchmarkSummary.benchmark_endpoint === ep.path;
                  const isSelected = selectedEndpoint === ep.path;
                  return (
                    <button
                      key={i}
                      onClick={() => onSelectEndpoint(ep.path)}
                      className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded text-xs font-mono transition-all ${
                        isSelected
                          ? 'bg-purple-600 text-white ring-2 ring-purple-300 shadow-sm'
                          : isBenchmarked
                            ? 'bg-green-100 text-green-800 ring-1 ring-green-300 hover:bg-green-200'
                            : 'bg-white text-gray-600 border border-gray-200 hover:border-purple-300 hover:bg-purple-50'
                      }`}
                    >
                      <span className={`font-medium ${isSelected ? 'text-purple-200' : 'text-blue-600'}`}>
                        {ep.method}
                      </span>
                      <span>{ep.path}</span>
                      {isSelected && <CheckCircle className="size-3 ml-1" />}
                      {!isSelected && isBenchmarked && (
                        <span className="ml-1 text-green-600 text-[10px]">last tested</span>
                      )}
                    </button>
                  );
                })}
                {apiSpecLoading && (
                  <div className="flex items-center gap-1 text-xs text-gray-400">
                    <Loader2 className="size-3 animate-spin" />
                    Discovering APIs...
                  </div>
                )}
                {!apiSpecLoading && (!apiSpec?.endpoints || apiSpec.endpoints.length === 0) && (
                  <div className="text-xs text-gray-400">No APIs discovered</div>
                )}
              </div>

              {/* Last Benchmarked Endpoint & Model Type */}
              <div className="flex items-center gap-2 text-xs flex-wrap">
                {benchmarkSummary.benchmark_endpoint && (
                  <>
                    <span className="text-gray-500">Last Tested:</span>
                    <span className="px-2 py-0.5 rounded font-medium bg-blue-100 text-blue-700 font-mono">
                      {benchmarkSummary.benchmark_endpoint}
                    </span>
                  </>
                )}
                {benchmarkSummary.model_type && (
                  <>
                    <span className="text-gray-400">|</span>
                    <span className="text-gray-500">Model:</span>
                    <span className={`px-2 py-0.5 rounded font-medium ${
                      benchmarkSummary.model_type === 'audio' ? 'bg-purple-100 text-purple-700' :
                      benchmarkSummary.model_type === 'multimodal' ? 'bg-blue-100 text-blue-700' :
                      'bg-green-100 text-green-700'
                    }`}>
                      {benchmarkSummary.model_type}
                    </span>
                  </>
                )}
              </div>
            </div>

            {/* TTFT/TPS */}
            {(benchmarkSummary.ttft_avg_ms !== null && benchmarkSummary.ttft_avg_ms !== undefined) && (
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gradient-to-br from-cyan-50 to-cyan-100/50 p-4 rounded-lg border border-cyan-100">
                  <div className="text-xs text-cyan-600 font-medium uppercase tracking-wider mb-1">TTFT (Time to First Token)</div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-bold text-cyan-700">
                      {benchmarkSummary.ttft_avg_ms?.toFixed(0) ?? '-'}
                    </span>
                    <span className="text-sm text-cyan-500">ms avg</span>
                    {benchmarkSummary.ttft_p95_ms && (
                      <span className="text-xs text-cyan-400 ml-2">p95: {benchmarkSummary.ttft_p95_ms.toFixed(0)}ms</span>
                    )}
                  </div>
                </div>
                <div className="bg-gradient-to-br from-orange-50 to-orange-100/50 p-4 rounded-lg border border-orange-100">
                  <div className="text-xs text-orange-600 font-medium uppercase tracking-wider mb-1">TPS (Tokens/Second)</div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-bold text-orange-700">
                      {benchmarkSummary.tokens_per_second_avg?.toFixed(1) ?? '-'}
                    </span>
                    <span className="text-sm text-orange-500">tok/s</span>
                    {benchmarkSummary.total_tokens_generated && (
                      <span className="text-xs text-orange-400 ml-2">{benchmarkSummary.total_tokens_generated.toLocaleString()} total</span>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Latency & Throughput Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
              <div className="bg-gradient-to-br from-blue-50 to-blue-100/50 p-4 rounded-lg border border-blue-100">
                <div className="text-xs text-blue-600 font-medium uppercase tracking-wider mb-1">Latency Avg</div>
                <div className="flex items-baseline gap-1">
                  <span className="text-2xl font-bold text-blue-700">
                    {benchmarkSummary.latency_avg_ms?.toFixed(1) ?? '-'}
                  </span>
                  <span className="text-xs text-blue-500">ms</span>
                </div>
              </div>
              <div className="bg-gradient-to-br from-green-50 to-green-100/50 p-4 rounded-lg border border-green-100">
                <div className="text-xs text-green-600 font-medium uppercase tracking-wider mb-1">P50</div>
                <div className="flex items-baseline gap-1">
                  <span className="text-2xl font-bold text-green-700">
                    {benchmarkSummary.latency_p50_ms?.toFixed(1) ?? '-'}
                  </span>
                  <span className="text-xs text-green-500">ms</span>
                </div>
              </div>
              <div className="bg-gradient-to-br from-yellow-50 to-yellow-100/50 p-4 rounded-lg border border-yellow-100">
                <div className="text-xs text-yellow-600 font-medium uppercase tracking-wider mb-1">P95</div>
                <div className="flex items-baseline gap-1">
                  <span className="text-2xl font-bold text-yellow-700">
                    {benchmarkSummary.latency_p95_ms?.toFixed(1) ?? '-'}
                  </span>
                  <span className="text-xs text-yellow-500">ms</span>
                </div>
              </div>
              <div className="bg-gradient-to-br from-red-50 to-red-100/50 p-4 rounded-lg border border-red-100">
                <div className="text-xs text-red-600 font-medium uppercase tracking-wider mb-1">P99</div>
                <div className="flex items-baseline gap-1">
                  <span className="text-2xl font-bold text-red-700">
                    {benchmarkSummary.latency_p99_ms?.toFixed(1) ?? '-'}
                  </span>
                  <span className="text-xs text-red-500">ms</span>
                </div>
              </div>
              <div className="bg-gradient-to-br from-purple-50 to-purple-100/50 p-4 rounded-lg border border-purple-100">
                <div className="text-xs text-purple-600 font-medium uppercase tracking-wider mb-1">Req/sec</div>
                <div className="flex items-baseline gap-1">
                  <span className="text-2xl font-bold text-purple-700">
                    {benchmarkSummary.requests_per_second?.toFixed(0) ?? '-'}
                  </span>
                </div>
              </div>
              <div className="bg-gradient-to-br from-indigo-50 to-indigo-100/50 p-4 rounded-lg border border-indigo-100">
                <div className="text-xs text-indigo-600 font-medium uppercase tracking-wider mb-1">Total</div>
                <div className="flex items-baseline gap-1">
                  <span className="text-2xl font-bold text-indigo-700">
                    {benchmarkSummary.total_requests?.toLocaleString() ?? '-'}
                  </span>
                </div>
              </div>
              <div className={`bg-gradient-to-br p-4 rounded-lg border ${
                benchmarkSummary.error_rate && benchmarkSummary.error_rate > 0
                  ? 'from-red-50 to-red-100/50 border-red-100'
                  : 'from-emerald-50 to-emerald-100/50 border-emerald-100'
              }`}>
                <div className={`text-xs font-medium uppercase tracking-wider mb-1 ${
                  benchmarkSummary.error_rate && benchmarkSummary.error_rate > 0 ? 'text-red-600' : 'text-emerald-600'
                }`}>Errors</div>
                <div className="flex items-baseline gap-1">
                  <span className={`text-2xl font-bold ${
                    benchmarkSummary.error_rate && benchmarkSummary.error_rate > 0 ? 'text-red-700' : 'text-emerald-700'
                  }`}>
                    {benchmarkSummary.error_rate?.toFixed(1) ?? '0'}
                  </span>
                  <span className={`text-xs ${
                    benchmarkSummary.error_rate && benchmarkSummary.error_rate > 0 ? 'text-red-500' : 'text-emerald-500'
                  }`}>%</span>
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between pt-4 border-t border-gray-100">
              <div className="text-xs text-gray-500 space-x-3">
                {benchmarkSummary.model_id && (
                  <span className="font-mono bg-gray-100 px-1.5 py-0.5 rounded">{benchmarkSummary.model_id}</span>
                )}
                <span>Last run: {benchmarkSummary.last_run_at ? formatDate(benchmarkSummary.last_run_at) : 'N/A'}</span>
              </div>
              <div className="flex items-center gap-3">
                {selectedEndpoint && selectedEndpoint !== benchmarkSummary.benchmark_endpoint && (
                  <span className="text-xs text-purple-600">
                    Will test: <code className="bg-purple-50 px-1 rounded">{selectedEndpoint}</code>
                  </span>
                )}
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onRunBenchmark}
                  disabled={runBenchmarkPending || !selectedEndpoint}
                >
                  <RotateCw className="size-3 mr-2" />
                  {selectedEndpoint && selectedEndpoint !== benchmarkSummary.benchmark_endpoint
                    ? `Benchmark ${selectedEndpoint}`
                    : 'Run Again'}
                </Button>
              </div>
            </div>
          </div>
        ) : (
          /* State: No benchmark data - show available APIs and CTA */
          <div className="space-y-6">
            {/* Available APIs - Selectable */}
            {(apiSpec?.endpoints?.length ?? 0) > 0 && (
              <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium text-gray-700">Select API to Benchmark</div>
                  {apiSpec?.api_type && apiSpec.api_type !== 'unknown' && (
                    <span className="px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-700">
                      {apiSpec.api_type}
                    </span>
                  )}
                </div>
                {/* Filter out endpoints that require file uploads (can't benchmark with JSON) */}
                <div className="flex flex-wrap gap-2">
                  {apiSpec?.endpoints?.filter(ep =>
                    ep.method === 'POST' && !ep.path.includes('/health') && !ep.path.includes('/metrics') && !ep.path.includes('/batch')
                  ).slice(0, 8).map((ep, i) => {
                    const isSelected = selectedEndpoint === ep.path;
                    const isRecommended = apiSpec.recommended_benchmark_endpoint === ep.path;
                    return (
                      <button
                        key={i}
                        onClick={() => onSelectEndpoint(ep.path)}
                        className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded text-xs font-mono transition-all ${
                          isSelected
                            ? 'bg-purple-600 text-white ring-2 ring-purple-300 shadow-sm'
                            : isRecommended
                              ? 'bg-purple-100 text-purple-800 ring-1 ring-purple-300 hover:bg-purple-200'
                              : 'bg-white text-gray-600 border border-gray-200 hover:border-purple-300 hover:bg-purple-50'
                        }`}
                      >
                        <span className={`font-medium ${isSelected ? 'text-purple-200' : 'text-blue-600'}`}>
                          {ep.method}
                        </span>
                        <span>{ep.path}</span>
                        {isSelected && <CheckCircle className="size-3 ml-1" />}
                        {!isSelected && isRecommended && (
                          <span className="ml-1 text-purple-600 text-[10px]">recommended</span>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
            {apiSpecLoading && (
              <div className="flex items-center justify-center gap-2 text-sm text-gray-400 py-4">
                <Loader2 className="size-4 animate-spin" />
                Discovering APIs...
              </div>
            )}

            {/* CTA - only show if there are benchmarkable endpoints */}
            {(apiSpec?.endpoints?.filter(ep =>
              ep.method === 'POST' && !ep.path.includes('/health') && !ep.path.includes('/metrics') && !ep.path.includes('/batch')
            ).length ?? 0) > 0 && (
              <div className="flex flex-col items-center justify-center py-8">
                <div className="size-16 bg-purple-100 rounded-full flex items-center justify-center mb-4">
                  <Gauge className="size-8 text-purple-600" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Benchmark Data</h3>
                <p className="text-sm text-gray-500 text-center max-w-sm mb-4">
                  Run a benchmark to measure latency, throughput, and performance metrics.
                </p>
                {selectedEndpoint && (
                  <div className="mb-4 px-3 py-2 bg-purple-50 rounded-lg border border-purple-200">
                    <span className="text-xs text-purple-600">Testing endpoint:</span>
                    <code className="ml-2 text-sm font-mono text-purple-800">{selectedEndpoint}</code>
                  </div>
                )}
                <Button
                  onClick={onRunBenchmark}
                  disabled={runBenchmarkPending || !selectedEndpoint}
                  className="bg-purple-600 hover:bg-purple-700"
                >
                  <Zap className="size-4 mr-2" />
                  {selectedEndpoint ? `Benchmark ${selectedEndpoint}` : 'Select an API endpoint'}
                </Button>
              </div>
            )}

            {/* Show default CTA if no API spec yet and not loading */}
            {!apiSpecLoading && !apiSpec?.endpoints?.length && (
              <div className="flex flex-col items-center justify-center py-8">
                <div className="size-16 bg-gray-100 rounded-full flex items-center justify-center mb-4">
                  <Gauge className="size-8 text-gray-400" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">No APIs Discovered</h3>
                <p className="text-sm text-gray-500 text-center max-w-sm">
                  Make sure the deployment is running and healthy to discover available endpoints.
                </p>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
