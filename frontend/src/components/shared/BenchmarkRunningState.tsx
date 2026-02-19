import { Loader2, Zap, StopCircle } from 'lucide-react';
import { Button } from '../ui/button';
import { Benchmark } from '../../lib/api';

interface StageCompleted {
  stage: string;
  success: boolean;
  message?: string;
  model_id?: string;
  model_type?: string;
  ttft_avg_ms?: number;
  tokens_per_second_avg?: number;
  latency_avg_ms?: number;
  requests_per_second?: number;
  successful?: number;
  failed?: number;
  endpoint?: string;
}

interface BenchmarkRunningStateProps {
  benchmarkId: string | null;
  benchmark: Benchmark | null | undefined;
  onCancel?: () => void;
  cancelPending?: boolean;
}

export function BenchmarkRunningState({
  benchmarkId,
  benchmark,
  onCancel,
  cancelPending,
}: BenchmarkRunningStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-8">
      {/* Benchmark ID */}
      {benchmarkId && (
        <div className="text-xs text-gray-400 font-mono mb-4">
          Benchmark: {benchmarkId.slice(0, 8)}...
        </div>
      )}

      {/* Progress indicator */}
      <div className="relative mb-4">
        <Loader2 className="size-12 text-purple-600 animate-spin" />
        <Zap className="size-5 text-purple-600 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
      </div>

      {/* Current stage */}
      <p className="text-sm font-medium text-gray-700">
        {benchmark?.current_stage === 'discovering_model' && 'Discovering model...'}
        {benchmark?.current_stage === 'health_check' && 'Running health check...'}
        {benchmark?.current_stage === 'inference_test' && 'Testing inference...'}
        {benchmark?.current_stage === 'ttft_benchmark' && 'Measuring TTFT/TPS...'}
        {benchmark?.current_stage === 'stress_test' && 'Running stress test...'}
        {benchmark?.current_stage === 'inference_benchmark' && 'Running inference benchmark...'}
        {benchmark?.current_stage === 'load_test' && 'Running load test...'}
        {benchmark?.current_stage === 'finalizing' && 'Finalizing results...'}
        {!benchmark?.current_stage && 'Starting benchmark...'}
      </p>

      {/* Stage progress */}
      {benchmark?.stage_progress && (
        <p className="text-sm text-purple-600 font-medium mt-1">
          {benchmark.stage_progress}
        </p>
      )}

      {/* Completed stages */}
      {benchmark?.stages_completed && benchmark.stages_completed.length > 0 && (
        <div className="mt-4 w-full max-w-xs space-y-1">
          {(benchmark.stages_completed as StageCompleted[]).map((stage, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <span className={stage.success ? 'text-green-500' : 'text-red-500'}>
                {stage.success ? '✓' : '✗'}
              </span>
              <span className="text-gray-600">
                {stage.stage === 'discovering_model' && (
                  <>Model: {stage.model_id || 'discovered'} {stage.model_type && `(${stage.model_type})`}</>
                )}
                {stage.stage === 'health_check' && 'Health check passed'}
                {stage.stage === 'inference_test' && (stage.message || 'Inference test passed')}
                {stage.stage === 'ttft_benchmark' && stage.ttft_avg_ms && (
                  <>TTFT: {stage.ttft_avg_ms.toFixed(0)}ms, TPS: {stage.tokens_per_second_avg?.toFixed(1) || '-'}</>
                )}
                {stage.stage === 'inference_benchmark' && (
                  <>{stage.endpoint}: {stage.latency_avg_ms?.toFixed(0)}ms, {stage.requests_per_second?.toFixed(1)} req/s</>
                )}
                {stage.stage === 'load_test' && (
                  <>{stage.endpoint}: {stage.latency_avg_ms?.toFixed(0)}ms, {stage.requests_per_second?.toFixed(1)} req/s</>
                )}
                {stage.stage === 'stress_test' && (
                  <>{stage.requests_per_second?.toFixed(1) || '-'} req/s ({stage.successful}/{(stage.successful || 0) + (stage.failed || 0)})</>
                )}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Cancel button */}
      {onCancel && benchmarkId && (
        <Button
          variant="outline"
          size="sm"
          onClick={onCancel}
          disabled={cancelPending}
          className="mt-4 text-red-600 border-red-200 hover:bg-red-50"
        >
          <StopCircle className="size-4 mr-2" />
          {cancelPending ? 'Cancelling...' : 'Cancel'}
        </Button>
      )}
    </div>
  );
}
