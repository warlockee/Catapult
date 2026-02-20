import { Loader2, Activity, RotateCw, StopCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Evaluation, EvaluationSummary } from '../../lib/api';

interface ASREvaluationCardProps {
  evaluationRunning: boolean;
  activeEvaluation: Evaluation | undefined;
  evaluationSummary: EvaluationSummary | undefined;
  isRunning: boolean;
  asrEvalLimit: number;
  onLimitChange: (limit: number) => void;
  onRunEvaluation: () => void;
  onCancelEvaluation?: () => void;
}

export function ASREvaluationCard({
  evaluationRunning,
  activeEvaluation,
  evaluationSummary,
  isRunning,
  asrEvalLimit,
  onLimitChange,
  onRunEvaluation,
  onCancelEvaluation,
}: ASREvaluationCardProps) {
  const hasWerData = evaluationSummary?.wer !== null && evaluationSummary?.wer !== undefined;

  return (
    <Card className="min-w-0">
      <CardHeader className="pb-4">
        <CardTitle className="text-lg font-semibold flex items-center gap-2">
          <Activity className="size-5 text-teal-600" />
          ASR Evaluation
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        {/* Running evaluation */}
        {evaluationRunning ? (
          <div className="flex flex-col items-center justify-center py-8">
            <div className="relative mb-4">
              <Loader2 className="size-12 text-teal-600 animate-spin" />
              <Activity className="size-5 text-teal-600 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
            </div>
            <p className="text-sm font-medium text-gray-700">
              {activeEvaluation?.status === 'running' ? 'Evaluating WER/CER...' : 'Starting evaluation...'}
            </p>
            {activeEvaluation?.stage_progress && (
              <p className="text-lg font-bold text-teal-600 mt-2">
                {activeEvaluation.stage_progress}
              </p>
            )}
            <p className="text-xs text-gray-500 mt-2">This may take a while for large datasets</p>
            {onCancelEvaluation && (
              <Button
                variant="outline"
                size="sm"
                onClick={onCancelEvaluation}
                className="mt-4 border-red-200 text-red-600 hover:bg-red-50"
              >
                <StopCircle className="size-3 mr-2" />
                Stop Evaluation
              </Button>
            )}
          </div>
        ) : hasWerData ? (
          /* Has WER data */
          <div className="space-y-4">
            {/* WER/CER Results */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="bg-gradient-to-br from-teal-50 to-teal-100/50 p-4 rounded-lg border border-teal-100">
                <div className="text-xs text-teal-600 font-medium uppercase tracking-wider mb-1">WER (Word Error Rate)</div>
                <div className="flex items-baseline gap-2">
                  <span className="text-3xl font-bold text-teal-700">
                    {(evaluationSummary!.wer! * 100).toFixed(2)}
                  </span>
                  <span className="text-sm text-teal-500">%</span>
                </div>
              </div>
              <div className="bg-gradient-to-br from-sky-50 to-sky-100/50 p-4 rounded-lg border border-sky-100">
                <div className="text-xs text-sky-600 font-medium uppercase tracking-wider mb-1">CER (Character Error Rate)</div>
                <div className="flex items-baseline gap-2">
                  <span className="text-3xl font-bold text-sky-700">
                    {evaluationSummary!.cer !== null && evaluationSummary!.cer !== undefined
                      ? (evaluationSummary!.cer * 100).toFixed(2)
                      : '-'}
                  </span>
                  <span className="text-sm text-sky-500">%</span>
                </div>
              </div>
              <div className="bg-gradient-to-br from-slate-50 to-slate-100/50 p-4 rounded-lg border border-slate-200">
                <div className="text-xs text-slate-600 font-medium uppercase tracking-wider mb-1">Samples Evaluated</div>
                <div className="flex items-baseline gap-2">
                  <span className="text-3xl font-bold text-slate-700">
                    {evaluationSummary!.samples_evaluated?.toLocaleString() ?? '-'}
                  </span>
                </div>
              </div>
            </div>

            {/* Evaluation Options & Re-run */}
            {isRunning && (
              <div className="pt-4 border-t border-gray-100">
                <div className="flex flex-wrap items-end gap-4">
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Samples (0=all)</label>
                    <input
                      type="number"
                      value={asrEvalLimit}
                      onChange={(e) => onLimitChange(parseInt(e.target.value) || 0)}
                      min={0}
                      max={10000}
                      className="w-24 px-2 py-1.5 text-sm border border-gray-300 rounded-md focus:ring-teal-500 focus:border-teal-500"
                    />
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={onRunEvaluation}
                    disabled={evaluationRunning}
                    className="shrink-0 border-teal-200 text-teal-700 hover:bg-teal-50"
                  >
                    <RotateCw className="size-3 mr-2" />
                    Run Evaluation
                  </Button>
                </div>
              </div>
            )}
          </div>
        ) : (
          /* No WER data yet */
          <div className="flex flex-col items-center justify-center py-8">
            <div className="size-12 bg-teal-100 rounded-full flex items-center justify-center mb-4">
              <Activity className="size-6 text-teal-600" />
            </div>
            <h3 className="text-base font-medium text-gray-900 mb-2">No Evaluation Data</h3>
            <p className="text-sm text-gray-500 text-center max-w-sm mb-4">
              {isRunning
                ? 'Run an evaluation to measure Word Error Rate (WER) and Character Error Rate (CER).'
                : 'Start the deployment to enable ASR evaluation.'}
            </p>
            {isRunning && (
              <div className="w-full max-w-md space-y-4">
                <div className="flex flex-wrap gap-4 justify-center">
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Samples (0=all)</label>
                    <input
                      type="number"
                      value={asrEvalLimit}
                      onChange={(e) => onLimitChange(parseInt(e.target.value) || 0)}
                      min={0}
                      max={10000}
                      className="w-24 px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:ring-teal-500 focus:border-teal-500"
                    />
                  </div>
                </div>
                <Button
                  onClick={onRunEvaluation}
                  disabled={evaluationRunning}
                  className="w-full bg-teal-600 hover:bg-teal-700"
                >
                  <Activity className="size-4 mr-2" />
                  Run Evaluation
                </Button>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
