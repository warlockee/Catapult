import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ArrowLeft, Cpu, Copy, Check, Terminal, Gauge, Activity, Clock, Tag } from 'lucide-react';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Separator } from './ui/separator';
import { Skeleton } from './ui/skeleton';
import { api } from '../lib/api';
import { formatBytes, cn } from '../lib/utils';
import { generateCurlExample } from './shared/benchmark-utils';
import type { Image, Version, BenchmarkSummary, EvaluationSummary, Deployment } from '../lib/api';

// --- Server type → API endpoint mapping ---

interface ApiExample {
  method: string;
  path: string;
  description: string;
  curlExtra?: string; // extra flags for curl (e.g. -F for file upload)
  sampleResponse?: Record<string, unknown>;
}

function getApiExamples(serverType: string | undefined): ApiExample[] {
  switch (serverType) {
    case 'vllm':
      return [
        {
          method: 'POST', path: '/v1/chat/completions',
          description: 'Send a chat completion request',
          sampleResponse: { choices: [{ message: { role: 'assistant', content: 'Hello! How can I help you today?' } }] },
        },
        {
          method: 'GET', path: '/v1/models',
          description: 'List available models',
        },
      ];
    case 'whisper':
    case 'asr-vllm':
    case 'asr-allinone':
    case 'stt':
      return [
        {
          method: 'POST', path: '/v1/audio/transcriptions',
          description: 'Transcribe an audio file',
          curlExtra: '-F "file=@audio.wav"',
          sampleResponse: { text: 'Hello world', language: 'en' },
        },
      ];
    case 'tts':
    case 'audio-generation':
      return [
        {
          method: 'POST', path: '/v1/audio/speech',
          description: 'Generate speech from text',
          sampleResponse: { _note: 'Returns WAV audio binary' },
        },
      ];
    case 'embedding':
      return [
        {
          method: 'POST', path: '/v1/embeddings',
          description: 'Generate text embeddings',
          sampleResponse: { object: 'list', data: [{ embedding: [0.1, 0.2, '...'], index: 0 }], model: 'model-name' },
        },
      ];
    case 'multimodal':
      return [
        {
          method: 'POST', path: '/v1/chat/completions',
          description: 'Send a multimodal (text + image) request',
          sampleResponse: { choices: [{ message: { role: 'assistant', content: 'I can see an image...' } }] },
        },
      ];
    case 'codec':
      return [
        {
          method: 'POST', path: '/encode',
          description: 'Encode audio to compressed codes',
          curlExtra: '-F "file=@audio.wav"',
          sampleResponse: { codes: [['...']], sample_rate: 24000 },
        },
      ];
    default:
      return [
        {
          method: 'GET', path: '/health',
          description: 'Check model health',
          sampleResponse: { status: 'healthy' },
        },
      ];
  }
}

function getPythonExample(serverType: string | undefined, modelName: string): string {
  const name = modelName || 'model-name';
  switch (serverType) {
    case 'vllm':
    case 'multimodal':
      return `from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="na")

response = client.chat.completions.create(
    model="${name}",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
)
print(response.choices[0].message.content)`;

    case 'whisper':
    case 'asr-vllm':
    case 'asr-allinone':
    case 'stt':
      return `from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="na")

result = client.audio.transcriptions.create(
    model="${name}",
    file=open("audio.wav", "rb"),
)
print(result.text)`;

    case 'tts':
    case 'audio-generation':
      return `from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="na")

response = client.audio.speech.create(
    model="${name}",
    input="Hello, this is a test of the text to speech system.",
    voice="default",
)
response.stream_to_file("output.wav")`;

    case 'embedding':
      return `from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="na")

result = client.embeddings.create(
    model="${name}",
    input="Hello world",
)
print(f"Embedding dimension: {len(result.data[0].embedding)}")
print(result.data[0].embedding[:5])`;

    default:
      return `import requests

response = requests.get("http://localhost:8000/health")
print(response.json())`;
  }
}

// --- Copyable code block ---

function CodeBlock({ code, language }: { code: string; language?: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <div className="relative group">
      {language && (
        <div className="absolute top-2 left-3 text-[10px] font-mono text-gray-500 uppercase tracking-wider">{language}</div>
      )}
      <pre className={cn(
        "bg-gray-900 text-gray-100 rounded-lg overflow-x-auto text-sm font-mono",
        language ? "pt-7 pb-4 px-4" : "p-4",
      )}>
        {code}
      </pre>
      <Button
        variant="secondary"
        size="sm"
        className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity h-7"
        onClick={handleCopy}
      >
        {copied ? <Check className="size-3 mr-1 text-green-600" /> : <Copy className="size-3 mr-1" />}
        {copied ? 'Copied' : 'Copy'}
      </Button>
    </div>
  );
}

// --- Stat card ---

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-gray-50 rounded-lg p-4 border">
      <div className="text-xs text-gray-500 font-medium uppercase tracking-wider mb-1">{label}</div>
      <div className="text-xl font-semibold text-gray-900">{value}</div>
      {sub && <div className="text-xs text-gray-400 mt-0.5">{sub}</div>}
    </div>
  );
}

// --- Main component ---

export function ModelCard() {
  const { modelId } = useParams<{ modelId: string }>();
  const navigate = useNavigate();

  // 1. Fetch model
  const { data: model, isLoading: modelLoading } = useQuery({
    queryKey: ['model', modelId],
    queryFn: ({ signal }) => api.getImage(modelId!, signal),
    enabled: !!modelId,
  });

  // 2. Fetch latest versions (up to 10 for version history)
  const { data: versionsData, isLoading: versionsLoading } = useQuery({
    queryKey: ['modelVersions', modelId, 'card'],
    queryFn: () => api.getImageReleases(modelId!, { page: 1, size: 10 }),
    enabled: !!modelId,
  });

  const versions = versionsData?.items || [];
  const latestVersion = versions[0] as Version | undefined;

  // 3. Fetch deployments for latest version (to get benchmark/eval data)
  const { data: deployments } = useQuery({
    queryKey: ['versionDeployments', latestVersion?.id],
    queryFn: () => api.getVersionDeployments(latestVersion!.id),
    enabled: !!latestVersion?.id,
  });

  // Find the most recent deployment (prefer one that ran benchmarks)
  const latestDeployment = deployments?.sort(
    (a: Deployment, b: Deployment) => new Date(b.deployed_at).getTime() - new Date(a.deployed_at).getTime()
  )[0];

  // 4. Fetch benchmark summary from latest deployment
  const { data: benchmarkSummary } = useQuery({
    queryKey: ['benchmarkSummary', latestDeployment?.id],
    queryFn: () => api.getDeploymentBenchmarkSummary(latestDeployment!.id),
    enabled: !!latestDeployment?.id,
  });

  // 5. Fetch evaluation summary from latest deployment
  const { data: evalSummary } = useQuery({
    queryKey: ['evalSummary', latestDeployment?.id],
    queryFn: () => api.getDeploymentEvaluationSummary(latestDeployment!.id),
    enabled: !!latestDeployment?.id,
  });

  // Loading state
  if (modelLoading || versionsLoading) {
    return (
      <div className="max-w-4xl mx-auto space-y-6 py-8 px-4">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-6 w-full" />
        <div className="grid grid-cols-3 gap-4">
          <Skeleton className="h-20" />
          <Skeleton className="h-20" />
          <Skeleton className="h-20" />
        </div>
        <Skeleton className="h-40" />
      </div>
    );
  }

  if (!model) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">Model not found</p>
        <Button onClick={() => navigate(-1)} className="mt-4">Go Back</Button>
      </div>
    );
  }

  const serverType = model.server_type;
  const modelName = model.name;
  const apiExamples = getApiExamples(serverType);
  const pythonCode = getPythonExample(serverType, modelName);
  const hasBenchmark = benchmarkSummary?.has_data;
  const hasEval = evalSummary?.has_data;
  const pullCommand = latestVersion
    ? `docker pull ${model.name}:${latestVersion.tag}`
    : `docker pull ${model.name}:latest`;

  return (
    <div className="max-w-4xl mx-auto space-y-8 py-8 px-4">
      {/* Back button */}
      <Button variant="ghost" onClick={() => navigate(-1)} className="mb-2">
        <ArrowLeft className="size-4 mr-2" />
        Back
      </Button>

      {/* --- Section 1: Hero --- */}
      <div className="space-y-3">
        <h1 className="text-3xl font-bold">{model.name}</h1>
        {model.description && (
          <p className="text-lg text-gray-600">{model.description}</p>
        )}
        <div className="flex flex-wrap gap-2">
          {serverType && (
            <Badge variant="outline" className="text-sm">{serverType}</Badge>
          )}
          {model.parameter_count && (
            <Badge variant="outline" className="text-sm">{model.parameter_count} params</Badge>
          )}
          {model.requires_gpu && (
            <Badge variant="outline" className="text-sm text-purple-700 border-purple-300">
              <Cpu className="size-3 mr-1" />GPU Required
            </Badge>
          )}
          {model.tags?.map(tag => (
            <Badge key={tag} variant="secondary" className="text-sm">{tag}</Badge>
          ))}
        </div>
        {model.company && (
          <p className="text-sm text-gray-500">by {model.company}</p>
        )}
      </div>

      <Separator />

      {/* --- Section 2: Overview --- */}
      {latestVersion && (
        <>
          <section className="space-y-4">
            <h2 className="text-xl font-semibold flex items-center gap-2">
              <Tag className="size-5 text-blue-600" />
              Overview
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <SpecItem label="Parameters" value={model.parameter_count} />
              <SpecItem label="Architecture" value={latestVersion.metadata?.model_architecture} />
              <SpecItem label="Platform" value={`${latestVersion.os}/${latestVersion.architecture}`} />
              <SpecItem label="Quantization" value={latestVersion.quantization} />
              <SpecItem label="Size" value={latestVersion.size_bytes ? formatBytes(latestVersion.size_bytes) : undefined} />
              <SpecItem label="Server Type" value={serverType} />
              {model.base_model && <SpecItem label="Base Model" value={model.base_model} />}
              <SpecItem label="Latest Version" value={`v${latestVersion.version}`} />
              <SpecItem label="Status" value={latestVersion.status} />
            </div>
          </section>
          <Separator />
        </>
      )}

      {/* --- Section 3: Getting Started --- */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          <Terminal className="size-5 text-green-600" />
          Getting Started
        </h2>
        <div className="space-y-3">
          <p className="text-sm text-gray-600">Pull the model and start serving:</p>
          <CodeBlock code={pullCommand} language="bash" />
          {serverType === 'vllm' && (
            <div className="text-sm text-gray-600 space-y-1">
              <p>Start the server with vLLM:</p>
              <CodeBlock
                code={`python -m vllm.entrypoints.openai.api_server \\
  --model ${model.storage_path || `/path/to/${model.name}`} \\
  --port 8000${model.requires_gpu ? '' : ' --device cpu'}`}
                language="bash"
              />
            </div>
          )}
          {model.requires_gpu && (
            <p className="text-sm text-amber-600">
              This model requires GPU. Ensure NVIDIA drivers and CUDA are installed.
            </p>
          )}
        </div>
      </section>

      <Separator />

      {/* --- Section 4: API Reference --- */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          <Terminal className="size-5 text-blue-600" />
          API Reference
        </h2>
        <p className="text-sm text-gray-600">
          The model exposes an HTTP API on port 8000. All endpoints accept and return JSON unless noted.
        </p>
        <div className="space-y-6">
          {apiExamples.map((ep, i) => {
            const curl = ep.curlExtra
              ? `curl -X ${ep.method} "http://localhost:8000${ep.path}" \\\n  ${ep.curlExtra}`
              : generateCurlExample('http://localhost:8000', ep.method, ep.path, modelName);
            return (
              <Card key={i}>
                <CardContent className="p-5 space-y-3">
                  <div className="flex items-center gap-3">
                    <Badge variant={ep.method === 'GET' ? 'secondary' : 'default'} className="font-mono">
                      {ep.method}
                    </Badge>
                    <code className="text-sm font-semibold">{ep.path}</code>
                  </div>
                  <p className="text-sm text-gray-600">{ep.description}</p>
                  <div>
                    <div className="text-xs text-gray-500 font-medium mb-1 uppercase tracking-wider">Request</div>
                    <CodeBlock code={curl} language="bash" />
                  </div>
                  {ep.sampleResponse && (
                    <div>
                      <div className="text-xs text-gray-500 font-medium mb-1 uppercase tracking-wider">Response</div>
                      <pre className="bg-gray-50 border p-4 rounded-lg overflow-x-auto text-xs font-mono max-h-48">
                        {JSON.stringify(ep.sampleResponse, null, 2)}
                      </pre>
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>

      <Separator />

      {/* --- Section 5: Python SDK --- */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          <Terminal className="size-5 text-indigo-600" />
          Python SDK
        </h2>
        <p className="text-sm text-gray-600">
          Use the OpenAI Python SDK (compatible with vLLM and most model servers):
        </p>
        <CodeBlock code={`pip install openai`} language="bash" />
        <CodeBlock code={pythonCode} language="python" />
      </section>

      {/* --- Section 6: Performance Benchmarks --- */}
      {hasBenchmark && benchmarkSummary && (
        <>
          <Separator />
          <BenchmarkSection summary={benchmarkSummary} />
        </>
      )}

      {/* --- Section 7: Quality Metrics --- */}
      {hasEval && evalSummary && (
        <>
          <Separator />
          <EvalSection summary={evalSummary} />
        </>
      )}

      {/* --- Section 8: Version History --- */}
      {versions.length > 0 && (
        <>
          <Separator />
          <section className="space-y-4">
            <h2 className="text-xl font-semibold flex items-center gap-2">
              <Clock className="size-5 text-gray-600" />
              Version History
            </h2>
            <div className="border rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-50 border-b">
                    <th className="text-left px-4 py-2 font-medium text-gray-600">Version</th>
                    <th className="text-left px-4 py-2 font-medium text-gray-600">Tag</th>
                    <th className="text-left px-4 py-2 font-medium text-gray-600">Size</th>
                    <th className="text-left px-4 py-2 font-medium text-gray-600">Date</th>
                    <th className="text-left px-4 py-2 font-medium text-gray-600">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {versions.map((v: Version) => (
                    <tr key={v.id} className="border-b last:border-0 hover:bg-gray-50">
                      <td className="px-4 py-2 font-mono">v{v.version}</td>
                      <td className="px-4 py-2">
                        <code className="text-xs bg-gray-100 px-1.5 py-0.5 rounded">{v.tag}</code>
                      </td>
                      <td className="px-4 py-2 text-gray-600">
                        {v.size_bytes ? formatBytes(v.size_bytes) : '-'}
                      </td>
                      <td className="px-4 py-2 text-gray-600">
                        {new Date(v.created_at).toLocaleDateString()}
                      </td>
                      <td className="px-4 py-2">
                        <div className="flex items-center gap-1.5">
                          {v.is_release && (
                            <Badge className="bg-blue-600 text-xs">Release</Badge>
                          )}
                          <Badge variant="outline" className="text-xs">{v.status}</Badge>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </div>
  );
}

// --- Sub-components ---

function SpecItem({ label, value }: { label: string; value?: string | null }) {
  if (!value) return null;
  return (
    <div className="bg-gray-50 rounded-lg p-3 border">
      <div className="text-xs text-gray-500 mb-0.5">{label}</div>
      <div className="text-sm font-medium">{value}</div>
    </div>
  );
}

function BenchmarkSection({ summary }: { summary: BenchmarkSummary }) {
  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold flex items-center gap-2">
        <Gauge className="size-5 text-orange-600" />
        Performance Benchmarks
      </h2>
      {summary.benchmark_endpoint && (
        <p className="text-sm text-gray-500">
          Benchmarked on <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs">{summary.benchmark_endpoint}</code>
          {summary.last_run_at && <> at {new Date(summary.last_run_at).toLocaleDateString()}</>}
        </p>
      )}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {summary.latency_avg_ms != null && (
          <StatCard label="Avg Latency" value={`${summary.latency_avg_ms.toFixed(0)}ms`} />
        )}
        {summary.latency_p95_ms != null && (
          <StatCard label="p95 Latency" value={`${summary.latency_p95_ms.toFixed(0)}ms`} />
        )}
        {summary.requests_per_second != null && (
          <StatCard label="Throughput" value={`${summary.requests_per_second.toFixed(1)} req/s`} />
        )}
        {summary.error_rate != null && (
          <StatCard label="Error Rate" value={`${summary.error_rate.toFixed(1)}%`} />
        )}
      </div>
      {(summary.ttft_avg_ms != null || summary.tokens_per_second_avg != null) && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {summary.ttft_avg_ms != null && (
            <StatCard label="TTFT (avg)" value={`${summary.ttft_avg_ms.toFixed(0)}ms`} sub="Time to first token" />
          )}
          {summary.ttft_p95_ms != null && (
            <StatCard label="TTFT (p95)" value={`${summary.ttft_p95_ms.toFixed(0)}ms`} />
          )}
          {summary.tokens_per_second_avg != null && (
            <StatCard label="Tokens/s" value={`${summary.tokens_per_second_avg.toFixed(0)}`} sub="Average generation speed" />
          )}
          {summary.total_tokens_generated != null && (
            <StatCard label="Total Tokens" value={summary.total_tokens_generated.toLocaleString()} />
          )}
        </div>
      )}
    </section>
  );
}

function EvalSection({ summary }: { summary: EvaluationSummary }) {
  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold flex items-center gap-2">
        <Activity className="size-5 text-teal-600" />
        Quality Metrics
      </h2>
      {summary.evaluation_type && (
        <p className="text-sm text-gray-500">
          Evaluation type: <Badge variant="outline">{summary.evaluation_type.toUpperCase()}</Badge>
          {summary.dataset_path && <> on <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs">{summary.dataset_path}</code></>}
        </p>
      )}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {summary.wer != null && (
          <StatCard label="WER" value={`${(summary.wer * 100).toFixed(2)}%`} sub="Word Error Rate" />
        )}
        {summary.cer != null && (
          <StatCard label="CER" value={`${(summary.cer * 100).toFixed(2)}%`} sub="Character Error Rate" />
        )}
        {summary.primary_metric != null && summary.primary_metric_name && summary.primary_metric_name !== 'wer' && (
          <StatCard label={summary.primary_metric_name.toUpperCase()} value={`${(summary.primary_metric * 100).toFixed(2)}%`} />
        )}
        {summary.samples_evaluated != null && (
          <StatCard label="Samples" value={summary.samples_evaluated.toLocaleString()} sub="Evaluated" />
        )}
      </div>
    </section>
  );
}
