/**
 * Generate a cURL example command for an API endpoint.
 *
 * For file upload endpoints (requires_file_upload=true), generates multipart
 * form curl with -F flags using field names from the request_schema.
 * For JSON endpoints, uses schema-derived sample body or path-based fallback.
 */
export function generateCurlExample(
  endpointUrl: string,
  method: string,
  path: string,
  modelName: string,
  sampleBody?: Record<string, any> | null,
  requiresFileUpload?: boolean,
  requestSchema?: Record<string, any> | null,
): string {
  const url = `${endpointUrl}${path}`;
  if (method === 'GET') {
    return `curl -X GET "${url}"`;
  }

  // File upload endpoints use -F (multipart form) syntax
  if (requiresFileUpload) {
    return generateMultipartCurl(url, method, path, modelName, requestSchema);
  }

  let body: string;
  if (sampleBody && Object.keys(sampleBody).length > 0) {
    // Use schema-derived sample body, substituting model name
    const bodyWithModel = { ...sampleBody };
    if ('model' in bodyWithModel && typeof bodyWithModel.model === 'string') {
      bodyWithModel.model = modelName;
    }
    body = JSON.stringify(bodyWithModel, null, 2);
  } else if (path.includes('/chat/completions')) {
    body = JSON.stringify({
      model: modelName,
      messages: [{ role: 'user', content: 'Hello!' }],
      max_tokens: 100,
    }, null, 2);
  } else if (path.includes('/completions')) {
    body = JSON.stringify({
      model: modelName,
      prompt: 'Once upon a time',
      max_tokens: 100,
    }, null, 2);
  } else if (path.includes('/embeddings')) {
    body = JSON.stringify({
      model: modelName,
      input: 'Hello world',
    }, null, 2);
  } else if (path.includes('/synthesize') || path.includes('/tts')) {
    body = JSON.stringify({ text: 'Hello, world!', voice: 'default' }, null, 2);
  } else if (path.includes('/predict') || path.includes('/inference')) {
    body = JSON.stringify({ input: 'your data here' }, null, 2);
  } else {
    body = '{}';
  }

  return `curl -X ${method} "${url}" \\
  -H "Content-Type: application/json" \\
  -d '${body}'`;
}

/**
 * Generate a multipart/form-data cURL command for file upload endpoints.
 *
 * Uses OpenAPI request_schema to determine field names and types.
 * Falls back to common patterns based on path.
 */
function generateMultipartCurl(
  url: string,
  method: string,
  path: string,
  modelName: string,
  requestSchema?: Record<string, any> | null,
): string {
  const parts: string[] = [`curl -X ${method} "${url}"`];

  if (requestSchema?.properties) {
    const props = requestSchema.properties as Record<string, any>;
    const required = new Set(requestSchema.required || []);

    for (const [name, spec] of Object.entries(props)) {
      if (spec?.format === 'binary' || spec?.type === 'string' && spec?.format === 'binary') {
        // Binary file field
        parts.push(`-F "${name}=@audio.wav"`);
      } else if (spec?.type === 'array' && name.toLowerCase().includes('file')) {
        // Array of files
        parts.push(`-F "${name}=@audio1.wav" -F "${name}=@audio2.wav"`);
      } else if (spec?.default !== undefined) {
        // Field with default value
        parts.push(`-F "${name}=${spec.default}"`);
      } else if (name === 'model') {
        parts.push(`-F "model=${modelName}"`);
      } else if (required.has(name)) {
        // Required field without default — show placeholder
        parts.push(`-F "${name}=value"`);
      }
    }
  } else {
    // Fallback: use path-based patterns
    if (path.includes('/transcribe') && path.includes('/batch')) {
      parts.push('-F "files=@audio1.wav" -F "files=@audio2.wav"');
      parts.push('-F "language=English"');
    } else if (path.includes('/transcribe') || path.includes('/audio/transcriptions') || path.includes('/stt')) {
      parts.push('-F "file=@audio.wav"');
      if (path.includes('/audio/transcriptions')) {
        parts.push(`-F "model=${modelName}"`);
      } else {
        parts.push('-F "language=English"');
      }
    } else if (path.includes('/encode') || path.includes('/decode')) {
      parts.push('-F "file=@audio.wav"');
    } else {
      parts.push('-F "file=@input_file"');
    }
  }

  return parts.join(' \\\n  ');
}

/**
 * Generate appropriate request body for benchmarking an endpoint
 */
export function getRequestBodyForEndpoint(
  endpointPath: string,
  modelName: string
): Record<string, unknown> {
  if (endpointPath === '/v1/audio/speech') {
    return {
      model: modelName,
      input: 'The quick brown fox jumps over the lazy dog. This is a test of the text to speech system.',
      voice: 'en_woman',
    };
  } else if (endpointPath === '/v1/audio/transcriptions') {
    return {
      model: modelName,
    };
  } else if (endpointPath === '/v1/completions') {
    return {
      model: modelName,
      prompt: 'Once upon a time',
      max_tokens: 50,
      stream: true,
    };
  } else if (endpointPath === '/v1/embeddings') {
    return {
      model: modelName,
      input: 'Hello, world!',
    };
  } else if (endpointPath === '/v1/inference') {
    return {
      input: 'The quick brown fox jumps over the lazy dog. This is a test.',
      parameters: {},
    };
  } else if (endpointPath === '/v1/generate') {
    return {
      model: modelName,
      prompt: 'Once upon a time',
      max_tokens: 50,
    };
  } else {
    // Default: chat completions format
    return {
      model: modelName,
      messages: [{ role: 'user', content: 'Say hello' }],
      max_tokens: 50,
      stream: true,
    };
  }
}

/**
 * Copy text to clipboard
 */
export function copyToClipboard(text: string): void {
  navigator.clipboard.writeText(text);
}
