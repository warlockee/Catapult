/**
 * Generate a cURL example command for an API endpoint
 */
export function generateCurlExample(
  endpointUrl: string,
  method: string,
  path: string,
  modelName: string
): string {
  const url = `${endpointUrl}${path}`;
  if (method === 'GET') {
    return `curl -X GET "${url}"`;
  }

  let body = '{}';
  if (path.includes('/chat/completions')) {
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
  }

  return `curl -X ${method} "${url}" \\
  -H "Content-Type: application/json" \\
  -d '${body}'`;
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
