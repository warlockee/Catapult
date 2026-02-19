/**
 * Shared utilities for deployment components
 * Following DRY principle - centralized status colors, helpers, and types
 */

// Status badge colors for deployments
export const deploymentStatusColors: Record<string, string> = {
  pending: 'bg-yellow-100 text-yellow-800',
  deploying: 'bg-blue-100 text-blue-800',
  running: 'bg-green-100 text-green-800',
  stopping: 'bg-orange-100 text-orange-800',
  stopped: 'bg-gray-100 text-gray-800',
  failed: 'bg-red-100 text-red-800',
  success: 'bg-green-100 text-green-800',
};

// Health status colors
export const healthStatusColors: Record<string, string> = {
  healthy: 'bg-green-100 text-green-800',
  unhealthy: 'bg-red-100 text-red-800',
  unknown: 'bg-gray-100 text-gray-800',
};

// Deployment type colors
export const deploymentTypeColors: Record<string, string> = {
  local: 'bg-purple-100 text-purple-800',
  metadata: 'bg-gray-100 text-gray-800',
  k8s: 'bg-blue-100 text-blue-800',
};

// Format date to locale string
export const formatDate = (dateString: string): string => {
  return new Date(dateString).toLocaleString();
};

// Get relative time string (e.g., "5m ago", "2h ago")
export const getRelativeTime = (dateString: string): string => {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  return `${diffDays}d ago`;
};

// Check if deployment is in a transitional state
export const isTransitionalStatus = (status: string): boolean => {
  return ['pending', 'deploying', 'stopping'].includes(status);
};

// Check if deployment status could change (needs frequent polling)
export const needsStatusPolling = (status: string): boolean => {
  // Running deployments can stop/fail at any time, transitional states are actively changing
  return ['pending', 'deploying', 'stopping', 'running'].includes(status);
};

// Check if deployment can be started
export const canStartDeployment = (status: string, deploymentType: string): boolean => {
  return status === 'stopped' && deploymentType === 'local';
};

// Check if deployment can be stopped
export const canStopDeployment = (status: string, deploymentType: string): boolean => {
  return status === 'running' && deploymentType === 'local';
};

// Check if deployment can be restarted
export const canRestartDeployment = (status: string, deploymentType: string): boolean => {
  return status === 'running' && deploymentType === 'local';
};

// Generate endpoint URL from deployment data
export const getEndpointUrl = (deployment: { endpoint_url?: string | null; host_port?: number | null }): string | null => {
  if (deployment.endpoint_url) return deployment.endpoint_url;
  if (deployment.host_port) return `http://localhost:${deployment.host_port}`;
  return null;
};
