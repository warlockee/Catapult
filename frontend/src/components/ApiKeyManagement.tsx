import { useState, useEffect } from 'react';
import { Key, Plus, Copy, Trash2, CheckCircle, XCircle, AlertTriangle, Eye, EyeOff, Loader2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import { Alert, AlertDescription } from './ui/alert';
import { api, ApiKey, ApiError } from '../lib/api';

export function ApiKeyManagement() {
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [newKeyName, setNewKeyName] = useState('');
  const [newlyCreatedKey, setNewlyCreatedKey] = useState<string | null>(null);
  const [visibleKeys, setVisibleKeys] = useState<Record<string, boolean>>({});
  const [isCreating, setIsCreating] = useState(false);

  const loadApiKeys = async () => {
    try {
      setLoading(true);
      setError(null);
      const keys = await api.getApiKeys();
      setApiKeys(keys);
    } catch (err) {
      console.error('Failed to load API keys:', err);
      setError('Failed to load API keys. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadApiKeys();
  }, []);

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleString();
  };

  const getRelativeTime = (dateString?: string) => {
    if (!dateString) return 'Never';
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffMs < 0) return 'Just now';
    if (diffMins < 60) return `${diffMins} minutes ago`;
    if (diffHours < 24) return `${diffHours} hours ago`;
    if (diffDays < 30) return `${diffDays} days ago`;
    return date.toLocaleDateString();
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const toggleKeyVisibility = (keyId: string) => {
    setVisibleKeys(prev => ({
      ...prev,
      [keyId]: !prev[keyId]
    }));
  };

  const maskKey = (key: string) => {
    if (!key) return '••••••••••••••••••••••••';
    if (key.length < 10) return '••••••';
    return `${key.substring(0, 8)}${'•'.repeat(20)}${key.substring(key.length - 4)}`;
  };

  const handleCreateKey = async () => {
    try {
      setIsCreating(true);
      const newKey = await api.createApiKey({ name: newKeyName });

      // Update local state with the new key (showing the secret once)
      setApiKeys([newKey, ...apiKeys]);
      setNewlyCreatedKey(newKey.key || 'Error: Key content missing');
      setNewKeyName('');

      // Only keep the secret visible in specific state, main list won't show it after refresh usually
      // but for now, we just rely on `newlyCreatedKey` for the dialog.
    } catch (err) {
      console.error('Failed to create API key:', err);
      // Optional: show error toast
    } finally {
      setIsCreating(false);
    }
  };

  const handleRevokeKey = async (keyId: string) => {
    // In our simplified API client, we map revoke to delete because current backend lacks explicit revoke endpoint distinct from delete
    // But verify if backend supports soft delete or revoke? Backend `delete_api_key` calls `repo.delete`.
    // So "Revoke" in UI effectively completely removes it for now.
    await handleDeleteKey(keyId);
  };

  const handleDeleteKey = async (keyId: string) => {
    if (!confirm('Are you sure you want to delete this API Key? Using applications will immediately lose access.')) {
      return;
    }

    try {
      await api.deleteApiKey(keyId);
      setApiKeys(apiKeys.filter(key => key.id !== keyId));
    } catch (err) {
      console.error('Failed to delete API key:', err);
      if (err instanceof ApiError && err.status === 400 && err.message.includes('self_delete')) {
        alert("You cannot delete your own API Key.");
      }
    }
  };

  const activeKeys = apiKeys.filter(k => k.is_active);
  const revokedKeys = apiKeys.filter(k => !k.is_active);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="size-8 text-blue-600 animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-red-50 text-red-700 rounded-lg border border-red-200">
        <h3 className="font-semibold mb-2">Error</h3>
        <p>{error}</p>
        <Button variant="outline" className="mt-4" onClick={loadApiKeys}>Try Again</Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1>API Keys</h1>
          <p className="text-gray-500 mt-1">Manage authentication keys for the registry</p>
        </div>
        <Dialog open={isCreateDialogOpen} onOpenChange={(open: boolean) => {
          setIsCreateDialogOpen(open);
          if (!open) setNewlyCreatedKey(null);
        }}>
          <DialogTrigger asChild>
            <Button className="bg-blue-600 hover:bg-blue-700">
              <Plus className="size-4 mr-2" />
              Create API Key
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New API Key</DialogTitle>
              <DialogDescription>
                Generate a new API key for authenticating with the registry.
              </DialogDescription>
            </DialogHeader>

            {newlyCreatedKey ? (
              <div className="space-y-4">
                <Alert>
                  <AlertTriangle className="size-4" />
                  <AlertDescription>
                    Make sure to copy your API key now. You won't be able to see it again!
                  </AlertDescription>
                </Alert>

                <div>
                  <Label htmlFor="new-api-key">Your New API Key</Label>
                  <div className="flex gap-2 mt-2">
                    <Input id="new-api-key" value={newlyCreatedKey} readOnly className="font-mono text-sm" />
                    <Button onClick={() => copyToClipboard(newlyCreatedKey)}>
                      <Copy className="size-4" />
                    </Button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div>
                  <Label htmlFor="keyName">Key Name</Label>
                  <Input
                    id="keyName"
                    placeholder="e.g., ci-cd-pipeline, admin-key"
                    value={newKeyName}
                    onChange={(e) => setNewKeyName(e.target.value)}
                    className="mt-2"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Choose a descriptive name to identify this key
                  </p>
                </div>
              </div>
            )}

            <DialogFooter>
              {newlyCreatedKey ? (
                <Button onClick={() => {
                  setNewlyCreatedKey(null);
                  setIsCreateDialogOpen(false);
                }}>
                  Done
                </Button>
              ) : (
                <>
                  <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
                    Cancel
                  </Button>
                  <Button
                    onClick={handleCreateKey}
                    disabled={!newKeyName.trim() || isCreating}
                    className="bg-blue-600 hover:bg-blue-700"
                  >
                    {isCreating && <Loader2 className="mr-2 size-4 animate-spin" />}
                    Generate Key
                  </Button>
                </>
              )}
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-500 mb-1">Total Keys</div>
                <div className="text-3xl">{apiKeys.length}</div>
              </div>
              <Key className="size-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-500 mb-1">Active</div>
                <div className="text-3xl text-green-600">{activeKeys.length}</div>
              </div>
              <CheckCircle className="size-8 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-500 mb-1">Revoked</div>
                <div className="text-3xl text-red-600">{revokedKeys.length}</div>
              </div>
              <XCircle className="size-8 text-red-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Active Keys */}
      <Card>
        <CardHeader>
          <CardTitle>Active API Keys</CardTitle>
        </CardHeader>
        <CardContent>
          {activeKeys.length > 0 ? (
            <div className="space-y-4">
              {activeKeys.map((apiKey) => (
                <div key={apiKey.id} className="p-4 border rounded-lg hover:border-blue-500 transition-colors">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className="size-10 bg-blue-100 rounded-lg flex items-center justify-center">
                        <Key className="size-5 text-blue-600" />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h3>{apiKey.name}</h3>
                          <Badge className="bg-green-100 text-green-700">Active</Badge>
                        </div>
                        <div className="text-sm text-gray-500 mt-1">
                          Created {formatDate(apiKey.created_at)}
                        </div>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleDeleteKey(apiKey.id)}
                      >
                        <Trash2 className="size-4 mr-2" />
                        Delete
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      {/* Only show/mask key if we actually HAVE the key value (which we usually don't for list). 
                           The current backend implementation sends NO key in list, only in create.
                           So showing "..." is the only option here. Toggle visibility won't work for existing keys. */}
                      <code className="flex-1 px-3 py-2 bg-gray-50 rounded text-sm font-mono text-gray-500">
                        {/* We don't have the key secret for listed keys, checking visibleKeys is moot unless we stored it locally (not implemented) */}
                        ••••••••••••••••••••••••••••••••
                      </code>
                    </div>

                    <div className="flex items-center gap-6 text-sm text-gray-500">
                      <span>
                        Last used: {getRelativeTime(apiKey.last_used_at)}
                      </span>
                      {apiKey.expires_at && (
                        <span>
                          Expires: {formatDate(apiKey.expires_at)}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              No active API keys
            </div>
          )}
        </CardContent>
      </Card>

      {/* Revoked Keys */}
      {revokedKeys.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Revoked API Keys</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {revokedKeys.map((apiKey) => (
                <div key={apiKey.id} className="p-4 border border-gray-200 rounded-lg opacity-60">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className="size-10 bg-gray-100 rounded-lg flex items-center justify-center">
                        <Key className="size-5 text-gray-400" />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h3>{apiKey.name}</h3>
                          <Badge variant="secondary">Revoked</Badge>
                        </div>
                        <div className="text-sm text-gray-500 mt-1">
                          Created {formatDate(apiKey.created_at)}
                        </div>
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleDeleteKey(apiKey.id)}
                    >
                      <Trash2 className="size-4 mr-2" />
                      Delete
                    </Button>
                  </div>

                  <div className="space-y-2">
                    <div className="text-sm text-gray-500">
                      Last used: {getRelativeTime(apiKey.last_used_at)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Security Notice */}
      <Card className="border-blue-200 bg-blue-50">
        <CardContent className="p-6">
          <div className="flex gap-4">
            <AlertTriangle className="size-6 text-blue-600 flex-shrink-0" />
            <div>
              <h3 className="text-blue-900 mb-2">Security Best Practices</h3>
              <ul className="text-sm text-blue-800 space-y-1 list-disc list-inside">
                <li>Never share your API keys in public repositories or client-side code</li>
                <li>Rotate keys regularly and revoke unused keys</li>
                <li>Use environment variables to store API keys in your applications</li>
                <li>Set expiration dates for temporary access</li>
                <li>Monitor API key usage in the audit logs</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
