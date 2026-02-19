import { useState, useEffect } from 'react';
import { Upload, X, FileText, Link } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { DialogFooter } from './ui/dialog';
import { api } from '../lib/api';
import { formatBytes } from '../lib/utils';
import type { Artifact, Release } from '../lib/api';

interface UploadArtifactFormProps {
  onSuccess: (artifact: Artifact) => void;
  onCancel: () => void;
}

export function UploadArtifactForm({ onSuccess, onCancel }: UploadArtifactFormProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [mode, setMode] = useState<'upload' | 'register'>('upload');
  const [filePath, setFilePath] = useState('');
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (mode === 'upload' && !file) {
      setError('Please select a file to upload');
      return;
    }

    if (mode === 'register' && !filePath) {
      setError('Please enter the file path');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      let artifact;
      if (mode === 'upload' && file) {
        const formData = new FormData();
        formData.append('file', file);
        // Default values for simplified upload
        formData.append('platform', 'any');
        artifact = await api.uploadArtifact(formData);
      } else {
        artifact = await api.registerArtifact({
          file_path: filePath,
          // Default values for simplified registration
          platform: 'any',
        });
      }

      onSuccess(artifact);
    } catch (err) {
      console.error('Upload failed:', err);
      setError(err instanceof Error ? err.message : 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };


  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      <Tabs value={mode} onValueChange={(v) => setMode(v as 'upload' | 'register')}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="upload">Upload File</TabsTrigger>
          <TabsTrigger value="register">Register Existing</TabsTrigger>
        </TabsList>

        <TabsContent value="upload" className="mt-4">
          <Label htmlFor="file-upload">File</Label>
          <div className="mt-2">
            <label
              htmlFor="file-upload"
              className="flex items-center justify-center w-full h-32 px-4 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer hover:border-blue-500 transition-colors"
            >
              {file ? (
                <div className="flex items-center gap-3">
                  <div className="flex-1">
                    <p className="text-sm font-medium">{file.name}</p>
                    <p className="text-xs text-gray-500">{formatBytes(file.size)}</p>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={(e: React.MouseEvent) => {
                      e.preventDefault();
                      setFile(null);
                    }}
                  >
                    <X className="size-4" />
                  </Button>
                </div>
              ) : (
                <div className="text-center">
                  <Upload className="mx-auto size-8 text-gray-400 mb-2" />
                  <p className="text-sm text-gray-600">Click to upload or drag and drop</p>
                  <p className="text-xs text-gray-500 mt-1">
                    .whl, .tar.gz, .zip, or other artifact files
                  </p>
                </div>
              )}
            </label>
            <input
              id="file-upload"
              type="file"
              className="hidden"
              onChange={handleFileChange}
              accept=".whl,.tar.gz,.zip,.tgz"
            />
          </div>
        </TabsContent>

        <TabsContent value="register" className="space-y-4 mt-4">
          <div>
            <Label htmlFor="file-path">File Path</Label>
            <div className="relative mt-2">
              <Input
                id="file-path"
                type="text"
                placeholder="/path/to/artifact/on/server/file.whl"
                value={filePath}
                onChange={(e) => setFilePath(e.target.value)}
              />
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Absolute path to the file on the server
            </p>
          </div>
        </TabsContent>
      </Tabs>

      <DialogFooter>
        <Button type="button" variant="outline" onClick={onCancel} disabled={uploading}>
          Cancel
        </Button>
        <Button type="submit" disabled={uploading || (mode === 'upload' ? !file : !filePath)}>
          {uploading ? (mode === 'upload' ? 'Uploading...' : 'Registering...') : (mode === 'upload' ? 'Add Artifact' : 'Add Artifact')}
        </Button>
      </DialogFooter>
    </form >
  );
}
