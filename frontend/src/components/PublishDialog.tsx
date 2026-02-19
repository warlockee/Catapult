import { useState } from 'react';
import { Globe, ArrowLeft, ChevronRight, FileText } from 'lucide-react';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from './ui/dialog';

type Platform = 'select' | 'eigen' | 'deepinfra';

interface PublishDialogProps {
  versionId: string;
  modelId?: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function PublishDialog({ versionId, modelId, open, onOpenChange }: PublishDialogProps) {
  const [platform, setPlatform] = useState<Platform>('select');

  const handleClose = (isOpen: boolean) => {
    if (!isOpen) setPlatform('select');
    onOpenChange(isOpen);
  };

  // Coming soon sub-view
  if (platform === 'eigen' || platform === 'deepinfra') {
    const name = platform === 'eigen' ? 'eigen.ai' : 'deepinfra.ai';
    return (
      <Dialog open={open} onOpenChange={handleClose}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="sm" onClick={() => setPlatform('select')}>
                <ArrowLeft className="size-4" />
              </Button>
              <DialogTitle>{name}</DialogTitle>
            </div>
            <DialogDescription>Publishing to {name}</DialogDescription>
          </DialogHeader>
          <div className="py-12 text-center text-gray-500">
            <Globe className="size-12 mx-auto mb-4 text-gray-300" />
            <p className="text-lg font-medium">Coming Soon</p>
            <p className="text-sm mt-1">{name} integration is under development.</p>
          </div>
        </DialogContent>
      </Dialog>
    );
  }

  // Platform selector
  const platforms = [
    {
      id: 'eigen' as Platform,
      name: 'eigen.ai',
      description: 'Publish to eigen.ai platform',
      icon: Globe,
      available: false,
      status: 'Coming Soon',
      statusColor: 'bg-gray-100 text-gray-600',
    },
    {
      id: 'deepinfra' as Platform,
      name: 'deepinfra.ai',
      description: 'Publish to DeepInfra inference platform',
      icon: Globe,
      available: false,
      status: 'Coming Soon',
      statusColor: 'bg-gray-100 text-gray-600',
    },
  ];

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Publish</DialogTitle>
          <DialogDescription>Select a publishing platform</DialogDescription>
        </DialogHeader>
        <div className="space-y-3 py-2">
          {modelId && (
            <a
              href={`/models/${modelId}/card`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-800 hover:underline mb-1"
            >
              <FileText className="size-4" />
              View Model Card
            </a>
          )}
          {platforms.map((p) => {
            const Icon = p.icon;
            return (
              <button
                key={p.id}
                className={`w-full flex items-center gap-3 p-4 border rounded-lg text-left transition-colors ${
                  p.available
                    ? 'hover:bg-gray-50 hover:border-gray-300 cursor-pointer'
                    : 'opacity-60 cursor-not-allowed'
                }`}
                onClick={() => p.available && setPlatform(p.id)}
                disabled={!p.available}
              >
                <div className="size-10 bg-gray-100 rounded-lg flex items-center justify-center shrink-0">
                  <Icon className="size-5 text-gray-600" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm">{p.name}</span>
                    <Badge className={p.statusColor}>{p.status}</Badge>
                  </div>
                  <p className="text-xs text-gray-500 mt-0.5">{p.description}</p>
                </div>
                {p.available && <ChevronRight className="size-4 text-gray-400 shrink-0" />}
              </button>
            );
          })}
        </div>
      </DialogContent>
    </Dialog>
  );
}
