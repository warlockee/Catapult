import { AlertCircle, RefreshCw } from 'lucide-react';
import { Button } from './ui/button';

interface ErrorStateProps {
    title: string;
    description?: string;
    onRetry: () => void;
}

export function ErrorState({
    title,
    description = "There was an error communicating with the server.",
    onRetry
}: ErrorStateProps) {
    return (
        <div className="p-6 border border-red-200 bg-red-50 rounded-lg text-red-800 mt-4">
            <div className="flex items-start gap-4">
                <AlertCircle className="size-5 mt-0.5 text-red-600 shrink-0" />
                <div className="flex-1">
                    <h3 className="font-semibold mb-2 text-red-900">{title}</h3>
                    <p className="text-sm mb-4 text-red-700">{description}</p>
                    <Button
                        onClick={onRetry}
                        variant="outline"
                        className="bg-red-100 hover:bg-red-200 border-red-200 text-red-900 hover:text-red-950"
                    >
                        <RefreshCw className="size-4 mr-2" />
                        Retry
                    </Button>
                </div>
            </div>
        </div>
    );
}
