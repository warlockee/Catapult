interface LoadingStateProps {
  title: string;
  message?: string;
}

/**
 * Reusable loading state component for list views.
 */
export function LoadingState({ title, message }: LoadingStateProps) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1>{title}</h1>
          <p className="text-gray-500 mt-1">{message || `Loading ${title.toLowerCase()}...`}</p>
        </div>
      </div>
      <div className="flex justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    </div>
  );
}
