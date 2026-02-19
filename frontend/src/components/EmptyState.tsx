import { type LucideIcon } from 'lucide-react';
import { Card, CardContent } from './ui/card';

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description: string;
}

/**
 * Reusable empty state component for list views with no data.
 */
export function EmptyState({ icon: Icon, title, description }: EmptyStateProps) {
  return (
    <Card>
      <CardContent className="p-12 text-center">
        <Icon className="size-12 text-gray-300 mx-auto mb-4" />
        <h3 className="text-gray-500 mb-2">{title}</h3>
        <p className="text-sm text-gray-400">{description}</p>
      </CardContent>
    </Card>
  );
}
