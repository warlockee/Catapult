import { useQueryClient, useQuery } from '@tanstack/react-query';
import { Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, Image, Package, GitBranch, Key, Settings, HelpCircle, FileArchive } from 'lucide-react';
import { api } from '../lib/api';
import { formatBytes } from '../lib/utils';

interface NavItem {
  path: string;
  icon: typeof LayoutDashboard;
  label: string;
  prefetchKey?: string;
}

const navItems: NavItem[] = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/artifacts', icon: FileArchive, label: 'Artifacts', prefetchKey: 'artifacts' },
  { path: '/models', icon: Image, label: 'Models', prefetchKey: 'images' },
  { path: '/releases', icon: Package, label: 'Releases', prefetchKey: 'releases' },
  { path: '/deployments', icon: GitBranch, label: 'Deployments', prefetchKey: 'deployments' },
  { path: '/api-keys', icon: Key, label: 'API Keys' },
];

const bottomNavItems: NavItem[] = [
  { path: '/settings', icon: Settings, label: 'Settings' },
  { path: '/help', icon: HelpCircle, label: 'Help' },
];


export function Sidebar() {
  const location = useLocation();
  const queryClient = useQueryClient();
  const { data: storageStats } = useQuery({
    queryKey: ['storage'],
    queryFn: () => api.getSystemStorage(),
    refetchInterval: 60000, // Refresh every minute
  });

  const usedPercent = storageStats ? Math.round((storageStats.used / storageStats.total) * 100) : 0;

  const isActive = (path: string) => {
    if (path === '/') {
      return location.pathname === '/';
    }
    if (path === '/deployments') {
      return location.pathname.startsWith('/deployments');
    }
    return location.pathname.startsWith(path);
  };

  const handleMouseEnter = (prefetchKey?: string) => {
    if (!prefetchKey) return;

    switch (prefetchKey) {
      case 'images':
        queryClient.prefetchQuery({ queryKey: ['images', 1, ''], queryFn: () => api.listImages({ page: 1, size: 12 }) });
        break;
      case 'releases':
        queryClient.prefetchQuery({ queryKey: ['releases'], queryFn: () => api.listReleases({ page: 1, size: 10 }) });
        break;
      case 'deployments':
        queryClient.prefetchQuery({ queryKey: ['deployments'], queryFn: () => api.listDeployments({ page: 1, size: 10 }) });
        break;
      case 'artifacts':
        queryClient.prefetchQuery({
          queryKey: ['artifacts', 'all', 'all'],
          queryFn: () => api.listArtifacts({})
        });
        break;
    }
  };

  const renderNavItem = (item: NavItem) => {
    const Icon = item.icon;
    const active = isActive(item.path);

    return (
      <Link
        key={item.path}
        to={item.path}
        onMouseEnter={() => handleMouseEnter(item.prefetchKey)}
        className={`
          w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors
          ${active
            ? 'bg-blue-600 text-white'
            : 'text-gray-300 hover:bg-gray-800 hover:text-white'
          }
        `}
      >
        <Icon className="size-5" />
        <span>{item.label}</span>
      </Link>
    );
  };

  return (
    <aside className="w-64 bg-gray-900 text-white flex flex-col">
      <div className="p-6 border-b border-gray-800">
        <Link to="/" className="flex items-center gap-2">
          <Package className="size-8 text-blue-400" />
          <div>
            <div>Catapult</div>
            <div className="text-xs text-gray-400">v1.0.1</div>
          </div>
        </Link>
      </div>

      <nav className="flex-1 p-4">
        <div className="space-y-1">
          {navItems.map(renderNavItem)}
        </div>

        <div className="mt-8 pt-8 border-t border-gray-800 space-y-1">
          {bottomNavItems.map(renderNavItem)}
        </div>
      </nav>

      <div className="p-4 border-t border-gray-800">
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="text-xs text-gray-400 mb-1">Storage Used</div>
          {storageStats ? (
            <>
              <div className="mb-2 text-sm">
                {formatBytes(storageStats.used)} / {formatBytes(storageStats.total)}
              </div>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all duration-500"
                  style={{ width: `${usedPercent}%` }}
                />
              </div>
            </>
          ) : (
            <div className="animate-pulse">
              <div className="h-4 bg-gray-700 rounded mb-2 w-2/3"></div>
              <div className="h-2 bg-gray-700 rounded-full"></div>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
