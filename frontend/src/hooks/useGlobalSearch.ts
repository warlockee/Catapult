import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '../lib/api';
import type { Image, Release, Deployment } from '../lib/api';

export interface SearchResult {
  id: string;
  type: 'model' | 'release' | 'deployment';
  title: string;
  subtitle: string;
  url: string;
  metadata?: {
    version?: string;
    environment?: string;
    status?: string;
    tags?: string[];
  };
}

export interface SearchResults {
  models: SearchResult[];
  releases: SearchResult[];
  deployments: SearchResult[];
  total: number;
}

const EMPTY_RESULTS: SearchResults = {
  models: [],
  releases: [],
  deployments: [],
  total: 0,
};

interface UseGlobalSearchOptions {
  debounceMs?: number;
  maxResultsPerCategory?: number;
}

export function useGlobalSearch(options: UseGlobalSearchOptions = {}) {
  const { debounceMs = 300, maxResultsPerCategory = 5 } = options;

  const [results, setResults] = useState<SearchResults>(EMPTY_RESULTS);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearResults = useCallback(() => {
    setResults(EMPTY_RESULTS);
    setError(null);
  }, []);

  // Core search function - called with debounced query
  const performSearch = useCallback(async (searchQuery: string) => {
    // Cancel any in-flight requests
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    if (!searchQuery.trim()) {
      setResults(EMPTY_RESULTS);
      setIsSearching(false);
      return;
    }

    setIsSearching(true);
    setError(null);

    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    try {
      // Search all three endpoints in parallel
      const [modelsResponse, releasesResponse, deploymentsResponse] = await Promise.all([
        api.listImages({ search: searchQuery, size: maxResultsPerCategory, signal }).catch(() => ({ items: [] })),
        api.listReleases({ version: searchQuery, size: maxResultsPerCategory, signal }).catch(() => ({ items: [] })),
        api.listDeployments({ size: 20, signal }).catch(() => ({ items: [] })),
      ]);

      // Check if request was aborted
      if (signal.aborted) return;

      // Transform models to search results
      const modelResults: SearchResult[] = (modelsResponse.items || []).map((model: Image) => ({
        id: model.id,
        type: 'model' as const,
        title: model.name,
        subtitle: model.description || model.storage_path,
        url: `/models/${model.id}`,
        metadata: {
          tags: model.tags,
        },
      }));

      // Transform releases to search results
      const releaseResults: SearchResult[] = (releasesResponse.items || []).map((release: Release) => ({
        id: release.id,
        type: 'release' as const,
        title: `${release.image_name || 'Unknown'} v${release.version}`,
        subtitle: `${release.tag} - ${release.platform}`,
        url: `/releases/${release.id}`,
        metadata: {
          version: release.version,
          status: release.status,
        },
      }));

      // Filter deployments client-side (API doesn't support text search)
      const searchLower = searchQuery.toLowerCase();
      const filteredDeployments = (deploymentsResponse.items || []).filter((d: Deployment) => {
        return (
          (d.image_name || '').toLowerCase().includes(searchLower) ||
          (d.environment || '').toLowerCase().includes(searchLower) ||
          (d.release_version || '').toLowerCase().includes(searchLower)
        );
      }).slice(0, maxResultsPerCategory);

      const deploymentResults: SearchResult[] = filteredDeployments.map((deployment: Deployment) => ({
        id: deployment.id,
        type: 'deployment' as const,
        title: `${deployment.image_name || 'Unknown'} → ${deployment.environment}`,
        subtitle: `v${deployment.release_version || 'unknown'} - ${new Date(deployment.deployed_at).toLocaleDateString()}`,
        url: `/deployments`,
        metadata: {
          version: deployment.release_version,
          environment: deployment.environment,
          status: deployment.status,
        },
      }));

      const total = modelResults.length + releaseResults.length + deploymentResults.length;

      setResults({
        models: modelResults,
        releases: releaseResults,
        deployments: deploymentResults,
        total,
      });
    } catch (err: any) {
      if (err.name === 'AbortError') return;
      console.error('Search error:', err);
      setError('Search failed. Please try again.');
    } finally {
      if (!signal.aborted) {
        setIsSearching(false);
      }
    }
  }, [maxResultsPerCategory]);

  // Debounced search trigger - called from component
  const search = useCallback((query: string) => {
    // Clear existing debounce timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
      debounceTimerRef.current = null;
    }

    if (!query.trim()) {
      // Immediately clear if query is empty
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      setResults(EMPTY_RESULTS);
      setIsSearching(false);
      return;
    }

    // Show searching state immediately for better UX
    setIsSearching(true);

    // Debounce the actual API call
    debounceTimerRef.current = setTimeout(() => {
      performSearch(query);
    }, debounceMs);
  }, [debounceMs, performSearch]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  return {
    results,
    isSearching,
    error,
    search,
    clearResults,
    hasResults: results.total > 0,
  };
}
