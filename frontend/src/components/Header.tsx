import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Bell, Search, User, Package, GitBranch, Rocket, Loader2, X, Command } from 'lucide-react';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { useGlobalSearch, type SearchResult } from '../hooks/useGlobalSearch';
import { cn } from '../lib/utils';

export function Header() {
  const navigate = useNavigate();

  // Local input state - updates immediately without blocking
  const [inputValue, setInputValue] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);

  const inputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const {
    results,
    isSearching,
    hasResults,
    search,
    clearResults,
  } = useGlobalSearch({ debounceMs: 250, maxResultsPerCategory: 5 });

  // Memoize flattened results for keyboard navigation
  const allResults = useMemo<SearchResult[]>(() => [
    ...results.models,
    ...results.releases,
    ...results.deployments,
  ], [results.models, results.releases, results.deployments]);

  // Store allResults in ref for use in callbacks without causing re-renders
  const allResultsRef = useRef(allResults);
  allResultsRef.current = allResults;

  // Handle click outside to close dropdown
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Keyboard shortcut: Cmd+K or Ctrl+K to focus search
  useEffect(() => {
    function handleGlobalKeyDown(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
        event.preventDefault();
        inputRef.current?.focus();
        setIsOpen(true);
      }
    }

    document.addEventListener('keydown', handleGlobalKeyDown);
    return () => document.removeEventListener('keydown', handleGlobalKeyDown);
  }, []);

  // Scroll selected item into view
  useEffect(() => {
    if (selectedIndex >= 0 && resultsRef.current) {
      const selectedElement = resultsRef.current.querySelector(`[data-index="${selectedIndex}"]`);
      selectedElement?.scrollIntoView({ block: 'nearest' });
    }
  }, [selectedIndex]);

  // Reset selection when results change
  useEffect(() => {
    setSelectedIndex(-1);
  }, [results]);

  const handleResultClick = useCallback((result: SearchResult) => {
    navigate(result.url);
    setIsOpen(false);
    setInputValue('');
    clearResults();
  }, [navigate, clearResults]);

  // Handle input changes - update local state immediately, trigger debounced search
  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setInputValue(value);
    setIsOpen(true);
    setSelectedIndex(-1);
    search(value); // This is debounced internally
  }, [search]);

  const handleClear = useCallback(() => {
    setInputValue('');
    clearResults();
    setSelectedIndex(-1);
    inputRef.current?.focus();
  }, [clearResults]);

  // Handle keyboard navigation - use refs to avoid stale closures
  const handleKeyDown = useCallback((event: React.KeyboardEvent<HTMLInputElement>) => {
    const currentResults = allResultsRef.current;

    if (event.key === 'Escape') {
      setIsOpen(false);
      inputRef.current?.blur();
      return;
    }

    if (!isOpen || currentResults.length === 0) return;

    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        setSelectedIndex(prev =>
          prev < currentResults.length - 1 ? prev + 1 : 0
        );
        break;
      case 'ArrowUp':
        event.preventDefault();
        setSelectedIndex(prev =>
          prev > 0 ? prev - 1 : currentResults.length - 1
        );
        break;
      case 'Enter':
        event.preventDefault();
        setSelectedIndex(currentIdx => {
          if (currentIdx >= 0 && currentIdx < currentResults.length) {
            // Use setTimeout to avoid state update during render
            setTimeout(() => handleResultClick(currentResults[currentIdx]), 0);
          }
          return currentIdx;
        });
        break;
    }
  }, [isOpen, handleResultClick]);

  const getTypeIcon = useCallback((type: SearchResult['type']) => {
    switch (type) {
      case 'model':
        return <Package className="size-4 text-blue-500" />;
      case 'release':
        return <GitBranch className="size-4 text-purple-500" />;
      case 'deployment':
        return <Rocket className="size-4 text-green-500" />;
    }
  }, []);

  const getStatusBadge = useCallback((result: SearchResult) => {
    if (result.type === 'deployment' && result.metadata?.environment) {
      const env = result.metadata.environment;
      return (
        <Badge
          variant="outline"
          className={cn(
            "text-xs",
            env === 'production' && "border-green-500 text-green-700",
            env === 'staging' && "border-yellow-500 text-yellow-700",
            env === 'development' && "border-blue-500 text-blue-700"
          )}
        >
          {env}
        </Badge>
      );
    }
    if (result.type === 'release' && result.metadata?.status) {
      return (
        <Badge variant="secondary" className="text-xs">
          {result.metadata.status}
        </Badge>
      );
    }
    return null;
  }, []);

  const showDropdown = isOpen && (inputValue.trim().length > 0 || isSearching);

  return (
    <header className="bg-white border-b border-gray-200 px-8 py-4">
      <div className="flex items-center justify-between">
        {/* Search */}
        <div ref={containerRef} className="relative flex-1 max-w-2xl">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-gray-400" />
            <Input
              ref={inputRef}
              type="text"
              placeholder="Search models, releases, deployments..."
              className="pl-10 pr-20"
              value={inputValue}
              onChange={handleInputChange}
              onFocus={() => inputValue.trim() && setIsOpen(true)}
              onKeyDown={handleKeyDown}
              autoComplete="off"
              aria-label="Global search"
              aria-expanded={showDropdown}
              aria-haspopup="listbox"
              aria-controls="search-results"
            />
            {/* Right side indicators */}
            <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
              {isSearching && (
                <Loader2 className="size-4 text-gray-400 animate-spin" />
              )}
              {inputValue && !isSearching && (
                <button
                  type="button"
                  onClick={handleClear}
                  className="p-0.5 hover:bg-gray-100 rounded"
                  aria-label="Clear search"
                >
                  <X className="size-4 text-gray-400" />
                </button>
              )}
              {!inputValue && (
                <kbd className="hidden sm:inline-flex items-center gap-1 px-1.5 py-0.5 text-xs text-gray-400 bg-gray-100 rounded border">
                  <Command className="size-3" />K
                </kbd>
              )}
            </div>
          </div>

          {/* Search Results Dropdown */}
          {showDropdown && (
            <div
              ref={resultsRef}
              id="search-results"
              role="listbox"
              className="absolute top-full left-0 right-0 mt-2 bg-white rounded-lg shadow-lg border border-gray-200 max-h-[70vh] overflow-y-auto z-50"
            >
              {isSearching && !hasResults && (
                <div className="p-4 text-center text-gray-500">
                  <Loader2 className="size-5 animate-spin mx-auto mb-2" />
                  Searching...
                </div>
              )}

              {!isSearching && !hasResults && inputValue.trim() && (
                <div className="p-4 text-center text-gray-500">
                  <Search className="size-8 text-gray-300 mx-auto mb-2" />
                  <p>No results found for "{inputValue}"</p>
                  <p className="text-sm mt-1">Try different keywords</p>
                </div>
              )}

              {hasResults && (
                <div className="py-2">
                  {/* Models Section */}
                  {results.models.length > 0 && (
                    <div>
                      <div className="px-3 py-1.5 text-xs font-semibold text-gray-500 uppercase tracking-wider bg-gray-50">
                        Models ({results.models.length})
                      </div>
                      {results.models.map((result, idx) => {
                        const globalIdx = idx;
                        return (
                          <button
                            key={result.id}
                            type="button"
                            data-index={globalIdx}
                            role="option"
                            aria-selected={selectedIndex === globalIdx}
                            className={cn(
                              "w-full px-3 py-2 flex items-center gap-3 text-left hover:bg-gray-50 transition-colors",
                              selectedIndex === globalIdx && "bg-blue-50"
                            )}
                            onClick={() => handleResultClick(result)}
                            onMouseEnter={() => setSelectedIndex(globalIdx)}
                          >
                            {getTypeIcon(result.type)}
                            <div className="flex-1 min-w-0">
                              <div className="font-medium text-gray-900 truncate">
                                {result.title}
                              </div>
                              <div className="text-sm text-gray-500 truncate">
                                {result.subtitle}
                              </div>
                            </div>
                            {result.metadata?.tags && result.metadata.tags.length > 0 && (
                              <div className="flex gap-1">
                                {result.metadata.tags.slice(0, 2).map(tag => (
                                  <Badge key={tag} variant="outline" className="text-xs">
                                    {tag}
                                  </Badge>
                                ))}
                              </div>
                            )}
                          </button>
                        );
                      })}
                    </div>
                  )}

                  {/* Releases Section */}
                  {results.releases.length > 0 && (
                    <div>
                      <div className="px-3 py-1.5 text-xs font-semibold text-gray-500 uppercase tracking-wider bg-gray-50">
                        Releases ({results.releases.length})
                      </div>
                      {results.releases.map((result, idx) => {
                        const globalIdx = results.models.length + idx;
                        return (
                          <button
                            key={result.id}
                            type="button"
                            data-index={globalIdx}
                            role="option"
                            aria-selected={selectedIndex === globalIdx}
                            className={cn(
                              "w-full px-3 py-2 flex items-center gap-3 text-left hover:bg-gray-50 transition-colors",
                              selectedIndex === globalIdx && "bg-blue-50"
                            )}
                            onClick={() => handleResultClick(result)}
                            onMouseEnter={() => setSelectedIndex(globalIdx)}
                          >
                            {getTypeIcon(result.type)}
                            <div className="flex-1 min-w-0">
                              <div className="font-medium text-gray-900 truncate">
                                {result.title}
                              </div>
                              <div className="text-sm text-gray-500 truncate">
                                {result.subtitle}
                              </div>
                            </div>
                            {getStatusBadge(result)}
                          </button>
                        );
                      })}
                    </div>
                  )}

                  {/* Deployments Section */}
                  {results.deployments.length > 0 && (
                    <div>
                      <div className="px-3 py-1.5 text-xs font-semibold text-gray-500 uppercase tracking-wider bg-gray-50">
                        Deployments ({results.deployments.length})
                      </div>
                      {results.deployments.map((result, idx) => {
                        const globalIdx = results.models.length + results.releases.length + idx;
                        return (
                          <button
                            key={result.id}
                            type="button"
                            data-index={globalIdx}
                            role="option"
                            aria-selected={selectedIndex === globalIdx}
                            className={cn(
                              "w-full px-3 py-2 flex items-center gap-3 text-left hover:bg-gray-50 transition-colors",
                              selectedIndex === globalIdx && "bg-blue-50"
                            )}
                            onClick={() => handleResultClick(result)}
                            onMouseEnter={() => setSelectedIndex(globalIdx)}
                          >
                            {getTypeIcon(result.type)}
                            <div className="flex-1 min-w-0">
                              <div className="font-medium text-gray-900 truncate">
                                {result.title}
                              </div>
                              <div className="text-sm text-gray-500 truncate">
                                {result.subtitle}
                              </div>
                            </div>
                            {getStatusBadge(result)}
                          </button>
                        );
                      })}
                    </div>
                  )}

                  {/* Keyboard hints */}
                  <div className="px-3 py-2 border-t border-gray-100 flex items-center gap-4 text-xs text-gray-400">
                    <span className="flex items-center gap-1">
                      <kbd className="px-1.5 py-0.5 bg-gray-100 rounded">↑</kbd>
                      <kbd className="px-1.5 py-0.5 bg-gray-100 rounded">↓</kbd>
                      to navigate
                    </span>
                    <span className="flex items-center gap-1">
                      <kbd className="px-1.5 py-0.5 bg-gray-100 rounded">Enter</kbd>
                      to select
                    </span>
                    <span className="flex items-center gap-1">
                      <kbd className="px-1.5 py-0.5 bg-gray-100 rounded">Esc</kbd>
                      to close
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right side actions */}
        <div className="flex items-center gap-4 ml-4">
          <Button variant="ghost" size="icon" className="relative" aria-label="Notifications">
            <Bell className="size-5" />
            <span className="absolute top-1 right-1 size-2 bg-blue-600 rounded-full" />
          </Button>

          <div className="flex items-center gap-3 pl-4 border-l">
            <div className="flex flex-col items-end">
              <span className="text-sm font-medium">Admin User</span>
              <span className="text-xs text-gray-500">admin@example.com</span>
            </div>
            <div className="size-10 bg-blue-600 rounded-full flex items-center justify-center">
              <User className="size-5 text-white" />
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
