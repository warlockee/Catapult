import { useState, useEffect, useRef } from 'react';

/**
 * A hook that provides a debounced value that updates after a delay.
 * Useful for search inputs where you want to avoid making API calls on every keystroke.
 *
 * @param initialValue - The initial value
 * @param delay - The debounce delay in milliseconds (default: 300)
 * @returns [inputValue, debouncedValue, setInputValue] - The immediate value, debounced value, and setter
 *
 * @example
 * const [inputValue, searchQuery, setInputValue] = useDebouncedState('', 300);
 *
 * // Use inputValue for the input field (updates immediately)
 * // Use searchQuery for API calls (updates after delay)
 */
export function useDebouncedState<T>(
  initialValue: T,
  delay: number = 300
): [T, T, React.Dispatch<React.SetStateAction<T>>] {
  const [inputValue, setInputValue] = useState<T>(initialValue);
  const [debouncedValue, setDebouncedValue] = useState<T>(initialValue);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }
    debounceRef.current = setTimeout(() => {
      setDebouncedValue(inputValue);
    }, delay);

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, [inputValue, delay]);

  return [inputValue, debouncedValue, setInputValue];
}
