/**
 * Search Input Component
 *
 * TAG: FUNCTION-TAG-017
 * Search input with autocomplete and recent searches
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';

interface SearchInputProps {
  value: string;
  onSearch: (query: string) => void;
  placeholder?: string;
  recent_searches?: string[];
  is_searching?: boolean;
  debounce_ms?: number;
}

export const SearchInput: React.FC<SearchInputProps> = ({
  value,
  onSearch,
  placeholder = 'Search...',
  recent_searches = [],
  is_searching = false,
  debounce_ms = 300,
}) => {
  const [input_value, setInputValue] = useState(value);
  const [show_suggestions, setShowSuggestions] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [selected_index, setSelectedIndex] = useState(-1);
  const input_ref = useRef<HTMLInputElement>(null);
  const debounce_ref = useRef<NodeJS.Timeout>();

  useEffect(() => {
    setInputValue(value);
  }, [value]);

  useEffect(() => {
    // Clean up debounce timer
    return () => {
      if (debounce_ref.current) {
        clearTimeout(debounce_ref.current);
      }
    };
  }, []);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const new_value = e.target.value;
    setInputValue(new_value);

    // Clear existing debounce timer
    if (debounce_ref.current) {
      clearTimeout(debounce_ref.current);
    }

    // Set new debounce timer
    debounce_ref.current = setTimeout(() => {
      onSearch(new_value);
    }, debounce_ms);

    // Update suggestions
    if (new_value.trim()) {
      const matching_recent = recent_searches.filter(search =>
        search.toLowerCase().includes(new_value.toLowerCase())
      ).slice(0, 5);
      setSuggestions(matching_recent);
      setShowSuggestions(true);
    } else {
      setSuggestions(recent_searches.slice(0, 5));
      setShowSuggestions(true);
    }
  }, [onSearch, debounce_ms, recent_searches]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (!show_suggestions || suggestions.length === 0) {
      if (e.key === 'Enter') {
        onSearch(input_value);
        setShowSuggestions(false);
      }
      return;
    }

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex(prev =>
          prev < suggestions.length - 1 ? prev + 1 : prev
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex(prev => (prev > 0 ? prev - 1 : -1));
        break;
      case 'Enter':
        e.preventDefault();
        if (selected_index >= 0) {
          const selected = suggestions[selected_index];
          setInputValue(selected);
          onSearch(selected);
          setShowSuggestions(false);
          setSelectedIndex(-1);
        } else {
          onSearch(input_value);
          setShowSuggestions(false);
        }
        break;
      case 'Escape':
        setShowSuggestions(false);
        setSelectedIndex(-1);
        break;
    }
  }, [show_suggestions, suggestions, selected_index, input_value, onSearch]);

  const handleFocus = useCallback(() => {
    if (input_value.trim()) {
      const matching_recent = recent_searches.filter(search =>
        search.toLowerCase().includes(input_value.toLowerCase())
      ).slice(0, 5);
      setSuggestions(matching_recent);
    } else {
      setSuggestions(recent_searches.slice(0, 5));
    }
    setShowSuggestions(true);
  }, [input_value, recent_searches]);

  const handleBlur = useCallback(() => {
    // Delay hiding suggestions to allow clicking on them
    setTimeout(() => {
      setShowSuggestions(false);
      setSelectedIndex(-1);
    }, 200);
  }, []);

  const handleSuggestionClick = useCallback((suggestion: string) => {
    setInputValue(suggestion);
    onSearch(suggestion);
    setShowSuggestions(false);
    input_ref.current?.focus();
  }, [onSearch]);

  const handleClear = useCallback(() => {
    setInputValue('');
    onSearch('');
    input_ref.current?.focus();
  }, [onSearch]);

  return (
    <div className="search-input-container">
      <div className="search-input-wrapper">
        <span className="search-icon">\u1F50D</span>
        <input
          ref={input_ref}
          type="text"
          value={input_value}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onFocus={handleFocus}
          onBlur={handleBlur}
          placeholder={placeholder}
          className="search-input"
          aria-label="Search"
          aria-autocomplete="list"
          aria-controls="search-suggestions-list"
          aria-activedescendant={selected_index >= 0 ? `suggestion-${selected_index}` : undefined}
        />
        {input_value && (
          <button
            className="clear-button"
            onClick={handleClear}
            aria-label="Clear search"
          >
            \u00D7
          </button>
        )}
        {is_searching && (
          <span className="search-spinner" aria-label="Searching...">
            \u21BB
          </span>
        )}
      </div>

      {show_suggestions && suggestions.length > 0 && (
        <ul
          id="search-suggestions-list"
          className="search-suggestions"
          role="listbox"
        >
          {suggestions.map((suggestion, index) => (
            <li
              key={index}
              id={`suggestion-${index}`}
              className={`suggestion-item ${index === selected_index ? 'selected' : ''}`}
              onClick={() => handleSuggestionClick(suggestion)}
              role="option"
              aria-selected={index === selected_index}
            >
              <span className="suggestion-icon">\u1F552</span>
              <span className="suggestion-text">{suggestion}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default SearchInput;
