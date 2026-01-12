/**
 * Search Interface Component
 *
 * TAG: FUNCTION-TAG-016
 * Unified search interface with filters and results display
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import {
  SearchQuery,
  SearchFilters,
  SearchResult,
  SearchInterfaceProps,
  SearchInterfaceState,
  SavedSearch,
} from './types';
import { SearchInput } from './SearchInput';
import { SearchFiltersPanel } from './SearchFiltersPanel';
import { SearchResults } from './SearchResults';
import { SavedSearchesPanel } from './SavedSearchesPanel';

export const SearchInterface: React.FC<SearchInterfaceProps> = ({
  onSearch,
  onResultClick,
  recent_searches = [],
  saved_searches = [],
  max_results = 50,
}) => {
  const [state, setState] = useState<SearchInterfaceState>({
    query_text: '',
    filters: {
      content_types: [],
      tags: [],
      speakers: [],
    },
    is_searching: false,
    results: [],
    selected_result_id: null,
    show_filters: false,
    show_saved_searches: false,
  });

  const [debounced_query, setDebouncedQuery] = useState('');

  // Debounce search input
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(state.query_text);
    }, 300);

    return () => clearTimeout(timer);
  }, [state.query_text]);

  // Execute search when debounced query changes
  useEffect(() => {
    if (debounced_query.trim()) {
      executeSearch(debounced_query);
    } else {
      setState(prev => ({ ...prev, results: [] }));
    }
  }, [debounced_query]);

  const executeSearch = useCallback(async (query_text: string) => {
    setState(prev => ({ ...prev, is_searching: true }));

    const query: SearchQuery = {
      text: query_text,
      filters: state.filters,
    };

    try {
      // This would call the actual search API
      const results = await performSearch(query);
      setState(prev => ({
        ...prev,
        results: results.slice(0, max_results),
        is_searching: false,
      }));
    } catch (error) {
      console.error('Search failed:', error);
      setState(prev => ({ ...prev, is_searching: false, results: [] }));
    }
  }, [state.filters, max_results]);

  const handleSearch = useCallback((query_text: string) => {
    setState(prev => ({ ...prev, query_text }));
  }, []);

  const handleFilterChange = useCallback((filters: Partial<SearchFilters>) => {
    setState(prev => ({
      ...prev,
      filters: { ...prev.filters, ...filters },
    }));
  }, []);

  const handleResultClick = useCallback((result: SearchResult) => {
    setState(prev => ({ ...prev, selected_result_id: result.id }));
    onResultClick(result);
  }, [onResultClick]);

  const handleClearFilters = useCallback(() => {
    setState(prev => ({
      ...prev,
      filters: {
        content_types: [],
        tags: [],
        speakers: [],
      },
    }));
  }, []);

  const handleToggleFilters = useCallback(() => {
    setState(prev => ({ ...prev, show_filters: !prev.show_filters }));
  }, []);

  const handleToggleSavedSearches = useCallback(() => {
    setState(prev => ({ ...prev, show_saved_searches: !prev.show_saved_searches }));
  }, []);

  const handleLoadSavedSearch = useCallback((saved_search: SavedSearch) => {
    setState(prev => ({
      ...prev,
      query_text: saved_search.query.text,
      filters: saved_search.query.filters,
    }));
  }, []);

  const handleSaveSearch = useCallback((name: string) => {
    // This would save the current search
    const new_saved_search: SavedSearch = {
      id: `saved-${Date.now()}`,
      name,
      query: {
        text: state.query_text,
        filters: state.filters,
      },
      created_at: new Date().toISOString(),
    };
    // In real implementation, this would be saved to backend
    console.log('Saving search:', new_saved_search);
  }, [state.query_text, state.filters]);

  // Calculate filter count for badge
  const active_filter_count = useMemo(() => {
    let count = 0;
    if (state.filters.content_types.length > 0) count++;
    if (state.filters.tags.length > 0) count++;
    if (state.filters.speakers.length > 0) count++;
    if (state.filters.min_gaslighting_probability !== undefined) count++;
    if (state.filters.emotion_types && state.filters.emotion_types.length > 0) count++;
    if (state.filters.date_range?.start || state.filters.date_range?.end) count++;
    return count;
  }, [state.filters]);

  const has_active_filters = active_filter_count > 0;

  return (
    <div className="search-interface">
      <SearchInput
        value={state.query_text}
        onSearch={handleSearch}
        placeholder="Search transcripts, evidence, forensic results..."
        recent_searches={recent_searches}
        is_searching={state.is_searching}
      />

      <div className="search-controls">
        <button
          className={`filter-toggle-button ${state.show_filters ? 'active' : ''}`}
          onClick={handleToggleFilters}
          aria-label="Toggle filters"
        >
          <span>Filters</span>
          {has_active_filters && (
            <span className="filter-count-badge">{active_filter_count}</span>
          )}
        </button>

        <button
          className={`saved-searches-toggle-button ${state.show_saved_searches ? 'active' : ''}`}
          onClick={handleToggleSavedSearches}
          aria-label="Toggle saved searches"
        >
          Saved Searches
        </button>

        {has_active_filters && (
          <button
            className="clear-filters-button"
            onClick={handleClearFilters}
            aria-label="Clear all filters"
          >
            Clear Filters
          </button>
        )}
      </div>

      {state.show_filters && (
        <SearchFiltersPanel
          filters={state.filters}
          onFilterChange={handleFilterChange}
          onClear={handleClearFilters}
        />
      )}

      {state.show_saved_searches && (
        <SavedSearchesPanel
          saved_searches={saved_searches}
          onLoadSearch={handleLoadSavedSearch}
          onSaveSearch={handleSaveSearch}
          current_query={{
            text: state.query_text,
            filters: state.filters,
          }}
        />
      )}

      <SearchResults
        results={state.results}
        selected_id={state.selected_result_id}
        is_loading={state.is_searching}
        onResultClick={handleResultClick}
      />

      {!state.is_searching && state.query_text && state.results.length === 0 && (
        <div className="search-no-results">
          <p>No results found for "{state.query_text}"</p>
          {has_active_filters && (
            <p>Try adjusting your filters or search terms.</p>
          )}
        </div>
      )}
    </div>
  );
};

// Mock search function - would be replaced with actual API call
async function performSearch(query: SearchQuery): Promise<SearchResult[]> {
  // This is a placeholder - actual implementation would call the backend API
  return [];
}

export default SearchInterface;
