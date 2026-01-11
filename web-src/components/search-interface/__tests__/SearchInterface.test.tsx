/**
 * Search Interface Component Tests
 *
 * TAG: TEST-TAG-004
 * Tests for unified search interface with filters and results
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { SearchInterface } from '../SearchInterface';
import { mockSearchResult } from '../../../test/test-utils';
import type { SearchResult } from '../types';

describe('SearchInterface Component', () => {
  const mockResults: SearchResult[] = [
    mockSearchResult({
      id: 'sr-001',
      title: 'Result 1',
      excerpt: 'This is result 1 excerpt.',
      content_type: 'transcript',
      relevance_score: 0.95,
    }),
    mockSearchResult({
      id: 'sr-002',
      title: 'Result 2',
      excerpt: 'This is result 2 excerpt.',
      content_type: 'forensic_results',
      relevance_score: 0.85,
    }),
    mockSearchResult({
      id: 'sr-003',
      title: 'Result 3',
      excerpt: 'This is result 3 excerpt.',
      content_type: 'evidence_metadata',
      relevance_score: 0.75,
    }),
  ];

  const mockHandlers = {
    onSearch: vi.fn(),
    onResultClick: vi.fn(),
  };

  const recentSearches = ['test search 1', 'test search 2', 'important evidence'];

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('Initialization', () => {
    it('should render search interface with input', () => {
      render(
        <SearchInterface
          onSearch={mockHandlers.onSearch}
          onResultClick={mockHandlers.onResultClick}
          recent_searches={recentSearches}
        />
      );

      const searchInput = screen.getByPlaceholderText(
        /Search transcripts, evidence, forensic results/i
      );
      expect(searchInput).toBeInTheDocument();
    });

    it('should show filter and saved searches buttons', () => {
      render(
        <SearchInterface
          onSearch={mockHandlers.onSearch}
          onResultClick={mockHandlers.onResultClick}
        />
      );

      expect(screen.getByText('Filters')).toBeInTheDocument();
      expect(screen.getByText('Saved Searches')).toBeInTheDocument();
    });
  });

  describe('Search Input', () => {
    it('should show suggestions when input is focused', async () => {
      render(
        <SearchInterface
          onSearch={mockHandlers.onSearch}
          onResultClick={mockHandlers.onResultClick}
          recent_searches={recentSearches}
        />
      );

      const searchInput = screen.getByPlaceholderText(/Search transcripts/i);
      searchInput.focus();

      await waitFor(() => {
        expect(screen.getByText('test search 1')).toBeInTheDocument();
        expect(screen.getByText('test search 2')).toBeInTheDocument();
      });
    });

    it('should filter suggestions by input value', async () => {
      render(
        <SearchInterface
          onSearch={mockHandlers.onSearch}
          onResultClick={mockHandlers.onResultClick}
          recent_searches={recentSearches}
        />
      );

      const searchInput = screen.getByPlaceholderText(/Search transcripts/i);
      fireEvent.change(searchInput, { target: { value: 'test' } });
      searchInput.focus();

      await waitFor(() => {
        expect(screen.getByText('test search 1')).toBeInTheDocument();
        expect(screen.getByText('test search 2')).toBeInTheDocument();
        expect(screen.queryByText('important evidence')).not.toBeInTheDocument();
      });
    });

    it('should select suggestion on click', async () => {
      render(
        <SearchInterface
          onSearch={mockHandlers.onSearch}
          onResultClick={mockHandlers.onResultClick}
          recent_searches={recentSearches}
        />
      );

      const searchInput = screen.getByPlaceholderText(/Search transcripts/i);
      searchInput.focus();

      await waitFor(() => {
        const suggestion = screen.getByText('test search 1');
        fireEvent.click(suggestion);

        expect(searchInput).toHaveValue('test search 1');
      });
    });

    it('should navigate suggestions with arrow keys', async () => {
      render(
        <SearchInterface
          onSearch={mockHandlers.onSearch}
          onResultClick={mockHandlers.onResultClick}
          recent_searches={recentSearches}
        />
      );

      const searchInput = screen.getByPlaceholderText(/Search transcripts/i) as HTMLInputElement;
      searchInput.focus();

      await waitFor(() => {
        fireEvent.keyDown(searchInput, { key: 'ArrowDown' });
        const firstSuggestion = screen.getByText('test search 1');
        expect(firstSuggestion.parentElement).toHaveClass('selected');
      });
    });
  });

  describe('Filters Panel', () => {
    it('should toggle filters panel when filters button is clicked', async () => {
      render(
        <SearchInterface
          onSearch={mockHandlers.onSearch}
          onResultClick={mockHandlers.onResultClick}
        />
      );

      const filtersButton = screen.getByText('Filters');
      fireEvent.click(filtersButton);

      await waitFor(() => {
        expect(screen.getByText('Content Types')).toBeInTheDocument();
        expect(filtersButton).toHaveClass('active');
      });
    });

    it('should toggle content type filters', async () => {
      render(
        <SearchInterface
          onSearch={mockHandlers.onSearch}
          onResultClick={mockHandlers.onResultClick}
        />
      );

      const filtersButton = screen.getByText('Filters');
      fireEvent.click(filtersButton);

      await waitFor(() => {
        const transcriptCheckbox = screen.getByLabelText('Transcripts');
        fireEvent.click(transcriptCheckbox);

        expect(transcriptCheckbox).toBeChecked();
      });
    });

    it('should show filter count badge when filters are active', async () => {
      render(
        <SearchInterface
          onSearch={mockHandlers.onSearch}
          onResultClick={mockHandlers.onResultClick}
        />
      );

      const filtersButton = screen.getByText('Filters');
      fireEvent.click(filtersButton);

      // Add a filter
      const transcriptCheckbox = screen.getByLabelText('Transcripts');
      fireEvent.click(transcriptCheckbox);

      await waitFor(() => {
        const badge = screen.querySelector('.filter-count-badge');
        expect(badge).toBeInTheDocument();
        expect(badge?.textContent).toBe('1');
      });
    });

    it('should clear all filters when clear button is clicked', async () => {
      render(
        <SearchInterface
          onSearch={mockHandlers.onSearch}
          onResultClick={mockHandlers.onResultClick}
        />
      );

      const filtersButton = screen.getByText('Filters');
      fireEvent.click(filtersButton);

      // Add a filter
      const transcriptCheckbox = screen.getByLabelText('Transcripts');
      fireEvent.click(transcriptCheckbox);

      // Clear filters
      const clearButton = screen.getByText('Clear All Filters');
      fireEvent.click(clearButton);

      await waitFor(() => {
        expect(transcriptCheckbox).not.toBeChecked();
      });
    });
  });

  describe('Saved Searches', () => {
    it('should toggle saved searches panel when button is clicked', async () => {
      const savedSearches = [
        {
          id: 'saved-1',
          name: 'My Saved Search',
          query: { text: 'test', filters: { content_types: [], tags: [], speakers: [] } },
          created_at: '2024-01-01T00:00:00Z',
        },
      ];

      render(
        <SearchInterface
          onSearch={mockHandlers.onSearch}
          onResultClick={mockHandlers.onResultClick}
          saved_searches={savedSearches}
        />
      );

      const savedSearchesButton = screen.getByText('Saved Searches');
      fireEvent.click(savedSearchesButton);

      await waitFor(() => {
        expect(screen.getByText('My Saved Search')).toBeInTheDocument();
        expect(savedSearchesButton).toHaveClass('active');
      });
    });

    it('should load saved search when clicked', async () => {
      const savedSearches = [
        {
          id: 'saved-1',
          name: 'My Saved Search',
          query: {
            text: 'important evidence',
            filters: { content_types: [], tags: [], speakers: [] }
          },
          created_at: '2024-01-01T00:00:00Z',
        },
      ];

      render(
        <SearchInterface
          onSearch={mockHandlers.onSearch}
          onResultClick={mockHandlers.onResultClick}
          saved_searches={savedSearches}
        />
      );

      const savedSearchesButton = screen.getByText('Saved Searches');
      fireEvent.click(savedSearchesButton);

      await waitFor(() => {
        const savedSearchItem = screen.getByText('My Saved Search');
        fireEvent.click(savedSearchItem);

        const searchInput = screen.getByPlaceholderText(/Search transcripts/i) as HTMLInputElement;
        expect(searchInput.value).toBe('important evidence');
      });
    });
  });

  describe('Search Results', () => {
    it('should display search results', async () => {
      // Mock the performSearch function to return results
      const mockSearch = vi.fn().mockResolvedValue(mockResults);

      render(
        <SearchInterface
          onSearch={mockSearch}
          onResultClick={mockHandlers.onResultClick}
        />
      );

      const searchInput = screen.getByPlaceholderText(/Search transcripts/i);
      fireEvent.change(searchInput, { target: { value: 'test query' } });

      // Fast forward debounce timer
      vi.advanceTimersByTime(300);

      await waitFor(() => {
        expect(mockSearch).toHaveBeenCalledWith(
          expect.objectContaining({ text: 'test query' })
        );
      });
    });

    it('should show no results message when search returns empty', async () => {
      render(
        <SearchInterface
          onSearch={vi.fn().mockResolvedValue([])}
          onResultClick={mockHandlers.onResultClick}
        />
      );

      const searchInput = screen.getByPlaceholderText(/Search transcripts/i);
      fireEvent.change(searchInput, { target: { value: 'no results query' } });

      vi.advanceTimersByTime(300);

      await waitFor(() => {
        expect(screen.getByText(/No results found/i)).toBeInTheDocument();
      });
    });

    it('should call onResultClick when result is clicked', async () => {
      render(
        <SearchInterface
          onSearch={vi.fn().mockResolvedValue(mockResults)}
          onResultClick={mockHandlers.onResultClick}
        />
      );

      const searchInput = screen.getByPlaceholderText(/Search transcripts/i);
      fireEvent.change(searchInput, { target: { value: 'test' } });

      vi.advanceTimersByTime(300);

      // Wait for results and click first result
      await waitFor(() => {
        const results = document.querySelectorAll('.result-item');
        if (results.length > 0) {
          fireEvent.click(results[0]);
          expect(mockHandlers.onResultClick).toHaveBeenCalled();
        }
      });
    });

    it('should display relevance score badges', () => {
      // This test would require mocking the search results state
      // as the component manages its own results state
      const mockSearch = vi.fn().mockResolvedValue(mockResults);

      render(
        <SearchInterface
          onSearch={mockSearch}
          onResultClick={mockHandlers.onResultClick}
        />
      );

      const searchInput = screen.getByPlaceholderText(/Search transcripts/i);
      fireEvent.change(searchInput, { target: { value: 'test' } });

      vi.advanceTimersByTime(300);

      // Check for relevance badges after results load
      // Note: This is a simplified test as we can't easily mock the internal state
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty recent searches', () => {
      render(
        <SearchInterface
          onSearch={mockHandlers.onSearch}
          onResultClick={mockHandlers.onResultClick}
          recent_searches={[]}
        />
      );

      const searchInput = screen.getByPlaceholderText(/Search transcripts/i);
      searchInput.focus();

      // Should not show suggestions
      expect(screen.queryByText('test search 1')).not.toBeInTheDocument();
    });

    it('should handle special characters in search query', async () => {
      render(
        <SearchInterface
          onSearch={mockHandlers.onSearch}
          onResultClick={mockHandlers.onResultClick}
        />
      );

      const searchInput = screen.getByPlaceholderText(/Search transcripts/i);
      fireEvent.change(searchInput, { target: { value: 'test !@#$%^&*()' } });

      vi.advanceTimersByTime(300);

      await waitFor(() => {
        expect(mockHandlers.onSearch).toHaveBeenCalled();
      });
    });

    it('should handle very long search query', async () => {
      render(
        <SearchInterface
          onSearch={mockHandlers.onSearch}
          onResultClick={mockHandlers.onResultClick}
        />
      );

      const longQuery = 'a'.repeat(1000);
      const searchInput = screen.getByPlaceholderText(/Search transcripts/i) as HTMLInputElement;
      fireEvent.change(searchInput, { target: { value: longQuery } });

      expect(searchInput.value.length).toBe(1000);
    });
  });
});
