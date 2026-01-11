/**
 * Search Results Component
 *
 * TAG: FUNCTION-TAG-019
 * Display search results with highlights and metadata
 */

import React from 'react';
import { SearchResult } from './types';

interface SearchResultsProps {
  results: SearchResult[];
  selected_id: string | null;
  is_loading: boolean;
  onResultClick: (result: SearchResult) => void;
}

export const SearchResults: React.FC<SearchResultsProps> = ({
  results,
  selected_id,
  is_loading,
  onResultClick,
}) => {
  const getContentTypeColor = (type: SearchResult['content_type']): string => {
    switch (type) {
      case 'transcript': return '#10b981';
      case 'evidence_metadata': return '#3b82f6';
      case 'forensic_results': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const getContentTypeLabel = (type: SearchResult['content_type']): string => {
    switch (type) {
      case 'transcript': return 'Transcript';
      case 'evidence_metadata': return 'Evidence';
      case 'forensic_results': return 'Forensic';
      default: return type;
    }
  };

  const renderHighlightedExcerpt = (excerpt: string, highlights: SearchResult['highlight_ranges']) => {
    if (!highlights || highlights.length === 0) {
      return <span>{excerpt}</span>;
    }

    const parts: React.ReactNode[] = [];
    let last_end = 0;

    highlights.forEach((range, index) => {
      // Add text before highlight
      if (range.start > last_end) {
        parts.push(
          <span key={`before-${index}`}>
            {excerpt.substring(last_end, range.start)}
          </span>
        );
      }

      // Add highlighted text
      const highlight_class = range.type === 'exact' ? 'highlight-exact' :
                             range.type === 'fuzzy' ? 'highlight-fuzzy' :
                             'highlight-semantic';
      parts.push(
        <mark key={`highlight-${index}`} className={highlight_class}>
          {excerpt.substring(range.start, range.end)}
        </mark>
      );

      last_end = range.end;
    });

    // Add remaining text
    if (last_end < excerpt.length) {
      parts.push(
        <span key="after">
          {excerpt.substring(last_end)}
        </span>
      );
    }

    return <>{parts}</>;
  };

  const getRelevanceBadgeColor = (score: number): string => {
    if (score >= 0.8) return 'bg-green-500';
    if (score >= 0.5) return 'bg-yellow-500';
    return 'bg-gray-500';
  };

  if (is_loading) {
    return (
      <div className="search-results-loading">
        <div className="loading-spinner" aria-label="Loading results...">
          \u21BB
        </div>
        <p>Searching...</p>
      </div>
    );
  }

  if (results.length === 0) {
    return null;
  }

  return (
    <div className="search-results">
      <div className="results-header">
        <span className="results-count">{results.length} results found</span>
      </div>

      <div className="results-list">
        {results.map((result, index) => (
          <div
            key={result.id}
            className={`result-item ${selected_id === result.id ? 'selected' : ''}`}
            onClick={() => onResultClick(result)}
            role="button"
            tabIndex={0}
            aria-label={`${result.title}, relevance: ${Math.round(result.relevance_score * 100)}%`}
          >
            <div className="result-header">
              <div className="result-type-badge" style={{
                backgroundColor: getContentTypeColor(result.content_type),
              }}>
                {getContentTypeLabel(result.content_type)}
              </div>
              <div
                className={`relevance-badge ${getRelevanceBadgeColor(result.relevance_score)}`}
                title={`Relevance: ${Math.round(result.relevance_score * 100)}%`}
              >
                {Math.round(result.relevance_score * 100)}%
              </div>
              {result.metadata.timestamp && (
                <span className="result-timestamp">
                  {Math.floor(result.metadata.timestamp)}s
                </span>
              )}
            </div>

            <h4 className="result-title">{result.title}</h4>

            <div className="result-excerpt">
              {renderHighlightedExcerpt(result.excerpt, result.highlight_ranges)}
            </div>

            <div className="result-metadata">
              {result.metadata.speaker && (
                <span className="metadata-speaker">
                  Speaker: {result.metadata.speaker}
                </span>
              )}
              {result.metadata.gaslighting_probability !== undefined && (
                <span className={`metadata-gaslighting ${
                  result.metadata.gaslighting_probability > 0.7 ? 'high' :
                  result.metadata.gaslighting_probability > 0.4 ? 'medium' : 'low'
                }`}>
                  Gaslighting: {Math.round(result.metadata.gaslighting_probability * 100)}%
                </span>
              )}
              {result.metadata.emotion && (
                <span className="metadata-emotion">
                  Emotion: {result.metadata.emotion}
                </span>
              )}
              {result.metadata.tags.length > 0 && (
                <div className="metadata-tags">
                  {result.metadata.tags.slice(0, 3).map((tag, tagIndex) => (
                    <span key={tagIndex} className="tag-badge-small">{tag}</span>
                  ))}
                  {result.metadata.tags.length > 3 && (
                    <span className="tag-badge-small">+{result.metadata.tags.length - 3}</span>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SearchResults;
