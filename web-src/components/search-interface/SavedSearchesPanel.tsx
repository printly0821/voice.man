/**
 * Saved Searches Panel Component
 *
 * TAG: FUNCTION-TAG-020
 * Panel for managing saved searches
 */

import React, { useState } from 'react';
import { SavedSearch, SearchQuery } from './types';

interface SavedSearchesPanelProps {
  saved_searches: SavedSearch[];
  onLoadSearch: (search: SavedSearch) => void;
  onSaveSearch: (name: string) => void;
  current_query: SearchQuery;
}

export const SavedSearchesPanel: React.FC<SavedSearchesPanelProps> = ({
  saved_searches,
  onLoadSearch,
  onSaveSearch,
  current_query,
}) => {
  const [is_saving, setIsSaving] = useState(false);
  const [save_name, setSaveName] = useState('');

  const handleSaveClick = () => {
    setIsSaving(true);
  };

  const handleSaveSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (save_name.trim()) {
      onSaveSearch(save_name.trim());
      setSaveName('');
      setIsSaving(false);
    }
  };

  const handleSaveCancel = () => {
    setSaveName('');
    setIsSaving(false);
  };

  const has_query = current_query.text.trim().length > 0 ||
                   current_query.filters.content_types.length > 0 ||
                   current_query.filters.tags.length > 0 ||
                   current_query.filters.speakers.length > 0;

  return (
    <div className="saved-searches-panel">
      <div className="saved-searches-header">
        <h3>Saved Searches</h3>
        {has_query && !is_saving && (
          <button
            className="save-search-button"
            onClick={handleSaveClick}
            aria-label="Save current search"
          >
            Save Current Search
          </button>
        )}
      </div>

      {is_saving && (
        <form className="save-search-form" onSubmit={handleSaveSubmit}>
          <input
            type="text"
            value={save_name}
            onChange={(e) => setSaveName(e.target.value)}
            placeholder="Enter search name..."
            className="save-name-input"
            autoFocus
          />
          <div className="save-form-actions">
            <button
              type="submit"
              className="save-submit-button"
              disabled={!save_name.trim()}
            >
              Save
            </button>
            <button
              type="button"
              className="save-cancel-button"
              onClick={handleSaveCancel}
            >
              Cancel
            </button>
          </div>
        </form>
      )}

      <div className="saved-searches-list">
        {saved_searches.length === 0 ? (
          <p className="no-saved-searches">No saved searches yet.</p>
        ) : (
          saved_searches.map(search => (
            <div
              key={search.id}
              className="saved-search-item"
              role="button"
              tabIndex={0}
              onClick={() => onLoadSearch(search)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onLoadSearch(search);
                }
              }}
              aria-label={`Load saved search: ${search.name}`}
            >
              <div className="saved-search-header">
                <span className="saved-search-name">{search.name}</span>
                <span className="saved-search-date">
                  {new Date(search.created_at).toLocaleDateString('ko-KR')}
                </span>
              </div>
              <div className="saved-search-query">
                <span className="query-text">{search.query.text}</span>
                <span className="query-filters">
                  {search.query.filters.content_types.length > 0 && (
                    <span className="filter-badge">
                      {search.query.filters.content_types.length} types
                    </span>
                  )}
                  {search.query.filters.tags.length > 0 && (
                    <span className="filter-badge">
                      {search.query.filters.tags.length} tags
                    </span>
                  )}
                  {search.query.filters.speakers.length > 0 && (
                    <span className="filter-badge">
                      {search.query.filters.speakers.length} speakers
                    </span>
                  )}
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default SavedSearchesPanel;
