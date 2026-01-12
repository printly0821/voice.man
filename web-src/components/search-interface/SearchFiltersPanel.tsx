/**
 * Search Filters Panel Component
 *
 * TAG: FUNCTION-TAG-018
 * Collapsible panel with search filter controls
 */

import React from 'react';
import { SearchFilters, ContentType } from './types';

interface SearchFiltersPanelProps {
  filters: SearchFilters;
  onFilterChange: (filters: Partial<SearchFilters>) => void;
  onClear: () => void;
}

const CONTENT_TYPES: { value: ContentType; label: string; icon: string }[] = [
  { value: 'transcript', label: 'Transcripts', icon: '\u1F4DD' },
  { value: 'evidence_metadata', label: 'Evidence Metadata', icon: '\u1F4CE' },
  { value: 'forensic_results', label: 'Forensic Results', icon: '\u26A0' },
];

const EMOTION_TYPES = [
  'anger', 'sadness', 'fear', 'joy', 'disgust', 'surprise', 'neutral'
];

export const SearchFiltersPanel: React.FC<SearchFiltersPanelProps> = ({
  filters,
  onFilterChange,
  onClear,
}) => {
  const handleContentTypeToggle = (type: ContentType) => {
    const new_types = filters.content_types.includes(type)
      ? filters.content_types.filter(t => t !== type)
      : [...filters.content_types, type];
    onFilterChange({ content_types: new_types });
  };

  const handleTagAdd = (tag: string) => {
    if (!filters.tags.includes(tag)) {
      onFilterChange({ tags: [...filters.tags, tag] });
    }
  };

  const handleTagRemove = (tag: string) => {
    onFilterChange({
      tags: filters.tags.filter(t => t !== tag),
    });
  };

  const handleSpeakerAdd = (speaker: string) => {
    if (!filters.speakers.includes(speaker)) {
      onFilterChange({ speakers: [...filters.speakers, speaker] });
    }
  };

  const handleSpeakerRemove = (speaker: string) => {
    onFilterChange({
      speakers: filters.speakers.filter(s => s !== speaker),
    });
  };

  const handleEmotionToggle = (emotion: string) => {
    const current_emotions = filters.emotion_types || [];
    const new_emotions = current_emotions.includes(emotion)
      ? current_emotions.filter(e => e !== emotion)
      : [...current_emotions, emotion];
    onFilterChange({ emotion_types: new_emotions });
  };

  return (
    <div className="search-filters-panel">
      <div className="filters-section">
        <h3>Content Types</h3>
        <div className="filter-options">
          {CONTENT_TYPES.map(type => (
            <label key={type.value} className="filter-checkbox">
              <input
                type="checkbox"
                checked={filters.content_types.includes(type.value)}
                onChange={() => handleContentTypeToggle(type.value)}
              />
              <span className="checkbox-icon">{type.icon}</span>
              <span>{type.label}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="filters-section">
        <h3>Date Range</h3>
        <div className="date-range-inputs">
          <input
            type="date"
            value={filters.date_range?.start || ''}
            onChange={(e) => onFilterChange({
              date_range: { ...filters.date_range, start: e.target.value || null }
            })}
            className="date-input"
            placeholder="Start date"
          />
          <span>to</span>
          <input
            type="date"
            value={filters.date_range?.end || ''}
            onChange={(e) => onFilterChange({
              date_range: { ...filters.date_range, end: e.target.value || null }
            })}
            className="date-input"
            placeholder="End date"
          />
        </div>
      </div>

      <div className="filters-section">
        <h3>Tags</h3>
        <div className="tags-input-container">
          <div className="selected-tags">
            {filters.tags.map(tag => (
              <span key={tag} className="selected-tag">
                {tag}
                <button
                  className="remove-tag-button"
                  onClick={() => handleTagRemove(tag)}
                  aria-label={`Remove ${tag}`}
                >
                  \u00D7
                </button>
              </span>
            ))}
          </div>
          <input
            type="text"
            className="tag-input"
            placeholder="Add tag and press Enter"
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                const target = e.target as HTMLInputElement;
                if (target.value.trim()) {
                  handleTagAdd(target.value.trim());
                  target.value = '';
                }
              }
            }}
          />
        </div>
      </div>

      <div className="filters-section">
        <h3>Speakers</h3>
        <div className="speakers-input-container">
          <div className="selected-speakers">
            {filters.speakers.map(speaker => (
              <span key={speaker} className="selected-speaker">
                {speaker}
                <button
                  className="remove-speaker-button"
                  onClick={() => handleSpeakerRemove(speaker)}
                  aria-label={`Remove ${speaker}`}
                >
                  \u00D7
                </button>
              </span>
            ))}
          </div>
          <input
            type="text"
            className="speaker-input"
            placeholder="Add speaker and press Enter"
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                const target = e.target as HTMLInputElement;
                if (target.value.trim()) {
                  handleSpeakerAdd(target.value.trim());
                  target.value = '';
                }
              }
            }}
          />
        </div>
      </div>

      <div className="filters-section">
        <h3>Gaslighting Probability</h3>
        <div className="probability-slider-container">
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={filters.min_gaslighting_probability ?? 0}
            onChange={(e) => onFilterChange({
              min_gaslighting_probability: parseFloat(e.target.value),
            })}
            className="probability-slider"
          />
          <span className="probability-value">
            {filters.min_gaslighting_probability
              ? `${Math.round(filters.min_gaslighting_probability * 100)}%+`
              : 'Any'}
          </span>
        </div>
      </div>

      <div className="filters-section">
        <h3>Emotions</h3>
        <div className="emotion-options">
          {EMOTION_TYPES.map(emotion => (
            <label key={emotion} className="filter-checkbox">
              <input
                type="checkbox"
                checked={filters.emotion_types?.includes(emotion) || false}
                onChange={() => handleEmotionToggle(emotion)}
              />
              <span className="emotion-label">{emotion}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="filters-actions">
        <button
          className="clear-filters-button"
          onClick={onClear}
          aria-label="Clear all filters"
        >
          Clear All Filters
        </button>
      </div>
    </div>
  );
};

export default SearchFiltersPanel;
