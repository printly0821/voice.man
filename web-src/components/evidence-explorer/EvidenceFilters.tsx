/**
 * Evidence Filters Component
 *
 * TAG: FUNCTION-TAG-010
 * Filter panel for evidence search and filtering
 */

import React from 'react';
import { EvidenceFilter, EvidenceSortOption } from './types';

interface EvidenceFiltersProps {
  filters: EvidenceFilter;
  sort: EvidenceSortOption;
  view_mode: 'grid' | 'list';
  onFilterChange: (filters: Partial<EvidenceFilter>) => void;
  onSortChange: (sort: EvidenceSortOption) => void;
  onViewModeChange: (mode: 'grid' | 'list') => void;
  readonly?: boolean;
}

export const EvidenceFilters: React.FC<EvidenceFiltersProps> = ({
  filters,
  sort,
  view_mode,
  onFilterChange,
  onSortChange,
  onViewModeChange,
  readonly,
}) => {
  const file_types = [
    { value: 'audio' as const, label: 'Audio', icon: '\u266A' },
    { value: 'video' as const, label: 'Video', icon: '\u25B6' },
    { value: 'image' as const, label: 'Image', icon: '\u1F5BC' },
    { value: 'document' as const, label: 'Document', icon: '\u1F4C4' },
  ];

  const sort_options = [
    { value: 'created_at' as const, label: 'Date Created' },
    { value: 'filename' as const, label: 'File Name' },
    { value: 'size_bytes' as const, label: 'File Size' },
    { value: 'duration_seconds' as const, label: 'Duration' },
  ];

  const handleFileTypeToggle = (type: EvidenceFilter['file_types'][0]) => {
    const new_types = filters.file_types.includes(type)
      ? filters.file_types.filter(t => t !== type)
      : [...filters.file_types, type];
    onFilterChange({ file_types: new_types });
  };

  const clearFilters = () => {
    onFilterChange({
      search_query: '',
      file_types: [],
      date_range: { start: null, end: null },
      tags: [],
      min_duration: null,
      max_duration: null,
    });
  };

  const hasActiveFilters = () => {
    return filters.search_query !== '' ||
           filters.file_types.length > 0 ||
           filters.date_range.start !== null ||
           filters.date_range.end !== null ||
           filters.tags.length > 0 ||
           filters.min_duration !== null ||
           filters.max_duration !== null;
  };

  return (
    <div className="evidence-filters">
      <div className="filters-row">
        <div className="search-box">
          <input
            type="text"
            placeholder="Search evidence..."
            value={filters.search_query}
            onChange={(e) => onFilterChange({ search_query: e.target.value })}
            className="search-input"
          />
        </div>

        <div className="filter-group file-type-filters">
          {file_types.map(type => (
            <button
              key={type.value}
              className={`filter-type-button ${
                filters.file_types.includes(type.value) ? 'active' : ''
              }`}
              onClick={() => handleFileTypeToggle(type.value)}
              title={type.label}
            >
              <span>{type.icon}</span>
              <span>{type.label}</span>
            </button>
          ))}
        </div>

        <div className="sort-controls">
          <select
            value={`${sort.field}-${sort.direction}`}
            onChange={(e) => {
              const [field, direction] = e.target.value.split('-') as [
                EvidenceSortOption['field'],
                EvidenceSortOption['direction']
              ];
              onSortChange({ field, direction });
            }}
            className="sort-select"
          >
            {sort_options.map(option => (
              <option key={`asc-${option.value}`} value={`${option.value}-asc`}>
                {option.label} (Asc)
              </option>
              <option key={`desc-${option.value}`} value={`${option.value}-desc`}>
                {option.label} (Desc)
              </option>
            ))}
          </select>
        </div>

        <div className="view-mode-controls">
          <button
            className={`view-mode-button ${view_mode === 'grid' ? 'active' : ''}`}
            onClick={() => onViewModeChange('grid')}
            title="Grid view"
          >
            \u25A3
          </button>
          <button
            className={`view-mode-button ${view_mode === 'list' ? 'active' : ''}`}
            onClick={() => onViewModeChange('list')}
            title="List view"
          >
            \u2630
          </button>
        </div>

        {hasActiveFilters() && (
          <button
            className="clear-filters-button"
            onClick={clearFilters}
          >
            Clear Filters
          </button>
        )}
      </div>

      <div className="filters-row advanced-filters">
        <div className="date-range-filter">
          <label>Date Range:</label>
          <input
            type="date"
            value={filters.date_range.start || ''}
            onChange={(e) => onFilterChange({
              date_range: { ...filters.date_range, start: e.target.value || null }
            })}
            className="date-input"
          />
          <span>to</span>
          <input
            type="date"
            value={filters.date_range.end || ''}
            onChange={(e) => onFilterChange({
              date_range: { ...filters.date_range, end: e.target.value || null }
            })}
            className="date-input"
          />
        </div>

        <div className="duration-filter">
          <label>Duration (sec):</label>
          <input
            type="number"
            placeholder="Min"
            value={filters.min_duration || ''}
            onChange={(e) => onFilterChange({
              min_duration: e.target.value ? parseInt(e.target.value) : null
            })}
            className="duration-input"
            min="0"
          />
          <span>to</span>
          <input
            type="number"
            placeholder="Max"
            value={filters.max_duration || ''}
            onChange={(e) => onFilterChange({
              max_duration: e.target.value ? parseInt(e.target.value) : null
            })}
            className="duration-input"
            min="0"
          />
        </div>
      </div>
    </div>
  );
};

export default EvidenceFilters;
