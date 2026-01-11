/**
 * Evidence Explorer Component
 *
 * TAG: FUNCTION-TAG-007
 * Browse, filter, and manage evidence files with grid/list views
 */

import React, { useState, useMemo, useCallback } from 'react';
import {
  EvidenceFile,
  EvidenceExplorerProps,
  EvidenceExplorerState,
  EvidenceFilter,
  EvidenceSortOption,
} from './types';
import { EvidenceGrid } from './EvidenceGrid';
import { EvidenceList } from './EvidenceList';
import { EvidenceFilters } from './EvidenceFilters';
import { EvidencePlayer } from './EvidencePlayer';

export const EvidenceExplorer: React.FC<EvidenceExplorerProps> = ({
  evidence,
  onSelectEvidence,
  onDeleteEvidence,
  onTagEvidence,
  readonly = false,
}) => {
  const [state, setState] = useState<EvidenceExplorerState>({
    selected_evidence: null,
    filters: {
      search_query: '',
      file_types: [],
      date_range: { start: null, end: null },
      tags: [],
      min_duration: null,
      max_duration: null,
    },
    sort: { field: 'created_at', direction: 'desc' },
    view_mode: 'grid',
    is_playing: false,
  });

  // Filter and sort evidence
  const filtered_evidence = useMemo(() => {
    let filtered = [...evidence];

    // Apply search filter
    if (state.filters.search_query) {
      const query = state.filters.search_query.toLowerCase();
      filtered = filtered.filter(file =>
        file.filename.toLowerCase().includes(query) ||
        file.tags.some(tag => tag.toLowerCase().includes(query))
      );
    }

    // Apply file type filter
    if (state.filters.file_types.length > 0) {
      filtered = filtered.filter(file =>
        state.filters.file_types.includes(file.file_type)
      );
    }

    // Apply date range filter
    if (state.filters.date_range.start) {
      filtered = filtered.filter(file =>
        new Date(file.created_at) >= new Date(state.filters.date_range.start!)
      );
    }
    if (state.filters.date_range.end) {
      filtered = filtered.filter(file =>
        new Date(file.created_at) <= new Date(state.filters.date_range.end!)
      );
    }

    // Apply tags filter
    if (state.filters.tags.length > 0) {
      filtered = filtered.filter(file =>
        state.filters.tags.some(tag => file.tags.includes(tag))
      );
    }

    // Apply duration filter
    if (state.filters.min_duration !== null) {
      filtered = filtered.filter(file =>
        (file.duration_seconds || 0) >= state.filters.min_duration!
      );
    }
    if (state.filters.max_duration !== null) {
      filtered = filtered.filter(file =>
        (file.duration_seconds || 0) <= state.filters.max_duration!
      );
    }

    // Apply sorting
    filtered.sort((a, b) => {
      const a_value = a[state.sort.field];
      const b_value = b[state.sort.field];

      if (a_value === undefined) return 1;
      if (b_value === undefined) return -1;

      if (state.sort.direction === 'asc') {
        return a_value > b_value ? 1 : -1;
      } else {
        return a_value < b_value ? 1 : -1;
      }
    });

    return filtered;
  }, [evidence, state.filters, state.sort]);

  const handleSelectEvidence = useCallback((id: string) => {
    setState(prev => ({ ...prev, selected_evidence: id }));
    onSelectEvidence(id);
  }, [onSelectEvidence]);

  const handleFilterChange = useCallback((filters: Partial<EvidenceFilter>) => {
    setState(prev => ({
      ...prev,
      filters: { ...prev.filters, ...filters },
    }));
  }, []);

  const handleSortChange = useCallback((sort: EvidenceSortOption) => {
    setState(prev => ({ ...prev, sort }));
  }, []);

  const handleViewModeChange = useCallback((view_mode: 'grid' | 'list') => {
    setState(prev => ({ ...prev, view_mode }));
  }, []);

  const handleDeleteEvidence = useCallback((id: string) => {
    if (onDeleteEvidence && !readonly) {
      if (confirm('정말 이 증거를 삭제하시겠습니까?')) {
        onDeleteEvidence(id);
        if (state.selected_evidence === id) {
          setState(prev => ({ ...prev, selected_evidence: null }));
        }
      }
    }
  }, [onDeleteEvidence, readonly, state.selected_evidence]);

  const selectedEvidenceData = useMemo(() => {
    return evidence.find(file => file.id === state.selected_evidence) || null;
  }, [evidence, state.selected_evidence]);

  return (
    <div className="evidence-explorer">
      <EvidenceFilters
        filters={state.filters}
        sort={state.sort}
        view_mode={state.view_mode}
        onFilterChange={handleFilterChange}
        onSortChange={handleSortChange}
        onViewModeChange={handleViewModeChange}
        readonly={readonly}
      />

      <div className="evidence-explorer-content">
        <div className="evidence-browser">
          {state.view_mode === 'grid' ? (
            <EvidenceGrid
              evidence={filtered_evidence}
              selected_id={state.selected_evidence}
              onSelect={handleSelectEvidence}
              onDelete={handleDeleteEvidence}
              readonly={readonly}
            />
          ) : (
            <EvidenceList
              evidence={filtered_evidence}
              selected_id={state.selected_evidence}
              onSelect={handleSelectEvidence}
              onDelete={handleDeleteEvidence}
              readonly={readonly}
            />
          )}
        </div>

        {selectedEvidenceData && (
          <div className="evidence-detail-panel">
            <EvidencePlayer
              evidence={selectedEvidenceData}
              onClose={() => setState(prev => ({ ...prev, selected_evidence: null }))}
            />
          </div>
        )}
      </div>

      <div className="evidence-explorer-footer">
        <span>Total: {filtered_evidence.length} files</span>
      </div>
    </div>
  );
};

export default EvidenceExplorer;
