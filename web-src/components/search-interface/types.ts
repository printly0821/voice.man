/**
 * Search Interface Component Types
 *
 * TAG: DESIGN-TAG-004
 * Defines data structures for unified search across evidence and transcripts
 */

export interface SearchQuery {
  text: string;
  filters: SearchFilters;
}

export interface SearchFilters {
  content_types: ContentType[];
  date_range?: DateRange;
  tags: string[];
  min_gaslighting_probability?: number;
  speakers: string[];
  emotion_types?: string[];
}

export type ContentType = 'transcript' | 'evidence_metadata' | 'forensic_results';

export interface DateRange {
  start: string | null;
  end: string | null;
}

export interface SearchResult {
  id: string;
  content_type: ContentType;
  title: string;
  excerpt: string;
  highlight_ranges: HighlightRange[];
  relevance_score: number;
  metadata: SearchResultMetadata;
}

export interface HighlightRange {
  start: number;
  end: number;
  type: 'exact' | 'fuzzy' | 'semantic';
}

export interface SearchResultMetadata {
  file_id?: string;
  timestamp?: number;
  speaker?: string;
  gaslighting_probability?: number;
  emotion?: string;
  tags: string[];
}

export interface SearchInterfaceProps {
  onSearch: (query: SearchQuery) => void;
  onResultClick: (result: SearchResult) => void;
  recent_searches?: string[];
  saved_searches?: SavedSearch[];
  max_results?: number;
}

export interface SavedSearch {
  id: string;
  name: string;
  query: SearchQuery;
  created_at: string;
}

export interface SearchInterfaceState {
  query_text: string;
  filters: SearchFilters;
  is_searching: boolean;
  results: SearchResult[];
  selected_result_id: string | null;
  show_filters: boolean;
  show_saved_searches: boolean;
}

export interface SearchSuggestion {
  text: string;
  type: 'history' | 'semantic' | 'correction';
  frequency?: number;
}
