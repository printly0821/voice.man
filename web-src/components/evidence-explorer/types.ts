/**
 * Evidence Explorer Component Types
 *
 * TAG: DESIGN-TAG-002
 * Defines data structures for evidence browsing and filtering
 */

export interface EvidenceFile {
  id: string;
  filename: string;
  filepath: string;
  file_type: 'audio' | 'video' | 'image' | 'document';
  size_bytes: number;
  duration_seconds?: number;
  created_at: string;
  transcript?: string;
  forensic_data?: ForensicData;
  tags: string[];
}

export interface ForensicData {
  gaslighting_probability: number;
  emotion_events: EmotionEvent[];
  speakers: SpeakerInfo[];
}

export interface EmotionEvent {
  timestamp: number;
  emotion: string;
  intensity: number;
  speaker?: string;
}

export interface SpeakerInfo {
  id: string;
  label: string;
  segments_count: number;
  total_duration: number;
}

export interface EvidenceExplorerProps {
  evidence: EvidenceFile[];
  onSelectEvidence: (id: string) => void;
  onDeleteEvidence?: (id: string) => void;
  onTagEvidence?: (id: string, tags: string[]) => void;
  readonly?: boolean;
}

export interface EvidenceFilter {
  search_query: string;
  file_types: EvidenceFile['file_type'][];
  date_range: {
    start: string | null;
    end: string | null;
  };
  tags: string[];
  min_duration: number | null;
  max_duration: number | null;
}

export interface EvidenceSortOption {
  field: 'filename' | 'created_at' | 'size_bytes' | 'duration_seconds';
  direction: 'asc' | 'desc';
}

export interface EvidenceExplorerState {
  selected_evidence: string | null;
  filters: EvidenceFilter;
  sort: EvidenceSortOption;
  view_mode: 'grid' | 'list';
  is_playing: boolean;
}
