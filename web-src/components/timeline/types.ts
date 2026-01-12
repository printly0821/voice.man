/**
 * Timeline Visualization Component Types
 *
 * TAG: DESIGN-TAG-003
 * Defines data structures for timeline event visualization
 */

export interface TimelineEvent {
  id: string;
  timestamp: number;
  event_type: 'audio_segment' | 'transcript' | 'gaslighting' | 'emotion' | 'speaker_change';
  title: string;
  description?: string;
  metadata: Record<string, unknown>;
  color?: string;
  icon?: string;
}

export interface TimelineRange {
  start: number;
  end: number;
}

export interface TimelineViewProps {
  events: TimelineEvent[];
  duration: number;
  onEventClick?: (event: TimelineEvent) => void;
  onRangeSelect?: (range: TimelineRange) => void;
  selected_event_id?: string;
  filter_types?: TimelineEvent['event_type'][];
}

export interface TimelineState {
  view_range: TimelineRange;
  zoom_level: number;
  hover_event_id: string | null;
  is_dragging: boolean;
  drag_start_x: number;
}

export interface TimelineMarker {
  timestamp: number;
  label: string;
  type: 'chapter' | 'bookmark' | 'annotation';
}

export interface TimelineExportFormat {
  format: 'json' | 'csv' | 'pdf';
  include_transcripts: boolean;
  include_forensics: boolean;
}
