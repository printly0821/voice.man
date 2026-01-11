/**
 * Complaint Viewer Component Types
 *
 * TAG: DESIGN-TAG-001
 * Defines data structures for complaint document viewing
 */

export interface ComplaintDocument {
  id: string;
  case_number: string;
  title: string;
  content: string;
  created_at: string;
  updated_at: string;
  status: 'draft' | 'submitted' | 'review' | 'approved' | 'rejected';
  metadata: ComplaintMetadata;
}

export interface ComplaintMetadata {
  author: string;
  department: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  tags: string[];
  related_cases: string[];
  evidence_count: number;
}

export interface ComplaintViewerProps {
  complaint: ComplaintDocument;
  onEdit?: (id: string) => void;
  onDelete?: (id: string) => void;
  onStatusChange?: (id: string, status: ComplaintDocument['status']) => void;
  readonly?: boolean;
}

export interface ComplaintViewerState {
  isLoading: boolean;
  isEditing: boolean;
  selectedTab: 'content' | 'metadata' | 'history' | 'evidence';
  zoom: number;
}
