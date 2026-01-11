/**
 * Evidence Grid Component
 *
 * TAG: FUNCTION-TAG-008
 * Grid view for evidence files with thumbnails
 */

import React from 'react';
import { EvidenceFile } from './types';

interface EvidenceGridProps {
  evidence: EvidenceFile[];
  selected_id: string | null;
  onSelect: (id: string) => void;
  onDelete?: (id: string) => void;
  readonly?: boolean;
}

export const EvidenceGrid: React.FC<EvidenceGridProps> = ({
  evidence,
  selected_id,
  onSelect,
  onDelete,
  readonly,
}) => {
  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getIconForType = (type: EvidenceFile['file_type']): string => {
    switch (type) {
      case 'audio': return '\u266A';
      case 'video': return '\u25B6';
      case 'image': return '\u1F5BC';
      case 'document': return '\u1F4C4';
      default: return '\u1F4CE';
    }
  };

  return (
    <div className="evidence-grid">
      {evidence.map(file => (
        <div
          key={file.id}
          className={`evidence-card ${selected_id === file.id ? 'selected' : ''}`}
          onClick={() => onSelect(file.id)}
        >
          <div className={`evidence-thumbnail thumbnail-${file.file_type}`}>
            <span className="thumbnail-icon">{getIconForType(file.file_type)}</span>
          </div>
          <div className="evidence-card-info">
            <div className="evidence-filename" title={file.filename}>
              {file.filename}
            </div>
            <div className="evidence-meta">
              <span className="evidence-type">{file.file_type}</span>
              <span className="evidence-size">{formatFileSize(file.size_bytes)}</span>
              {file.duration_seconds && (
                <span className="evidence-duration">
                  {formatDuration(file.duration_seconds)}
                </span>
              )}
            </div>
            {file.forensic_data && (
              <div className="evidence-forensic-badge">
                Gaslighting: {Math.round(file.forensic_data.gaslighting_probability * 100)}%
              </div>
            )}
            {file.tags.length > 0 && (
              <div className="evidence-tags">
                {file.tags.slice(0, 2).map((tag, index) => (
                  <span key={index} className="tag-badge">{tag}</span>
                ))}
                {file.tags.length > 2 && (
                  <span className="tag-badge">+{file.tags.length - 2}</span>
                )}
              </div>
            )}
          </div>
          {!readonly && onDelete && (
            <button
              className="evidence-delete-button"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(file.id);
              }}
              aria-label="Delete evidence"
            >
              \u00D7
            </button>
          )}
        </div>
      ))}
    </div>
  );
};

export default EvidenceGrid;
