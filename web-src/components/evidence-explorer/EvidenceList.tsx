/**
 * Evidence List Component
 *
 * TAG: FUNCTION-TAG-009
 * List view for evidence files with detailed information
 */

import React from 'react';
import { EvidenceFile } from './types';

interface EvidenceListProps {
  evidence: EvidenceFile[];
  selected_id: string | null;
  onSelect: (id: string) => void;
  onDelete?: (id: string) => void;
  readonly?: boolean;
}

export const EvidenceList: React.FC<EvidenceListProps> = ({
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

  return (
    <div className="evidence-list">
      <table className="evidence-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Type</th>
            <th>Size</th>
            <th>Duration</th>
            <th>Created</th>
            <th>Gaslighting</th>
            <th>Tags</th>
            {!readonly && <th>Actions</th>}
          </tr>
        </thead>
        <tbody>
          {evidence.map(file => (
            <tr
              key={file.id}
              className={`evidence-row ${selected_id === file.id ? 'selected' : ''}`}
              onClick={() => onSelect(file.id)}
            >
              <td className="evidence-name-cell" title={file.filename}>
                {file.filename}
              </td>
              <td className="evidence-type-cell">{file.file_type}</td>
              <td className="evidence-size-cell">{formatFileSize(file.size_bytes)}</td>
              <td className="evidence-duration-cell">
                {file.duration_seconds ? formatDuration(file.duration_seconds) : '-'}
              </td>
              <td className="evidence-date-cell">
                {new Date(file.created_at).toLocaleDateString('ko-KR')}
              </td>
              <td className="evidence-gaslighting-cell">
                {file.forensic_data ? (
                  <span className={`gaslighting-badge ${
                    file.forensic_data.gaslighting_probability > 0.7 ? 'high' :
                    file.forensic_data.gaslighting_probability > 0.4 ? 'medium' : 'low'
                  }`}>
                    {Math.round(file.forensic_data.gaslighting_probability * 100)}%
                  </span>
                ) : '-'}
              </td>
              <td className="evidence-tags-cell">
                {file.tags.length > 0 ? (
                  <div className="tags-inline">
                    {file.tags.slice(0, 2).map((tag, index) => (
                      <span key={index} className="tag-badge-small">{tag}</span>
                    ))}
                    {file.tags.length > 2 && (
                      <span className="tag-badge-small">+{file.tags.length - 2}</span>
                    )}
                  </div>
                ) : '-'}
              </td>
              {!readonly && onDelete && (
                <td className="evidence-actions-cell">
                  <button
                    className="delete-button-small"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(file.id);
                    }}
                    aria-label="Delete evidence"
                  >
                    Delete
                  </button>
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default EvidenceList;
