/**
 * Complaint Metadata View Component
 *
 * TAG: FUNCTION-TAG-003
 * Displays metadata information for a complaint document
 */

import React from 'react';
import { ComplaintDocument } from './types';

interface ComplaintMetadataProps {
  complaint: ComplaintDocument;
}

export const ComplaintMetadataView: React.FC<ComplaintMetadataProps> = ({
  complaint,
}) => {
  const { metadata } = complaint;

  return (
    <div className="complaint-metadata">
      <h3>Metadata</h3>
      <div className="metadata-grid">
        <div className="metadata-item">
          <label>Author</label>
          <span>{metadata.author || 'N/A'}</span>
        </div>
        <div className="metadata-item">
          <label>Department</label>
          <span>{metadata.department || 'N/A'}</span>
        </div>
        <div className="metadata-item">
          <label>Priority</label>
          <span className={`priority-badge priority-${metadata.priority}`}>
            {metadata.priority}
          </span>
        </div>
        <div className="metadata-item">
          <label>Evidence Count</label>
          <span>{metadata.evidence_count}</span>
        </div>
        <div className="metadata-item full-width">
          <label>Tags</label>
          <div className="tags-container">
            {metadata.tags.length > 0 ? (
              metadata.tags.map((tag, index) => (
                <span key={index} className="tag-badge">
                  {tag}
                </span>
              ))
            ) : (
              <span className="no-tags">No tags</span>
            )}
          </div>
        </div>
        {metadata.related_cases.length > 0 && (
          <div className="metadata-item full-width">
            <label>Related Cases</label>
            <ul className="related-cases-list">
              {metadata.related_cases.map((caseId, index) => (
                <li key={index}>{caseId}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default ComplaintMetadataView;
