/**
 * Complaint Content Component
 *
 * TAG: FUNCTION-TAG-002
 * Displays the main content of a complaint document with zoom support
 */

import React from 'react';
import { ComplaintDocument } from './types';

interface ComplaintContentProps {
  complaint: ComplaintDocument;
  zoom: number;
}

export const ComplaintContent: React.FC<ComplaintContentProps> = ({
  complaint,
  zoom,
}) => {
  if (!complaint.content) {
    return (
      <div className="complaint-content-empty">
        <p>No content available</p>
      </div>
    );
  }

  return (
    <div
      className="complaint-content"
      style={{ fontSize: `${16 * zoom}px` }}
    >
      <div className="complaint-content-header">
        <h2>{complaint.title}</h2>
        <div className="complaint-meta">
          <span className="case-number">{complaint.case_number}</span>
          <span className="status-badge">{complaint.status}</span>
        </div>
      </div>
      <div className="complaint-content-body">
        {complaint.content.split('\n').map((paragraph, index) => (
          <p key={index}>{paragraph || '\u00A0'}</p>
        ))}
      </div>
      <div className="complaint-content-footer">
        <span className="created-at">
          Created: {new Date(complaint.created_at).toLocaleString('ko-KR')}
        </span>
        <span className="updated-at">
          Updated: {new Date(complaint.updated_at).toLocaleString('ko-KR')}
        </span>
      </div>
    </div>
  );
};

export default ComplaintContent;
