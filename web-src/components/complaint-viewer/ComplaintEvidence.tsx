/**
 * Complaint Evidence Component
 *
 * TAG: FUNCTION-TAG-005
 * Displays evidence files linked to a complaint document
 */

import React, { useEffect, useState } from 'react';

interface EvidenceItem {
  id: string;
  filename: string;
  type: 'audio' | 'video' | 'image' | 'document';
  size: number;
  uploaded_at: string;
  url: string;
}

interface ComplaintEvidenceProps {
  complaintId: string;
}

export const ComplaintEvidence: React.FC<ComplaintEvidenceProps> = ({
  complaintId,
}) => {
  const [evidence, setEvidence] = useState<EvidenceItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Fetch evidence from API
    fetch(`/api/complaints/${complaintId}/evidence`)
      .then(res => res.json())
      .then(data => {
        setEvidence(data);
        setIsLoading(false);
      })
      .catch(() => {
        setIsLoading(false);
      });
  }, [complaintId]);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  if (isLoading) {
    return <div className="complaint-evidence-loading">Loading evidence...</div>;
  }

  if (evidence.length === 0) {
    return <div className="complaint-evidence-empty">No evidence attached</div>;
  }

  return (
    <div className="complaint-evidence">
      <h3>Evidence ({evidence.length})</h3>
      <div className="evidence-grid">
        {evidence.map(item => (
          <div key={item.id} className="evidence-card">
            <div className={`evidence-icon evidence-${item.type}`}>
              {item.type === 'audio' && '\u266A'}
              {item.type === 'video' && '\u25B6'}
              {item.type === 'image' && '\u1F5BC'}
              {item.type === 'document' && '\u1F4C4'}
            </div>
            <div className="evidence-info">
              <div className="evidence-filename">{item.filename}</div>
              <div className="evidence-meta">
                <span className="evidence-type">{item.type}</span>
                <span className="evidence-size">{formatFileSize(item.size)}</span>
              </div>
              <div className="evidence-date">
                {new Date(item.uploaded_at).toLocaleDateString('ko-KR')}
              </div>
            </div>
            <a href={item.url} className="evidence-download" download>
              Download
            </a>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ComplaintEvidence;
