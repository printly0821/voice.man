/**
 * Complaint History Component
 *
 * TAG: FUNCTION-TAG-004
 * Displays revision history for a complaint document
 */

import React, { useEffect, useState } from 'react';

interface HistoryEntry {
  id: string;
  timestamp: string;
  user: string;
  action: string;
  changes: string[];
}

interface ComplaintHistoryProps {
  complaintId: string;
}

export const ComplaintHistory: React.FC<ComplaintHistoryProps> = ({
  complaintId,
}) => {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Fetch history from API
    fetch(`/api/complaints/${complaintId}/history`)
      .then(res => res.json())
      .then(data => {
        setHistory(data);
        setIsLoading(false);
      })
      .catch(() => {
        setIsLoading(false);
      });
  }, [complaintId]);

  if (isLoading) {
    return <div className="complaint-history-loading">Loading history...</div>;
  }

  if (history.length === 0) {
    return <div className="complaint-history-empty">No history available</div>;
  }

  return (
    <div className="complaint-history">
      <h3>Revision History</h3>
      <div className="history-timeline">
        {history.map((entry, index) => (
          <div key={entry.id} className="history-entry">
            <div className="history-marker">{index + 1}</div>
            <div className="history-content">
              <div className="history-header">
                <span className="history-user">{entry.user}</span>
                <span className="history-timestamp">
                  {new Date(entry.timestamp).toLocaleString('ko-KR')}
                </span>
              </div>
              <div className="history-action">{entry.action}</div>
              {entry.changes.length > 0 && (
                <ul className="history-changes">
                  {entry.changes.map((change, changeIndex) => (
                    <li key={changeIndex}>{change}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ComplaintHistory;
