/**
 * Complaint Toolbar Component
 *
 * TAG: FUNCTION-TAG-006
 * Toolbar for complaint viewer with tab navigation and action buttons
 */

import React from 'react';
import { ComplaintDocument } from './types';

interface ComplaintToolbarProps {
  complaint: ComplaintDocument;
  selectedTab: 'content' | 'metadata' | 'history' | 'evidence';
  zoom: number;
  onTabChange: (tab: 'content' | 'metadata' | 'history' | 'evidence') => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onEdit: () => void;
  onDelete: () => void;
  onStatusChange: (status: ComplaintDocument['status']) => void;
  readonly: boolean;
}

export const ComplaintToolbar: React.FC<ComplaintToolbarProps> = ({
  complaint,
  selectedTab,
  zoom,
  onTabChange,
  onZoomIn,
  onZoomOut,
  onEdit,
  onDelete,
  onStatusChange,
  readonly,
}) => {
  const tabs = [
    { id: 'content' as const, label: 'Content', icon: '\u1F4D6' },
    { id: 'metadata' as const, label: 'Metadata', icon: '\u2139' },
    { id: 'history' as const, label: 'History', icon: '\u1F552' },
    { id: 'evidence' as const, label: 'Evidence', icon: '\u1F4CE' },
  ];

  return (
    <div className="complaint-toolbar">
      <div className="toolbar-tabs">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`tab-button ${selectedTab === tab.id ? 'active' : ''}`}
            onClick={() => onTabChange(tab.id)}
            role="tab"
            aria-selected={selectedTab === tab.id}
          >
            <span className="tab-icon">{tab.icon}</span>
            <span className="tab-label">{tab.label}</span>
          </button>
        ))}
      </div>

      <div className="toolbar-actions">
        <div className="zoom-controls">
          <button
            className="zoom-button"
            onClick={onZoomOut}
            disabled={zoom <= 0.5}
            aria-label="Zoom out"
          >
            -
          </button>
          <span className="zoom-level">{Math.round(zoom * 100)}%</span>
          <button
            className="zoom-button"
            onClick={onZoomIn}
            disabled={zoom >= 2.0}
            aria-label="Zoom in"
          >
            +
          </button>
        </div>

        {!readonly && (
          <>
            <button
              className="action-button edit-button"
              onClick={onEdit}
              aria-label="Edit complaint"
            >
              Edit
            </button>
            <button
              className="action-button delete-button"
              onClick={onDelete}
              aria-label="Delete complaint"
            >
              Delete
            </button>
            <select
              className="status-select"
              value={complaint.status}
              onChange={(e) => onStatusChange(e.target.value as ComplaintDocument['status'])}
              aria-label="Change status"
            >
              <option value="draft">Draft</option>
              <option value="submitted">Submitted</option>
              <option value="review">Review</option>
              <option value="approved">Approved</option>
              <option value="rejected">Rejected</option>
            </select>
          </>
        )}
      </div>
    </div>
  );
};

export default ComplaintToolbar;
