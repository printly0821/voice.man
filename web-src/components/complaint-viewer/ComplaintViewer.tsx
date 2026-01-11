/**
 * Complaint Viewer Component
 *
 * TAG: FUNCTION-TAG-001
 * Displays and manages complaint documents with tabs for content, metadata, history, and evidence
 */

import React, { useState, useCallback } from 'react';
import {
  ComplaintDocument,
  ComplaintViewerProps,
  ComplaintViewerState,
} from './types';
import { ComplaintContent } from './ComplaintContent';
import { ComplaintMetadata } from './ComplaintMetadataView';
import { ComplaintHistory } from './ComplaintHistory';
import { ComplaintEvidence } from './ComplaintEvidence';
import { ComplaintToolbar } from './ComplaintToolbar';

export const ComplaintViewer: React.FC<ComplaintViewerProps> = ({
  complaint,
  onEdit,
  onDelete,
  onStatusChange,
  readonly = false,
}) => {
  const [state, setState] = useState<ComplaintViewerState>({
    isLoading: false,
    isEditing: false,
    selectedTab: 'content',
    zoom: 1.0,
  });

  const handleTabChange = useCallback((tab: ComplaintViewerState['selectedTab']) => {
    setState(prev => ({ ...prev, selectedTab: tab }));
  }, []);

  const handleZoomIn = useCallback(() => {
    setState(prev => ({ ...prev, zoom: Math.min(prev.zoom + 0.1, 2.0) }));
  }, []);

  const handleZoomOut = useCallback(() => {
    setState(prev => ({ ...prev, zoom: Math.max(prev.zoom - 0.1, 0.5) }));
  }, []);

  const handleEdit = useCallback(() => {
    if (onEdit && !readonly) {
      onEdit(complaint.id);
      setState(prev => ({ ...prev, isEditing: true }));
    }
  }, [complaint.id, onEdit, readonly]);

  const handleDelete = useCallback(() => {
    if (onDelete && !readonly && confirm('정말 삭제하시겠습니까?')) {
      onDelete(complaint.id);
    }
  }, [complaint.id, onDelete, readonly]);

  const handleStatusChange = useCallback((status: ComplaintDocument['status']) => {
    if (onStatusChange && !readonly) {
      onStatusChange(complaint.id, status);
    }
  }, [complaint.id, onStatusChange, readonly]);

  const renderTabContent = () => {
    const { selectedTab, zoom } = state;

    switch (selectedTab) {
      case 'content':
        return <ComplaintContent complaint={complaint} zoom={zoom} />;
      case 'metadata':
        return <ComplaintMetadata complaint={complaint} />;
      case 'history':
        return <ComplaintHistory complaintId={complaint.id} />;
      case 'evidence':
        return <ComplaintEvidence complaintId={complaint.id} />;
      default:
        return null;
    }
  };

  return (
    <div className="complaint-viewer">
      <ComplaintToolbar
        complaint={complaint}
        selectedTab={state.selectedTab}
        zoom={state.zoom}
        onTabChange={handleTabChange}
        onZoomIn={handleZoomOut}
        onZoomOut={handleZoomIn}
        onEdit={handleEdit}
        onDelete={handleDelete}
        onStatusChange={handleStatusChange}
        readonly={readonly}
      />
      <div className="complaint-viewer-content">
        {renderTabContent()}
      </div>
    </div>
  );
};

export default ComplaintViewer;
