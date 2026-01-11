/**
 * Timeline Event Popup Component
 *
 * TAG: FUNCTION-TAG-015
 * Popup displaying details for hovered timeline events
 */

import React, { useEffect, useRef } from 'react';
import { TimelineEvent } from './types';

interface TimelineEventPopupProps {
  event: TimelineEvent;
  formatTime: (seconds: number) => string;
}

export const TimelineEventPopup: React.FC<TimelineEventPopupProps> = ({
  event,
  formatTime,
}) => {
  const popup_ref = useRef<HTMLDivElement>(null);

  // Position popup near cursor or element
  useEffect(() => {
    if (popup_ref.current) {
      const rect = popup_ref.current.getBoundingClientRect();
      // Ensure popup stays within viewport
      if (rect.right > window.innerWidth) {
        popup_ref.current.style.left = `${window.innerWidth - rect.width - 10}px`;
      }
      if (rect.bottom > window.innerHeight) {
        popup_ref.current.style.top = `${window.innerHeight - rect.height - 10}px`;
      }
    }
  }, []);

  const getEventTypeLabel = (type: TimelineEvent['event_type']): string => {
    switch (type) {
      case 'audio_segment': return 'Audio Segment';
      case 'transcript': return 'Transcript';
      case 'gaslighting': return 'Gaslighting Detection';
      case 'emotion': return 'Emotion Event';
      case 'speaker_change': return 'Speaker Change';
      default: return type;
    }
  };

  return (
    <div
      ref={popup_ref}
      className="timeline-event-popup"
      style={{
        backgroundColor: event.color || '#6b7280',
      }}
    >
      <div className="popup-header">
        <span className="popup-type">{getEventTypeLabel(event.event_type)}</span>
        <span className="popup-time">{formatTime(event.timestamp)}</span>
      </div>
      <div className="popup-title">{event.title}</div>
      {event.description && (
        <div className="popup-description">{event.description}</div>
      )}
      {Object.keys(event.metadata).length > 0 && (
        <div className="popup-metadata">
          {Object.entries(event.metadata).map(([key, value]) => (
            <div key={key} className="metadata-item">
              <span className="metadata-key">{key}:</span>
              <span className="metadata-value">
                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default TimelineEventPopup;
