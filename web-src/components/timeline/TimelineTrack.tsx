/**
 * Timeline Track Component
 *
 * TAG: FUNCTION-TAG-013
 * Single track displaying events of a specific type
 */

import React from 'react';
import { TimelineEvent } from './types';

interface TimelineTrackProps {
  event_type: TimelineEvent['event_type'];
  events: TimelineEvent[];
  selected_id?: string;
  time_to_x: (time: number) => number;
  onEventClick: (event: TimelineEvent) => void;
  onEventHover: (event_id: string | null) => void;
}

export const TimelineTrack: React.FC<TimelineTrackProps> = ({
  event_type,
  events,
  selected_id,
  time_to_x,
  onEventClick,
  onEventHover,
}) => {
  const getEventTypeColor = (type: TimelineEvent['event_type']): string => {
    switch (type) {
      case 'audio_segment': return '#3b82f6';
      case 'transcript': return '#10b981';
      case 'gaslighting': return '#ef4444';
      case 'emotion': return '#f59e0b';
      case 'speaker_change': return '#8b5cf6';
      default: return '#6b7280';
    }
  };

  const getEventTypeIcon = (type: TimelineEvent['event_type']): string => {
    switch (type) {
      case 'audio_segment': return '\u266A';
      case 'transcript': return '\u1F4DD';
      case 'gaslighting': return '\u26A0';
      case 'emotion': return '\u1F60E';
      case 'speaker_change': return '\u1F465';
      default: return '\u25CF';
    }
  };

  const getEventLabel = (type: TimelineEvent['event_type']): string => {
    switch (type) {
      case 'audio_segment': return 'Audio';
      case 'transcript': return 'Transcript';
      case 'gaslighting': return 'Gaslighting';
      case 'emotion': return 'Emotion';
      case 'speaker_change': return 'Speaker';
      default: return type;
    }
  };

  const color = getEventTypeColor(event_type);
  const icon = getEventTypeIcon(event_type);
  const label = getEventLabel(event_type);

  // Sort events by timestamp
  const sorted_events = [...events].sort((a, b) => a.timestamp - b.timestamp);

  return (
    <div className="timeline-track">
      <div className="track-header">
        <span className="track-icon" style={{ backgroundColor: color }}>
          {icon}
        </span>
        <span className="track-label">{label}</span>
        <span className="track-count">{events.length}</span>
      </div>

      <div className="track-content">
        {sorted_events.map(event => {
          const left = time_to_x(event.timestamp);
          const is_selected = selected_id === event.id;

          return (
            <div
              key={event.id}
              className={`timeline-event ${is_selected ? 'selected' : ''}`}
              style={{
                left: `${left}%`,
                backgroundColor: event.color || color,
              }}
              onClick={() => onEventClick(event)}
              onMouseEnter={() => onEventHover(event.id)}
              onMouseLeave={() => onEventHover(null)}
              title={`${event.title} at ${Math.floor(event.timestamp)}s`}
            >
              {event.icon && (
                <span className="event-icon">{event.icon}</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default TimelineTrack;
