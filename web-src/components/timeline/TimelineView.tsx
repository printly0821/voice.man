/**
 * Timeline View Component
 *
 * TAG: FUNCTION-TAG-012
 * Interactive timeline visualization for audio evidence events
 */

import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import {
  TimelineEvent,
  TimelineViewProps,
  TimelineState,
  TimelineRange,
} from './types';
import { TimelineTrack } from './TimelineTrack';
import { TimelineControls } from './TimelineControls';
import { TimelineEventPopup } from './TimelineEventPopup';

export const TimelineView: React.FC<TimelineViewProps> = ({
  events,
  duration,
  onEventClick,
  onRangeSelect,
  selected_event_id,
  filter_types = [],
}) => {
  const [state, setState] = useState<TimelineState>({
    view_range: { start: 0, end: duration },
    zoom_level: 1.0,
    hover_event_id: null,
    is_dragging: false,
    drag_start_x: 0,
  });

  const timeline_ref = useRef<HTMLDivElement>(null);
  const selection_start = useRef<number | null>(null);

  // Filter events by type
  const filtered_events = useMemo(() => {
    if (filter_types.length === 0) return events;
    return events.filter(event => filter_types.includes(event.event_type));
  }, [events, filter_types]);

  // Group events by type for tracks
  const events_by_type = useMemo(() => {
    const grouped: Record<TimelineEvent['event_type'], TimelineEvent[]> = {
      audio_segment: [],
      transcript: [],
      gaslighting: [],
      emotion: [],
      speaker_change: [],
    };

    filtered_events.forEach(event => {
      grouped[event.event_type].push(event);
    });

    return grouped;
  }, [filtered_events]);

  // Format time as MM:SS
  const formatTime = useCallback((seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }, []);

  // Convert timestamp to x position
  const timeToX = useCallback((time: number): number => {
    const { start, end } = state.view_range;
    const range = end - start;
    if (range === 0) return 0;
    return ((time - start) / range) * 100;
  }, [state.view_range]);

  // Convert x position to timestamp
  const xToTime = useCallback((x: number, element_width: number): number => {
    const { start, end } = state.view_range;
    const range = end - start;
    const percent = x / element_width;
    return start + (range * percent);
  }, [state.view_range]);

  // Handle zoom in
  const handleZoomIn = useCallback(() => {
    setState(prev => {
      const new_zoom = Math.min(prev.zoom_level * 1.5, 10);
      const center = (prev.view_range.start + prev.view_range.end) / 2;
      const new_range = (prev.view_range.end - prev.view_range.start) / 1.5;
      return {
        ...prev,
        zoom_level: new_zoom,
        view_range: {
          start: Math.max(0, center - new_range / 2),
          end: Math.min(duration, center + new_range / 2),
        },
      };
    });
  }, [duration]);

  // Handle zoom out
  const handleZoomOut = useCallback(() => {
    setState(prev => {
      const new_zoom = Math.max(prev.zoom_level / 1.5, 0.1);
      const center = (prev.view_range.start + prev.view_range.end) / 2;
      const new_range = (prev.view_range.end - prev.view_range.start) * 1.5;
      return {
        ...prev,
        zoom_level: new_zoom,
        view_range: {
          start: Math.max(0, center - new_range / 2),
          end: Math.min(duration, center + new_range / 2),
        },
      };
    });
  }, [duration]);

  // Handle fit to screen
  const handleFitToScreen = useCallback(() => {
    setState(prev => ({
      ...prev,
      zoom_level: 1.0,
      view_range: { start: 0, end: duration },
    }));
  }, [duration]);

  // Handle pan
  const handlePan = useCallback((delta_x: number, element_width: number) => {
    setState(prev => {
      const { start, end } = prev.view_range;
      const range = end - start;
      const time_delta = (delta_x / element_width) * range;
      return {
        ...prev,
        view_range: {
          start: Math.max(0, start - time_delta),
          end: Math.min(duration, end - time_delta),
        },
      };
    });
  }, [duration]);

  // Handle mouse down on timeline (for dragging/selection)
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 0) { // Left click
      setState(prev => ({ ...prev, is_dragging: true, drag_start_x: e.clientX }));
      selection_start.current = e.clientX;
    }
  }, []);

  // Handle mouse move
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (state.is_dragging && timeline_ref.current) {
      const delta_x = state.drag_start_x - e.clientX;
      handlePan(delta_x, timeline_ref.current.offsetWidth);
      setState(prev => ({ ...prev, drag_start_x: e.clientX }));
    }
  }, [state.is_dragging, state.drag_start_x, handlePan]);

  // Handle mouse up
  const handleMouseUp = useCallback(() => {
    setState(prev => ({ ...prev, is_dragging: false }));
    selection_start.current = null;
  }, []);

  // Handle event click
  const handleEventClick = useCallback((event: TimelineEvent) => {
    if (onEventClick) {
      onEventClick(event);
    }
  }, [onEventClick]);

  // Handle event hover
  const handleEventHover = useCallback((event_id: string | null) => {
    setState(prev => ({ ...prev, hover_event_id: event_id }));
  }, []);

  // Get hovered event
  const hovered_event = useMemo(() => {
    return state.hover_event_id
      ? filtered_events.find(e => e.id === state.hover_event_id) || null
      : null;
  }, [state.hover_event_id, filtered_events]);

  return (
    <div className="timeline-view">
      <TimelineControls
        zoom_level={state.zoom_level}
        view_range={state.view_range}
        duration={duration}
        formatTime={formatTime}
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        onFitToScreen={handleFitToScreen}
      />

      <div
        ref={timeline_ref}
        className={`timeline-container ${state.is_dragging ? 'dragging' : ''}`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {/* Time ruler */}
        <div className="time-ruler">
          {Array.from({ length: 11 }, (_, i) => {
            const percent = (i / 10) * 100;
            const time = xToTime((percent / 100) * (timeline_ref.current?.offsetWidth || 1));
            return (
              <div key={i} className="ruler-mark" style={{ left: `${percent}%` }}>
                <span className="ruler-line"></span>
                <span className="ruler-label">{formatTime(time)}</span>
              </div>
            );
          })}
        </div>

        {/* Event tracks */}
        <div className="timeline-tracks">
          {Object.entries(events_by_type).map(([type, type_events]) => (
            type_events.length > 0 && (
              <TimelineTrack
                key={type}
                event_type={type as TimelineEvent['event_type']}
                events={type_events}
                selected_id={selected_event_id}
                time_to_x={timeToX}
                onEventClick={handleEventClick}
                onEventHover={handleEventHover}
              />
            )
          ))}
        </div>
      </div>

      {/* Event popup */}
      {hovered_event && (
        <TimelineEventPopup
          event={hovered_event}
          formatTime={formatTime}
        />
      )}
    </div>
  );
};

export default TimelineView;
