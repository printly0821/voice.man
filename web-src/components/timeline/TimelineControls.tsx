/**
 * Timeline Controls Component
 *
 * TAG: FUNCTION-TAG-014
 * Control panel for timeline zoom and navigation
 */

import React from 'react';
import { TimelineRange } from './types';

interface TimelineControlsProps {
  zoom_level: number;
  view_range: TimelineRange;
  duration: number;
  formatTime: (seconds: number) => string;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFitToScreen: () => void;
}

export const TimelineControls: React.FC<TimelineControlsProps> = ({
  zoom_level,
  view_range,
  duration,
  formatTime,
  onZoomIn,
  onZoomOut,
  onFitToScreen,
}) => {
  const handleZoomSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(e.target.value);
    // Calculate new zoom level
    // This is a simplified implementation
    if (value > zoom_level) {
      onZoomIn();
    } else if (value < zoom_level) {
      onZoomOut();
    }
  };

  return (
    <div className="timeline-controls">
      <div className="controls-row">
        <div className="zoom-controls">
          <button
            className="control-button"
            onClick={onZoomOut}
            disabled={zoom_level <= 0.1}
            aria-label="Zoom out"
            title="Zoom out"
          >
            \u2212
          </button>
          <input
            type="range"
            min="0.1"
            max="10"
            step="0.1"
            value={zoom_level}
            onChange={handleZoomSliderChange}
            className="zoom-slider"
            aria-label="Zoom level"
          />
          <button
            className="control-button"
            onClick={onZoomIn}
            disabled={zoom_level >= 10}
            aria-label="Zoom in"
            title="Zoom in"
          >
            +
          </button>
          <span className="zoom-label">{Math.round(zoom_level * 100)}%</span>
        </div>

        <button
          className="control-button fit-button"
          onClick={onFitToScreen}
          aria-label="Fit to screen"
          title="Fit to screen"
        >
          \u25B2 \u25BC
        </button>
      </div>

      <div className="controls-row time-range">
        <span className="time-range-label">
          {formatTime(view_range.start)} - {formatTime(view_range.end)}
        </span>
        <span className="time-range-total">
          of {formatTime(duration)}
        </span>
      </div>
    </div>
  );
};

export default TimelineControls;
