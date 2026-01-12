/**
 * Evidence Player Component
 *
 * TAG: FUNCTION-TAG-011
 * Audio/video player with transcript and forensic data display
 */

import React, { useState, useRef, useEffect } from 'react';
import { EvidenceFile } from './types';

interface EvidencePlayerProps {
  evidence: EvidenceFile;
  onClose: () => void;
}

export const EvidencePlayer: React.FC<EvidencePlayerProps> = ({
  evidence,
  onClose,
}) => {
  const [is_playing, setIsPlaying] = useState(false);
  const [current_time, setCurrentTime] = useState(0);
  const [playback_rate, setPlaybackRate] = useState(1.0);
  const [volume, setVolume] = useState(1.0);
  const mediaRef = useRef<HTMLAudioElement | HTMLVideoElement>(null);

  useEffect(() => {
    const media = mediaRef.current;
    if (media) {
      media.playbackRate = playback_rate;
      media.volume = volume;
    }
  }, [playback_rate, volume]);

  const handleTogglePlay = () => {
    const media = mediaRef.current;
    if (media) {
      if (is_playing) {
        media.pause();
      } else {
        media.play();
      }
      setIsPlaying(!is_playing);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    setCurrentTime(time);
    if (mediaRef.current) {
      mediaRef.current.currentTime = time;
    }
  };

  const handleTimeUpdate = () => {
    if (mediaRef.current) {
      setCurrentTime(mediaRef.current.currentTime);
    }
  };

  const handleEnded = () => {
    setIsPlaying(false);
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const renderMediaElement = () => {
    const common_props = {
      ref: mediaRef as any,
      onTimeUpdate: handleTimeUpdate,
      onEnded: handleEnded,
      onPlay: () => setIsPlaying(true),
      onPause: () => setIsPlaying(false),
    };

    if (evidence.file_type === 'audio') {
      return <audio src={evidence.filepath} {...common_props} />;
    }
    if (evidence.file_type === 'video') {
      return <video src={evidence.filepath} controls {...common_props} />;
    }
    return null;
  };

  return (
    <div className="evidence-player">
      <div className="player-header">
        <h3>{evidence.filename}</h3>
        <button
          className="close-button"
          onClick={onClose}
          aria-label="Close player"
        >
          \u00D7
        </button>
      </div>

      <div className="player-media">
        {renderMediaElement()}
        {(evidence.file_type === 'audio' || evidence.file_type === 'video') && (
          <div className="player-controls">
            <button
              className="control-button play-button"
              onClick={handleTogglePlay}
              aria-label={is_playing ? 'Pause' : 'Play'}
            >
              {is_playing ? '\u23F8' : '\u25B6'}
            </button>

            <div className="time-display">
              <span>{formatTime(current_time)}</span>
              <span>/</span>
              <span>
                {evidence.duration_seconds
                  ? formatTime(evidence.duration_seconds)
                  : '--:--'}
              </span>
            </div>

            <input
              type="range"
              min="0"
              max={evidence.duration_seconds || 100}
              value={current_time}
              onChange={handleSeek}
              className="seek-bar"
            />

            <select
              value={playback_rate}
              onChange={(e) => setPlaybackRate(parseFloat(e.target.value))}
              className="speed-select"
            >
              <option value="0.5">0.5x</option>
              <option value="0.75">0.75x</option>
              <option value="1.0">1.0x</option>
              <option value="1.25">1.25x</option>
              <option value="1.5">1.5x</option>
              <option value="2.0">2.0x</option>
            </select>

            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={volume}
              onChange={(e) => setVolume(parseFloat(e.target.value))}
              className="volume-control"
              aria-label="Volume"
            />
          </div>
        )}
      </div>

      {evidence.transcript && (
        <div className="player-transcript">
          <h4>Transcript</h4>
          <div className="transcript-content">
            {evidence.transcript.split('\n').map((line, index) => (
              <p key={index}>{line}</p>
            ))}
          </div>
        </div>
      )}

      {evidence.forensic_data && (
        <div className="player-forensic">
          <h4>Forensic Analysis</h4>
          <div className="forensic-summary">
            <div className="forensic-metric">
              <label>Gaslighting Probability</label>
              <div className={`metric-value ${
                evidence.forensic_data.gaslighting_probability > 0.7 ? 'high' :
                evidence.forensic_data.gaslighting_probability > 0.4 ? 'medium' : 'low'
              }`}>
                {Math.round(evidence.forensic_data.gaslighting_probability * 100)}%
              </div>
            </div>
          </div>

          {evidence.forensic_data.emotion_events.length > 0 && (
            <div className="emotion-events">
              <h5>Emotion Events</h5>
              <div className="emotion-timeline">
                {evidence.forensic_data.emotion_events.map((event, index) => (
                  <div
                    key={index}
                    className="emotion-event"
                    style={{ left: `${(event.timestamp / (evidence.duration_seconds || 1)) * 100}%` }}
                    title={`${event.emotion}: ${Math.round(event.intensity * 100)}%`}
                  >
                    <span className="emotion-marker">{event.emotion[0]}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {evidence.forensic_data.speakers.length > 0 && (
            <div className="speaker-info">
              <h5>Speakers</h5>
              {evidence.forensic_data.speakers.map((speaker, index) => (
                <div key={index} className="speaker-item">
                  <span className="speaker-label">{speaker.label}</span>
                  <span className="speaker-stats">
                    {speaker.segments_count} segments, {Math.round(speaker.total_duration / 60)} min
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {evidence.tags.length > 0 && (
        <div className="player-tags">
          <h4>Tags</h4>
          <div className="tags-list">
            {evidence.tags.map((tag, index) => (
              <span key={index} className="tag-badge">{tag}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default EvidencePlayer;
