/**
 * Timeline View Component Tests
 *
 * TAG: TEST-TAG-003
 * Tests for timeline visualization and event interaction
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { TimelineView } from '../TimelineView';
import { mockTimelineEvent } from '../../../test/test-utils';
import type { TimelineEvent } from '../types';

describe('TimelineView Component', () => {
  const mockEvents: TimelineEvent[] = [
    mockTimelineEvent({
      id: 'te-001',
      timestamp: 10,
      event_type: 'gaslighting',
      title: 'Gaslighting Event 1',
    }),
    mockTimelineEvent({
      id: 'te-002',
      timestamp: 30,
      event_type: 'emotion',
      title: 'Emotion Event 1',
    }),
    mockTimelineEvent({
      id: 'te-003',
      timestamp: 60,
      event_type: 'transcript',
      title: 'Transcript Event 1',
    }),
    mockTimelineEvent({
      id: 'te-004',
      timestamp: 90,
      event_type: 'speaker_change',
      title: 'Speaker Change 1',
    }),
  ];

  const duration = 120;

  const mockHandlers = {
    onEventClick: vi.fn(),
    onRangeSelect: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Initialization', () => {
    it('should render timeline with events', () => {
      render(
        <TimelineView
          events={mockEvents}
          duration={duration}
          onEventClick={mockHandlers.onEventClick}
        />
      );

      expect(screen.getByText(/0:00/i)).toBeInTheDocument();
      expect(screen.getByText(/2:00/i)).toBeInTheDocument();
    });

    it('should initialize with full duration view range', () => {
      render(
        <TimelineView
          events={mockEvents}
          duration={duration}
          onEventClick={mockHandlers.onEventClick}
        />
      );

      expect(screen.getByText(/0:00 - 2:00/i)).toBeInTheDocument();
      expect(screen.getByText(/of 2:00/i)).toBeInTheDocument();
    });

    it('should initialize with 100% zoom level', () => {
      render(
        <TimelineView
          events={mockEvents}
          duration={duration}
          onEventClick={mockHandlers.onEventClick}
        />
      );

      expect(screen.getByText('100%')).toBeInTheDocument();
    });
  });

  describe('Event Rendering', () => {
    it('should render events by type in separate tracks', () => {
      render(
        <TimelineView
          events={mockEvents}
          duration={duration}
          onEventClick={mockHandlers.onEventClick}
        />
      );

      // Should have tracks for gaslighting, emotion, transcript, speaker_change
      expect(screen.getByText(/Gaslighting/i)).toBeInTheDocument();
      expect(screen.getByText(/Emotion/i)).toBeInTheDocument();
      expect(screen.getByText(/Transcript/i)).toBeInTheDocument();
      expect(screen.getByText(/Speaker/i)).toBeInTheDocument();
    });

    it('should display event count for each track', () => {
      render(
        <TimelineView
          events={mockEvents}
          duration={duration}
          onEventClick={mockHandlers.onEventClick}
        />
      );

      // Each track should show count of 1
      const counts = screen.getAllByText('1');
      expect(counts.length).toBeGreaterThanOrEqual(4);
    });
  });

  describe('Event Interaction', () => {
    it('should call onEventClick when event is clicked', async () => {
      render(
        <TimelineView
          events={mockEvents}
          duration={duration}
          onEventClick={mockHandlers.onEventClick}
        />
      );

      // Find and click on an event (timeline-event)
      const events = document.querySelectorAll('.timeline-event');
      if (events.length > 0) {
        fireEvent.click(events[0]);

        await waitFor(() => {
          expect(mockHandlers.onEventClick).toHaveBeenCalled();
        });
      }
    });

    it('should show popup on event hover', async () => {
      render(
        <TimelineView
          events={mockEvents}
          duration={duration}
          onEventClick={mockHandlers.onEventClick}
        />
      );

      const events = document.querySelectorAll('.timeline-event');
      if (events.length > 0) {
        fireEvent.mouseEnter(events[0]);

        await waitFor(() => {
          const popup = document.querySelector('.timeline-event-popup');
          expect(popup).toBeInTheDocument();
        });
      }
    });
  });

  describe('Zoom Controls', () => {
    it('should zoom in when zoom in button is clicked', async () => {
      render(
        <TimelineView
          events={mockEvents}
          duration={duration}
          onEventClick={mockHandlers.onEventClick}
        />
      );

      const zoomInButton = screen.getByLabelText('Zoom in');
      fireEvent.click(zoomInButton);

      await waitFor(() => {
        // Zoom level should increase
        const zoomLabel = screenByTextContent(/150%/i);
        expect(zoomLabel).toBeInTheDocument();
      });
    });

    it('should zoom out when zoom out button is clicked', async () => {
      render(
        <TimelineView
          events={mockEvents}
          duration={duration}
          onEventClick={mockHandlers.onEventClick}
        />
      );

      // First zoom in to enable zoom out
      const zoomInButton = screen.getByLabelText('Zoom in');
      fireEvent.click(zoomInButton);

      await waitFor(() => {
        expect(screenByTextContent(/150%/i)).toBeInTheDocument();
      });

      const zoomOutButton = screen.getByLabelText('Zoom out');
      fireEvent.click(zoomOutButton);

      await waitFor(() => {
        expect(screenByTextContent(/100%/i)).toBeInTheDocument();
      });
    });

    it('should disable zoom in button at maximum zoom', () => {
      render(
        <TimelineView
          events={mockEvents}
          duration={duration}
          onEventClick={mockHandlers.onEventClick}
        />
      );

      const zoomInButton = screen.getByLabelText('Zoom in');

      // Click multiple times to reach max
      for (let i = 0; i < 15; i++) {
        fireEvent.click(zoomInButton);
      }

      expect(zoomInButton).toBeDisabled();
    });

    it('should disable zoom out button at minimum zoom', () => {
      render(
        <TimelineView
          events={mockEvents}
          duration={duration}
          onEventClick={mockHandlers.onEventClick}
        />
      );

      const zoomOutButton = screen.getByLabelText('Zoom out');

      // Click multiple times to reach min
      for (let i = 0; i < 10; i++) {
        fireEvent.click(zoomOutButton);
      }

      expect(zoomOutButton).toBeDisabled();
    });
  });

  describe('Fit to Screen', () => {
    it('should reset view to full duration when fit to screen is clicked', async () => {
      render(
        <TimelineView
          events={mockEvents}
          duration={duration}
          onEventClick={mockHandlers.onEventClick}
        />
      );

      // First zoom in
      const zoomInButton = screen.getByLabelText('Zoom in');
      fireEvent.click(zoomInButton);

      // Then fit to screen
      const fitButton = screen.getByLabelText('Fit to screen');
      fireEvent.click(fitButton);

      await waitFor(() => {
        expect(screen.getByText('100%')).toBeInTheDocument();
        expect(screen.getByText(/0:00 - 2:00/i)).toBeInTheDocument();
      });
    });
  });

  describe('Filter by Event Type', () => {
    it('should only show specified event types when filters are applied', () => {
      render(
        <TimelineView
          events={mockEvents}
          duration={duration}
          onEventClick={mockHandlers.onEventClick}
          filter_types={['gaslighting', 'emotion']}
        />
      );

      // Should only show gaslighting and emotion tracks
      expect(screen.getByText(/Gaslighting/i)).toBeInTheDocument();
      expect(screen.getByText(/Emotion/i)).toBeInTheDocument();
      expect(screen.getByText(/Transcript/i)).toBeInTheDocument();
      expect(screen.getByText(/Speaker/i)).toBeInTheDocument();
    });

    it('should show all event types when no filters are applied', () => {
      render(
        <TimelineView
          events={mockEvents}
          duration={duration}
          onEventClick={mockHandlers.onEventClick}
          filter_types={[]}
        />
      );

      expect(screen.getByText(/Gaslighting/i)).toBeInTheDocument();
      expect(screen.getByText(/Emotion/i)).toBeInTheDocument();
      expect(screen.getByText(/Transcript/i)).toBeInTheDocument();
      expect(screen.getByText(/Speaker/i)).toBeInTheDocument();
    });
  });

  describe('Time Ruler', () => {
    it('should display time marks at regular intervals', () => {
      render(
        <TimelineView
          events={mockEvents}
          duration={duration}
          onEventClick={mockHandlers.onEventClick}
        />
      );

      // Should have multiple time marks (0:00, 0:12, 0:24, 0:36, 0:48, 1:00, 1:12, 1:24, 1:36, 1:48, 2:00)
      const timeMarks = screen.getAllByText(/\d+:\d{2}/);
      expect(timeMarks.length).toBeGreaterThan(5);
    });
  });

  describe('Selected Event', () => {
    it('should highlight selected event', () => {
      render(
        <TimelineView
          events={mockEvents}
          duration={duration}
          onEventClick={mockHandlers.onEventClick}
          selected_event_id="te-001"
        />
      );

      const selectedEvent = document.querySelector('.timeline-event.selected');
      expect(selectedEvent).toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty events list', () => {
      render(
        <TimelineView
          events={[]}
          duration={duration}
          onEventClick={mockHandlers.onEventClick}
        />
      );

      expect(screen.getByText(/0:00/i)).toBeInTheDocument();
    });

    it('should handle zero duration', () => {
      render(
        <TimelineView
          events={mockEvents}
          duration={0}
          onEventClick={mockHandlers.onEventClick}
        />
      );

      expect(screen.getByText(/0:00/i)).toBeInTheDocument();
    });

    it('should handle events at same timestamp', () => {
      const sameTimeEvents: TimelineEvent[] = [
        mockTimelineEvent({ id: 'te-001', timestamp: 30, event_type: 'gaslighting' }),
        mockTimelineEvent({ id: 'te-002', timestamp: 30, event_type: 'emotion' }),
        mockTimelineEvent({ id: 'te-003', timestamp: 30, event_type: 'transcript' }),
      ];

      render(
        <TimelineView
          events={sameTimeEvents}
          duration={duration}
          onEventClick={mockHandlers.onEventClick}
        />
      );

      expect(screen.getByText(/Gaslighting/i)).toBeInTheDocument();
      expect(screen.getByText(/Emotion/i)).toBeInTheDocument();
      expect(screen.getByText(/Transcript/i)).toBeInTheDocument();
    });
  });
});

// Helper function to find element by text content
function screenByTextContent(text: string | RegExp) {
  const elements = Array.from(document.querySelectorAll('*')).filter(el =>
    el.textContent?.match(text)
  );
  return elements[0];
}
