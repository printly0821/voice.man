/**
 * Test Utilities
 *
 * Helper functions and components for testing
 */

import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';

// Custom render function with providers
interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  wrapper?: ({ children }: { children: React.ReactNode }) => ReactElement;
}

export function renderWithProviders(
  ui: ReactElement,
  options?: CustomRenderOptions
) {
  const { wrapper, ...rest } = options || {};

  const Wrapper = wrapper || (({ children }: { children: React.ReactNode }) => <>{children}</>);

  return render(ui, { wrapper: Wrapper, ...rest });
}

// Mock data generators
export const mockComplaintDocument = (overrides = {}) => ({
  id: 'test-001',
  case_number: 'CASE-2024-001',
  title: 'Test Complaint',
  content: 'This is a test complaint content.',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-02T00:00:00Z',
  status: 'draft' as const,
  metadata: {
    author: 'Test Author',
    department: 'Test Department',
    priority: 'high' as const,
    tags: ['test', 'complaint'],
    related_cases: ['CASE-2024-002'],
    evidence_count: 5,
  },
  ...overrides,
});

export const mockEvidenceFile = (overrides = {}) => ({
  id: 'ev-001',
  filename: 'test-audio.mp3',
  filepath: '/path/to/test-audio.mp3',
  file_type: 'audio' as const,
  size_bytes: 1024000,
  duration_seconds: 120,
  created_at: '2024-01-01T00:00:00Z',
  tags: ['important', 'witness'],
  transcript: 'This is a test transcript.',
  forensic_data: {
    gaslighting_probability: 0.75,
    emotion_events: [
      { timestamp: 10, emotion: 'anger', intensity: 0.8, speaker: 'Speaker 1' },
      { timestamp: 30, emotion: 'sadness', intensity: 0.6, speaker: 'Speaker 2' },
    ],
    speakers: [
      { id: 'spk-1', label: 'Speaker 1', segments_count: 5, total_duration: 60 },
      { id: 'spk-2', label: 'Speaker 2', segments_count: 3, total_duration: 40 },
    ],
  },
  ...overrides,
});

export const mockTimelineEvent = (overrides = {}) => ({
  id: 'te-001',
  timestamp: 10,
  event_type: 'gaslighting' as const,
  title: 'Gaslighting Detected',
  description: 'High probability gaslighting event',
  metadata: { probability: 0.8 },
  color: '#ef4444',
  icon: '\u26A0',
  ...overrides,
});

export const mockSearchResult = (overrides = {}) => ({
  id: 'sr-001',
  content_type: 'transcript' as const,
  title: 'Matching Transcript',
  excerpt: 'This is a matching excerpt from the transcript.',
  highlight_ranges: [
    { start: 5, end: 15, type: 'exact' as const },
  ],
  relevance_score: 0.9,
  metadata: {
    file_id: 'ev-001',
    timestamp: 45,
    speaker: 'Speaker 1',
    gaslighting_probability: 0.7,
    emotion: 'anger',
    tags: ['important'],
  },
  ...overrides,
});

// Re-export testing library utilities
export * from '@testing-library/react';
export { default as userEvent } from '@testing-library/user-event';
