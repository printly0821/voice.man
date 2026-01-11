/**
 * Evidence Explorer Component Tests
 *
 * TAG: TEST-TAG-002
 * Tests for evidence browsing, filtering, and management
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { EvidenceExplorer } from '../EvidenceExplorer';
import { mockEvidenceFile } from '../../../test/test-utils';
import type { EvidenceFile } from '../types';

describe('EvidenceExplorer Component', () => {
  const mockEvidence: EvidenceFile[] = [
    mockEvidenceFile({
      id: 'ev-001',
      filename: 'audio-1.mp3',
      file_type: 'audio',
      size_bytes: 1024000,
      duration_seconds: 120,
      created_at: '2024-01-01T00:00:00Z',
    }),
    mockEvidenceFile({
      id: 'ev-002',
      filename: 'video-1.mp4',
      file_type: 'video',
      size_bytes: 5120000,
      duration_seconds: 300,
      created_at: '2024-01-02T00:00:00Z',
    }),
    mockEvidenceFile({
      id: 'ev-003',
      filename: 'document-1.pdf',
      file_type: 'document',
      size_bytes: 204800,
      created_at: '2024-01-03T00:00:00Z',
    }),
  ];

  const mockHandlers = {
    onSelectEvidence: vi.fn(),
    onDeleteEvidence: vi.fn(),
    onTagEvidence: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Initialization', () => {
    it('should render evidence explorer with evidence list', () => {
      render(
        <EvidenceExplorer
          evidence={mockEvidence}
          onSelectEvidence={mockHandlers.onSelectEvidence}
        />
      );

      expect(screen.getByText(/Total: 3 files/i)).toBeInTheDocument();
    });

    it('should initialize with grid view mode', () => {
      render(
        <EvidenceExplorer
          evidence={mockEvidence}
          onSelectEvidence={mockHandlers.onSelectEvidence}
        />
      );

      const gridViewButton = screen.getByTitle('Grid view');
      expect(gridViewButton).toHaveClass('active');
    });

    it('should initialize with default sort by created_at descending', () => {
      render(
        <EvidenceExplorer
          evidence={mockEvidence}
          onSelectEvidence={mockHandlers.onSelectEvidence}
        />
      );

      // Sort select should have created_at-desc as default
      const sortSelect = screen.getByRole('combobox');
      expect(sortSelect).toHaveValue('created_at-desc');
    });
  });

  describe('Evidence Selection', () => {
    it('should select evidence when clicking on evidence card', async () => {
      render(
        <EvidenceExplorer
          evidence={mockEvidence}
          onSelectEvidence={mockHandlers.onSelectEvidence}
        />
      );

      const firstCard = screen.getByText('audio-1.mp3');
      fireEvent.click(firstCard);

      await waitFor(() => {
        expect(mockHandlers.onSelectEvidence).toHaveBeenCalledWith('ev-001');
      });
    });

    it('should show detail panel when evidence is selected', async () => {
      render(
        <EvidenceExplorer
          evidence={mockEvidence}
          onSelectEvidence={mockHandlers.onSelectEvidence}
        />
      );

      const firstCard = screen.getByText('audio-1.mp3');
      fireEvent.click(firstCard);

      await waitFor(() => {
        expect(screen.getByText('audio-1.mp3')).toBeInTheDocument();
        expect(screen.getByText(/Transcript/i)).toBeInTheDocument();
      });
    });
  });

  describe('View Mode Toggle', () => {
    it('should switch to list view when list view button is clicked', async () => {
      render(
        <EvidenceExplorer
          evidence={mockEvidence}
          onSelectEvidence={mockHandlers.onSelectEvidence}
        />
      );

      const listViewButton = screen.getByTitle('List view');
      fireEvent.click(listViewButton);

      await waitFor(() => {
        expect(listViewButton).toHaveClass('active');
        expect(screen.getByRole('table')).toBeInTheDocument();
      });
    });

    it('should switch back to grid view when grid view button is clicked', async () => {
      render(
        <EvidenceExplorer
          evidence={mockEvidence}
          onSelectEvidence={mockHandlers.onSelectEvidence}
        />
      );

      const listViewButton = screen.getByTitle('List view');
      const gridViewButton = screen.getByTitle('Grid view');

      fireEvent.click(listViewButton);
      await waitFor(() => {
        expect(listViewButton).toHaveClass('active');
      });

      fireEvent.click(gridViewButton);
      await waitFor(() => {
        expect(gridViewButton).toHaveClass('active');
      });
    });
  });

  describe('Search Filtering', () => {
    it('should filter evidence by search query', async () => {
      render(
        <EvidenceExplorer
          evidence={mockEvidence}
          onSelectEvidence={mockHandlers.onSelectEvidence}
        />
      );

      const searchInput = screen.getByPlaceholderText('Search evidence...');
      fireEvent.change(searchInput, { target: { value: 'audio' } });

      await waitFor(() => {
        expect(screen.getByText('audio-1.mp3')).toBeInTheDocument();
        expect(screen.queryByText('video-1.mp4')).not.toBeInTheDocument();
      });
    });

    it('should filter evidence by file type', async () => {
      render(
        <EvidenceExplorer
          evidence={mockEvidence}
          onSelectEvidence={mockHandlers.onSelectEvidence}
        />
      );

      const audioTypeButton = screen.getByTitle('Audio');
      fireEvent.click(audioTypeButton);

      await waitFor(() => {
        expect(audioTypeButton).toHaveClass('active');
        expect(screen.getByText(/Total: 1 files/i)).toBeInTheDocument();
      });
    });

    it('should combine multiple filters', async () => {
      render(
        <EvidenceExplorer
          evidence={mockEvidence}
          onSelectEvidence={mockHandlers.onSelectEvidence}
        />
      );

      const searchInput = screen.getByPlaceholderText('Search evidence...');
      const audioTypeButton = screen.getByTitle('Audio');

      fireEvent.change(searchInput, { target: { value: 'audio' } });
      fireEvent.click(audioTypeButton);

      await waitFor(() => {
        expect(screen.getByText(/Total: 1 files/i)).toBeInTheDocument();
      });
    });
  });

  describe('Date Range Filtering', () => {
    it('should filter evidence by date range', async () => {
      render(
        <EvidenceExplorer
          evidence={mockEvidence}
          onSelectEvidence={mockHandlers.onSelectEvidence}
        />
      );

      const startDateInput = screen.getAllByPlaceholderText('yyyy-mm-dd')[0];
      const endDateInput = screen.getAllByPlaceholderText('yyyy-mm-dd')[1];

      fireEvent.change(startDateInput, { target: { value: '2024-01-02' } });
      fireEvent.change(endDateInput, { target: { value: '2024-01-03' } });

      await waitFor(() => {
        expect(screen.queryByText('audio-1.mp3')).not.toBeInTheDocument();
        expect(screen.getByText('video-1.mp4')).toBeInTheDocument();
      });
    });
  });

  describe('Duration Filtering', () => {
    it('should filter evidence by minimum duration', async () => {
      render(
        <EvidenceExplorer
          evidence={mockEvidence}
          onSelectEvidence={mockHandlers.onSelectEvidence}
        />
      );

      const durationInputs = screen.getAllByPlaceholderText('Min');
      fireEvent.change(durationInputs[1], { target: { value: '200' } });

      await waitFor(() => {
        expect(screen.queryByText('audio-1.mp3')).not.toBeInTheDocument();
        expect(screen.getByText('video-1.mp4')).toBeInTheDocument();
      });
    });
  });

  describe('Evidence Deletion', () => {
    it('should show confirmation dialog when delete button is clicked', async () => {
      global.confirm = vi.fn(() => true);

      render(
        <EvidenceExplorer
          evidence={mockEvidence}
          onSelectEvidence={mockHandlers.onSelectEvidence}
          onDeleteEvidence={mockHandlers.onDeleteEvidence}
        />
      );

      const deleteButtons = screen.getAllByTitle(/Delete evidence/i);
      fireEvent.click(deleteButtons[0]);

      await waitFor(() => {
        expect(global.confirm).toHaveBeenCalledWith('정말 이 증거를 삭제하시겠습니까?');
        expect(mockHandlers.onDeleteEvidence).toHaveBeenCalledWith('ev-001');
      });
    });

    it('should not call onDelete when confirmation is cancelled', async () => {
      global.confirm = vi.fn(() => false);

      render(
        <EvidenceExplorer
          evidence={mockEvidence}
          onSelectEvidence={mockHandlers.onSelectEvidence}
          onDeleteEvidence={mockHandlers.onDeleteEvidence}
        />
      );

      const deleteButtons = screen.getAllByTitle(/Delete evidence/i);
      fireEvent.click(deleteButtons[0]);

      await waitFor(() => {
        expect(global.confirm).toHaveBeenCalled();
        expect(mockHandlers.onDeleteEvidence).not.toHaveBeenCalled();
      });
    });

    it('should not show delete buttons in readonly mode', () => {
      render(
        <EvidenceExplorer
          evidence={mockEvidence}
          onSelectEvidence={mockHandlers.onSelectEvidence}
          onDeleteEvidence={mockHandlers.onDeleteEvidence}
          readonly
        />
      );

      const deleteButtons = screen.queryAllByTitle(/Delete evidence/i);
      expect(deleteButtons.length).toBe(0);
    });
  });

  describe('Sort Functionality', () => {
    it('should sort evidence by file name ascending', async () => {
      render(
        <EvidenceExplorer
          evidence={mockEvidence}
          onSelectEvidence={mockHandlers.onSelectEvidence}
        />
      );

      const sortSelect = screen.getByRole('combobox');
      fireEvent.change(sortSelect, { target: { value: 'filename-asc' } });

      await waitFor(() => {
        expect(sortSelect).toHaveValue('filename-asc');
      });
    });

    it('should sort evidence by file size descending', async () => {
      render(
        <EvidenceExplorer
          evidence={mockEvidence}
          onSelectEvidence={mockHandlers.onSelectEvidence}
        />
      );

      const sortSelect = screen.getByRole('combobox');
      fireEvent.change(sortSelect, { target: { value: 'size_bytes-desc' } });

      await waitFor(() => {
        expect(sortSelect).toHaveValue('size_bytes-desc');
      });
    });
  });

  describe('Clear Filters', () => {
    it('should clear all active filters when clear button is clicked', async () => {
      render(
        <EvidenceExplorer
          evidence={mockEvidence}
          onSelectEvidence={mockHandlers.onSelectEvidence}
        />
      );

      // Apply some filters
      const searchInput = screen.getByPlaceholderText('Search evidence...');
      fireEvent.change(searchInput, { target: { value: 'audio' } });

      const audioTypeButton = screen.getByTitle('Audio');
      fireEvent.click(audioTypeButton);

      // Clear filters
      const clearButton = screen.getByText('Clear Filters');
      fireEvent.click(clearButton);

      await waitFor(() => {
        expect(searchInput).toHaveValue('');
        expect(audioTypeButton).not.toHaveClass('active');
      });
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty evidence list', () => {
      render(
        <EvidenceExplorer
          evidence={[]}
          onSelectEvidence={mockHandlers.onSelectEvidence}
        />
      );

      expect(screen.getByText(/Total: 0 files/i)).toBeInTheDocument();
    });

    it('should handle evidence without duration', () => {
      const evidenceWithoutDuration: EvidenceFile[] = [
        mockEvidenceFile({
          id: 'ev-001',
          filename: 'document.pdf',
          file_type: 'document',
          duration_seconds: undefined,
        }),
      ];

      render(
        <EvidenceExplorer
          evidence={evidenceWithoutDuration}
          onSelectEvidence={mockHandlers.onSelectEvidence}
        />
      );

      expect(screen.getByText('document.pdf')).toBeInTheDocument();
    });

    it('should handle evidence without forensic data', () => {
      const evidenceWithoutForensic: EvidenceFile[] = [
        mockEvidenceFile({
          id: 'ev-001',
          forensic_data: undefined,
        }),
      ];

      render(
        <EvidenceExplorer
          evidence={evidenceWithoutForensic}
          onSelectEvidence={mockHandlers.onSelectEvidence}
        />
      );

      expect(screen.queryByText(/Gaslighting:/i)).not.toBeInTheDocument();
    });
  });
});
