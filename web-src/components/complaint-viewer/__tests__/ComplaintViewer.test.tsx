/**
 * Complaint Viewer Component Tests
 *
 * TAG: TEST-TAG-001
 * Tests for complaint document viewing and management
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ComplaintViewer } from '../ComplaintViewer';
import type { ComplaintDocument } from '../types';

describe('ComplaintViewer Component', () => {
  const mockComplaint: ComplaintDocument = {
    id: 'test-001',
    case_number: 'CASE-2024-001',
    title: 'Test Complaint',
    content: 'This is a test complaint content.',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-02T00:00:00Z',
    status: 'draft',
    metadata: {
      author: 'Test Author',
      department: 'Test Department',
      priority: 'high',
      tags: ['test', 'complaint'],
      related_cases: ['CASE-2024-002'],
      evidence_count: 5,
    },
  };

  const mockHandlers = {
    onEdit: vi.fn(),
    onDelete: vi.fn(),
    onStatusChange: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Initialization', () => {
    it('should render complaint viewer with default state', () => {
      render(<ComplaintViewer complaint={mockComplaint} />);

      expect(screen.getByText('Test Complaint')).toBeInTheDocument();
      expect(screen.getByText('CASE-2024-001')).toBeInTheDocument();
    });

    it('should initialize with content tab selected', () => {
      render(<ComplaintViewer complaint={mockComplaint} />);

      expect(screen.getByRole('tab', { name: /content/i })).toHaveAttribute('aria-selected', 'true');
    });

    it('should initialize with default zoom level of 1.0', () => {
      render(<ComplaintViewer complaint={mockComplaint} />);

      const zoomIndicator = screen.getByText(/100%/i);
      expect(zoomIndicator).toBeInTheDocument();
    });
  });

  describe('Tab Navigation', () => {
    it('should switch to metadata tab when clicked', async () => {
      render(<ComplaintViewer complaint={mockComplaint} />);

      const metadataTab = screen.getByRole('tab', { name: /metadata/i });
      fireEvent.click(metadataTab);

      await waitFor(() => {
        expect(metadataTab).toHaveAttribute('aria-selected', 'true');
      });
    });

    it('should switch to history tab when clicked', async () => {
      render(<ComplaintViewer complaint={mockComplaint} />);

      const historyTab = screen.getByRole('tab', { name: /history/i });
      fireEvent.click(historyTab);

      await waitFor(() => {
        expect(historyTab).toHaveAttribute('aria-selected', 'true');
      });
    });

    it('should switch to evidence tab when clicked', async () => {
      render(<ComplaintViewer complaint={mockComplaint} />);

      const evidenceTab = screen.getByRole('tab', { name: /evidence/i });
      fireEvent.click(evidenceTab);

      await waitFor(() => {
        expect(evidenceTab).toHaveAttribute('aria-selected', 'true');
      });
    });
  });

  describe('Zoom Controls', () => {
    it('should increase zoom level when zoom in is clicked', async () => {
      render(<ComplaintViewer complaint={mockComplaint} />);

      const zoomInButton = screen.getByRole('button', { name: /zoom in/i });
      fireEvent.click(zoomInButton);

      await waitFor(() => {
        expect(screen.getByText(/110%/i)).toBeInTheDocument();
      });
    });

    it('should decrease zoom level when zoom out is clicked', async () => {
      render(<ComplaintViewer complaint={mockComplaint} />);

      const zoomOutButton = screen.getByRole('button', { name: /zoom out/i });
      fireEvent.click(zoomOutButton);

      await waitFor(() => {
        expect(screen.getByText(/90%/i)).toBeInTheDocument();
      });
    });

    it('should not exceed maximum zoom level of 2.0', async () => {
      render(<ComplaintViewer complaint={mockComplaint} />);

      const zoomInButton = screen.getByRole('button', { name: /zoom in/i });

      // Click 11 times to exceed max
      for (let i = 0; i < 11; i++) {
        fireEvent.click(zoomInButton);
      }

      await waitFor(() => {
        expect(screen.getByText(/200%/i)).toBeInTheDocument();
      });
    });

    it('should not go below minimum zoom level of 0.5', async () => {
      render(<ComplaintViewer complaint={mockComplaint} />);

      const zoomOutButton = screen.getByRole('button', { name: /zoom out/i });

      // Click 6 times to go below min
      for (let i = 0; i < 6; i++) {
        fireEvent.click(zoomOutButton);
      }

      await waitFor(() => {
        expect(screen.getByText(/50%/i)).toBeInTheDocument();
      });
    });
  });

  describe('Edit Functionality', () => {
    it('should call onEdit handler when edit button is clicked', async () => {
      render(
        <ComplaintViewer
          complaint={mockComplaint}
          onEdit={mockHandlers.onEdit}
        />
      );

      const editButton = screen.getByRole('button', { name: /edit/i });
      fireEvent.click(editButton);

      await waitFor(() => {
        expect(mockHandlers.onEdit).toHaveBeenCalledWith('test-001');
      });
    });

    it('should not show edit button when readonly is true', () => {
      render(
        <ComplaintViewer
          complaint={mockComplaint}
          onEdit={mockHandlers.onEdit}
          readonly
        />
      );

      const editButton = screen.queryByRole('button', { name: /edit/i });
      expect(editButton).not.toBeInTheDocument();
    });
  });

  describe('Delete Functionality', () => {
    it('should show confirmation dialog when delete button is clicked', async () => {
      global.confirm = vi.fn(() => true);

      render(
        <ComplaintViewer
          complaint={mockComplaint}
          onDelete={mockHandlers.onDelete}
        />
      );

      const deleteButton = screen.getByRole('button', { name: /delete/i });
      fireEvent.click(deleteButton);

      await waitFor(() => {
        expect(global.confirm).toHaveBeenCalledWith('정말 삭제하시겠습니까?');
        expect(mockHandlers.onDelete).toHaveBeenCalledWith('test-001');
      });
    });

    it('should not call onDelete when confirmation is cancelled', async () => {
      global.confirm = vi.fn(() => false);

      render(
        <ComplaintViewer
          complaint={mockComplaint}
          onDelete={mockHandlers.onDelete}
        />
      );

      const deleteButton = screen.getByRole('button', { name: /delete/i });
      fireEvent.click(deleteButton);

      await waitFor(() => {
        expect(global.confirm).toHaveBeenCalled();
        expect(mockHandlers.onDelete).not.toHaveBeenCalled();
      });
    });

    it('should not show delete button when readonly is true', () => {
      render(
        <ComplaintViewer
          complaint={mockComplaint}
          onDelete={mockHandlers.onDelete}
          readonly
        />
      );

      const deleteButton = screen.queryByRole('button', { name: /delete/i });
      expect(deleteButton).not.toBeInTheDocument();
    });
  });

  describe('Status Change', () => {
    it('should call onStatusChange when status is changed', async () => {
      render(
        <ComplaintViewer
          complaint={mockComplaint}
          onStatusChange={mockHandlers.onStatusChange}
        />
      );

      const statusSelect = screen.getByRole('combobox', { name: /status/i });
      fireEvent.change(statusSelect, { target: { value: 'review' } });

      await waitFor(() => {
        expect(mockHandlers.onStatusChange).toHaveBeenCalledWith('test-001', 'review');
      });
    });

    it('should not show status selector when readonly is true', () => {
      render(
        <ComplaintViewer
          complaint={mockComplaint}
          onStatusChange={mockHandlers.onStatusChange}
          readonly
        />
      );

      const statusSelect = screen.queryByRole('combobox', { name: /status/i });
      expect(statusSelect).not.toBeInTheDocument();
    });
  });

  describe('Content Rendering', () => {
    it('should display complaint content in content tab', () => {
      render(<ComplaintViewer complaint={mockComplaint} />);

      expect(screen.getByText('This is a test complaint content.')).toBeInTheDocument();
    });

    it('should display metadata in metadata tab', async () => {
      render(<ComplaintViewer complaint={mockComplaint} />);

      const metadataTab = screen.getByRole('tab', { name: /metadata/i });
      fireEvent.click(metadataTab);

      await waitFor(() => {
        expect(screen.getByText('Test Author')).toBeInTheDocument();
        expect(screen.getByText('Test Department')).toBeInTheDocument();
      });
    });

    it('should display evidence count in metadata', async () => {
      render(<ComplaintViewer complaint={mockComplaint} />);

      const metadataTab = screen.getByRole('tab', { name: /metadata/i });
      fireEvent.click(metadataTab);

      await waitFor(() => {
        expect(screen.getByText(/5.*evidence/i)).toBeInTheDocument();
      });
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty complaint content', () => {
      const emptyComplaint: ComplaintDocument = {
        ...mockComplaint,
        content: '',
      };

      render(<ComplaintViewer complaint={emptyComplaint} />);

      expect(screen.getByText(/no content/i)).toBeInTheDocument();
    });

    it('should handle missing metadata fields', () => {
      const incompleteComplaint: ComplaintDocument = {
        ...mockComplaint,
        metadata: {
          author: '',
          department: '',
          priority: 'low',
          tags: [],
          related_cases: [],
          evidence_count: 0,
        },
      };

      render(<ComplaintViewer complaint={incompleteComplaint} />);

      const metadataTab = screen.getByRole('tab', { name: /metadata/i });
      fireEvent.click(metadataTab);

      expect(screen.getByText(/no author/i)).toBeInTheDocument();
    });

    it('should handle very long content', () => {
      const longContent = 'A'.repeat(10000);
      const longComplaint: ComplaintDocument = {
        ...mockComplaint,
        content: longContent,
      };

      render(<ComplaintViewer complaint={longComplaint} />);

      expect(screen.getByText(longContent.substring(0, 100))).toBeInTheDocument();
    });
  });
});
