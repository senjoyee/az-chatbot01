import { useState, useEffect, useCallback, useRef } from 'react';
import { FileProcessingStatus, FileItem } from '../types';
import { getFilesStatus } from '../api';

interface UseFileStatusPollingProps {
  files: FileItem[];
  onStatusUpdate: (updates: Record<string, Partial<FileItem>>) => void;
  pollingInterval?: number;
  maxRetries?: number;
}

export const useFileStatusPolling = ({
  files,
  onStatusUpdate,
  pollingInterval = 10000, // 10 seconds default
  maxRetries = 3
}: UseFileStatusPollingProps) => {
  const [isPolling, setIsPolling] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const retryCountRef = useRef<Record<string, number>>({});
  const timeoutRef = useRef<NodeJS.Timeout>();

  // Keep track of which files need polling
  const getPendingFiles = useCallback(() => {
    return files.filter(file => 
      file.status === FileProcessingStatus.NOT_STARTED || 
      file.status === FileProcessingStatus.IN_PROGRESS
    );
  }, [files]);

  const stopPolling = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = undefined;
    }
    setIsPolling(false);
  }, []);

  const pollFileStatuses = useCallback(async () => {
    const pendingFiles = getPendingFiles();
    
    if (pendingFiles.length === 0) {
      stopPolling();
      return;
    }

    try {
      const fileNames = pendingFiles.map(file => file.name);
      const statusUpdates = await getFilesStatus(fileNames);
      
      // Process updates and track retries
      const updates: Record<string, Partial<FileItem>> = {};
      
      Object.entries(statusUpdates).forEach(([fileName, status]) => {
        updates[fileName] = {
          status: status.status,
          errorMessage: status.errorMessage
        };

        // Reset retry count on successful status update
        if (status.status === FileProcessingStatus.COMPLETED || 
            status.status === FileProcessingStatus.FAILED) {
          delete retryCountRef.current[fileName];
        }
      });

      // Update all statuses at once
      onStatusUpdate(updates);

      // Schedule next poll if there are still pending files
      if (getPendingFiles().length > 0) {
        timeoutRef.current = setTimeout(pollFileStatuses, pollingInterval);
      } else {
        stopPolling();
      }
    } catch (err) {
      console.error('Error polling file statuses:', err);
      setError(err instanceof Error ? err : new Error('Failed to poll file statuses'));
      
      // Increment retry count for all pending files
      pendingFiles.forEach(file => {
        retryCountRef.current[file.name] = (retryCountRef.current[file.name] || 0) + 1;
      });

      // Check if we should stop polling due to max retries
      const shouldStopPolling = pendingFiles.every(
        file => (retryCountRef.current[file.name] || 0) >= maxRetries
      );

      if (shouldStopPolling) {
        stopPolling();
      } else {
        // Exponential backoff for retries
        const nextInterval = pollingInterval * Math.pow(2, Math.min(...Object.values(retryCountRef.current)));
        timeoutRef.current = setTimeout(pollFileStatuses, nextInterval);
      }
    }
  }, [getPendingFiles, onStatusUpdate, pollingInterval, maxRetries, stopPolling]);

  const startPolling = useCallback(() => {
    if (!isPolling && getPendingFiles().length > 0) {
      setIsPolling(true);
      setError(null);
      retryCountRef.current = {};
      pollFileStatuses();
    }
  }, [isPolling, getPendingFiles, pollFileStatuses]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopPolling();
    };
  }, [stopPolling]);

  return {
    isPolling,
    error,
    startPolling,
    stopPolling
  };
};
