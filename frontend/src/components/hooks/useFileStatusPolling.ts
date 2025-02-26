import { useState, useEffect, useCallback, useRef } from 'react';
import { FileProcessingStatus, FileItem } from '../types';
import { listFiles } from '../api';

interface UseFileStatusPollingProps {
  files: FileItem[];
  onStatusUpdate: (updates: Record<string, Partial<FileItem>>) => void;
  pollingInterval?: number;
  maxRetries?: number;
  currentPage?: number;
  pageSize?: number;
}

export const useFileStatusPolling = ({
  files,
  onStatusUpdate,
  pollingInterval = 10000, // 10 seconds default
  maxRetries = 3,
  currentPage = 1,
  pageSize = 10
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
      // Use listFiles to get the current status of all files
      const response = await listFiles(currentPage, pageSize);
      
      if (!response || !Array.isArray(response.files)) {
        throw new Error('Invalid response from server');
      }
      
      // Process updates and track retries
      const updates: Record<string, Partial<FileItem>> = {};
      
      response.files.forEach((file: any) => {
        // Only process files that we're currently tracking
        const pendingFile = pendingFiles.find(pf => pf.name === file.name);
        if (pendingFile) {
          // Helper function to convert backend status to frontend enum
          const convertStatus = (statusStr: string): FileProcessingStatus => {
            // Convert snake_case to UPPER_CASE
            const upperCaseStatus = statusStr.toUpperCase();
            
            // Map to the enum or default to NOT_STARTED
            return FileProcessingStatus[upperCaseStatus as keyof typeof FileProcessingStatus] || 
                   FileProcessingStatus.NOT_STARTED;
          };
          
          const status = file.status ? 
            (typeof file.status === 'string' ? 
              convertStatus(file.status) : 
              file.status) : 
            FileProcessingStatus.NOT_STARTED;
            
          updates[file.name] = {
            status: status,
            errorMessage: file.errorMessage
          };

          // Reset retry count on successful status update
          if (status === FileProcessingStatus.COMPLETED || 
              status === FileProcessingStatus.FAILED) {
            delete retryCountRef.current[file.name];
          }
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
  }, [getPendingFiles, onStatusUpdate, pollingInterval, maxRetries, stopPolling, currentPage, pageSize]);

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
