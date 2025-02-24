'use client'

import React from 'react'
import { FileItem } from '../types'
import { formatFileSize } from '../utils'
import { StatusIndicator } from './status-indicator'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from './dialog'

interface FileDetailsProps {
  file: FileItem
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function FileDetails({ file, open, onOpenChange }: FileDetailsProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>File Details</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-2">
            <div className="text-sm font-medium">Name</div>
            <div className="text-sm">{file.name}</div>
            
            <div className="text-sm font-medium">Size</div>
            <div className="text-sm">{formatFileSize(file.size)}</div>
            
            <div className="text-sm font-medium">Upload Date</div>
            <div className="text-sm">{file.uploadDate.toLocaleString()}</div>
            
            <div className="text-sm font-medium">Status</div>
            <div className="text-sm">
              <StatusIndicator 
                status={file.status} 
                errorMessage={file.errorMessage}
              />
            </div>

            {file.processingStartTime && (
              <>
                <div className="text-sm font-medium">Processing Started</div>
                <div className="text-sm">
                  {new Date(file.processingStartTime).toLocaleString()}
                </div>
              </>
            )}

            {file.processingEndTime && (
              <>
                <div className="text-sm font-medium">Processing Completed</div>
                <div className="text-sm">
                  {new Date(file.processingEndTime).toLocaleString()}
                </div>
              </>
            )}

            {file.errorMessage && (
              <>
                <div className="text-sm font-medium">Error Details</div>
                <div className="text-sm text-red-500">{file.errorMessage}</div>
              </>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
