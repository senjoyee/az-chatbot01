'use client'

import React, { useState, useEffect } from 'react'
import { listFiles } from '../api'
import { FileItem, FileProcessingStatus } from '../types'
import { Checkbox } from './checkbox'
import { ScrollArea } from './scroll-area'
import { Button } from './button'
import { ChevronLeft, ChevronRight, RefreshCw, FileQuestion } from 'lucide-react'

interface FileSidebarProps {
  onFileSelectionChange?: (selectedFiles: string[]) => void
}

export function FileSidebar({ onFileSelectionChange }: FileSidebarProps) {
  const [files, setFiles] = useState<FileItem[]>([])
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [collapsed, setCollapsed] = useState(false)
  
  // Only show completed files
  const indexedFiles = files.filter(file => file.status === FileProcessingStatus.COMPLETED)

  const fetchFiles = async () => {
    try {
      setLoading(true)
      setError(null)
      // Get all files, with a large page size to ensure we get everything
      const response = await listFiles(1, 100)
      
      if (!response || !Array.isArray(response.files)) {
        throw new Error('Invalid response from server')
      }
      
      const mappedFiles = response.files.map((file: any) => ({
        id: file.name,
        name: file.name,
        size: file.size || 0,
        uploadDate: new Date(file.lastModified || Date.now()),
        status: file.status ? 
          (typeof file.status === 'string' ? 
            FileProcessingStatus[file.status.toUpperCase() as keyof typeof FileProcessingStatus] : 
            file.status) : 
          FileProcessingStatus.NOT_STARTED,
        errorMessage: file.errorMessage,
        processingStartTime: file.processingStartTime,
        processingEndTime: file.processingEndTime
      }))
      
      setFiles(mappedFiles)
    } catch (error) {
      console.error('Error fetching files:', error)
      setError('Failed to fetch files')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchFiles()
    // Set up polling interval to refresh the file list every 30 seconds
    const interval = setInterval(fetchFiles, 30000)
    return () => clearInterval(interval)
  }, [])

  const handleFileSelection = (fileId: string) => {
    setSelectedFiles(prev => {
      const newSelection = prev.includes(fileId)
        ? prev.filter(id => id !== fileId)
        : [...prev, fileId]
      
      // Notify parent component if callback is provided
      if (onFileSelectionChange) {
        onFileSelectionChange(newSelection)
      }
      
      return newSelection
    })
  }

  const handleSelectAll = () => {
    const allFileIds = indexedFiles.map(file => file.id)
    const newSelection = selectedFiles.length === indexedFiles.length ? [] : allFileIds
    
    setSelectedFiles(newSelection)
    
    // Notify parent component if callback is provided
    if (onFileSelectionChange) {
      onFileSelectionChange(newSelection)
    }
  }

  if (collapsed) {
    return (
      <div className="h-full flex flex-col border-r bg-gray-50">
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={() => setCollapsed(false)}
          className="m-2"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col border-r bg-gray-50 w-1/3 font-sans">
      <div className="p-4 border-b flex justify-between items-center">
        <h3 className="font-medium text-sm">Indexed Files</h3>
        <div className="flex gap-1">
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={fetchFiles}
            className="h-8 w-8"
            title="Refresh files"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={() => setCollapsed(true)}
            className="h-8 w-8"
            title="Collapse sidebar"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
        </div>
      </div>
      
      <div className="p-2 border-b">
        <div className="flex items-center space-x-2">
          <Checkbox 
            id="select-all" 
            checked={selectedFiles.length === indexedFiles.length && indexedFiles.length > 0}
            onCheckedChange={handleSelectAll}
          />
          <label 
            htmlFor="select-all" 
            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
          >
            Select All ({indexedFiles.length})
          </label>
        </div>
      </div>
      
      {/* Help text */}
      <div className="p-2 bg-blue-50 text-blue-700 text-xs border-b">
        <div className="flex items-start gap-2">
          <FileQuestion className="h-4 w-4 flex-shrink-0 mt-0.5" />
          <div>
            <p>Select one or more files to provide context for the chatbot. At least one file must be selected to enable chat.</p>
            <p className="mt-1">For best results:</p>
            <ul className="list-disc ml-4 mt-0.5">
              <li>Select up to 50 specific files for targeted searches</li>
              <li>Use "Select All" to search across the entire database</li>
            </ul>
          </div>
        </div>
      </div>
      
      <ScrollArea className="flex-1">
        {loading ? (
          <div className="flex justify-center items-center p-4">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
          </div>
        ) : error ? (
          <div className="p-4 text-sm text-red-500">{error}</div>
        ) : indexedFiles.length === 0 ? (
          <div className="p-4 text-sm text-gray-500">No indexed files found</div>
        ) : (
          <div className="p-2 space-y-1">
            {indexedFiles.map((file) => (
              <div key={file.id} className="flex items-center space-x-2 p-2 hover:bg-gray-100 rounded">
                <Checkbox 
                  id={`file-${file.id}`}
                  checked={selectedFiles.includes(file.id)}
                  onCheckedChange={() => handleFileSelection(file.id)}
                />
                <label 
                  htmlFor={`file-${file.id}`}
                  className="text-sm leading-none truncate flex-1 cursor-pointer"
                  title={file.name}
                >
                  {file.name}
                </label>
              </div>
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  )
}
