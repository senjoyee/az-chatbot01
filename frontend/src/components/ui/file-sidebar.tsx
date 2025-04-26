'use client'

import React, { useState, useEffect } from 'react'
import { listFiles, summarizeDocument } from '../api'
import { FileItem, FileProcessingStatus } from '../types'
import { Button } from './button'
import { ScrollArea } from './scroll-area'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from './dialog'
import { RefreshCw, FileIcon, CheckCircle, ChevronRight, FileText, Copy, Check } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { copyToClipboard } from '../../utils/copy-utils'
import { toast } from 'sonner'

interface FileSidebarProps {
  onFileSelectionChange?: (selectedFiles: string[]) => void
}

interface CopyButtonProps {
  text: string
}

function CopyButton({ text }: CopyButtonProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await copyToClipboard(text, true) // true to remove markdown formatting
      setCopied(true)
      toast.success('Copied to clipboard')
      setTimeout(() => setCopied(false), 2000)
    } catch (error) {
      toast.error('Failed to copy')
    }
  }

  return (
    <Button
      variant="ghost"
      size="sm"
      className="h-8 w-8 p-0 text-gray-500 hover:text-gray-700"
      onClick={handleCopy}
      title="Copy to clipboard"
    >
      {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
    </Button>
  )
}

export function FileSidebar({ onFileSelectionChange }: FileSidebarProps) {
  const [files, setFiles] = useState<FileItem[]>([])
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [collapsed, setCollapsed] = useState(false)
  const [summaryOpen, setSummaryOpen] = useState(false)
  const [summarizing, setSummarizing] = useState(false)
  const [summaryContent, setSummaryContent] = useState<string>('')
  const [summaryFileName, setSummaryFileName] = useState<string>('')

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
    // Remove the polling interval - only refresh when button is clicked
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

  const handleClearSelection = () => {
    setSelectedFiles([])
    if (onFileSelectionChange) {
      onFileSelectionChange([])
    }
  }

  const handleRefresh = () => {
    fetchFiles()
  }

  if (collapsed) {
    return (
      <div className="h-full flex flex-col">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setCollapsed(false)}
          className="m-2 text-white hover:bg-gray-700"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    )
  }

  // Handle document summarization
  const handleSummarize = async (fileId: string) => {
    try {
      // Don't allow summarization if already summarizing
      if (summarizing) return;
      
      setSummarizing(true);
      setSummaryFileName(fileId);
      setSummaryContent('');
      setSummaryOpen(true);
      
      const result = await summarizeDocument(fileId);
      setSummaryContent(result.summary);
    } catch (error) {
      console.error('Error summarizing document:', error);
      setSummaryContent('Error retrieving document summary. Please try again later.');
    } finally {
      setSummarizing(false);
    }
  };

  return (
    <ScrollArea className="h-full flex flex-col w-full font-sans">
      <Dialog open={summaryOpen} onOpenChange={setSummaryOpen}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto font-noto-sans">
          <DialogHeader>
            <div className="flex items-center justify-between">
              <div className="font-noto-sans">
                <DialogTitle className="font-noto-sans">Document Summary</DialogTitle>
                <DialogDescription className="text-sm text-gray-500 font-noto-sans">
                  {summaryFileName}
                </DialogDescription>
              </div>
              {!summarizing && summaryContent && (
                <CopyButton text={summaryContent} />
              )}
            </div>
          </DialogHeader>
          <div className="py-4">
            {summarizing ? (
              <div className="flex flex-col items-center justify-center py-8 font-noto-sans">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-700 mb-4"></div>
                <p className="text-gray-600">Generating summary...</p>
              </div>
            ) : (
              <div className="markdown-content font-noto-sans text-gray-900">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {summaryContent}
                </ReactMarkdown>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center">
            <Button
              variant="outline"
              size="sm"
              className={`mr-2 ${selectedFiles.length === indexedFiles.length ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-white text-gray-700 hover:bg-gray-100'}`}
              onClick={handleSelectAll}
            >
              Select All ({indexedFiles.length})
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="bg-white text-gray-700 hover:bg-gray-100"
              onClick={handleClearSelection}
              disabled={selectedFiles.length === 0}
            >
              Clear
            </Button>
          </div>
          <Button
            variant="outline"
            size="sm"
            className="bg-white text-gray-700 hover:bg-gray-100"
            onClick={handleRefresh}
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>

        <div className="mb-4 text-sm text-gray-600 bg-gray-100 p-3 rounded-lg">
          <p className="font-medium mb-1">For best results:</p>
          <ul className="list-disc pl-5 space-y-1">
            <li>Select up to 50 specific files for targeted searches</li>
            <li>Use "Select All" to search across the entire database</li>
            <li>For summarization, select only one document at a time</li>
          </ul>
        </div>

        {loading ? (
          <div className="flex justify-center items-center h-20">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-700"></div>
          </div>
        ) : error ? (
          <div className="text-red-500 p-4 border border-red-200 rounded-md bg-red-50">
            {error}
          </div>
        ) : indexedFiles.length === 0 ? (
          <div className="text-gray-500 p-4 text-center">
            No indexed files found
          </div>
        ) : (
          <div className="space-y-1">
            {indexedFiles.map((file) => (
              <div
                key={file.id}
                className={`grid grid-cols-[auto_1fr_auto_auto] gap-2 items-center p-2 rounded-md cursor-pointer transition-colors ${
                  selectedFiles.includes(file.id)
                    ? 'bg-blue-100 text-blue-800'
                    : 'hover:bg-gray-100 text-gray-700'
                }`}
              >
                {/* File icon */}
                <div className="flex-shrink-0" onClick={() => handleFileSelection(file.id)}>
                  <FileIcon className="h-5 w-5 text-gray-500" />
                </div>
                
                {/* Filename with tooltip */}
                <div 
                  className="truncate min-w-0"
                  onClick={() => handleFileSelection(file.id)}
                  title={file.name}
                >
                  <p className="text-sm font-medium truncate">
                    {file.name}
                  </p>
                </div>
                
                {/* Selection checkbox - always visible */}
                <div 
                  className="flex-shrink-0"
                  onClick={() => handleFileSelection(file.id)}
                >
                  <CheckCircle
                    className={`h-4 w-4 ${
                      selectedFiles.includes(file.id)
                        ? 'text-blue-600'
                        : 'text-gray-300'
                    }`}
                  />
                </div>
                
                {/* Summarize button - always visible and fixed position */}
                <div className="flex-shrink-0">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="p-1 h-7 w-7"
                    title="Summarize document"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleSummarize(file.id);
                    }}
                  >
                    <FileText className="h-4 w-4 text-gray-500" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </ScrollArea>
  )
}
