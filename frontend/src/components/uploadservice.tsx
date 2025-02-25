'use client'

import React, { useState, useCallback, useEffect } from 'react'
import { useDropzone } from 'react-dropzone'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Input } from "@/components/ui/input"
import { Upload, Trash2, File, Info } from 'lucide-react'
import { toast } from 'sonner'
import { uploadFiles, deleteFile, listFiles, getFileStatus, getFilesStatus } from './api'
import { FileProcessingStatus, FileItem, FileWithCustomer } from './types'
import { useFileStatusPolling } from './hooks/useFileStatusPolling'
import { StatusIndicator } from './ui/status-indicator'
import { FileDetails } from './ui/file-details'
import { formatFileSize } from './utils'

export default function DocumentUploadService() {
  const [files, setFiles] = useState<FileItem[]>([])
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [uploading, setUploading] = useState(false)
  const [loading, setLoading] = useState(true)
  const [filesToUpload, setFilesToUpload] = useState<FileWithCustomer[]>([])
  const [selectedFileDetails, setSelectedFileDetails] = useState<string | null>(null)
  
  // Pagination states
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [totalFiles, setTotalFiles] = useState(0)
  const [pageSize, setPageSize] = useState(10)

  useEffect(() => {
    fetchFiles(currentPage, pageSize)
  }, [currentPage, pageSize])

  useEffect(() => {
    setSelectedFiles([]);
  }, [currentPage]);

  const fetchFiles = async (page: number = 1, pageSize: number = 10) => {
    try {
      setLoading(true)
      console.log(`Fetching files: page ${page}, pageSize ${pageSize}`)
      const response = await listFiles(page, pageSize)
      console.log('listFiles response:', response)
      
      if (!response || typeof response !== 'object') {
        throw new Error('Invalid response from server')
      }

      if (!Array.isArray(response.files)) {
        throw new Error('Invalid response format - files property is not an array')
      }
      
      const mappedFiles = response.files.map((file: any) => ({
        id: file.name,
        name: file.name,
        size: file.size || 0,
        uploadDate: new Date(file.lastModified || Date.now()),
        status: file.status || FileProcessingStatus.COMPLETED, // Default to COMPLETED if status is not provided
        errorMessage: file.errorMessage,
        processingStartTime: file.processingStartTime,
        processingEndTime: file.processingEndTime
      }))
      
      setFiles(mappedFiles)
      setTotalFiles(response.total_files || 0)
      setTotalPages(response.total_pages || 1)
      setCurrentPage(response.page || 1)
    } catch (error) {
      console.error('Error in fetchFiles:', error)
      let errorMessage = 'Failed to fetch files'
      if (error instanceof Error) {
        errorMessage += ': ' + error.message
      } else if (typeof error === 'string') {
        errorMessage += ': ' + error
      } else {
        errorMessage += ': Unknown error occurred'
      }
      toast.error(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFilesToUpload(acceptedFiles.map(file => ({ file, customerName: '' })))
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop })

  const handleCustomerNameChange = (index: number, customerName: string) => {
    setFilesToUpload(prev => 
      prev.map((item, i) => i === index ? { ...item, customerName } : item)
    )
  }

  const handleDeleteFile = async (id: string) => {
    try {
      setLoading(true)
      await deleteFile(id)
      toast.success('Document deleted successfully')
      await fetchFiles(currentPage, pageSize)
    } catch (error) {
      console.error('Error deleting file:', error)
      let errorMessage = 'Failed to delete document'
      if (error instanceof Error) {
        errorMessage += ': ' + error.message
      } else if (typeof error === 'string') {
        errorMessage += ': ' + error
      } else {
        errorMessage += ': Unknown error occurred'
      }
      toast.error(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  const handleStatusUpdate = useCallback((updates: Record<string, Partial<FileItem>>) => {
    setFiles(prevFiles => 
      prevFiles.map(file => {
        const update = updates[file.name];
        if (update) {
          // If status changed to COMPLETED or FAILED, show toast
          if (
            (update.status === FileProcessingStatus.COMPLETED || 
             update.status === FileProcessingStatus.FAILED) && 
            file.status !== update.status
          ) {
            const message = update.status === FileProcessingStatus.COMPLETED
              ? `File ${file.name} has been processed successfully`
              : `Failed to process file ${file.name}${update.errorMessage ? `: ${update.errorMessage}` : ''}`;
            
            toast[update.status === FileProcessingStatus.COMPLETED ? 'success' : 'error'](message);
          }
          
          return { ...file, ...update };
        }
        return file;
      })
    );
  }, []);

  const { isPolling, error: pollingError, startPolling, stopPolling } = useFileStatusPolling({
    files,
    onStatusUpdate: handleStatusUpdate,
    pollingInterval: 10000, // 10 seconds
    maxRetries: 3
  });

  // Start polling when files are uploaded
  useEffect(() => {
    const pendingFiles = files.filter(file => 
      file.status === FileProcessingStatus.NOT_STARTED || 
      file.status === FileProcessingStatus.IN_PROGRESS
    );
    
    if (pendingFiles.length > 0 && !isPolling) {
      startPolling();
    }
  }, [files, isPolling, startPolling]);

  const handleUpload = async () => {
    setUploading(true)
    try {
      await uploadFiles(filesToUpload)
      toast.success('Documents uploaded successfully')
      await fetchFiles(currentPage, pageSize)
      setFilesToUpload([])
      // Start polling for the newly uploaded files
      startPolling();
    } catch (error) {
      console.error('Error uploading files:', error)
      let errorMessage = 'Failed to upload documents'
      if (error instanceof Error) {
        errorMessage += ': ' + error.message
      } else if (typeof error === 'string') {
        errorMessage += ': ' + error
      } else {
        errorMessage += ': Unknown error occurred'
      }
      toast.error(errorMessage)
    } finally {
      setUploading(false)
    }
  }

  // Show polling error if it occurs
  useEffect(() => {
    if (pollingError) {
      toast.error(`Error checking file status: ${pollingError.message}`);
    }
  }, [pollingError]);

  const handleSelectFile = (fileId: string) => {
    setSelectedFiles(prev => 
      prev.includes(fileId) 
        ? prev.filter(id => id !== fileId)
        : [...prev, fileId]
    );
  };

  const handleSelectAllFiles = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSelectedFiles(event.target.checked ? files.map(file => file.id) : []);
  };

  const handleDeleteSelected = async () => {
    if (!selectedFiles.length) return;
    
    try {
      setLoading(true);
      await Promise.all(selectedFiles.map(fileId => deleteFile(fileId)));
      toast.success(`${selectedFiles.length} document(s) deleted successfully`);
      setSelectedFiles([]);
      await fetchFiles(currentPage, pageSize);
    } catch (error) {
      console.error('Error deleting files:', error);
      toast.error('Failed to delete some documents');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4 max-w-4xl">
      <Card>
        <CardHeader>
          <CardTitle>Document Upload Service</CardTitle>
          <CardDescription>Upload, manage, and delete your documents securely in Azure Blob Storage</CardDescription>
        </CardHeader>
        <CardContent>
          {/* Dropzone */}
          <div 
            {...getRootProps()} 
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive ? 'border-primary bg-primary/10' : 'border-muted-foreground'
            }`}
          >
            <input {...getInputProps()} />
            <Upload className="mx-auto h-12 w-12 text-muted-foreground" />
            <p className="mt-2 text-sm text-muted-foreground">
              Drag 'n' drop some documents here, or click to select files
            </p>
          </div>

          {/* Files to Upload */}
          {filesToUpload.length > 0 && (
            <div className="mt-4">
              <h3 className="text-lg font-semibold mb-2">Files to Upload</h3>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Document Name</TableHead>
                    <TableHead>Customer Name</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filesToUpload.map((file, index) => (
                    <TableRow key={index}>
                      <TableCell className="font-medium">
                        <div className="flex items-center">
                          <File className="mr-2 h-4 w-4" />
                          {file.file.name}
                        </div>
                      </TableCell>
                      <TableCell>
                        <Input
                          type="text"
                          placeholder="Customer Name"
                          value={file.customerName}
                          onChange={(e) => handleCustomerNameChange(index, e.target.value)}
                          className="w-48"
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <Button 
                onClick={handleUpload} 
                disabled={uploading} 
                className="mt-4"
              >
                {uploading ? 'Uploading...' : 'Upload Files'}
              </Button>
            </div>
          )}

          {/* Uploaded Files */}
          {loading ? (
            <div className="flex justify-center items-center mt-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            </div>
          ) : (
            files.length > 0 && (
              <div className="mt-8">
                <h3 className="text-lg font-semibold mb-2">Uploaded Files</h3>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-12">
                        <input
                          type="checkbox"
                          className="h-4 w-4 rounded border-gray-300"
                          checked={selectedFiles.length === files.length && files.length > 0}
                          onChange={handleSelectAllFiles}
                        />
                      </TableHead>
                      <TableHead>Name</TableHead>
                      <TableHead>Size</TableHead>
                      <TableHead>Upload Date</TableHead>
                      <TableHead>Indexing Status</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {files.map((file) => (
                      <TableRow key={file.id}>
                        <TableCell>
                          <input
                            type="checkbox"
                            className="h-4 w-4 rounded border-gray-300"
                            checked={selectedFiles.includes(file.id)}
                            onChange={() => handleSelectFile(file.id)}
                          />
                        </TableCell>
                        <TableCell className="font-medium">
                          <div className="flex items-center">
                            <File className="mr-2 h-4 w-4" />
                            {file.name}
                          </div>
                        </TableCell>
                        <TableCell>{formatFileSize(file.size)}</TableCell>
                        <TableCell>{file.uploadDate.toLocaleString()}</TableCell>
                        <TableCell>
                          <StatusIndicator 
                            status={file.status} 
                            errorMessage={file.errorMessage}
                          />
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-2">
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => setSelectedFileDetails(file.id)}
                              className="h-8 w-8 p-0"
                            >
                              <Info className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => handleDeleteFile(file.id)}
                              className="h-8 w-8 p-0"
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                {/* Delete Selected Button */}
                <div className="mt-4 flex items-center justify-between">
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={handleDeleteSelected}
                    disabled={selectedFiles.length === 0}
                  >
                    Delete Selected ({selectedFiles.length})
                  </Button>
                  {/* Pagination Controls */}
                  <div className="flex items-center space-x-2">
                    <div className="text-sm text-muted-foreground">
                      Showing {files.length} of {totalFiles} files
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                      disabled={currentPage === 1}
                    >
                      Previous
                    </Button>
                    <div className="text-sm">
                      Page {currentPage} of {totalPages}
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                      disabled={currentPage === totalPages}
                    >
                      Next
                    </Button>
                  </div>
                </div>
              </div>
            )
          )}
        </CardContent>
      </Card>

      {/* File Details Dialog */}
      {selectedFileDetails && (
        <FileDetails
          file={files.find(f => f.id === selectedFileDetails)!}
          open={!!selectedFileDetails}
          onOpenChange={(open) => !open && setSelectedFileDetails(null)}
        />
      )}
    </div>
  )
}
