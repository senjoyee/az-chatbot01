'use client'

import React, { useState, useCallback, useEffect } from 'react'
import { useDropzone } from 'react-dropzone'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Input } from "@/components/ui/input"
import { Upload, Trash2, File } from 'lucide-react'
import { toast } from 'sonner'
import { uploadFiles, deleteFile, listFiles, FileWithCustomer } from './api'

interface FileItem {
  id: string
  name: string
  size: number
  uploadDate: Date
}

export default function DocumentUploadService() {
  const [files, setFiles] = useState<FileItem[]>([])
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [uploading, setUploading] = useState(false)
  const [loading, setLoading] = useState(true)
  const [filesToUpload, setFilesToUpload] = useState<FileWithCustomer[]>([])
  
  // Pagination states
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [totalFiles, setTotalFiles] = useState(0)
  const [pageSize, setPageSize] = useState(10)

  useEffect(() => {
    fetchFiles(currentPage, pageSize)
  }, [currentPage, pageSize])

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

  const handleUpload = async () => {
    setUploading(true)
    try {
      await uploadFiles(filesToUpload)
      toast.success('Documents uploaded successfully')
      fetchFiles(currentPage, pageSize)
      setFilesToUpload([])
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

  return (
    <div className="container mx-auto p-4 max-w-4xl">
      <Card>
        <CardHeader>
          <CardTitle>Document Upload Service</CardTitle>
          <CardDescription>Upload, manage, and delete your documents securely in Azure Blob Storage</CardDescription>
        </CardHeader>
        <CardContent>
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

          {filesToUpload.length > 0 && (
            <div className="mt-4">
              <h3 className="text-lg font-semibold mb-2">Files to Upload</h3>
              {filesToUpload.map((file, index) => (
                <div key={index} className="mb-2 flex items-center space-x-2">
                  <File className="h-4 w-4" />
                  <span className="flex-grow">{file.file.name}</span>
                  <Input
                    type="text"
                    placeholder="Customer Name"
                    value={file.customerName}
                    onChange={(e) => handleCustomerNameChange(index, e.target.value)}
                    className="w-48"
                  />
                </div>
              ))}
              <Button onClick={handleUpload} disabled={uploading} className="mt-2">
                {uploading ? 'Uploading...' : 'Upload Files'}
              </Button>
            </div>
          )}

          {loading ? (
            <p className="text-center mt-4">Loading files...</p>
          ) : (
            files.length > 0 && (
              <Table className="mt-8">
                <TableHeader>
                  <TableRow>
                    <TableHead>Document Name</TableHead>
                    <TableHead>Size</TableHead>
                    <TableHead>Upload Date</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {files.map((file) => (
                    <TableRow key={file.id}>
                      <TableCell className="font-medium">
                        <div className="flex items-center">
                          <File className="mr-2 h-4 w-4" />
                          {file.name}
                        </div>
                      </TableCell>
                      <TableCell>{formatFileSize(file.size)}</TableCell>
                      <TableCell>{file.uploadDate.toLocaleString()}</TableCell>
                      <TableCell>
                        <Button variant="ghost" size="icon" onClick={() => handleDeleteFile(file.id)}>
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )
          )}

          {/* Pagination Controls */}
          <div className="mt-4 flex justify-between items-center">
            <p className="text-sm text-muted-foreground">
              Showing {files.length} of {totalFiles} documents
            </p>
            <div className="flex space-x-2">
              <Button 
                variant="outline" 
                onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))} 
                disabled={currentPage === 1}
              >
                Previous
              </Button>
              <span className="flex items-center">
                Page {currentPage} of {totalPages}
              </span>
              <Button 
                variant="outline" 
                onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))} 
                disabled={currentPage === totalPages}
              >
                Next
              </Button>
            </div>
          </div>
        </CardContent>
        <CardFooter>
          <p className="text-sm text-muted-foreground">
            {totalFiles} document{totalFiles !== 1 ? 's' : ''} uploaded
          </p>
        </CardFooter>
      </Card>
    </div>
  )
}

function formatFileSize(bytes: number) {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

