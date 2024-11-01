'use client'

import { useState, useCallback, useEffect } from 'react'
import { useDropzone } from 'react-dropzone'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Checkbox } from "@/components/ui/checkbox"
import { Upload, Trash2, File } from 'lucide-react'
import { toast } from 'sonner'
import { uploadFiles, deleteFile, listFiles } from './api' // Import the API functions

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

  // Fetch the list of files when the component mounts
  useEffect(() => {
    fetchFiles()
  }, [])

  const fetchFiles = async () => {
    try {
      console.log('Starting fetchFiles...');
      setLoading(true);
      const response = await listFiles();
      console.log('List files response:', response);
      
      if (!response.files) {
        console.error('No files property in response:', response);
        throw new Error('Invalid response format - missing files property');
      }
      
      const mappedFiles = response.files.map((file: any) => {
        console.log('Processing file:', file);
        return {
          id: file.name,
          name: file.name,
          size: file.size || 0,
          uploadDate: new Date(file.lastModified || Date.now())
        };
      });
      
      console.log('Mapped files:', mappedFiles);
      setFiles(mappedFiles);
    } catch (error) {
      console.error('Error in fetchFiles:', error);
      toast.error('Failed to fetch files: ' + (error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    setUploading(true);
    try {
      await uploadFiles(acceptedFiles);
      toast.success('Documents uploaded successfully');
      fetchFiles(); // Refresh the file list after upload
    } catch (error) {
      toast.error(`Failed to upload documents: ${error.message}`);
      console.error('Error uploading files:', error);
    } finally {
      setUploading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop })

  const handleDeleteFile = async (id: string) => {
    try {
      console.log('Deleting file:', id);
      setLoading(true);
      await deleteFile(id);
      toast.success('Document deleted successfully');
      await fetchFiles(); // Refresh the file list after deletion
    } catch (error) {
      console.error('Error deleting file:', error);
      toast.error(`Failed to delete document: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const deleteSelectedFiles = async () => {
    try {
      setLoading(true);
      await Promise.all(selectedFiles.map(fileId => deleteFile(fileId)));
      toast.success('Selected documents deleted successfully');
      await fetchFiles(); // Refresh the file list after deletion
      setSelectedFiles([]);
    } catch (error) {
      console.error('Error deleting selected files:', error);
      toast.error(`Failed to delete selected documents: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const toggleFileSelection = (id: string) => {
    setSelectedFiles(prev =>
      prev.includes(id) ? prev.filter(fileId => fileId !== id) : [...prev, id]
    )
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
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
          {uploading && <p className="text-center mt-4">Uploading...</p>}
          {loading ? (
            <p className="text-center mt-4">Loading files...</p>
          ) : (
            files.length > 0 && (
              <>
                <Table className="mt-8">
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[50px]">Select</TableHead>
                      <TableHead>Document Name</TableHead>
                      <TableHead>Size</TableHead>
                      <TableHead>Upload Date</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {files.map((file) => (
                      <TableRow key={file.id}>
                        <TableCell>
                          <Checkbox
                            checked={selectedFiles.includes(file.id)}
                            onCheckedChange={() => toggleFileSelection(file.id)}
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
                          <Button variant="ghost" size="icon" onClick={() => handleDeleteFile(file.id)}>
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                <div className="mt-4 flex justify-between items-center">
                  <p className="text-sm text-muted-foreground">
                    {selectedFiles.length} of {files.length} document{files.length !== 1 ? 's' : ''} selected
                  </p>
                  <div className="space-x-2">
                    <Button 
                      variant="destructive" 
                      onClick={deleteSelectedFiles}
                      disabled={selectedFiles.length === 0}
                    >
                      Delete Selected
                    </Button>
                    <Button variant="outline" onClick={() => setFiles([])}>Clear All</Button>
                  </div>
                </div>
              </>
            )
          )}
        </CardContent>
        <CardFooter>
          <p className="text-sm text-muted-foreground">
            {files.length} document{files.length !== 1 ? 's' : ''} uploaded
          </p>
        </CardFooter>
      </Card>
    </div>
  )
}