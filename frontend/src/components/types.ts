export enum FileProcessingStatus {
  NOT_STARTED = 'NOT_STARTED',
  IN_PROGRESS = 'IN_PROGRESS',
  COMPLETED = 'COMPLETED',
  FAILED = 'FAILED'
}

export interface FileItem {
  id: string
  name: string
  size: number
  uploadDate: Date
  status: FileProcessingStatus
  errorMessage?: string
  processingStartTime?: string
  processingEndTime?: string
}

export interface FileWithCustomer {
  file: File
  customerName: string
}
