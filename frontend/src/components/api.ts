// api.ts

const API_BASE_URL = 'https://jscbbackend01.azurewebsites.net';

// Error handling utility
const handleApiError = (error: any, operation: string) => {
  if (error instanceof TypeError && error.message === 'Failed to fetch') {
    console.error(`CORS or network error during ${operation}:`, error);
    throw new Error(`Network error: Please check your connection or CORS configuration`);
  }
  throw error;
};

export const uploadFiles = async (files: File[]) => {
  // Add file size validation
  const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB
  const MAX_TOTAL_SIZE = 500 * 1024 * 1024; // 500MB
  
  // Calculate total size
  const totalSize = files.reduce((sum, file) => sum + file.size, 0);
  
  // Validate total size
  if (totalSize > MAX_TOTAL_SIZE) {
    throw new Error(`Total file size exceeds ${MAX_TOTAL_SIZE / (1024 * 1024)}MB limit`);
  }
  
  // Validate individual files
  for (const file of files) {
    if (file.size > MAX_FILE_SIZE) {
      throw new Error(`File ${file.name} exceeds ${MAX_FILE_SIZE / (1024 * 1024)}MB limit`);
    }
  }

  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));

  try {
    const response = await fetch(`${API_BASE_URL}/uploadfiles`, {
      method: 'POST',
      body: formData,
      credentials: 'include',
      signal: AbortSignal.timeout(5 * 60 * 1000), // 5 minutes timeout
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Upload failed:', response.status, errorText);
      throw new Error(`Failed to upload files: ${response.status} ${errorText}`);
    }
    
    return response.json();
  } catch (error) {
    handleApiError(error, 'file upload');
    throw error;
  }
};

export const deleteFile = async (filename: string) => {
  console.log('Attempting to delete file:', filename);
  try {
    const response = await fetch(`${API_BASE_URL}/deletefile/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    console.log('Delete response status:', response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Delete failed:', response.status, errorText);
      throw new Error(`Failed to delete file: ${response.status} ${errorText}`);
    }
    
    const data = await response.json();
    console.log('Delete response data:', data);
    return data;
  } catch (error) {
    handleApiError(error, 'file deletion');
    throw error;
  }
};

export interface ListFilesResponse {
  files: any[];
  total_files: number;
  page: number;
  total_pages: number;
}

export const listFiles = async (page: number = 1, pageSize: number = 10): Promise<ListFilesResponse> => {
  const url = `${API_BASE_URL}/listfiles?page=${page}&page_size=${pageSize}`;
  console.log('Attempting to fetch files from:', url);
  
  try {
    const response = await fetch(url, {
      credentials: 'include',
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to fetch files: ${response.status} ${errorText}`);
    }
    
    const data: ListFilesResponse = await response.json();
    return data;
  } catch (error) {
    handleApiError(error, 'file listing');
    throw error;
  }
};

export const searchDocuments = async (query: string) => {
  try {
    const response = await fetch(`${API_BASE_URL}/search?query=${encodeURIComponent(query)}`, {
      method: 'GET',
      credentials: 'include',
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Search failed:', response.status, errorText);
      throw new Error(`Failed to search documents: ${response.status} ${errorText}`);
    }
    
    return response.json();
  } catch (error) {
    handleApiError(error, 'document search');
    throw error;
  }
};

export const getDocumentContent = async (filename: string) => {
  try {
    const response = await fetch(`${API_BASE_URL}/document/${encodeURIComponent(filename)}`, {
      method: 'GET',
      credentials: 'include',
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Get document content failed:', response.status, errorText);
      throw new Error(`Failed to get document content: ${response.status} ${errorText}`);
    }
    
    return response.text();
  } catch (error) {
    handleApiError(error, 'document content retrieval');
    throw error;
  }
};

export const sendMessage = async (
  question: string,
  conversation: Message[]
) => {
  try {
    const response = await fetch(`${API_BASE_URL}/conversation`, {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: question,
        conversation: {
          conversation: conversation.map(msg => ({
            role: msg.sender === 'user' ? 'user' : 'assistant',
            content: msg.text
          }))
        }
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    return data.answer;
  } catch (error) {
    handleApiError(error, 'message sending');
    throw error;
  }
};