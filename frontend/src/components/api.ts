import { EventSourcePolyfill } from 'event-source-polyfill';
const API_BASE_URL = 'https://jscbbackend01.azurewebsites.net';

// Error handling utility
const handleApiError = (error: any, operation: string) => {
  console.error(`Error during ${operation}:`, error);
  if (error instanceof TypeError && error.message === 'Failed to fetch') {
    throw new Error(`Network error: Please check your connection or CORS configuration`);
  }
  throw error;
};

export interface FileWithCustomer {
  file: File;
  customerName: string;
}

export const uploadFiles = async (filesWithCustomers: FileWithCustomer[]) => {
  const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB
  const MAX_TOTAL_SIZE = 500 * 1024 * 1024; // 500MB
  
  // Calculate total size
  const totalSize = filesWithCustomers.reduce((sum, { file }) => sum + file.size, 0);
  
  // Validate total size
  if (totalSize > MAX_TOTAL_SIZE) {
    throw new Error(`Total file size exceeds ${MAX_TOTAL_SIZE / (1024 * 1024)}MB limit`);
  }
  
  // Validate individual files
  for (const { file } of filesWithCustomers) {
    if (file.size > MAX_FILE_SIZE) {
      throw new Error(`File ${file.name} exceeds ${MAX_FILE_SIZE / (1024 * 1024)}MB limit`);
    }
  }

  const formData = new FormData();
  filesWithCustomers.forEach(({ file, customerName }, index) => {
    formData.append(`files`, file);
    formData.append(`customer_names`, customerName);
  });

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

export const listFiles = async (page: number = 1, pageSize: number = 10) => {
  console.log('Attempting to fetch files from:', `${API_BASE_URL}/listfiles?page=${page}&page_size=${pageSize}`);
  
  try {
    const response = await fetch(
      `${API_BASE_URL}/listfiles?page=${page}&page_size=${pageSize}`,
      {
        credentials: 'include',
      }
    );
    console.log('Raw response:', response);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('List files failed:', response.status, errorText);
      throw new Error(`Failed to fetch files: ${response.status} ${errorText}`);
    }
    
    const data = await response.json();
    console.log('Parsed response data:', data);
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

export interface Message {
  content: string;
  isBot: boolean;
  isStreaming?: boolean;
  error?: boolean;
}

// Keep only the EventSource version and modify to:
export const sendMessageStream = async (
  message: string,
  conversation: Message[],
  token: string
): Promise<EventSourcePolyfill> => {
  const formattedConversation = conversation.map(msg => ({
    role: msg.isBot ? 'assistant' : 'user',
    content: msg.content
  }));

  return new EventSourcePolyfill(`${API_BASE_URL}/conversation/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`
    },
    body: JSON.stringify({
      question: message,
      conversation: {
        conversation: formattedConversation
      }
    }),
    withCredentials: true
  });
};