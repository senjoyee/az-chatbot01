// api.ts

const API_BASE_URL = 'https://jscb-proxy-nginx.azurewebsites.net/api';

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
      // Add timeout
      signal: AbortSignal.timeout(5 * 60 * 1000), // 5 minutes timeout
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Upload failed:', response.status, errorText);
      throw new Error(`Failed to upload files: ${response.status} ${errorText}`);
    }
    
    return response.json();
  } catch (error) {
    console.error('Upload error:', error);
    throw error;
  }
};

export const deleteFile = async (filename: string) => {
  console.log('Attempting to delete file:', filename);
  try {
    const response = await fetch(`${API_BASE_URL}/deletefile/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
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
    console.error('Delete error:', error);
    throw error;
  }
};

export const listFiles = async (page: number = 1, pageSize: number = 10) => {
  console.log('Attempting to fetch files from:', `${API_BASE_URL}/listfiles?page=${page}&page_size=${pageSize}`);
  
  try {
    const response = await fetch(`${API_BASE_URL}/listfiles?page=${page}&page_size=${pageSize}`);
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
    console.error('List files error:', error);
    throw error;
  }
};

// You can add more API functions here as needed

export const searchDocuments = async (query: string) => {
  try {
    const response = await fetch(`${API_BASE_URL}/search?query=${encodeURIComponent(query)}`, {
      method: 'GET',
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Search failed:', response.status, errorText);
      throw new Error(`Failed to search documents: ${response.status} ${errorText}`);
    }
    
    return response.json();
  } catch (error) {
    console.error('Search error:', error);
    throw error;
  }
};

export const getDocumentContent = async (filename: string) => {
  try {
    const response = await fetch(`${API_BASE_URL}/document/${encodeURIComponent(filename)}`, {
      method: 'GET',
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Get document content failed:', response.status, errorText);
      throw new Error(`Failed to get document content: ${response.status} ${errorText}`);
    }
    
    return response.text();
  } catch (error) {
    console.error('Get document content error:', error);
    throw error;
  }
};