/**
 * Utility functions for copying text to clipboard
 */

/**
 * Remove markdown formatting from text
 * @param text The markdown text to be cleaned
 * @returns Plain text without markdown formatting
 */
export function removeMarkdown(text: string): string {
  if (!text) return '';
  
  // Remove headers (# Header)
  let plainText = text.replace(/#+\s+/g, '');
  
  // Remove bold/italic formatting
  plainText = plainText.replace(/[*_]{1,3}(.*?)[*_]{1,3}/g, '$1');
  
  // Remove code blocks
  plainText = plainText.replace(/```[\s\S]*?```/g, '');
  plainText = plainText.replace(/`([^`]+)`/g, '$1');
  
  // Remove links [text](url) -> text
  plainText = plainText.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
  
  // Remove images ![alt](src) -> alt
  plainText = plainText.replace(/!\[([^\]]+)\]\([^)]+\)/g, '$1');
  
  // Remove horizontal rules
  plainText = plainText.replace(/---+/g, '');
  
  // Replace bullet points with plain dashes
  plainText = plainText.replace(/^\s*[*+-]\s+/gm, '- ');
  
  // Replace numbered lists with plain numbers
  plainText = plainText.replace(/^\s*\d+\.\s+/gm, '');
  
  return plainText.trim();
}

/**
 * Copy text to clipboard
 * @param text The text to copy to clipboard
 * @param markdown Whether the text is markdown (if true, markdown will be removed)
 * @returns Promise that resolves when the text is copied
 */
export async function copyToClipboard(text: string, markdown = true): Promise<void> {
  try {
    const textToCopy = markdown ? removeMarkdown(text) : text;
    await navigator.clipboard.writeText(textToCopy);
    return Promise.resolve();
  } catch (error) {
    console.error('Failed to copy text:', error);
    return Promise.reject(error);
  }
}
