"""
Helper utility functions.
"""

from typing import Dict, Any

def escape_odata_filter_value(value: str) -> str:
    """
    Escape a string value for use in OData filter expressions.
    Handles special characters and follows OData string literal rules.
    """
    if not value:
        return value
        
    # First, escape single quotes (OData uses two single quotes for escaping)
    value = value.replace("'", "''")
    
    # List of characters that might need to be percent-encoded
    # Order matters: encode % first to prevent double-encoding
    value = value.replace("%", "%25")  # Encode % first
    special_chars = {
        '#': '%23',
        '+': '%2B',
        '?': '%3F',
        '\\': '%5C',
        '&': '%26'
    }
    
    # Replace special characters with their percent-encoded values
    for char, encoded in special_chars.items():
        value = value.replace(char, encoded)
    
    return value


def sanitize_id(filename: str) -> str:
    """
    Sanitize filename to create a valid Azure Search ID.
    Only keeps alphanumeric chars and underscores, replacing others with dashes.
    Also removes file extensions.
    """
    # Remove file extensions (handles multiple extensions like .docx.pdf)
    base_name = filename.split('.')[0]
    # Replace any non-alphanumeric chars (except underscore) with dash
    sanitized = ''.join(c if c.isalnum() or c == '_' else '-' for c in base_name)
    # Remove any duplicate dashes
    sanitized = '-'.join(filter(None, sanitized.split('-')))
    return sanitized.lower()


def serialize_metadata(metadata: dict) -> dict:
    """Convert metadata values to JSON-serializable format."""
    serialized = {}
    for key, value in metadata.items():
        if hasattr(value, 'text'):  # Handle Title objects
            serialized[key] = value.text
        elif isinstance(value, (str, int, float, bool, type(None))):
            serialized[key] = value
        # Skip any other types that can't be serialized
    return serialized


def extract_source(file_name: str) -> str:
    """
    Extracts the source identifier from the file name.
    Examples:
        - 'TBS SOM 2.3.pdf' -> 'TBS'
        - 'TBS_SOM 2.3.pdf' -> 'TBS'
        - 'BSW Document.docx' -> 'BSW'
    
    Args:
        file_name (str): The name of the file
        
    Returns:
        str: The extracted source identifier in uppercase, or 'UNKNOWN' if no valid source found
    """
    # Split on first occurrence of space, underscore, or dash
    parts = file_name.split()
    if not parts:
        return "UNKNOWN"
        
    # Get first part and clean it
    first_part = parts[0].strip()
    
    # If it contains underscore or dash, split on that
    if '_' in first_part:
        first_part = first_part.split('_')[0]
    elif '-' in first_part:
        first_part = first_part.split('-')[0]
        
    # Ensure it's uppercase and contains only letters
    source = ''.join(c for c in first_part if c.isalpha()).upper()
    
    return source if source else "UNKNOWN"
