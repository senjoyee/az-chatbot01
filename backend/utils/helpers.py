"""
Helper utility functions.
"""

from typing import Dict, Any
import re

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
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Serializing metadata: {metadata}")
    serialized = {}
    for key, value in metadata.items():
        if key == 'customer':
            logger.info(f"Processing customer field: {value}")
        
        if hasattr(value, 'text'):  # Handle Title objects
            serialized[key] = value.text
        elif isinstance(value, (str, int, float, bool, type(None))):
            serialized[key] = value
        # Skip any other types that can't be serialized
    
    logger.info(f"Serialized metadata: {serialized}")
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

def is_casual_conversation(message: str, llm=None) -> bool:
    """Determines if a message requires casual response.
    
    Args:
        message: User input text
        llm: Optional language model to use for classification
    
    Returns:
        bool: True if message is casual conversation
    """
    if llm is None:
        try:
            from services.agent import llm_4o_mini as llm
        except ImportError:
            raise ImportError("Failed to import llm from services.agent")

    classification_prompt = '''
    You will be given a user message and your task is to determine if it is casual talk like a greeting or something similar. Here is the user message:

    <user_message>
    {message}
    </user_message>

    Casual talk typically includes:
    - Greetings (e.g., "Hello", "Hi", "Hey")
    - Farewells (e.g., "Goodbye", "See you later", "Take care")
    - Small talk about weather, time of day, or general well-being (e.g., "Nice day, isn't it?", "How are you?")
    - Brief, friendly expressions (e.g., "Have a good one", "What's up?")

    Analyze the user message and determine if it fits the criteria for casual talk.

    Provide your reasoning first, then give your final answer. Your response should be structured as follows:

    <reasoning>
    [Explain your thought process here]
    </reasoning>

    <answer>
    [Respond with only "yes" if the message is casual talk, or "no" if it is not]
    </answer>
    '''
    
    response = llm.invoke(classification_prompt.format(message=message))
    match = re.search(r"<answer>\s*(yes)\s*</answer>", response.content, flags=re.IGNORECASE)
    return match is not None

def process_excel_elements(elements):
    """
    Process Excel table elements to ensure they're properly formatted for chunking.
    Each sheet in the Excel file is stored as a Table object.
    
    Args:
        elements: List of elements returned by partition_xlsx
        
    Returns:
        List of processed elements with enhanced text and metadata
    """
    processed_elements = []
    
    for i, element in enumerate(elements):
        # For Excel tables, we want to include both text and HTML representation
        if hasattr(element, 'metadata') and hasattr(element.metadata, 'text_as_html'):
            # Add sheet name or index to the element if available
            sheet_name = f"Sheet {i+1}"
            if hasattr(element.metadata, 'sheet_name') and element.metadata.sheet_name:
                sheet_name = element.metadata.sheet_name
                
            # Create a more descriptive text representation
            element.text = f"[{sheet_name}]\n{element.text}"
            
            # Store the HTML representation for potential future use
            if not hasattr(element, 'excel_html'):
                element.excel_html = element.metadata.text_as_html
                
        processed_elements.append(element)
        
    return processed_elements