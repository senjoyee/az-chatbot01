import os
import logging
import asyncio
from typing import List
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Initialize logging
logger = logging.getLogger(__name__)

class Contextualizer:
    def __init__(self):
        logger.debug("Initializing AzureChatOpenAI...")
        self.llm = AzureChatOpenAI(
            azure_deployment="gpt-4o-mini",
            openai_api_version="2023-03-15-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.0,
            streaming=False  # Ensure streaming is disabled for async calls
        )
        logger.debug("AzureChatOpenAI initialized successfully.")

        # Define the prompt template for context generation using message roles
        self.context_prompt = ChatPromptTemplate.from_template("""
        Here is the document:
        <document>
        {document}
        </document>
        Here is the chunk we want to situate within the whole document:
        <chunk>
        {chunk}
        </chunk>
        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
""")

    async def generate_context_async(self, document: str, chunk: str) -> str:
        """
        Asynchronously generates context for a given chunk.

        Args:
            document (str): The full document text.
            chunk (str): The specific chunk of text to contextualize.

        Returns:
            str: The generated context or an empty string in case of failure.
        """
        logger.debug(f"Generating context for chunk: {chunk[:50]}...")

        # Format the prompt using the template
        prompt = self.context_prompt.format(document=document, chunk=chunk)

        # Structure messages with roles
        messages = [
            SystemMessage(content="You are an AI assistant specializing in document analysis. Your task is to provide brief, relevant context for a chunk of text from the given document."),
            HumanMessage(content=prompt)
        ]
        logger.debug(f"Generated messages for LLM: {messages}")

        try:
            # Use the appropriate asynchronous method
            response = await self.llm.ainvoke(messages)  # Use 'arun' for asynchronous calls
            logger.debug(f"LLM Response: {response}")
            context = response.content.strip()
            logger.debug(f"Generated context: {context}")
            return context
        except Exception as e:
            logger.error(f"Error generating context: {str(e)}")
            return ""

    async def contextualize_chunks(self, document: str, chunks: List[str]) -> List[str]:
        """
        Asynchronously generates contextualized chunks by prepending context to each chunk.

        Args:
            document (str): The full document text.
            chunks (List[str]): The list of chunked texts.

        Returns:
            List[str]: List of contextualized chunks.
        """
        logger.debug(f"Contextualizing {len(chunks)} chunks for document.")
        contextualized_chunks = []
        tasks = [self.generate_context_async(document, chunk) for chunk in chunks]
        logger.debug(f"Created {len(tasks)} tasks for context generation.")

        # Gather responses concurrently
        contexts = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, context in enumerate(contexts):
            if isinstance(context, Exception):
                logger.error(f"Context generation failed for chunk {idx}: {str(context)}")
                contextualized_chunk = chunks[idx]  # Fallback to original chunk
            elif context:
                contextualized_chunk = f"{context}\n\n{chunks[idx]}"
            else:
                contextualized_chunk = chunks[idx]  # Fallback to original chunk
            contextualized_chunks.append(contextualized_chunk)

        logger.debug(f"Generated {len(contextualized_chunks)} contextualized chunks.")
        return contextualized_chunks