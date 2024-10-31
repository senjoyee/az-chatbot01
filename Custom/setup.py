from setuptools import setup, find_packages

setup(
    name="custom_doc_splitter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-openai",
        "langchain-core",
        "pymupdf4llm",
        "azure-storage-blob",
        "python-dotenv"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Custom document splitter with GPT and PyMuPDF support",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/custom_doc_splitter",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)