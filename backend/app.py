# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import utility functions and configuration
from config.logging_config import setup_logging
from routes import file_status, file_management, conversation

logger = setup_logging()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://documentchatbot01.azurewebsites.net",
        "https://jsragfunc01.azurewebsites.net",
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include routers
app.include_router(file_status.router, tags=["File Status"])
app.include_router(file_management.router, tags=["File Management"])
app.include_router(conversation.router, tags=["Conversation"])

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Document Chatbot API is running"}