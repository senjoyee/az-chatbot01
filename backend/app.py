from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# Import utility functions and configuration
from config.logging_config import setup_logging
from routes import file_status, conversation, file_routes, document_routes


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
app.include_router(file_status.router)
app.include_router(conversation.router)
app.include_router(file_routes.router)
app.include_router(document_routes.router)