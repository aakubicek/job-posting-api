from typing import Optional, Any
from pydantic import BaseModel

class ErrorDetail(BaseModel):
    """
    Standardized error response model
    """
    error_code: str
    message: str
    details: Optional[Any] = None

class JobDescriptionException(Exception):
    """
    Base exception for job description generation errors
    """
    def __init__(self, message: str, error_code: str = "JOB_DESCRIPTION_ERROR"):
        self.error_detail = ErrorDetail(
            error_code=error_code,
            message=message
        )
        super().__init__(message)

class InvalidJobIDError(JobDescriptionException):
    """
    Raised when an invalid job ID is provided
    """
    def __init__(self, job_id: int):
        super().__init__(
            message=f"Invalid job ID: {job_id}. No job posting found.",
            error_code="INVALID_JOB_ID"
        )
        self.error_detail.details = {"job_id": job_id}

class DatabaseConnectionError(JobDescriptionException):
    """
    Raised when there are database connection or query issues
    """
    def __init__(self, original_error: Exception):
        super().__init__(
            message=f"Database connection error: {str(original_error)}",
            error_code="DATABASE_CONNECTION_ERROR"
        )
        self.error_detail.details = {
            "original_error": str(original_error)
        }

class AIModelError(JobDescriptionException):
    """
    Raised when there are issues with the AI model generation
    """
    def __init__(self, error_message: str, model: str = "gpt-4o-mini"):
        super().__init__(
            message=f"AI model generation error: {error_message}",
            error_code="AI_MODEL_ERROR"
        )
        self.error_detail.details = {
            "model": model,
            "error_message": error_message
        }

class ValidationError(JobDescriptionException):
    """
    Raised when input validation fails
    """
    def __init__(self, validation_errors: dict):
        super().__init__(
            message="Input validation failed",
            error_code="VALIDATION_ERROR"
        )
        self.error_detail.details = validation_errors

class OpenAIConfigurationError(JobDescriptionException):
    """
    Raised when OpenAI API key or configuration is invalid
    """
    def __init__(self, message: str = "OpenAI API configuration is invalid"):
        super().__init__(
            message=message,
            error_code="OPENAI_CONFIG_ERROR"
        )

def handle_job_description_exception(exc: JobDescriptionException):
    """
    Convert JobDescriptionException to FastAPI HTTPException
    """
    from fastapi import HTTPException, status
    
    error_status_map = {
        "INVALID_JOB_ID": status.HTTP_404_NOT_FOUND,
        "DATABASE_CONNECTION_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
        "AI_MODEL_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "VALIDATION_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "OPENAI_CONFIG_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "JOB_DESCRIPTION_ERROR": status.HTTP_400_BAD_REQUEST
    }
    
    status_code = error_status_map.get(
        exc.error_detail.error_code, 
        status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    
    return HTTPException(
        status_code=status_code,
        detail=exc.error_detail.model_dump()
    ) 