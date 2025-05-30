from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError
import logging
import os
# from dotenv import load_dotenv

from src.models.models import engine
from src.endpoints.jobs import job_router, company_router
from src.exceptions import JobDescriptionException, handle_job_description_exception

# Load environment variables
# load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Global exception handler for JobDescriptionException
@app.exception_handler(JobDescriptionException)
async def job_description_exception_handler(request: Request, exc: JobDescriptionException):
    """
    Global exception handler for job description generation errors
    """
    http_exception = handle_job_description_exception(exc)
    return JSONResponse(
        status_code=http_exception.status_code,
        content=exc.error_detail.model_dump()
    )

# Include routers
app.include_router(job_router)
app.include_router(company_router)

# Connection and table verification on startup
try:
    # Test the connection and verify tables exist
    with engine.connect() as connection:
        # Check if Company table exists
        try:
            connection.execute(text('SELECT 1 FROM "Company" LIMIT 1'))
            logger.info("Company table exists")
        except ProgrammingError:
            logger.error("Company table does not exist - please ensure it is created in the database")
            raise

        # Check if JobPosting table exists
        try:
            connection.execute(text('SELECT 1 FROM "JobPosting" LIMIT 1'))
            logger.info("JobPosting table exists")
        except ProgrammingError:
            logger.error("JobPosting table does not exist - please ensure it is created in the database")
            raise

        logger.info("Database connection successful and tables verified!")
except SQLAlchemyError as e:
    logger.error(f"Database error during initialization: {str(e)}")
    raise

# Verify OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY is not set. OpenAI-related features will not work.") 