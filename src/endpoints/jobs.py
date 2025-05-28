from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import logging
import os
from openai import OpenAI

from src.models.models import get_db
from src.schemas.schemas import (
    JobPostingCreate, JobPostingUpdate, JobPosting, ToolsInput, JobDescriptionUpdate,
    CompanyCreate, CompanyUpdate, Company
)

# Configure OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# Set up logging
logger = logging.getLogger(__name__)

# =============================================================================
# Helper Functions
# =============================================================================

def get_job_posting(db: Session, job_id: int):
    query = text('SELECT * FROM "JobPosting" WHERE id = :job_id')
    result = db.execute(query, {"job_id": job_id})
    job = result.fetchone()
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job posting with id {job_id} not found")
    return job

def get_company(db: Session, company_id: int):
    query = text('SELECT * FROM "Company" WHERE id = :company_id')
    result = db.execute(query, {"company_id": company_id})
    company = result.fetchone()
    if company is None:
        raise HTTPException(status_code=404, detail=f"Company with id {company_id} not found")
    return company

# =============================================================================
# Job Posting Endpoints
# =============================================================================

job_router = APIRouter(prefix="/jobs", tags=["jobs"])

@job_router.get("")
def get_all_job_postings(db: Session = Depends(get_db)):
    try:
        result = db.execute(text('SELECT * FROM "JobPosting"'))
        rows = result.fetchall()
        
        if not rows:
            return {"message": "No job postings found"}

        output = []
        for row in rows:
            output.append(dict(row._mapping))

        return output
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@job_router.post("", response_model=JobPosting)
def create_job_posting(job: JobPostingCreate, db: Session = Depends(get_db)):
    try:
        query = text("""
            INSERT INTO "JobPosting" (
                title, company_id, compensation_min, compensation_max, 
                location_type, employment_type
            )
            VALUES (
                :title, :company_id, :compensation_min, :compensation_max, 
                :location_type, :employment_type
            )
            RETURNING id, title, company_id, compensation_min, compensation_max, 
                      location_type, employment_type
        """)
        
        result = db.execute(
            query,
            {
                "title": job.title,
                "company_id": job.company_id,
                "compensation_min": job.compensation_min,
                "compensation_max": job.compensation_max,
                "location_type": job.location_type,
                "employment_type": job.employment_type
            }
        )
        
        db.commit()
        created_job = result.fetchone()
        
        if created_job is None:
            raise HTTPException(status_code=500, detail="Failed to create job posting")
            
        return dict(created_job._mapping)
        
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@job_router.patch("/{job_id}", response_model=JobPosting)
def update_job_posting(job_id: int, job_update: JobPostingUpdate, db: Session = Depends(get_db)):
    try:
        # First check if the job posting exists
        existing_job = get_job_posting(db, job_id)
        
        # Build update query dynamically based on provided fields
        update_fields = {}
        update_parts = []
        
        if job_update.title is not None:
            update_fields["title"] = job_update.title
            update_parts.append("title = :title")
        if job_update.company_id is not None:
            update_fields["company_id"] = job_update.company_id
            update_parts.append("company_id = :company_id")
        if job_update.compensation_min is not None:
            update_fields["compensation_min"] = job_update.compensation_min
            update_parts.append("compensation_min = :compensation_min")
        if job_update.compensation_max is not None:
            update_fields["compensation_max"] = job_update.compensation_max
            update_parts.append("compensation_max = :compensation_max")
        if job_update.location_type is not None:
            update_fields["location_type"] = job_update.location_type
            update_parts.append("location_type = :location_type")
        if job_update.employment_type is not None:
            update_fields["employment_type"] = job_update.employment_type
            update_parts.append("employment_type = :employment_type")
            
        if not update_fields:
            return dict(existing_job._mapping)
            
        # Add job_id to update_fields for the WHERE clause
        update_fields["job_id"] = job_id
        
        # Create and execute update query
        update_query = text(f"""
            UPDATE "JobPosting"
            SET {", ".join(update_parts)}
            WHERE id = :job_id
            RETURNING id, title, company_id, compensation_min, compensation_max, 
                      location_type, employment_type
        """)
        
        result = db.execute(update_query, update_fields)
        db.commit()
        
        updated_job = result.fetchone()
        return dict(updated_job._mapping)
        
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@job_router.delete("/{job_id}")
def delete_job_posting(job_id: int, db: Session = Depends(get_db)):
    try:
        # First check if the job posting exists
        get_job_posting(db, job_id)
        
        # Delete the job posting
        query = text('DELETE FROM "JobPosting" WHERE id = :job_id')
        db.execute(query, {"job_id": job_id})
        db.commit()
        
        return {"message": f"Job posting with id {job_id} has been deleted"}
        
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@job_router.post("/{job_id}/description", response_model=dict)
def generate_job_description(
    job_id: int = Path(..., description="Job ID"), 
    tools: ToolsInput = None, 
    db: Session = Depends(get_db)
):
    try:
        # Retrieve job posting
        job = get_job_posting(db, job_id)
        job_dict = dict(job._mapping)

        # Retrieve company information
        company = get_company(db, job_dict['company_id'])
        company_dict = dict(company._mapping)

        # Prepare tools list
        tool_list = tools.required_tools if tools and tools.required_tools else []

        # Construct prompt for GPT
        tool_str = ", ".join(tool_list) if tool_list else "Not specified"
        prompt = (
            f"Write a detailed job description for the following job:\n\n"
            f"Title: {job_dict.get('title', 'N/A')}\n"
            f"Company: {company_dict.get('name', 'N/A')} ({company_dict.get('industry', 'N/A')})\n"
            f"Location Type: {job_dict.get('location_type', 'N/A')}\n"
            f"Employment Type: {job_dict.get('employment_type', 'N/A')}\n"
            f"Compensation Range: ${job_dict.get('compensation_min', 'N/A')} - ${job_dict.get('compensation_max', 'N/A')}\n"
            f"Required Tools: {tool_str}\n\n"
            f"Make it professional and informative."
        )

        # Generate description using OpenAI (new v1.x syntax)
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional job description writer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            job_description = response.choices[0].message.content.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

        # Update job posting with generated description
        update_query = text("""
            UPDATE "JobPosting"
            SET description = :description
            WHERE id = :job_id
            RETURNING id, description
        """)
        
        result = db.execute(
            update_query, 
            {
                "description": job_description, 
                "job_id": job_id
            }
        )
        db.commit()

        return {
            "job_id": job_id, 
            "description": job_description,
            "tools": tool_list
        }
        
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@job_router.patch("/{job_id}/description", response_model=dict)
def update_job_description(
    job_id: int = Path(..., description="Job ID"),
    description_update: JobDescriptionUpdate = None,
    db: Session = Depends(get_db)
):
    """
    Update the job description for a specific job posting
    """
    try:
        # First check if the job posting exists
        get_job_posting(db, job_id)
        
        # Update the job description
        update_query = text("""
            UPDATE "JobPosting"
            SET description = :description
            WHERE id = :job_id
            RETURNING id, title, description
        """)
        
        result = db.execute(
            update_query, 
            {
                "description": description_update.description, 
                "job_id": job_id
            }
        )
        db.commit()
        
        updated_job = result.fetchone()
        
        if updated_job is None:
            raise HTTPException(status_code=500, detail="Failed to update job description")
        
        return {
            "job_id": job_id,
            "title": updated_job.title,
            "description": updated_job.description,
            "message": "Job description updated successfully"
        }
        
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@job_router.get("/openai-test")
async def test_openai_connection():
    try:
        # Verify API key is set
        if not os.getenv("OPENAI_API_KEY"):
            return {"error": "OpenAI API key is not set"}

        # Test OpenAI API connection (new v1.x syntax)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'API connection successful'"}
            ],
            max_tokens=50
        )

        # Extract the response message
        ai_response = response.choices[0].message.content.strip()
        
        return {
            "status": "success", 
            "message": ai_response,
            "model": "gpt-3.5-turbo"
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": str(e)
        }

@job_router.get("/check-api-key")
async def check_api_key_status():
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return {
            "status": "not_set",
            "message": "OpenAI API key is not set in environment variables"
        }
    
    # Check if key looks valid (starts with sk- and has reasonable length)
    if api_key.startswith("sk-") and len(api_key) > 40:
        return {
            "status": "set",
            "message": "OpenAI API key appears to be set correctly",
            "key_preview": f"{api_key[:7]}...{api_key[-4:]}"  # Show only first 7 and last 4 chars
        }
    else:
        return {
            "status": "invalid_format",
            "message": "API key is set but doesn't appear to have correct format"
        }

# =============================================================================
# Company Endpoints
# =============================================================================

company_router = APIRouter(prefix="/company", tags=["company"])

@company_router.get("")
def get_all_companies(db: Session = Depends(get_db)):
    try:
        logger.info("Attempting to fetch all companies")
        result = db.execute(text('SELECT * FROM "Company"'))
        rows = result.fetchall()
        
        if not rows:
            logger.info("No companies found")
            return []

        companies = [dict(row._mapping) for row in rows]
        logger.info(f"Successfully fetched {len(companies)} companies")
        return companies
    except SQLAlchemyError as e:
        logger.error(f"Database error in get_all_companies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in get_all_companies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@company_router.get("/{company_id}", response_model=Company)
def get_company_by_id(company_id: int, db: Session = Depends(get_db)):
    try:
        company = get_company(db, company_id)
        return dict(company._mapping)
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@company_router.post("", response_model=Company)
def create_company(company: CompanyCreate, db: Session = Depends(get_db)):
    try:
        logger.info(f"Attempting to create company: {company.name}")
        query = text("""
            INSERT INTO "Company" (
                name, industry, url, headcount, 
                country, state, city, is_public
            )
            VALUES (
                :name, :industry, :url, :headcount,
                :country, :state, :city, :is_public
            )
            RETURNING id, name, industry, url, headcount,
                      country, state, city, is_public
        """)
        
        result = db.execute(
            query,
            {
                "name": company.name,
                "industry": company.industry,
                "url": company.url,
                "headcount": company.headcount,
                "country": company.country,
                "state": company.state,
                "city": company.city,
                "is_public": company.is_public
            }
        )
        
        db.commit()
        created_company = result.fetchone()
        
        if created_company is None:
            logger.error("Failed to create company - no row returned")
            raise HTTPException(status_code=500, detail="Failed to create company")
            
        company_dict = dict(created_company._mapping)
        logger.info(f"Successfully created company with id: {company_dict['id']}")
        return company_dict
        
    except SQLAlchemyError as e:
        logger.error(f"Database error in create_company: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in create_company: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@company_router.patch("/{company_id}", response_model=Company)
def update_company(company_id: int, company_update: CompanyUpdate, db: Session = Depends(get_db)):
    try:
        # First check if the company exists
        existing_company = get_company(db, company_id)
        
        # Build update query dynamically based on provided fields
        update_fields = {}
        update_parts = []
        
        if company_update.name is not None:
            update_fields["name"] = company_update.name
            update_parts.append("name = :name")
        if company_update.industry is not None:
            update_fields["industry"] = company_update.industry
            update_parts.append("industry = :industry")
        if company_update.url is not None:
            update_fields["url"] = company_update.url
            update_parts.append("url = :url")
        if company_update.headcount is not None:
            update_fields["headcount"] = company_update.headcount
            update_parts.append("headcount = :headcount")
        if company_update.country is not None:
            update_fields["country"] = company_update.country
            update_parts.append("country = :country")
        if company_update.state is not None:
            update_fields["state"] = company_update.state
            update_parts.append("state = :state")
        if company_update.city is not None:
            update_fields["city"] = company_update.city
            update_parts.append("city = :city")
        if company_update.is_public is not None:
            update_fields["is_public"] = company_update.is_public
            update_parts.append("is_public = :is_public")
            
        if not update_fields:
            return dict(existing_company._mapping)
            
        # Add company_id to update_fields for the WHERE clause
        update_fields["company_id"] = company_id
        
        # Create and execute update query
        update_query = text(f"""
            UPDATE "Company"
            SET {", ".join(update_parts)}
            WHERE id = :company_id
            RETURNING id, name, industry, url, headcount,
                      country, state, city, is_public
        """)
        
        result = db.execute(update_query, update_fields)
        db.commit()
        
        updated_company = result.fetchone()
        return dict(updated_company._mapping)
        
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@company_router.delete("/{company_id}")
def delete_company(company_id: int, db: Session = Depends(get_db)):
    try:
        # First check if the company exists
        get_company(db, company_id)
        
        # Delete the company
        query = text('DELETE FROM "Company" WHERE id = :company_id')
        db.execute(query, {"company_id": company_id})
        db.commit()
        
        return {"message": f"Company with id {company_id} has been deleted"}
        
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}") 