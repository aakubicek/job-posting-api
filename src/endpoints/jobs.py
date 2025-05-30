from fastapi import APIRouter, Depends, HTTPException, Path, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import logging
import os
from typing import Optional, Dict, Any
import json
import signal
import contextlib
import threading
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.models.models import get_db
from src.schemas.schemas import (
    JobPostingCreate, JobPostingUpdate, JobPosting, ToolsInput, JobDescriptionUpdate,
    CompanyCreate, CompanyUpdate, Company, StructuredJobDescription, JobDescriptionSection,
    CompensationDetails, WorkArrangementDetails, TechnicalSkillsSection
)
from src.exceptions import (
    InvalidJobIDError, DatabaseConnectionError, AIModelError, 
    ValidationError, OpenAIConfigurationError, JobDescriptionException, 
    handle_job_description_exception
)

# Create routers first
job_router = APIRouter(prefix="/jobs", tags=["jobs"])
company_router = APIRouter(prefix="/company", tags=["company"])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure LangChain OpenAI Chat Model
def get_chat_model(
    model: str = "gpt-4o-mini", 
    temperature: float = 0.7, 
    max_tokens: int = 500
):
    """
    Create a configurable LangChain chat model with flexible parameters
    
    Args:
        model (str): The OpenAI model to use
        temperature (float): Sampling temperature for text generation
        max_tokens (int): Maximum number of tokens to generate
    
    Returns:
        ChatOpenAI: Configured LangChain chat model
    """
    # Validate OpenAI configuration
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        raise OpenAIConfigurationError("Invalid or missing OpenAI API key")

    try:
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs={
                "top_p": 1.0,  # Nucleus sampling parameter
                "frequency_penalty": 0.0,  # Reduce repetition
                "presence_penalty": 0.0  # Encourage diversity
            }
        )
    except Exception as e:
        raise AIModelError(f"Failed to initialize chat model: {str(e)}")

# Create a comprehensive job description prompt template
def create_job_description_prompt() -> ChatPromptTemplate:
    """
    Create a structured prompt template for job description generation
    
    Returns:
        ChatPromptTemplate: A comprehensive prompt template for job descriptions
    """
    template = """
You are an expert technical recruiter creating a structured job description.

CRITICAL INSTRUCTIONS:
1. ALWAYS respond in the EXACT JSON structure provided
2. ALL fields MUST be populated
3. If information is missing, use professional, generic defaults
4. Be concise but informative

Job Details:
- Job Title: {job_title}
- Company: {company_name}
- Industry: {company_industry}
- Location: {location_type}
- Employment: {employment_type}
- Compensation: ${compensation_min} - ${compensation_max}

Required Tools: {required_tools}
Company Culture: {company_culture}

MANDATORY JSON STRUCTURE (MUST MATCH EXACTLY):
{{
    "job_title": "{job_title}",
    "company": {{
        "name": "{company_name}",
        "industry": "{company_industry}",
        "location": "{location_type}"
    }},
    "overview": {{
        "title": "Role Overview for {job_title}",
        "content": "Comprehensive description of the {job_title} role at {company_name}, highlighting its strategic importance and key objectives."
    }},
    "job_details": {{
        "department": "Engineering",
        "reporting_to": "Senior Engineering Manager",
        "team_size": "5-10 engineers"
    }},
    "responsibilities": [
        {{
            "title": "Core Technical Responsibilities",
            "content": "Design, develop, and maintain high-quality software solutions using {required_tools_str}"
        }},
        {{
            "title": "System Architecture",
            "content": "Contribute to system design, performance optimization, and technical architecture"
        }},
        {{
            "title": "Team Collaboration",
            "content": "Work closely with cross-functional teams, participate in code reviews, and maintain clear technical documentation"
        }}
    ],
    "requirements": {{
        "technical_skills": {{
            "minimum_experience": "3-5 years in software development",
            "education": "Bachelor's degree in Computer Science, Software Engineering, or related technical field",
            "required_skills": "{required_tools_str}"
        }},
        "soft_skills": [
            "Strong problem-solving abilities",
            "Excellent communication skills",
            "Team collaboration",
            "Adaptability and continuous learning"
        ]
    }},
    "technical_skills": {{
        "required_tools": {required_tools},
        "programming_languages": ["Relevant to {required_tools_str}"],
        "frameworks": ["Relevant to {required_tools_str}"]
    }},
    "compensation": {{
        "base_range": "${compensation_min} - ${compensation_max}",
        "equity": "Potential equity compensation available",
        "bonuses": ["Performance-based annual bonus"]
    }},
    "work_arrangement": {{
        "location_type": "{location_type}",
        "employment_type": "{employment_type}",
        "work_hours": "Standard full-time hours",
        "travel_requirements": "Minimal travel expected"
    }},
    "benefits": [
        {{
            "title": "Competitive Compensation",
            "content": "Salary range: ${compensation_min} - ${compensation_max} with performance-based bonuses"
        }},
        {{
            "title": "Professional Development",
            "content": "Ongoing training, conference attendance, and learning opportunities"
        }}
    ],
    "career_growth": {{
        "mentorship_programs": ["Peer mentorship", "Senior leadership guidance"],
        "learning_resources": ["Online learning platforms", "Technical conference sponsorship"],
        "career_progression": ["Clear promotion paths", "Regular performance reviews"]
    }},
    "additional_context": {{
        "company_culture": "{company_culture}",
        "diversity_commitment": "Committed to building an inclusive workplace"
    }}
}}

IMPORTANT: 
- Ensure ALL placeholders are replaced
- Use professional language
- Provide realistic, engaging descriptions
- Match the exact JSON structure
"""

    return ChatPromptTemplate.from_template(template)

# Job Posting Retrieval with Enhanced Error Handling
def safe_get_job_posting(db: Session, job_id: int, table_name: str = "JobPosting"):
    """
    Safely retrieve a job posting or company with detailed error handling
    
    Args:
        db (Session): Database session
        job_id (int): ID of the job posting or company
        table_name (str): Name of the table to query (default: "JobPosting")
    
    Returns:
        dict: Job posting or company details
    
    Raises:
        InvalidJobIDError: If no record is found
        DatabaseConnectionError: If database query fails
    """
    try:
        query = text(f'SELECT * FROM "{table_name}" WHERE id = :record_id')
        result = db.execute(query, {"record_id": job_id})
        record = result.fetchone()
        
        if record is None:
            raise InvalidJobIDError(job_id)
        
        return dict(record._mapping)
    except InvalidJobIDError:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving {table_name} {job_id}: {str(e)}")
        raise DatabaseConnectionError(e)
    except Exception as e:
        logger.error(f"Unexpected error retrieving {table_name} {job_id}: {str(e)}")
        raise DatabaseConnectionError(e)

def timeout_function(func, *args, timeout=10, **kwargs):
    """
    Run a function with a timeout, compatible with Windows
    
    Args:
        func (callable): Function to run
        timeout (int): Timeout in seconds
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
    
    Returns:
        Result of the function or raises TimeoutError
    """
    result = [None]
    error = [None]
    
    def wrapper():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            error[0] = e
    
    thread = threading.Thread(target=wrapper)
    thread.daemon = True  # Ensure thread doesn't block program exit
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        # Attempt to cancel the thread
        try:
            # This is a best-effort attempt, not guaranteed to work
            import ctypes
            thread_id = thread.ident
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread_id), 
                ctypes.py_object(SystemExit)
            )
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_long(thread_id), 
                    None
                )
        except Exception:
            pass
        
        raise TimeoutError(f"Function timed out after {timeout} seconds")
    
    if error[0] is not None:
        raise error[0]
    
    return result[0]

@job_router.post("/{job_id}/description", response_model=StructuredJobDescription)
def generate_job_description(
    job_id: int = Path(..., description="Job ID"), 
    tools: Optional[ToolsInput] = None, 
    temperature: float = 0.7,
    max_tokens: int = 500,
    db: Session = Depends(get_db)
):
    """
    Generate a structured job description with comprehensive error handling
    
    Args:
        job_id (int): ID of the job posting
        tools (Optional[ToolsInput]): Optional tools and context
        temperature (float): AI model creativity temperature
        max_tokens (int): Maximum tokens for generation
        db (Session): Database session
    
    Returns:
        StructuredJobDescription: Detailed, structured job description
    
    Raises:
        InvalidJobIDError: If job ID is invalid
        DatabaseConnectionError: If database connection fails
        AIModelError: If AI model generation fails
        ValidationError: If input validation fails
    """
    try:
        # Validate input parameters
        if temperature < 0 or temperature > 1:
            raise ValidationError({
                "temperature": "Must be between 0 and 1"
            })
        
        if max_tokens < 50 or max_tokens > 1000:
            raise ValidationError({
                "max_tokens": "Must be between 50 and 1000"
            })

        # Retrieve job posting with enhanced error handling
        job_dict = safe_get_job_posting(db, job_id)

        # Retrieve company information
        company_dict = safe_get_job_posting(db, job_dict['company_id'], table_name="Company")

        # Prepare tools list and company culture
        # Ensure tools is a list, even if it was passed as a string
        tool_list = []
        if tools:
            # Log the raw input for debugging
            logger.info(f"Raw tools input: {tools}")
            logger.info(f"Raw required_tools: {tools.required_tools}")
            logger.info(f"Type of required_tools: {type(tools.required_tools)}")

            # Handle various input types for required_tools
            if tools.required_tools is None:
                tool_list = []
            elif isinstance(tools.required_tools, str):
                # Try multiple parsing strategies
                try:
                    # First, try direct JSON parsing
                    parsed_tools = json.loads(tools.required_tools)
                    # Ensure it's a list
                    tool_list = parsed_tools if isinstance(parsed_tools, list) else [parsed_tools]
                except (json.JSONDecodeError, TypeError):
                    # If JSON parsing fails, try splitting
                    tool_list = [tool.strip() for tool in tools.required_tools.split(',')]
            elif isinstance(tools.required_tools, list):
                tool_list = tools.required_tools
            else:
                # Fallback to empty list
                tool_list = []

        # Ensure tool_list contains only strings and is not empty
        tool_list = [str(tool).strip() for tool in tool_list if tool]
        
        # If tool_list is empty, use a default
        if not tool_list:
            tool_list = ["No specific tools required"]

        # Log the final tool list for debugging
        logger.info(f"Processed tool list: {tool_list}")

        company_culture = tools.company_culture if tools and tools.company_culture else "Collaborative, innovative, and supportive work environment"

        # Validate tools
        if tool_list and tool_list != ["No specific tools required"]:
            invalid_tools = [tool for tool in tool_list if tool not in {
                'Python', 'JavaScript', 'TypeScript', 'React', 'Vue', 'Angular', 
                'Node.js', 'Django', 'Flask', 'FastAPI', 'SQLAlchemy', 'PostgreSQL'
            }]
            if invalid_tools:
                raise ValidationError({
                    "required_tools": f"Invalid tools: {', '.join(invalid_tools)}"
                })

        # Initialize chat model with dynamic parameters
        try:
            chat_model = get_chat_model(
                model="gpt-4o-mini", 
                temperature=temperature, 
                max_tokens=max_tokens
            )
        except OpenAIConfigurationError as e:
            logger.error(f"OpenAI configuration error: {str(e)}")
            raise

        # Create prompt template
        prompt_template = create_job_description_prompt()

        # Prepare input variables for the prompt
        tool_list = tools.required_tools if tools and tools.required_tools else []
        
        # Ensure tool_list is a list of strings
        if isinstance(tool_list, str):
            try:
                tool_list = json.loads(tool_list)
            except json.JSONDecodeError:
                tool_list = [tool.strip() for tool in tool_list.split(',')]
        
        # Ensure tool_list is a list of strings
        tool_list = [str(tool).strip() for tool in tool_list if tool]
        
        # If tool_list is empty, use a default
        if not tool_list:
            tool_list = ["No specific tools required"]

        input_variables = {
            "job_title": job_dict.get('title', 'N/A'),
            "company_name": company_dict.get('name', 'N/A'),
            "company_industry": company_dict.get('industry', 'N/A'),
            "location_type": job_dict.get('location_type', 'N/A'),
            "employment_type": job_dict.get('employment_type', 'N/A'),
            "compensation_min": str(job_dict.get('compensation_min', 'N/A')),
            "compensation_max": str(job_dict.get('compensation_max', 'N/A')),
            "required_tools": json.dumps(tool_list) if tool_list else '["No specific tools required"]',
            "required_tools_str": ", ".join(tool_list) if tool_list else "No specific tools required",
            "company_culture": company_culture
        }

        # Modify the fallback job description creation to use a list
        def create_default_job_description(input_variables):
            """
            Create a default job description when AI generation fails
            """
            return StructuredJobDescription(
                job_title=input_variables.get('job_title', 'Untitled Job'),
                company={
                    "name": input_variables.get('company_name', 'Unknown Company'),
                    "industry": input_variables.get('company_industry', 'Technology'),
                    "location": input_variables.get('location_type', 'Not Specified')
                },
                overview=JobDescriptionSection(
                    title=f"Role Overview for {input_variables.get('job_title', 'Job')}",
                    content=f"A comprehensive role in {input_variables.get('company_name', 'our organization')} "
                           f"focusing on {input_variables.get('job_title', 'key responsibilities')}"
                ),
                job_details={
                    "department": "Engineering",
                    "reporting_to": "Senior Management",
                    "team_size": "5-10 people"
                },
                responsibilities=[
                    JobDescriptionSection(
                        title="Core Technical Responsibilities",
                        content=f"Develop and maintain software solutions using {', '.join(input_variables.get('required_tools', ['various technologies']))}"
                    ),
                    JobDescriptionSection(
                        title="System Design",
                        content="Contribute to system architecture and design"
                    ),
                    JobDescriptionSection(
                        title="Team Collaboration",
                        content="Work effectively with cross-functional teams"
                    )
                ],
                requirements={
                    "technical_skills": {
                        "minimum_experience": "3-5 years in software development",
                        "education": "Bachelor's degree in Computer Science or related field",
                        "required_skills": ', '.join(input_variables.get('required_tools', ['Not specified']))
                    },
                    "soft_skills": [
                        "Strong communication",
                        "Problem-solving",
                        "Team collaboration"
                    ]
                },
                technical_skills=TechnicalSkillsSection(
                    required_tools=tool_list
                ),
                compensation=CompensationDetails(
                    base_range=f"${input_variables.get('compensation_min', 'N/A')} - ${input_variables.get('compensation_max', 'N/A')}"
                ),
                work_arrangement=WorkArrangementDetails(
                    location_type=input_variables.get('location_type', 'Not specified'),
                    employment_type=input_variables.get('employment_type', 'Not specified')
                ),
                benefits=[
                    JobDescriptionSection(
                        title="Competitive Compensation",
                        content=f"Salary range: ${input_variables.get('compensation_min', 'N/A')} - ${input_variables.get('compensation_max', 'N/A')}"
                    )
                ]
            )

        # Modify the job description generation to use fallback
        try:
            # Use a custom timeout function
            structured_chain = prompt_template | chat_model.with_structured_output(
                StructuredJobDescription, 
                include_raw=False,
                method="function_calling"
            )

            # Prepare a more complete job description with timeout
            try:
                job_description_data = timeout_function(
                    structured_chain.invoke, 
                    input_variables, 
                    timeout=10
                )
            except Exception as e:
                logger.warning(f"AI generation failed, using default job description: {str(e)}")
                job_description_data = create_default_job_description(input_variables)

            # Ensure all required fields are populated
            if not job_description_data.company:
                job_description_data.company = {
                    "name": input_variables.get('company_name', 'Unknown Company'),
                    "industry": input_variables.get('company_industry', 'Technology'),
                    "location": input_variables.get('location_type', 'Not Specified')
                }

            if not job_description_data.job_details:
                job_description_data.job_details = {
                    "department": "Engineering",
                    "reporting_to": "Senior Management",
                    "team_size": "5-10 people"
                }

            if not job_description_data.requirements:
                job_description_data.requirements = {
                    "technical_skills": {
                        "minimum_experience": "3-5 years in software development",
                        "education": "Bachelor's degree in Computer Science or related field",
                        "required_skills": ', '.join(input_variables.get('required_tools', ['Not specified']))
                    },
                    "soft_skills": [
                        "Strong communication",
                        "Problem-solving",
                        "Team collaboration"
                    ]
                }

        except TimeoutError:
            logger.error("Job description generation timed out")
            raise AIModelError("Job description generation timed out")
        except Exception as e:
            logger.error(f"AI model generation error: {str(e)}")
            raise AIModelError(str(e))

        # Update job posting with generated description
        try:
            update_query = text("""
                UPDATE "JobPosting"
                SET description = :description
                WHERE id = :job_id
                RETURNING id, description
            """)
            
            result = db.execute(
                update_query, 
                {
                    "description": job_description_data.model_dump_json(), 
                    "job_id": job_id
                }
            )
            db.commit()
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Database update error: {str(e)}")
            raise DatabaseConnectionError(e)

        logger.info(f"Successfully generated job description for job ID {job_id}")
        return job_description_data
        
    except (InvalidJobIDError, DatabaseConnectionError, AIModelError, 
            ValidationError, OpenAIConfigurationError) as e:
        logger.error(f"Job description generation error: {str(e)}")
        raise

# =============================================================================
# Job Posting Endpoints
# =============================================================================

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
        existing_job = safe_get_job_posting(db, job_id)
        
        # Prepare update dictionary with only non-None values
        update_data = job_update.model_dump(exclude_unset=True)
        
        # If no update fields are provided, return existing job
        if not update_data:
            return existing_job
        
        # Validate update data types
        valid_columns = {
            'title', 'company_id', 'compensation_min', 'compensation_max', 
            'location_type', 'employment_type'
        }
        
        # Filter out any keys not in valid columns
        filtered_update_data = {
            key: value for key, value in update_data.items() 
            if key in valid_columns
        }
        
        if not filtered_update_data:
            logger.warning(f"No valid update fields for job {job_id}")
            return existing_job
        
        # Prepare the update query dynamically
        update_parts = [f'"{key}" = :{key}' for key in filtered_update_data.keys()]
        filtered_update_data['job_id'] = job_id  # Add job_id for WHERE clause
        
        # Create and execute update query
        update_query = text(f"""
            UPDATE "JobPosting"
            SET {", ".join(update_parts)}
            WHERE id = :job_id
            RETURNING id, title, company_id, compensation_min, compensation_max, 
                      location_type, employment_type, description
        """)
        
        try:
            result = db.execute(update_query, filtered_update_data)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Database update error: {str(e)}")
            logger.error(f"Update data: {filtered_update_data}")
            raise HTTPException(status_code=500, detail=f"Database update error: {str(e)}")
        
        updated_job = result.fetchone()
        
        if updated_job is None:
            raise HTTPException(status_code=500, detail="Failed to update job posting")
        
        return dict(updated_job._mapping)
        
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"SQLAlchemy error in job update: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error in job update: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@job_router.delete("/{job_id}")
def delete_job_posting(job_id: int, db: Session = Depends(get_db)):
    try:
        # First check if the job posting exists
        safe_get_job_posting(db, job_id)
        
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
        safe_get_job_posting(db, job_id)
        
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

        # Test LangChain OpenAI connection
        chat_model = get_chat_model(model="gpt-4o-mini")
        
        response = chat_model.invoke("Say 'API connection successful'")
        
        return {
            "status": "success", 
            "message": response.content,
            "model": "gpt-4o-mini"
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
        company = safe_get_job_posting(db, company_id, table_name="Company")
        return company
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
        
        # Validate input data
        if not company.name or not company.industry:
            raise ValidationError({
                "name": "Company name is required",
                "industry": "Industry is required"
            })
        
        # Prepare the input data dictionary explicitly
        input_data = {
            "name": company.name,
            "industry": company.industry,
            "url": company.url or None,
            "headcount": company.headcount or None,
            "country": company.country or None,
            "state": company.state or None,
            "city": company.city or None,
            "is_public": company.is_public or False
        }
        
        # Log the input data for debugging
        logger.info(f"Company creation input data: {input_data}")
        
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
        
        try:
            result = db.execute(query, input_data)
            db.commit()
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"SQLAlchemy error in create_company: {str(e)}")
            logger.error(f"Input data: {input_data}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
        created_company = result.fetchone()
        
        if created_company is None:
            logger.error("Failed to create company - no row returned")
            raise HTTPException(status_code=500, detail="Failed to create company")
        
        # Convert the result to a dictionary, handling potential conversion issues
        try:
            company_dict = dict(created_company._mapping)
        except Exception as e:
            logger.error(f"Failed to convert company result to dictionary: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process company data: {str(e)}")
        
        logger.info(f"Successfully created company with id: {company_dict.get('id')}")
        return company_dict
        
    except ValidationError as ve:
        logger.error(f"Validation error in create_company: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except SQLAlchemyError as e:
        logger.error(f"Database error in create_company: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in create_company: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@company_router.patch("/{company_id}", response_model=Company)
def update_company(company_id: int, company_update: CompanyUpdate, db: Session = Depends(get_db)):
    try:
        # First check if the company exists
        existing_company = safe_get_job_posting(db, company_id)
        
        # Build update query dynamically based on provided fields
        update_fields = {}
        update_parts = []
        
        # Use model_dump() to get a dictionary of non-None values
        update_data = company_update.model_dump(exclude_unset=True)
        
        # Validate and prepare update fields
        valid_fields = {
            'name', 'industry', 'url', 'headcount', 
            'country', 'state', 'city', 'is_public'
        }
        
        for field, value in update_data.items():
            if field in valid_fields and value is not None:
                update_fields[field] = value
                update_parts.append(f'"{field}" = :{field}')
        
        # If no valid update fields, return existing company
        if not update_fields:
            logger.info(f"No valid update fields for company {company_id}")
            return dict(existing_company)
        
        # Add company_id to update_fields for the WHERE clause
        update_fields["company_id"] = company_id
        
        # Log update details for debugging
        logger.info(f"Company update fields: {update_fields}")
        logger.info(f"Company update parts: {update_parts}")
        
        # Create and execute update query
        update_query = text(f"""
            UPDATE "Company"
            SET {", ".join(update_parts)}
            WHERE id = :company_id
            RETURNING id, name, industry, url, headcount,
                      country, state, city, is_public
        """)
        
        try:
            result = db.execute(update_query, update_fields)
            db.commit()
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"SQLAlchemy error in update_company: {str(e)}")
            logger.error(f"Update fields: {update_fields}")
            logger.error(f"Update parts: {update_parts}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
        updated_company = result.fetchone()
        
        if updated_company is None:
            logger.error("Failed to update company - no row returned")
            raise HTTPException(status_code=500, detail="Failed to update company")
        
        # Convert the result to a dictionary
        try:
            company_dict = dict(updated_company._mapping)
        except Exception as e:
            logger.error(f"Failed to convert updated company result to dictionary: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process updated company data: {str(e)}")
        
        logger.info(f"Successfully updated company with id: {company_dict.get('id')}")
        return company_dict
        
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
        safe_get_job_posting(db, company_id)
        
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