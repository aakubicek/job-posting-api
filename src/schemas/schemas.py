from pydantic import BaseModel
from typing import Optional, List

# =============================================================================
# Candidate Schemas
# =============================================================================

class Candidate(BaseModel):
    candidate_id: str 
    name: str 
    email: str 
    job_id: Optional[str] = None

# =============================================================================
# Company Schemas
# =============================================================================

class CompanyBase(BaseModel):
    name: str
    industry: str
    url: str
    headcount: int
    country: str
    state: str
    city: str
    is_public: bool

class CompanyCreate(CompanyBase):
    pass

class CompanyUpdate(BaseModel):
    name: Optional[str] = None
    industry: Optional[str] = None
    url: Optional[str] = None
    headcount: Optional[int] = None
    country: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    is_public: Optional[bool] = None

class Company(CompanyBase):
    id: int
    
    class Config:
        from_attributes = True

# =============================================================================
# Job Description Schemas
# =============================================================================

class ToolsInput(BaseModel):
    required_tools: List[str]

class JobDescriptionUpdate(BaseModel):
    description: str

# =============================================================================
# Job Posting Schemas
# =============================================================================

class JobPostingBase(BaseModel):
    title: str
    company_id: int
    compensation_min: Optional[int] = None
    compensation_max: Optional[int] = None
    location_type: str
    employment_type: str
    description: Optional[str] = None

class JobPostingCreate(JobPostingBase):
    pass

class JobPostingUpdate(BaseModel):
    title: Optional[str] = None
    company_id: Optional[int] = None
    compensation_min: Optional[int] = None
    compensation_max: Optional[int] = None
    location_type: Optional[str] = None
    employment_type: Optional[str] = None
    description: Optional[str] = None

class JobPosting(JobPostingBase):
    id: int
    
    class Config:
        from_attributes = True 