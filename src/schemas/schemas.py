from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pydantic import validator

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

class JobDescriptionSection(BaseModel):
    """
    Represents a section of a job description with a title and content
    """
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Detailed content of the section")

class CompensationDetails(BaseModel):
    """
    Detailed compensation information
    """
    base_range: Optional[str] = Field(
        default=None, 
        description="Base salary range"
    )
    equity: Optional[str] = Field(
        default=None, 
        description="Equity compensation details"
    )
    bonuses: Optional[List[str]] = Field(
        default=None, 
        description="Potential bonus structures"
    )

class WorkArrangementDetails(BaseModel):
    """
    Comprehensive work arrangement information
    """
    location_type: Optional[str] = Field(
        default=None, 
        description="Work location type (remote, hybrid, on-site)"
    )
    employment_type: Optional[str] = Field(
        default=None, 
        description="Type of employment (full-time, part-time, contract)"
    )
    work_hours: Optional[str] = Field(
        default=None, 
        description="Expected work hours or flexibility"
    )
    travel_requirements: Optional[str] = Field(
        default=None, 
        description="Travel expectations for the role"
    )

class TechnicalSkillsSection(BaseModel):
    """
    Detailed technical skills and tool requirements
    """
    required_tools: Optional[List[str]] = Field(
        default=None, 
        description="List of required technical tools"
    )
    programming_languages: Optional[List[str]] = Field(
        default=None, 
        description="Programming languages required or preferred"
    )
    frameworks: Optional[List[str]] = Field(
        default=None, 
        description="Frameworks and libraries"
    )
    databases: Optional[List[str]] = Field(
        default=None, 
        description="Database technologies"
    )
    cloud_platforms: Optional[List[str]] = Field(
        default=None, 
        description="Cloud platforms and services"
    )

class CareerGrowthDetails(BaseModel):
    """
    Career development and growth opportunities
    """
    mentorship_programs: Optional[List[str]] = Field(
        default=None, 
        description="Available mentorship opportunities"
    )
    learning_resources: Optional[List[str]] = Field(
        default=None, 
        description="Professional development resources"
    )
    career_progression: Optional[List[str]] = Field(
        default=None, 
        description="Potential career paths"
    )

class StructuredJobDescription(BaseModel):
    """
    Comprehensive structured job description model with nested, logical sections
    """
    # Basic Job Information
    job_title: str = Field(..., description="Official job title")
    company: Dict[str, str] = Field(
        ..., 
        description="Company details",
        example={
            "name": "Tech Innovations Inc.",
            "industry": "Software Development",
            "location": "San Francisco, CA"
        }
    )

    # Detailed Job Description Sections
    overview: JobDescriptionSection = Field(
        ..., 
        description="Brief overview of the job and company"
    )
    
    job_details: Dict[str, Any] = Field(
        ...,
        description="Comprehensive job details",
        example={
            "department": "Engineering",
            "reporting_to": "Senior Engineering Manager",
            "team_size": "10-15 engineers"
        }
    )

    # Structured Responsibilities and Requirements
    responsibilities: List[JobDescriptionSection] = Field(
        ..., 
        description="Key responsibilities of the role",
        min_items=3,
        max_items=7
    )
    
    requirements: Dict[str, Any] = Field(
        ...,
        description="Comprehensive requirements section",
        example={
            "technical_skills": {
                "minimum_experience": "3-5 years",
                "education": "Bachelor's in Computer Science or related field"
            },
            "soft_skills": [
                "Strong communication skills",
                "Team collaboration",
                "Problem-solving ability"
            ]
        }
    )

    # Advanced Skill and Qualification Sections
    technical_skills: Optional[TechnicalSkillsSection] = Field(
        default=None,
        description="Detailed technical skills and tool requirements"
    )

    # Compensation and Benefits
    compensation: Optional[CompensationDetails] = Field(
        default=None,
        description="Comprehensive compensation information"
    )

    # Work Arrangement
    work_arrangement: Optional[WorkArrangementDetails] = Field(
        default=None,
        description="Detailed work arrangement information"
    )

    # Benefits and Perks
    benefits: List[JobDescriptionSection] = Field(
        ..., 
        description="Company benefits and perks",
        min_items=1,
        max_items=5
    )

    # Career Growth
    career_growth: Optional[CareerGrowthDetails] = Field(
        default=None,
        description="Career development and growth opportunities"
    )

    # Additional Context
    additional_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Any additional contextual information"
    )

    # Validation methods
    @validator('responsibilities', 'benefits')
    def validate_sections(cls, sections):
        """
        Validate that sections have meaningful content
        """
        for section in sections:
            if not section.title or not section.content:
                raise ValueError("Each section must have a title and content")
        return sections

# Update ToolsInput to be more flexible
class ToolsInput(BaseModel):
    required_tools: Optional[List[str]] = None
    company_culture: Optional[str] = None
    additional_context: Optional[Dict[str, Any]] = None

    @validator('required_tools', always=True)
    def validate_required_tools(cls, tools):
        # Predefined list of acceptable tools
        ALLOWED_TOOLS = {
            'Python', 'JavaScript', 'TypeScript', 'React', 'Vue', 'Angular', 
            'Node.js', 'Django', 'Flask', 'FastAPI', 'SQLAlchemy', 'PostgreSQL', 
            'MongoDB', 'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP', 
            'Git', 'GitHub', 'GitLab', 'Terraform', 'Ansible', 'Jenkins', 
            'Selenium', 'Jest', 'Pytest', 'GraphQL', 'Redis', 'Kafka'
        }

        if tools is None:
            return None

        # Validate each tool against the allowed list
        invalid_tools = [tool for tool in tools if tool not in ALLOWED_TOOLS]
        
        if invalid_tools:
            raise ValueError(f"Invalid tools: {', '.join(invalid_tools)}. Allowed tools are: {', '.join(ALLOWED_TOOLS)}")
        
        return tools

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