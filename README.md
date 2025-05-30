# Job Posting Management API

## Project Overview
This is a FastAPI-based backend service for managing job postings and companies, using:
- SQLAlchemy for database interactions with Supabase PostgreSQL
- LangChain for AI-powered job description generation
- OpenAI's GPT models with structured output
- Comprehensive error handling and logging

## Structured Job Description

The AI generates a comprehensive, nested job description with multiple logical sections:

### Job Description Structure
```json
{
  "job_title": "Senior Software Engineer",
  "company": {
    "name": "Tech Innovations Inc.",
    "industry": "Software Development",
    "location": "San Francisco, CA"
  },
  "overview": {
    "title": "Role Overview",
    "content": "We're seeking a talented Senior Software Engineer..."
  },
  "job_details": {
    "department": "Engineering",
    "reporting_to": "Senior Engineering Manager",
    "team_size": "10-15 engineers"
  },
  "responsibilities": [
    {
      "title": "Backend Development",
      "content": "Design and implement scalable backend systems..."
    }
  ],
  "requirements": {
    "technical_skills": {
      "minimum_experience": "3-5 years",
      "education": "Bachelor's in Computer Science"
    },
    "soft_skills": [
      "Strong communication",
      "Team collaboration",
      "Problem-solving"
    ]
  },
  "technical_skills": {
    "required_tools": ["Python", "Docker", "Kubernetes"],
    "programming_languages": ["Python", "JavaScript"],
    "frameworks": ["Django", "React"],
    "databases": ["PostgreSQL", "MongoDB"]
  },
  "compensation": {
    "base_range": "$120,000 - $150,000",
    "equity": "Stock options available",
    "bonuses": ["Annual performance bonus"]
  },
  "work_arrangement": {
    "location_type": "Hybrid",
    "employment_type": "Full-time",
    "work_hours": "Flexible 9-5",
    "travel_requirements": "Occasional on-site meetings"
  },
  "benefits": [
    {
      "title": "Health and Wellness",
      "content": "Comprehensive health insurance..."
    }
  ],
  "career_growth": {
    "mentorship_programs": ["Senior engineer mentorship"],
    "learning_resources": ["Annual conference budget"],
    "career_progression": ["Senior Engineer", "Lead Engineer", "Engineering Manager"]
  },
  "additional_context": {
    "diversity_statement": "We are an equal opportunity employer"
  }
}
```

### Key Features
- Nested, logical job description sections
- Comprehensive job and company details
- Flexible technical and soft skill requirements
- Detailed compensation and benefits information
- Career growth opportunities

## Prerequisites
- Python 3.9+
- pip

## Setup Instructions

1. Clone the repository
```bash
git clone <repository-url>
cd <project-directory>
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set OpenAI API Key
```bash
export OPENAI_API_KEY='your-api-key-here'
```

5. Run the application
```bash
uvicorn src.main:app --reload
```

## AI Job Description Generation

### Endpoint: `POST /jobs/{job_id}/description`

#### Parameters
- `job_id`: The ID of the job posting
- `tools` (optional): List of required tools
- `company_culture` (optional): Description of company culture
- `temperature` (optional, default=0.7): Controls randomness of output
- `max_tokens` (optional, default=500): Maximum tokens in generated description

#### Example Request
```json
{
  "required_tools": ["Python", "React", "PostgreSQL"],
  "company_culture": "Innovative, collaborative, fast-paced environment",
  "temperature": 0.6,
  "max_tokens": 600
}
```

## AI Model Configuration
The job description generation uses LangChain with OpenAI's GPT models. You can easily configure:
- Model (default: gpt-4o-mini)
- Temperature (creativity level)
- Maximum tokens
- Additional model parameters

## API Endpoints

### Companies
- `GET /company`: List all companies
- `GET /company/{company_id}`: Get a specific company
- `POST /company`: Create a new company
- `PATCH /company/{company_id}`: Update a company
- `DELETE /company/{company_id}`: Delete a company

### Job Postings
- `GET /jobs`: List all job postings
- `POST /jobs`: Create a new job posting
- `PATCH /jobs/{job_id}`: Update a job posting
- `DELETE /jobs/{job_id}`: Delete a job posting

## Database Configuration
The application uses a Supabase PostgreSQL database.

## Logging
Logging is configured to provide information about database connections and API operations.

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 