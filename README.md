# Job Posting Management API

## Project Overview
This is a FastAPI-based backend service for managing job postings and companies, using SQLAlchemy for database interactions with Supabase PostgreSQL.

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

4. Run the application
```bash
uvicorn src.main:app --reload
```

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
The application uses a Supabase PostgreSQL database. Ensure you have the correct connection string in `src/models/database.py`.

## Logging
Logging is configured to provide information about database connections and API operations.

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 