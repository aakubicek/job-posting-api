import httpx
import os
import json
import traceback

# Base URL for local FastAPI server
BASE_URL = "http://127.0.0.1:8000"

def test_create_company():
    """
    Test creating a new company
    """
    company_data = {
        "name": "Tech Innovations Inc.",
        "industry": "Software Development",
        "url": "https://techinnovations.com",
        "headcount": 50,
        "country": "USA",
        "state": "California", 
        "city": "San Francisco",
        "is_public": False
    }
    
    try:
        response = httpx.post(f"{BASE_URL}/company", json=company_data)
        print("Create Company Response:")
        print(f"Status Code: {response.status_code}")
        response_json = response.json()
        print("Response Body:", response_json)
        return response_json
    except Exception as e:
        print(f"Error creating company: {e}")
        return None

def test_get_companies():
    """
    Test retrieving all companies
    """
    try:
        response = httpx.get(f"{BASE_URL}/company")
        print("\nGet Companies Response:")
        print(f"Status Code: {response.status_code}")
        companies = response.json()
        print(f"Number of companies: {len(companies)}")
        for company in companies:
            print(f"Company ID: {company.get('id')}, Name: {company.get('name')}")
        return companies
    except Exception as e:
        print(f"Error getting companies: {e}")
        traceback.print_exc()
        return None

def test_create_job_posting(company_id):
    """
    Test creating a new job posting
    """
    job_data = {
        "title": "Senior Software Engineer",
        "company_id": company_id,
        "compensation_min": 120000,
        "compensation_max": 150000,
        "location_type": "Remote",
        "employment_type": "Full-time"
    }
    
    try:
        response = httpx.post(f"{BASE_URL}/jobs", json=job_data)
        print("\nCreate Job Posting Response:")
        print(f"Status Code: {response.status_code}")
        response_json = response.json()
        print("Response Body:", response_json)
        return response_json
    except Exception as e:
        print(f"Error creating job posting: {e}")
        return None

def test_get_job_postings():
    """
    Test retrieving all job postings
    """
    try:
        response = httpx.get(f"{BASE_URL}/jobs")
        print("\nGet Job Postings Response:")
        print(f"Status Code: {response.status_code}")
        jobs = response.json()
        print(f"Number of job postings: {len(jobs)}")
        for job in jobs:
            print(f"Job ID: {job.get('id')}, Title: {job.get('title')}")
        return jobs
    except Exception as e:
        print(f"Error getting job postings: {e}")
        traceback.print_exc()
        return None

def test_generate_job_description(job_id):
    """
    Test generating a job description
    """
    description_data = {
        "required_tools": ["Python", "PostgreSQL"],
        "company_culture": "Innovative and collaborative environment"
    }
    
    try:
        response = httpx.post(
            f"{BASE_URL}/jobs/{job_id}/description", 
            json=description_data,
            timeout=30.0  # Increase timeout to 30 seconds
        )
        print("\nGenerate Job Description Response:")
        print(f"Status Code: {response.status_code}")
        
        # Check for error response
        if response.status_code != 200:
            print("Error Response:")
            print(response.text)
            return None
        
        # Pretty print the JSON response
        response_json = response.json()
        print(json.dumps(response_json, indent=2))
        return response_json
    except Exception as e:
        print(f"Error generating job description: {e}")
        traceback.print_exc()
        return None

def test_check_openai_key():
    """
    Test checking OpenAI API key status
    """
    try:
        response = httpx.get(f"{BASE_URL}/jobs/check-api-key")
        print("\nCheck OpenAI API Key Response:")
        print(f"Status Code: {response.status_code}")
        response_json = response.json()
        print(json.dumps(response_json, indent=2))
        return response_json
    except Exception as e:
        print(f"Error checking OpenAI API key: {e}")
        traceback.print_exc()
        return None

def main():
    print("Starting endpoint tests...")
    
    # Check OpenAI API key status
    openai_key_status = test_check_openai_key()
    
    # Test retrieving companies
    companies = test_get_companies()
    if not companies:
        print("Failed to retrieve companies.")
        return
    
    # Use the first company ID for job posting
    company_id = companies[0].get('id')
    print(f"\nUsing Company ID: {company_id}")
    
    # Test job postings
    job_postings = test_get_job_postings()
    if not job_postings:
        print("Failed to retrieve job postings.")
        return
    
    # Use the first job posting ID for description generation
    job_id = job_postings[0].get('id')
    print(f"\nUsing Job ID: {job_id}")
    
    # Test job description generation only if API key is set
    if openai_key_status and openai_key_status.get('status') == 'set':
        test_generate_job_description(job_id)
    else:
        print("\nSkipping job description generation - OpenAI API key not set.")

if __name__ == "__main__":
    main() 