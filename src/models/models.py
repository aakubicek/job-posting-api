from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLAlchemy setup 
DATABASE_URL = "postgresql://postgres.jxepeqqrywwgjdqmdsot:supabasepassword123@aws-0-us-east-2.pooler.supabase.com:5432/postgres"

# Create engine with connection pooling settings
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True,
    pool_recycle=1800,
)

# Add event listeners to handle connection issues
@event.listens_for(engine, 'connect')
def connect(dbapi_connection, connection_record):
    logger.info('Connection established')

@event.listens_for(engine, 'checkout')
def checkout(dbapi_connection, connection_record, connection_proxy):
    logger.info('Connection retrieved from pool')

@event.listens_for(engine, 'checkin')
def checkin(dbapi_connection, connection_record):
    logger.info('Connection returned to pool')

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 