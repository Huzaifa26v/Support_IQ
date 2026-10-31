# Import Annotated for reusable FastAPI dependencies.
from typing import Annotated

# Import FastAPI tools for routing, validation,
# dependency injection, and HTTP error handling.
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)

# Import SQLAlchemy Session.
from sqlalchemy.orm import Session

# Import the database session dependency.
from app.db.session import get_db

# Import ticket schemas.
from app.schemas.ticket import TicketCreate, TicketResponse

# Import ticket service business logic.
from app.services import ticket_service


# ==========================================================
# ROUTER CONFIGURATION
# ==========================================================

# Create a router for ticket-related API endpoints.
router = APIRouter(
    prefix="/tickets",
    tags=["Tickets"],
)

# Create a reusable database dependency.
DatabaseSession = Annotated[Session, Depends(get_db)]


# ==========================================================
# 1. CREATE TICKET
# ==========================================================

# POST request creates a new customer support ticket.
@router.post(
    "/",
    response_model=TicketResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_ticket(
    ticket_data: TicketCreate,
    db: DatabaseSession,
):
    try:
        # Call the service layer to validate and create the ticket.
        return ticket_service.create_ticket(
            db,
            ticket_data,
        )

    except ValueError as exc:
        # Return HTTP 400 if business validation fails.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


# ==========================================================
# 2. GET ALL TICKETS
# ==========================================================

# GET request retrieves all tickets with pagination.
@router.get(
    "/",
    response_model=list[TicketResponse],
)
def get_all_tickets(
    db: DatabaseSession,

    # Number of records to skip.
    skip: Annotated[int, Query(ge=0)] = 0,

    # Maximum number of records to retrieve.
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    # Retrieve tickets through the service layer.
    return ticket_service.list_tickets(
        db,
        skip=skip,
        limit=limit,
    )


# ==========================================================
# 3. GET TICKETS BY CUSTOMER
# ==========================================================

# GET request retrieves tickets belonging to one customer.
@router.get(
    "/customer/{customer_id}",
    response_model=list[TicketResponse],
)
def get_customer_tickets(
    customer_id: int,
    db: DatabaseSession,
):
    try:
        # Retrieve the customer's tickets.
        return ticket_service.list_customer_tickets(
            db,
            customer_id,
        )

    except ValueError as exc:
        # Return HTTP 404 if the customer does not exist.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


# ==========================================================
# 4. GET TICKET BY ID
# ==========================================================

# GET request retrieves one ticket using its unique ID.
@router.get(
    "/{ticket_id}",
    response_model=TicketResponse,
)
def get_ticket_by_id(
    ticket_id: int,
    db: DatabaseSession,
):
    # Retrieve the requested ticket.
    ticket = ticket_service.get_ticket(
        db,
        ticket_id,
    )

    # Return HTTP 404 if the ticket does not exist.
    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ticket not found.",
        )

    return ticket