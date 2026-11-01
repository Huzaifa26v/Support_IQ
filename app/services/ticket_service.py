# Import SQLAlchemy Session for database communication.
from sqlalchemy.orm import Session

# Import the Ticket database model.
from app.models.ticket import Ticket

# Import repositories for database operations.
from app.repositories import ticket_repository
from app.repositories import user_repository

# Import schemas for ticket creation and status updates.
from app.schemas.ticket import TicketCreate, TicketStatusUpdate

# Import the trained ML prediction function.
from ml.predict import predict_category


# ==========================================================
# 1. CREATE TICKET
# ==========================================================

def create_ticket(
    db: Session,
    ticket_data: TicketCreate,
) -> Ticket:
    """
    Create a new support ticket after validating its customer.

    Business rules:
    1. The customer must exist.
    2. The customer account must be active.
    3. The user must have the customer role.
    4. The ML model automatically predicts the ticket category.
    """

    # Find the customer using the supplied customer ID.
    customer = user_repository.get_user_by_id(
        db,
        ticket_data.customer_id,
    )

    # Reject the request if the customer does not exist.
    if customer is None:
        raise ValueError("Customer not found.")

    # Prevent inactive accounts from creating tickets.
    if not customer.is_active:
        raise ValueError("This customer account is inactive.")

    # Only users with the customer role can own customer tickets.
    if customer.role != "customer":
        raise ValueError(
            "Only customers can create support tickets."
        )

    # Combine the ticket title and description for classification.
    ticket_text = (
        f"{ticket_data.title} {ticket_data.description}"
    )

    # Predict the ticket category using our trained ML model.
    predicted_category = predict_category(ticket_text)

    # Save the ticket with its automatically predicted category.
    return ticket_repository.create_ticket(
        db=db,
        title=ticket_data.title,
        description=ticket_data.description,
        customer_id=ticket_data.customer_id,
        priority=ticket_data.priority,
        category=predicted_category,
    )


# ==========================================================
# 2. GET TICKET BY ID
# ==========================================================

def get_ticket(
    db: Session,
    ticket_id: int,
) -> Ticket | None:
    """
    Retrieve a ticket using its unique ID.
    """

    # Delegate the database operation to the repository.
    return ticket_repository.get_ticket_by_id(
        db,
        ticket_id,
    )


# ==========================================================
# 3. GET ALL TICKETS
# ==========================================================

def list_tickets(
    db: Session,
    skip: int = 0,
    limit: int = 20,
) -> list[Ticket]:
    """
    Retrieve all support tickets with pagination.
    """

    # Delegate ticket retrieval to the repository.
    return ticket_repository.get_tickets(
        db,
        skip=skip,
        limit=limit,
    )


# ==========================================================
# 4. GET TICKETS BY CUSTOMER
# ==========================================================

def list_customer_tickets(
    db: Session,
    customer_id: int,
) -> list[Ticket]:
    """
    Retrieve all tickets belonging to a specific customer.
    """

    # First, check whether the customer exists.
    customer = user_repository.get_user_by_id(
        db,
        customer_id,
    )

    # Return an error if the customer does not exist.
    if customer is None:
        raise ValueError("Customer not found.")

    # Retrieve tickets belonging to this customer.
    return ticket_repository.get_tickets_by_customer(
        db,
        customer_id,
    )


# ==========================================================
# 5. UPDATE TICKET STATUS
# ==========================================================

def update_ticket_status(
    db: Session,
    ticket_id: int,
    status_data: TicketStatusUpdate,
) -> Ticket | None:
    """
    Update the status of an existing support ticket.

    Args:
        db: Active database session.
        ticket_id: ID of the ticket to update.
        status_data: Validated new ticket status.

    Returns:
        Updated Ticket object if found, otherwise None.
    """

    # First, find the ticket using its ID.
    ticket = ticket_repository.get_ticket_by_id(
        db,
        ticket_id,
    )

    # Return None if the requested ticket does not exist.
    if ticket is None:
        return None

    # Update the ticket status through the repository.
    return ticket_repository.update_ticket_status(
        db=db,
        ticket=ticket,
        new_status=status_data.status,
    )