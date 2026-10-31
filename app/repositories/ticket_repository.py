# Import select to construct SQL queries.
from sqlalchemy import select

# Import Session for database communication.
from sqlalchemy.orm import Session

# Import the Ticket database model.
from app.models.ticket import Ticket


# ==========================================================
# 1. CREATE TICKET
# ==========================================================

def create_ticket(
    db: Session,
    title: str,
    description: str,
    customer_id: int,
    priority: str,
) -> Ticket:
    """
    Create and save a new support ticket.
    """

    # Create a new Ticket database object.
    ticket = Ticket(
        title=title,
        description=description,
        customer_id=customer_id,
        priority=priority,
    )

    # Add the ticket to the current database session.
    db.add(ticket)

    # Save the ticket permanently.
    db.commit()

    # Refresh the object to retrieve generated values.
    db.refresh(ticket)

    # Return the newly created ticket.
    return ticket


# ==========================================================
# 2. GET TICKET BY ID
# ==========================================================

def get_ticket_by_id(
    db: Session,
    ticket_id: int,
) -> Ticket | None:
    """
    Retrieve a ticket using its unique ID.
    """

    # Construct a query to find the requested ticket.
    statement = select(Ticket).where(
        Ticket.id == ticket_id
    )

    # Execute the query and return the matching ticket.
    return db.scalar(statement)


# ==========================================================
# 3. GET ALL TICKETS
# ==========================================================

def get_tickets(
    db: Session,
    skip: int = 0,
    limit: int = 20,
) -> list[Ticket]:
    """
    Retrieve tickets with pagination.
    """

    # Construct a query to retrieve tickets.
    statement = (
        select(Ticket)
        .order_by(Ticket.id)
        .offset(skip)
        .limit(limit)
    )

    # Execute the query and return a list of tickets.
    return list(db.scalars(statement).all())


# ==========================================================
# 4. GET TICKETS BY CUSTOMER
# ==========================================================

def get_tickets_by_customer(
    db: Session,
    customer_id: int,
) -> list[Ticket]:
    """
    Retrieve all tickets belonging to a specific customer.
    """

    # Find tickets using the customer's database ID.
    statement = (
        select(Ticket)
        .where(Ticket.customer_id == customer_id)
        .order_by(Ticket.id)
    )

    # Execute the query and return matching tickets.
    return list(db.scalars(statement).all())



# ==========================================================
# 5. UPDATE TICKET STATUS
# ==========================================================

def update_ticket_status(
    db: Session,
    ticket: Ticket,
    new_status: str,
) -> Ticket:
    """
    Update the status of an existing support ticket.

    Args:
        db: Active database session.
        ticket: Ticket object that needs updating.
        new_status: New status for the ticket.

    Returns:
        The updated Ticket object.
    """

    # Change the ticket's current status.
    ticket.status = new_status

    # Save the changes to the database.
    db.commit()

    # Refresh the object to retrieve updated database values.
    db.refresh(ticket)

    # Return the updated ticket.
    return ticket
