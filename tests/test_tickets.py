# ==========================================================
# SUPPORTIQ - TICKET API AUTOMATED TESTS
# ==========================================================

# Helper function to create a customer for ticket tests.
# Each test gets its own clean database through conftest.py.

def create_test_customer(client):
    """Register and return a customer for testing."""

    response = client.post(
        "/api/v1/users/",
        json={
            "full_name": "Ali Ahmed",
            "email": "ali@example.com",
            "role": "customer",
        },
    )

    assert response.status_code == 201

    # Return the newly created customer's database ID.
    return response.json()["id"]


# ==========================================================
# 1. TEST TICKET CREATION
# ==========================================================

def test_create_ticket(client):
    """Verify that a customer can create a support ticket."""

    # First, create a customer.
    customer_id = create_test_customer(client)

    # Prepare ticket information.
    ticket_data = {
        "title": "Unable to access my account",
        "description": (
            "I cannot log in to my account. "
            "Please help me resolve this issue."
        ),
        "customer_id": customer_id,
        "priority": "high",
    }

    # Send a POST request to create the ticket.
    response = client.post(
        "/api/v1/tickets/",
        json=ticket_data,
    )

    # Verify successful ticket creation.
    assert response.status_code == 201

    data = response.json()

    # Verify the returned ticket information.
    assert data["title"] == ticket_data["title"]
    assert data["description"] == ticket_data["description"]
    assert data["customer_id"] == customer_id
    assert data["priority"] == "high"
    assert data["status"] == "open"
    assert data["id"] is not None


# ==========================================================
# 2. TEST TICKET CREATION WITH INVALID CUSTOMER
# ==========================================================

def test_create_ticket_with_nonexistent_customer(client):
    """Verify that tickets cannot be created for missing customers."""

    # Prepare ticket information with a customer ID that does not exist.
    ticket_data = {
        "title": "Account problem",
        "description": "I need help with my account.",
        "customer_id": 9999,
        "priority": "medium",
    }

    # Attempt to create the ticket.
    response = client.post(
        "/api/v1/tickets/",
        json=ticket_data,
    )

    # The API should reject the request.
    assert response.status_code == 400

    assert response.json()["detail"] == "Customer not found."


# ==========================================================
# 3. TEST GET ALL TICKETS
# ==========================================================

def test_get_all_tickets(client):
    """Verify that the API returns all registered tickets."""

    # Create a customer.
    customer_id = create_test_customer(client)

    # Prepare two tickets.
    tickets = [
        {
            "title": "Login issue",
            "description": "I cannot access my account.",
            "customer_id": customer_id,
            "priority": "high",
        },
        {
            "title": "Payment issue",
            "description": "My payment is not being processed.",
            "customer_id": customer_id,
            "priority": "medium",
        },
    ]

    # Create both tickets.
    for ticket_data in tickets:
        response = client.post(
            "/api/v1/tickets/",
            json=ticket_data,
        )

        assert response.status_code == 201

    # Request all tickets.
    response = client.get("/api/v1/tickets/")

    assert response.status_code == 200

    data = response.json()

    # Verify that both tickets are returned.
    assert len(data) == 2


# ==========================================================
# 4. TEST GET TICKET BY ID
# ==========================================================

def test_get_ticket_by_id(client):
    """Verify that a ticket can be retrieved using its ID."""

    # Create a customer.
    customer_id = create_test_customer(client)

    # Create a ticket.
    create_response = client.post(
        "/api/v1/tickets/",
        json={
            "title": "Internet connection issue",
            "description": "My internet connection is not working.",
            "customer_id": customer_id,
            "priority": "high",
        },
    )

    assert create_response.status_code == 201

    # Extract the generated ticket ID.
    ticket_id = create_response.json()["id"]

    # Retrieve the ticket using its ID.
    response = client.get(
        f"/api/v1/tickets/{ticket_id}"
    )

    assert response.status_code == 200

    # Verify the returned ticket.
    assert response.json()["id"] == ticket_id
    assert response.json()["title"] == "Internet connection issue"


# ==========================================================
# 5. TEST GET TICKETS BY CUSTOMER
# ==========================================================

def test_get_customer_tickets(client):
    """Verify that tickets belonging to a customer can be retrieved."""

    # Create a customer.
    customer_id = create_test_customer(client)

    # Create two tickets for this customer.
    for title in ["Login problem", "Payment problem"]:
        response = client.post(
            "/api/v1/tickets/",
            json={
                "title": title,
                "description": "Please help me resolve this issue.",
                "customer_id": customer_id,
                "priority": "medium",
            },
        )

        assert response.status_code == 201

    # Retrieve tickets belonging to this customer.
    response = client.get(
        f"/api/v1/tickets/customer/{customer_id}"
    )

    assert response.status_code == 200

    data = response.json()

    # Verify that both tickets belong to this customer.
    assert len(data) == 2

    for ticket in data:
        assert ticket["customer_id"] == customer_id


# ==========================================================
# 6. TEST TICKET STATUS UPDATE
# ==========================================================

def test_update_ticket_status(client):
    """Verify that an existing ticket's status can be updated."""

    # Create a customer.
    customer_id = create_test_customer(client)

    # Create a ticket.
    create_response = client.post(
        "/api/v1/tickets/",
        json={
            "title": "Account access problem",
            "description": "I need assistance accessing my account.",
            "customer_id": customer_id,
            "priority": "high",
        },
    )

    assert create_response.status_code == 201

    ticket_id = create_response.json()["id"]

    # Update the ticket status.
    response = client.patch(
        f"/api/v1/tickets/{ticket_id}/status",
        json={"status": "in_progress"},
    )

    assert response.status_code == 200

    # Verify that the status was updated.
    assert response.json()["status"] == "in_progress"


# ==========================================================
# 7. TEST INVALID TICKET STATUS
# ==========================================================

def test_update_ticket_with_invalid_status(client):
    """Verify that unsupported ticket statuses are rejected."""

    # Create a customer.
    customer_id = create_test_customer(client)

    # Create a ticket.
    create_response = client.post(
        "/api/v1/tickets/",
        json={
            "title": "Application issue",
            "description": "The application is not responding.",
            "customer_id": customer_id,
            "priority": "medium",
        },
    )

    assert create_response.status_code == 201

    ticket_id = create_response.json()["id"]

    # Attempt to update the ticket with an unsupported status.
    response = client.patch(
        f"/api/v1/tickets/{ticket_id}/status",
        json={"status": "pending"},
    )

    # Pydantic should reject the invalid status.
    assert response.status_code == 422