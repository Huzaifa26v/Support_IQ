# ==========================================================
# SUPPORTIQ - USER API AUTOMATED TESTS
# ==========================================================


# ==========================================================
# 1. TEST USER REGISTRATION
# ==========================================================

def test_create_user(client):
    """Verify that a new user can be registered successfully."""

    # Prepare valid user information.
    user_data = {
        "full_name": "Ali Ahmed",
        "email": "ali@example.com",
        "role": "customer",
    }

    # Send a POST request to register the user.
    response = client.post(
        "/api/v1/users/",
        json=user_data,
    )

    # Verify successful user creation.
    assert response.status_code == 201

    # Read the returned JSON data.
    data = response.json()

    # Verify the user's information.
    assert data["full_name"] == "Ali Ahmed"
    assert data["email"] == "ali@example.com"
    assert data["role"] == "customer"
    assert data["is_active"] is True
    assert data["id"] is not None

# ==========================================================
# 2. TEST DUPLICATE EMAIL PREVENTION
# ==========================================================

def test_duplicate_user_email(client):
    """
    Verify that duplicate email registration is rejected.
    """

    # Prepare user information.
    user_data = {
        "full_name": "Ali Ahmed",
        "email": "duplicate@example.com",
        "role": "customer",
    }

    # Register the first user.
    first_response = client.post(
        "/api/v1/users/",
        json=user_data,
    )

    # The first registration should succeed.
    assert first_response.status_code == 201

    # Attempt another registration using the same email.
    second_response = client.post(
        "/api/v1/users/",
        json=user_data,
    )

    # The second registration should be rejected.
    assert second_response.status_code == 409

    # Verify the error message.
    assert second_response.json()["detail"] == (
        "A user with this email already exists."
    )

    
# ==========================================================
# 3. TEST INVALID EMAIL VALIDATION
# ==========================================================

def test_create_user_with_invalid_email(client):
    """Verify that invalid email addresses are rejected."""

    # Prepare user information with an invalid email.
    user_data = {
        "full_name": "Ali Ahmed",
        "email": "invalid-email",
        "role": "customer",
    }

    # Send the registration request.
    response = client.post(
        "/api/v1/users/",
        json=user_data,
    )

    # Pydantic should reject the invalid email.
    assert response.status_code == 422


# ==========================================================
# 4. TEST GET ALL USERS
# ==========================================================

def test_get_all_users(client):
    """Verify that the API returns all registered users."""

    # Prepare two different users.
    users = [
        {
            "full_name": "Ali Ahmed",
            "email": "ali@example.com",
            "role": "customer",
        },
        {
            "full_name": "Sara Khan",
            "email": "sara@example.com",
            "role": "agent",
        },
    ]

    # Register both users.
    for user_data in users:
        response = client.post(
            "/api/v1/users/",
            json=user_data,
        )

        assert response.status_code == 201

    # Request all registered users.
    response = client.get("/api/v1/users/")

    assert response.status_code == 200

    # Read the returned users.
    data = response.json()

    # Verify that exactly two users were returned.
    assert len(data) == 2

    # Extract email addresses.
    emails = [user["email"] for user in data]

    assert "ali@example.com" in emails
    assert "sara@example.com" in emails


# ==========================================================
# 5. TEST GET USER BY ID
# ==========================================================

def test_get_user_by_id(client):
    """Verify that a user can be retrieved using their ID."""

    # Prepare user information.
    user_data = {
        "full_name": "Ahmed Khan",
        "email": "ahmed@example.com",
        "role": "customer",
    }

    # Register the user.
    create_response = client.post(
        "/api/v1/users/",
        json=user_data,
    )

    assert create_response.status_code == 201

    # Extract the generated user ID.
    user_id = create_response.json()["id"]

    # Retrieve the user using their ID.
    response = client.get(
        f"/api/v1/users/{user_id}"
    )

    assert response.status_code == 200

    # Verify the returned information.
    data = response.json()

    assert data["id"] == user_id
    assert data["full_name"] == "Ahmed Khan"
    assert data["email"] == "ahmed@example.com"


# ==========================================================
# 6. TEST USER NOT FOUND
# ==========================================================

def test_get_nonexistent_user(client):
    """Verify that a nonexistent user returns HTTP 404."""

    # Request a user ID that does not exist.
    response = client.get("/api/v1/users/9999")

    assert response.status_code == 404

    # Verify the error message.
    assert response.json()["detail"] == "User not found."