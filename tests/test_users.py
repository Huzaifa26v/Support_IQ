# ==========================================================

# SUPPORTIQ USER API TESTS

# ==========================================================

# These tests use the client fixture from conftest.py.

# The fixture provides an isolated testing database.

# ==========================================================

# 1. TEST USER REGISTRATION

# ==========================================================

def test_create_user(client):

# Verify that a new user can be registered successfully.



# Prepare valid user information.
 user_data = {
    "full_name": "Ali Ahmed",
    "email": "ali@example.com",
    "role": "customer",
}

# Send a POST request to create the user.
 response = client.post(
    "/api/v1/users/",
    json=user_data,
)

# Verify that the API returns HTTP 201 Created.
 assert response.status_code == 201

# Read the JSON response.
 data = response.json()

# Verify that the returned user information is correct.
 assert data["full_name"] == "Ali Ahmed"
 assert data["email"] == "ali@example.com"
 assert data["role"] == "customer"
 assert data["is_active"] is True

# Verify that the database generated a user ID.
 assert data["id"] is not None


# ==========================================================

# 2. TEST DUPLICATE EMAIL PREVENTION

# ==========================================================

 def test_duplicate_user_email(client):

# Verify that two users cannot register with the same email.



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

# Attempt to register another user with the same email.
 second_response = client.post(
    "/api/v1/users/",
    json=user_data,
)

# The API should reject the duplicate registration.
 assert second_response.status_code == 409

# Verify the expected error message.
 assert second_response.json()["detail"] == (
    "A user with this email already exists."
)


# ==========================================================

# 3. TEST INVALID EMAIL VALIDATION

# ==========================================================

def test_create_user_with_invalid_email(client):

# Verify that an invalid email address is rejected.



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

# Verify that the API returns all registered users.



# Prepare two users.
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
  client.post(
        "/api/v1/users/",
        json=user_data,
    )

# Request the complete user list.
 response = client.get("/api/v1/users/")

# Verify successful retrieval.
 assert response.status_code == 200

# Verify that both users are returned.
 data = response.json()
 assert len(data) == 2

# Verify the returned email addresses.
 emails = [user["email"] for user in data]

 assert "ali@example.com" in emails
 assert "sara@example.com" in emails


# ==========================================================

# 5. TEST GET USER BY ID

# ==========================================================

def test_get_user_by_id(client):

# Verify that a user can be retrieved using their ID.



# Prepare user information.
 user_data = {
    "full_name": "Ahmed Khan",
    "email": "ahmed@example.com",
    "role": "customer",
}

# Register the user first.
 create_response = client.post(
    "/api/v1/users/",
    json=user_data,
)

# Extract the generated user ID.
 user_id = create_response.json()["id"]

# Retrieve the user using the generated ID.
 response = client.get(
    f"/api/v1/users/{user_id}"
)

# Verify successful retrieval.
 assert response.status_code == 200

# Verify the returned user information.
 assert response.json()["id"] == user_id
 assert response.json()["email"] == "ahmed@example.com"


# ==========================================================

# 6. TEST USER NOT FOUND

# ==========================================================

def test_get_nonexistent_user(client):

# Verify that requesting a nonexistent user returns HTTP 404.



# Request a user ID that does not exist.
 response = client.get("/api/v1/users/9999")

# The API should return HTTP 404 Not Found.
 assert response.status_code == 404

# Verify the error message.
 assert response.json()["detail"] == "User not found."
