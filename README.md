# Swipe Project

## Overview

The Swipe Project is a smart experience recommendation system designed to provide users with hyper-relevant suggestions based on their preferences. Users can like, dislike, or skip experiences, and the system uses this feedback to refine future recommendations.

The core of the project is a FastAPI backend that interacts with a Supabase database for user data persistence and a Qdrant vector database for powerful semantic search.

**Tech Stack:**

- **Backend:** FastAPI
- **Database:** Supabase (PostgreSQL)
- **Vector Search:** Qdrant
- **Language:** Python

## How to Run

### Prerequisites

- Python 3.8+
- An active Supabase project
- A Qdrant cluster

### Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/swipe-project.git
    cd swipe-project
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root of the project and add the following keys:

    ```
    SUPABASE_URL="your_supabase_url"
    SUPABASE_SERVICE_ROLE_KEY="your_supabase_service_role_key"
    QDRANT_URL="your_qdrant_url"
    QDRANT_API_KEY="your_qdrant_api_key"
    ```

4.  **Run the FastAPI server:**
    ```bash
    uvicorn mainswipe:app --reload
    ```
    The application will be available at `http://127.0.0.1:8000`.

## Full Documentation

### `mainswipe.py`

This file contains the main FastAPI application logic, including the API endpoints for recommendations.

- **`/start`**

  - **Method:** `GET`
  - **Description:** This endpoint is called when a user starts a new session. It checks the user's history in Supabase. If the user has a history of liked items, it returns 7 recommendations based on their past preferences. Otherwise, it returns 7 diverse experiences with unique tags to establish a baseline.
  - **Query Parameters:**
    - `user_id` (str): The unique identifier for the user.
  - **Supabase Tables:** `swipes`, `experiences`

- **`/recommendation`**
  - **Method:** `POST`
  - **Description:** This endpoint accepts a user's swipe actions (likes, dislikes, skips), updates their profile in the `swipes` table in Supabase, and then calls the Qdrant search engine to retrieve 5 new matched experiences.
  - **Request Body:**
    ```json
    {
      "user_id": "string",
      "likes": ["uuid"],
      "dislikes": ["uuid"],
      "skips": ["uuid"]
    }
    ```
  - **Supabase Tables:** `swipes`, `experiences`

### `qdrant_search.py`

This module handles the vector-based semantic search functionality.

- **Core Functionality:** It performs semantic matching of user preferences against the `experiences` stored in the Qdrant database.
- **Method:** It uses a combination of tags and embeddings for efficient and accurate retrieval of experiences.

### Data Flow

1.  **User Session:** A user starts a session, and the client calls the `/start` endpoint.
2.  **Swipes:** The user interacts with the experiences, and their swipes are sent to the `/recommendation` endpoint.
3.  **Supabase:** The backend updates the user's swipe history in the `swipes` table in Supabase.
4.  **Vector Tag Mapping:** The backend creates a vector representation of the user's preferences based on the tags of their liked experiences.
5.  **Qdrant Search:** The vector is used to query the Qdrant database for similar experiences.
6.  **Response:** The backend returns a list of recommended experiences to the client.

### Design Choices

- **Stateless Endpoints:** The API endpoints are stateless, meaning they do not retain any data locally. All user data is fetched from and persisted to Supabase.
- **Separation of Concerns:** The session logic (handled by the FastAPI app) is separate from the vector querying logic (handled by `qdrant_search.py`).
- **Supabase as Source of Truth:** Supabase is the single source of truth for all user and experience data.

## Sample Request/Response Examples

### `/start`

**Request:**

```
GET /start?user_id=some_user_id
```

**Response:**

```json
[
  {
    "id": "experience_uuid_1",
    "name": "Experience Name 1",
    "tags": ["tag1", "tag2"]
  },
  {
    "id": "experience_uuid_2",
    "name": "Experience Name 2",
    "tags": ["tag3", "tag4"]
  }
]
```

### `/recommendation`

**Request:**

```json
{
  "user_id": "some_user_id",
  "likes": ["experience_uuid_3"],
  "dislikes": [],
  "skips": []
}
```

**Response:**

```json
[
  {
    "id": "experience_uuid_4",
    "name": "Experience Name 4",
    "tags": ["tag1", "tag5"]
  },
  {
    "id": "experience_uuid_5",
    "name": "Experience Name 5",
    "tags": ["tag2", "tag6"]
  }
]
```

## Roadmap/Future Scope

- **User Login System:** Implement a full-featured user authentication and authorization system.
- **Scale Recommendations:** Enhance the recommendation engine using multi-vector filters (e.g., price, location, time).
- **UI Front-end Integration:** Develop a user-friendly front-end to interact with the recommendation system.
