# Swipe Project – Smart Experience Recommender

**Version:** Stable Beta

**Authors/Maintainers:** Satvik Aderla

**Date:** 2025-07-15

---

## 1. Executive Summary

The Swipe Project is an intelligent recommendation system designed to solve the problem of information overload for users seeking new experiences. By capturing user preferences through a simple swipe interface (like, dislike, skip), the system delivers hyper-relevant, personalized suggestions. The intended user base includes anyone looking for curated activities, from dining and entertainment to travel and wellness.

**Key Features:**

- **AI-Driven Suggestions:** Leverages vector-based semantic search to understand user intent and provide accurate recommendations.
- **Tag-Based Filtering:** Uses a sophisticated tagging system to categorize and match experiences.
- **Session-Based History:** Tracks user interactions within a session to dynamically refine suggestions.
- **Semantic Search:** Powered by Qdrant, the system goes beyond simple keyword matching to understand the underlying meaning of user preferences.

**High-Level System Architecture:**
The system is built on a modular, three-tier architecture:

1.  **FastAPI Server:** A lightweight Python web server that exposes the API endpoints.
2.  **Supabase Database:** A PostgreSQL database for persistent storage of user and experience data.
3.  **Qdrant Semantic Engine:** A vector database that powers the semantic search and recommendation logic.

---

## 2. System Overview

### Architecture Diagram (Textual)

```
+-----------------+      +-------------------+      +----------------------+
|   User Client   |----->|   FastAPI Server  |----->|  Supabase Database   |
+-----------------+      +-------------------+      +----------------------+
                             |                      (experiences, swipes)
                             |
                             v
                    +----------------------+
                    |Qdrant Semantic Engine|
                    +----------------------+
```

### Modules

- **FastAPI Server (`mainswipe.py`):** The core of the application, responsible for handling HTTP requests, processing business logic, and interacting with the other modules.
- **Supabase DB:** The primary data store for the application. It houses the `experiences` and `swipes` tables, ensuring data persistence and integrity.
- **Qdrant Semantic Engine (`qdrant_search.py`):** The recommendation engine. It uses vector embeddings to perform semantic similarity searches and find the most relevant experiences for a user.

### Tech Stack

- **Python 3.8+**
- **FastAPI:** For building the RESTful API.
- **Supabase (PostgreSQL):** For data storage.
- **Qdrant:** For vector-based semantic search.
- **UUID-based Session Logic:** For tracking user sessions.

---

## 3. Setup & Deployment

### Environment Prerequisites

- Python 3.8 or higher
- `pip` for package management

### Folder Structure and File Roles

- `mainswipe.py`: The main FastAPI application file.
- `qdrant_search.py`: The Qdrant search engine module.
- `requirements.txt`: A list of all the Python packages required for the project.
- `.env`: A file to store environment variables (not included in the repository).

### Environment Variables (`.env` setup)

Create a `.env` file in the root directory with the following variables:

```
SUPABASE_URL="your_supabase_url"
SUPABASE_SERVICE_ROLE_KEY="your_supabase_service_role_key"
QDRANT_URL="your_qdrant_url"
QDRANT_API_KEY="your_qdrant_api_key"
```

### Local Run Instructions

1.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the FastAPI server:
    ```bash
    uvicorn mainswipe:app --reload
    ```

### Recommended Deployment Flow

1.  Set up a production environment with a dedicated server or a platform-as-a-service (PaaS) like Heroku or Render.
2.  Configure the environment variables in the production environment.
3.  Use a production-ready web server like Gunicorn to run the FastAPI application.

---

## 4. API Reference (FastAPI)

### `/start`

- **Function:** Initializes a user session and provides the first set of recommendations.
- **Behavior:**
  - If the user has a history of liked items in the `swipes` table, it fetches 7 experiences based on the tags of those liked items.
  - If the user has no history, it fetches 7 experiences with unique tags to establish a baseline of preferences.
- **Returns:** A JSON list of experience objects from the `experiences` table.

### `/recommendation`

- **Function:** Accepts a user's swipe actions and returns a new set of recommendations.
- **Behavior:**
  - Updates the `swipes` table in Supabase with the received actions (likes, dislikes, skips).
  - Passes the user's session preferences to the Qdrant engine to fetch the 5 best-matched experiences.
- **Returns:** A JSON list of 5 experience objects.

---

## 5. Supabase Schema Design

### Tables

- **`experiences`:** Stores all available experience objects.
  - `id` (uuid, primary key)
  - `name` (text)
  - `tags` (array of text)
  - Other relevant fields (e.g., `description`, `image_url`, `price`, `location`)
- **`swipes`:** Stores per-user swipe actions.
  - `user_id` (text, foreign key to a future `users` table)
  - `likes` (array of uuid)
  - `dislikes` (array of uuid)
  - `skips` (array of uuid)
  - `created_at` (timestamp)
  - `updated_at` (timestamp)

### Data Integrity Logic

- The `id` of each experience is a UUID to ensure uniqueness.
- The `swipes` table uses arrays of UUIDs to store the user's actions, which is an efficient way to manage lists of related items.

### Indexing/Optimization Notes

- Indexes should be created on the `user_id` column in the `swipes` table to speed up queries.
- The `tags` column in the `experiences` table should also be indexed to improve the performance of tag-based searches.

---

## 6. Qdrant Search Integration

### `qdrant_search.py` Functionality

This module is responsible for all interactions with the Qdrant vector database. It provides a `QdrantSearchEngine` class that encapsulates the logic for creating query vectors, performing similarity searches, and filtering results.

### Tag-Based + Vector-Based Similarity Logic

The search engine uses a hybrid approach to find the best matches for a user:

1.  **Tag-Based Filtering:** It first filters the experiences based on the tags of the user's liked items.
2.  **Vector-Based Similarity:** It then uses a pre-trained sentence transformer model to create a vector embedding of the user's preferences and performs a similarity search in the Qdrant database.

### How User History is Used for Semantic Matching

The user's history of liked items is used to create a profile of their preferences. The tags of these liked items are combined to create a query that is then used to search for similar experiences.

### Input/Output Format of Qdrant Functions

- **Input:** A query string containing the user's preferences (e.g., "Tags: adventure, outdoors").
- **Output:** A list of dictionaries, where each dictionary represents a recommended experience and includes the experience ID, similarity score, and payload (the experience data).

---

## 7. Data Flow & Session Management

### Stateless Design

Each API endpoint is stateless, meaning that it does not store any session information on the server. All the necessary data is passed in the request or retrieved from the Supabase database.

### No Local State Retention

After each API cycle, no local state is retained on the server. This makes the application more scalable and resilient.

### Flowchart (Textual)

```
Request -> Check Session -> Supabase Interaction -> Qdrant Search -> Response
```

---

## 8. Design Philosophy & Considerations

- **Clear Separation of Concerns:** The application is divided into three distinct modules (FastAPI server, Supabase DB, Qdrant engine), each with a clear responsibility.
- **Supabase as Single Source of Truth:** All the data is stored in the Supabase database, which acts as the single source of truth for the application.
- **Expandability:** The modular design of the application makes it easy to add new features, such as a user authentication layer or a more sophisticated recommendation algorithm.
- **Stable Beta Release:** The application is in a stable beta phase, which means that it is feature-complete but may still have some minor bugs or performance issues.

---

## 9. Known Issues / Limitations

- **Lack of Auth Layer:** The application does not currently have a user authentication layer, which means that all the data is publicly accessible.
- **Session Reset Logic:** There is no session reset logic in case of a Supabase wipe, which means that all the user data will be lost.

---

## 10. Roadmap & Future Work

- **Add Authentication or OAuth Flow:** Implement a user authentication layer to protect user data and provide a more personalized experience.
- **Improve Vector Diversity and Weight Tuning:** Fine-tune the recommendation algorithm to provide more diverse and relevant suggestions.
- **Session Recovery and Analytics Pipeline:** Implement a session recovery mechanism and an analytics pipeline to track user engagement and improve the recommendation algorithm.

---

## 11. Appendix

### Sample API Payloads

**`/start` Request:**

```
GET /start?user_id=some_user_id
```

**`/recommendation` Request:**

```json
{
  "user_id": "some_user_id",
  "likes": ["experience_uuid_1"],
  "dislikes": ["experience_uuid_2"],
  "skips": ["experience_uuid_3"]
}
```
