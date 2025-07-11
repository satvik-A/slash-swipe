from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import uuid
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Validate required environment variables
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables.")
    sys.exit(1)

if not QDRANT_URL or not QDRANT_API_KEY:
    logger.warning("Missing Qdrant credentials. Please set QDRANT_URL and QDRANT_API_KEY environment variables.")
    logger.warning("Will attempt to use direct database queries instead of vector search.")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "https://slash-rag-agent.onrender.com",
        "https://slash-experiences.netlify.app",
        "http://localhost:5173", 
        "https://slashexperiences.in",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SwipeRequest(BaseModel):
    user_id: str
    likes_str: Optional[str] = None  # Delimited string of UUIDs
    dislikes_str: Optional[str] = None  # Delimited string of UUIDs
    skips_str: Optional[str] = None  # Delimited string of UUIDs
    wishlist_str: Optional[str] = None  # Delimited string of UUIDs

# Create a dependency for the QdrantSearchEngine
def get_search_engine():
    """Dependency to create and return a QdrantSearchEngine instance."""
    try:
        # Only import if Qdrant credentials are available
        if QDRANT_URL and QDRANT_API_KEY:
            from qdrant_search import QdrantSearchEngine, create_search_engine_with_session
            search_engine = create_search_engine_with_session(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY
            )
            yield search_engine
        else:
            # Return None if Qdrant is not available
            yield None
    except Exception as e:
        logger.error(f"Error creating QdrantSearchEngine: {str(e)}")
        yield None

def parse_uuid_string(uuid_str: str) -> List[str]:
    """Parse a delimited string of UUIDs into a list."""
    if not uuid_str:
        return []
    
    # Split by backtick
    return [uid.strip() for uid in uuid_str.split('`') if uid.strip()]

def merge_swipe_arrays(existing_array: List[str], new_array: List[str]) -> List[str]:
    """
    Merge two arrays of UUIDs, removing duplicates and maintaining order.
    
    Args:
        existing_array: Current array from database
        new_array: New array to merge
        
    Returns:
        Merged array without duplicates
    """
    if not existing_array:
        existing_array = []
    if not new_array:
        new_array = []
    
    # Convert to set for deduplication, then back to list
    merged = list(set(existing_array + new_array))
    return merged

def remove_from_other_arrays(target_id: str, likes: List[str], dislikes: List[str], skips: List[str]) -> tuple:
    """
    Remove a target ID from all arrays except the one it should be in.
    This prevents an experience from being in multiple arrays.
    
    Args:
        target_id: The experience ID to clean up
        likes: Current likes array
        dislikes: Current dislikes array  
        skips: Current skips array
        
    Returns:
        Tuple of (cleaned_likes, cleaned_dislikes, cleaned_skips)
    """
    # Remove target_id from all arrays
    cleaned_likes = [id for id in likes if id != target_id]
    cleaned_dislikes = [id for id in dislikes if id != target_id]
    cleaned_skips = [id for id in skips if id != target_id]
    
    return cleaned_likes, cleaned_dislikes, cleaned_skips

def update_swipes_with_new_data(user_id: str, new_likes: List[str], new_dislikes: List[str], new_skips: List[str]) -> bool:
    """
    Update swipes record with new data, ensuring no redundancy across arrays.
    
    Args:
        user_id: The user ID
        new_likes: New likes to add
        new_dislikes: New dislikes to add
        new_skips: New skips to add
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get current record
        current_record = supabase.table("swipes").select("*").eq("user_id", user_id).execute()
        
        if not current_record.data:
            # Create new record if none exists
            logger.info(f"Creating new swipes record for user {user_id}")
            
            # For new record, just use the new data (no duplicates possible)
            final_likes = list(set(new_likes)) if new_likes else []
            final_dislikes = list(set(new_dislikes)) if new_dislikes else []
            final_skips = list(set(new_skips)) if new_skips else []
            
            supabase.table("swipes").insert({
                "user_id": user_id,
                "likes": final_likes,
                "dislikes": final_dislikes,
                "skips": final_skips
            }).execute()
            
            logger.info(f"Created swipes record: {len(final_likes)} likes, {len(final_dislikes)} dislikes, {len(final_skips)} skips")
            return True
        
        else:
            # Update existing record
            record = current_record.data[0]
            
            # Get current arrays (handle None values)
            current_likes = record.get("likes", []) or []
            current_dislikes = record.get("dislikes", []) or []
            current_skips = record.get("skips", []) or []
            
            # Start with current arrays
            final_likes = current_likes.copy()
            final_dislikes = current_dislikes.copy()
            final_skips = current_skips.copy()
            
            # Process new likes
            for like_id in new_likes:
                if like_id not in final_likes:
                    # Remove from other arrays if present
                    final_dislikes = [id for id in final_dislikes if id != like_id]
                    final_skips = [id for id in final_skips if id != like_id]
                    # Add to likes
                    final_likes.append(like_id)
            
            # Process new dislikes
            for dislike_id in new_dislikes:
                if dislike_id not in final_dislikes:
                    # Remove from other arrays if present
                    final_likes = [id for id in final_likes if id != dislike_id]
                    final_skips = [id for id in final_skips if id != dislike_id]
                    # Add to dislikes
                    final_dislikes.append(dislike_id)
            
            # Process new skips
            for skip_id in new_skips:
                if skip_id not in final_skips:
                    # Remove from other arrays if present
                    final_likes = [id for id in final_likes if id != skip_id]
                    final_dislikes = [id for id in final_dislikes if id != skip_id]
                    # Add to skips
                    final_skips.append(skip_id)
            
            # Update the record
            supabase.table("swipes").update({
                "likes": final_likes,
                "dislikes": final_dislikes,
                "skips": final_skips,
                "updated_at": "now()"
            }).eq("user_id", user_id).execute()
            
            logger.info(f"Updated swipes record: {len(final_likes)} likes, {len(final_dislikes)} dislikes, {len(final_skips)} skips")
            return True
    
    except Exception as e:
        logger.error(f"Error updating swipes with new data: {str(e)}")
        return False

def get_already_swiped_ids(user_id: str) -> List[str]:
    """Get IDs of experiences the user has already swiped on."""
    try:
        result = supabase.table("swipes").select("likes, dislikes, skips").eq("user_id", user_id).execute()
        
        already_swiped = []
        if result.data:
            for record in result.data:
                # Add all likes, dislikes, and skips to the list
                if "likes" in record and record["likes"]:
                    already_swiped.extend(record["likes"])
                if "dislikes" in record and record["dislikes"]:
                    already_swiped.extend(record["dislikes"])
                if "skips" in record and record["skips"]:
                    already_swiped.extend(record["skips"])
        
        return already_swiped
    except Exception as e:
        logger.error(f"Error getting swiped IDs: {str(e)}")
        return []

def save_experiences_to_swipes(user_id: str, experiences: List[Dict], status: str = "shown") -> bool:
    """
    Save experiences to the swipes table in Supabase.
    
    Args:
        user_id: The user ID
        experiences: List of experience objects to save
        status: Status to set for these experiences (default: "shown")
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not experiences:
            logger.warning(f"No experiences to save for user {user_id}")
            return True
        
        # Extract experience IDs
        exp_ids = [exp.get("id") for exp in experiences if exp.get("id")]
        
        if not exp_ids:
            logger.warning(f"No valid experience IDs found for user {user_id}")
            return True
        
        # Determine which array to update based on status
        if status == "liked":
            return update_swipes_with_new_data(user_id, exp_ids, [], [])
        elif status == "disliked":
            return update_swipes_with_new_data(user_id, [], exp_ids, [])
        elif status == "skipped":
            return update_swipes_with_new_data(user_id, [], [], exp_ids)
        else:
            # For "shown" status, we don't update any arrays yet
            logger.info(f"Experiences shown to user {user_id} but not yet swiped")
            return True
    
    except Exception as e:
        logger.error(f"Error saving experiences to swipes: {str(e)}")
        return False

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
        "qdrant_configured": bool(QDRANT_URL and QDRANT_API_KEY)
    }

@app.get("/user/{user_id}/swipes")
async def get_user_swipes(user_id: str):
    """Get current swipes data for a user."""
    try:
        result = supabase.table("swipes").select("*").eq("user_id", user_id).execute()
        
        if result.data:
            swipes_data = result.data[0]
            return {
                "user_id": user_id,
                "likes": swipes_data.get("likes", []),
                "dislikes": swipes_data.get("dislikes", []),
                "skips": swipes_data.get("skips", []),
                "created_at": swipes_data.get("created_at"),
                "updated_at": swipes_data.get("updated_at")
            }
        else:
            return {
                "user_id": user_id,
                "likes": [],
                "dislikes": [],
                "skips": [],
                "message": "No swipes data found for this user"
            }
    
    except Exception as e:
        logger.error(f"Error getting user swipes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting user swipes: {str(e)}")

@app.get("/start")
async def get_initial_experiences(
    user_id: str, 
    search_engine = Depends(get_search_engine)
):
    """
    Get initial experiences for a user.
    If user has previous data, return 7 experiences similar to their likes.
    Otherwise, return 7 diverse experiences with different tags.
    """
    try:
        logger.info(f"Getting initial experiences for user {user_id}")
        
        # Check if user already has history
        user_history = supabase.table("swipes").select("*").eq("user_id", user_id).execute()
        
        # Get IDs of experiences the user has already swiped on
        already_swiped = get_already_swiped_ids(user_id)
        
        experiences = []
        
        if user_history.data and len(user_history.data) > 0:
            logger.info(f"User {user_id} has previous swipe history. Finding similar experiences...")
            
            # Get liked experiences to use as seed for recommendations
            liked_ids = []
            for record in user_history.data:
                if "likes" in record and record["likes"]:
                    liked_ids.extend(record["likes"])
            
            # If we have Qdrant search available and liked experiences, use it
            if search_engine and liked_ids:
                # Fetch the full experience data for liked experiences
                liked_experiences = []
                for exp_id in liked_ids:
                    exp_data = supabase.table("experiences").select("*").eq("id", exp_id).execute()
                    if exp_data.data and len(exp_data.data) > 0:
                        liked_experiences.append(exp_data.data[0])
                
                # Extract tags from liked experiences
                liked_tags = []
                for exp in liked_experiences:
                    if "tags" in exp and exp["tags"]:
                        # Handle tags as string or list
                        if isinstance(exp["tags"], list):
                            liked_tags.extend(exp["tags"])
                        elif isinstance(exp["tags"], str):
                            # Split tags string by commas or other delimiters
                            tags = [t.strip() for t in exp["tags"].split(',')]
                            liked_tags.extend(tags)
                
                # Use Qdrant to find similar experiences based on tags
                if liked_tags:
                    # Join tags into a query string
                    tag_query = ", ".join(liked_tags)
                    logger.info(f"Searching for experiences with tags: {tag_query}")
                    
                    # Get recommendations using Qdrant
                    results = search_engine.search_similar(
                        query=f"Tags: {tag_query}",
                        top_k=15,  # Get more than needed to filter out already swiped
                        use_structured_parsing=True
                    )
                    
                    # Filter out already swiped experiences and get full data
                    filtered_results = []
                    for result in results:
                        exp_id = result.get("id")
                        if exp_id and exp_id not in already_swiped:
                            # Get the full experience data from Supabase
                            exp_data = supabase.table("experiences").select("*").eq("id", exp_id).execute()
                            if exp_data.data and len(exp_data.data) > 0:
                                filtered_results.append(exp_data.data[0])
                            
                            # Stop when we have 7 experiences
                            if len(filtered_results) >= 7:
                                break
                    
                    experiences = filtered_results[:7]
            
            # If we couldn't get recommendations based on likes or Qdrant is not available, get random experiences
            if not experiences:
                logger.info("No liked experiences found or couldn't get recommendations. Using random experiences.")
                # Get 7 random experiences that haven't been swiped on
                if already_swiped:
                    # Use !in filter to exclude already swiped experiences
                    random_exp = supabase.table("experiences").select("*").not_in("id", already_swiped).limit(7).execute()
                else:
                    random_exp = supabase.table("experiences").select("*").limit(7).execute()
                
                if random_exp.data:
                    experiences = random_exp.data
        else:
            logger.info(f"New user {user_id}. Providing diverse initial experiences.")
            
            # For new users, get 7 diverse experiences
            # If Qdrant is available, use it to get diverse experiences
            if search_engine:
                # Get all available tags from Qdrant
                try:
                    available_tags = search_engine.get_available_tags()
                    
                    # Select 7 diverse tags (or fewer if not enough available)
                    diverse_tags = available_tags[:min(7, len(available_tags))]
                    
                    # Get one experience for each diverse tag
                    for tag in diverse_tags:
                        # Search for an experience with this tag
                        results = search_engine.search_similar(
                            query=f"Tags: {tag}",
                            top_k=3,  # Get a few options to choose from
                            use_structured_parsing=True
                        )
                        
                        if results:
                            # Take the first result for this tag
                            result = results[0]
                            exp_id = result.get("id")
                            
                            # Get the full experience data from Supabase
                            exp_data = supabase.table("experiences").select("*").eq("id", exp_id).execute()
                            if exp_data.data and len(exp_data.data) > 0:
                                experiences.append(exp_data.data[0])
                                
                                # Stop when we have 7 experiences
                                if len(experiences) >= 7:
                                    break
                except Exception as e:
                    logger.error(f"Error getting diverse experiences: {str(e)}")
            
            # If we couldn't get enough diverse experiences or Qdrant is not available, fill with random ones
            if len(experiences) < 7:
                remaining = 7 - len(experiences)
                random_exp = supabase.table("experiences").select("*").limit(remaining).execute()
                if random_exp.data:
                    experiences.extend(random_exp.data)
        
        # Create an empty swipes record for this session
        update_swipes_record(user_id, session_id, [], [], [])
        
        return {
            "status": "existing" if user_history.data else "new",
            "experiences": experiences,
            "session_id": session_id
        }
    
    except Exception as e:
        logger.error(f"Error in get_initial_experiences: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting initial experiences: {str(e)}")
