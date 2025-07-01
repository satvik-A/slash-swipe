from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
import os
from swipetool import get_top_chunks

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

class SwipeRequest(BaseModel):
    user_id: str
    swiped_ids: list[str]

@app.get("/start")
def get_initial_experiences(user_id: str):
    # Check if user already has history
    user_history = supabase.table("swipes").select("*").eq("user_id", user_id).execute().data
    if user_history:
        return {"status": "existing", "message": "User has previous data."}

    # Get 7 random experiences
    experiences = supabase.table("experiences").select("*").limit(7).execute().data
    return {"status": "new", "experiences": experiences}

@app.post("/recommend")
def recommend_more(request: SwipeRequest):
    # Fetch swiped experience data
    experiences = []
    for eid in request.swiped_ids:
        res = supabase.table("experiences").select("*").eq("id", eid).execute()
        if res.data:
            experiences.append(res.data[0])

    if not experiences:
        raise HTTPException(status_code=404, detail="No valid experience data found.")

    prompt = " ".join([exp.get("name", "") + " " + exp.get("description", "") for exp in experiences])
    recommended = get_top_chunks(prompt, k=5)
    return {"recommended": recommended}
