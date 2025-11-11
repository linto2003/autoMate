import os
import json
import http.client
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

# ----------------------------------------------------------------------
# 🔧 Setup
# ----------------------------------------------------------------------
load_dotenv()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY"),
)

conn = http.client.HTTPSConnection("twitter241.p.rapidapi.com")

headers = {
    'x-rapidapi-key': "7a251f30a5msh74b996da2dac437p136eb2jsn709e2c2a61b9",
    'x-rapidapi-host': "twitter241.p.rapidapi.com"
}

# ----------------------------------------------------------------------
# 🧩 Tool 1: get_user_details
# ----------------------------------------------------------------------
class UserDetailsOutput(BaseModel):
    name: str
    description: str
    followers_count: int
    error: str | None = None


def get_user_details(user_id: str) -> dict:
    """Fetch user details (name, description, followers_count)"""
    try:
        conn.request("GET", f"/get-users?users={user_id}", headers=headers)
        res = conn.getresponse()
        data = res.read()
        json_data = json.loads(data.decode("utf-8"))
        user_data = json_data["result"]["data"]["users"][0]["result"]["legacy"]

        validated = UserDetailsOutput(
            name=user_data.get("name", ""),
            description=user_data.get("description", ""),
            followers_count=user_data.get("followers_count", 0)
        )
        return validated.model_dump()
    except ValidationError as ve:
        return {"error": f"Validation failed: {ve}"}
    except Exception as e:
        return {"error": str(e)}


# ----------------------------------------------------------------------
# 🧩 Tool 2: get_following_ids
# ----------------------------------------------------------------------
class FollowingOutput(BaseModel):
    ids: list[str]
    error: str | None = None


def get_following_ids(username: str, count: int = 20) -> dict:
    """Fetch list of following user IDs for a given username."""
    try:
        conn.request("GET", f"/following-ids?username={username}&count={count}", headers=headers)
        res = conn.getresponse()
        data = res.read()
        json_data = json.loads(data.decode("utf-8"))
        ids = json_data.get("ids", [])
        validated = FollowingOutput(ids=ids)
        print(f"✅ Got {len(ids)} following IDs for {username}")
        return validated.model_dump()
    except ValidationError as ve:
        return {"error": f"Validation failed: {ve}"}
    except Exception as e:
        return {"error": str(e)}


# ----------------------------------------------------------------------
# 🧩 Agent Response Schema
# ----------------------------------------------------------------------
class AgentResponse(BaseModel):
    query: str
    tool_used: str
    result: dict


# ----------------------------------------------------------------------
# 🧩 Create Agent
# ----------------------------------------------------------------------
agent = create_agent(
    model=llm,
    tools=[get_user_details, get_following_ids],
    system_prompt=(
        "You are a helpful assistant that can analyze Twitter data using the given tools. "
        "Use get_following_ids(username) to get the list of people a user follows. "
        "Use get_user_details(user_id) to get name, description, and followers_count. "
        "If asked to find which following has more followers than the given user, "
        "first get the user's followers_count, then compare it with the followings'. "
        "Return the names of those with higher follower counts."
    ),
)

# ----------------------------------------------------------------------
# 🧩 Run the agent
# ----------------------------------------------------------------------
query = "find from the following of yashc_twts who has greater number of followers than yashc_twts"
response = agent.invoke({"messages": [{"role": "user", "content": query}]})

final_message, tool_used = None, None

for msg in response["messages"]:
    if isinstance(msg, AIMessage) and msg.content.strip():
        final_message = msg.content.strip()
    elif hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_used = msg.tool_calls[0]["name"]

structured_response = AgentResponse(
    query=query,
    tool_used=tool_used or "N/A",
    result={"message": final_message or "No response generated"}
)

print(structured_response.model_dump_json(indent=2))
