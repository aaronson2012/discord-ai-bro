import discord
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timezone
import json
import asyncio
import google.generativeai as genai
from google.api_core.exceptions import (
    DeadlineExceeded,
    ServiceUnavailable,
    InvalidArgument,
    PermissionDenied,
    Unauthenticated,
    GoogleAPICallError
)
from supabase import create_client, Client
from typing import List, Dict, Optional
import collections

# Load environment variables
load_dotenv(override=True)

# --- Constants and Configurations ---

# From ppm.py
AI_PERSONA_SYSTEM_MESSAGE = "You are AIBro, a genuine friend. Your main goal is to be helpful and engage in thoughtful, inquisitive conversation. Speak naturally, like you're chatting with a good pal. Avoid any stiff, formal, or overly robotic language that might make you sound like a standard AI assistant. Be curious, ask clarifying questions if needed, and offer support or insights in a warm, approachable way. Keep your responses to a comfortable length for a friendly chat, typically 2-3 sentences unless more detail is clearly beneficial for the conversation."

# From original bot.py
MAX_RESPONSE_CHAR_LIMIT = 2000
CHANNEL_STM_MAX_LEN = 10
LLM_RESPONSE_CONTEXT_HISTORY_LIMIT = 10
LLM_SUMMARY_CONTEXT_HISTORY_LIMIT = 20

# Environment Variables
DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Logger setup
logger = logging.getLogger(__name__) # Main logger for the application

# Gemini API Configuration (from llmi.py, adapted)
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("Google Gemini API configured successfully.")
    except Exception as e:
        logger.error(f"Error configuring Google Gemini API: {e}", exc_info=True)
else:
    logger.warning("Warning: GOOGLE_API_KEY environment variable not found. LLM functionality will be limited.")

gemini_model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5-flash-latest')


# --- Prompt Generation Logic (from ppm.py) ---

def generate_prompt(current_message_content: str, history: List[Dict[str, str]], user_profile: Optional[Dict] = None) -> str:
    """
    Generates a prompt for the LLM, including recent conversation history, persona, and user context if available.
    """
    prompt_parts = [AI_PERSONA_SYSTEM_MESSAGE]

    if user_profile:
        profile_info_parts = []
        summary = user_profile.get("interaction_summary")
        if summary:
            profile_info_parts.append(f"Summary of your past interactions with this user: {summary}")

        preferences = user_profile.get("preferences")
        if preferences and isinstance(preferences, dict) and any(preferences.values()):
            pref_str_parts = []
            for key, value in preferences.items():
                if value:
                    display_key = key.replace("_", " ").capitalize()
                    if isinstance(value, list):
                        if value: # Ensure list is not empty
                             pref_str_parts.append(f"- {display_key}: {', '.join(map(str, value))}")
                    elif isinstance(value, dict):
                        dict_items = [f"{k_inner}: {v_inner}" for k_inner, v_inner in value.items() if v_inner]
                        if dict_items:
                            pref_str_parts.append(f"- {display_key}: {{{', '.join(dict_items)}}}")
                    else:
                        pref_str_parts.append(f"- {display_key}: {str(value)}")
            if pref_str_parts:
                profile_info_parts.append("User's known preferences:\n" + "\n".join(pref_str_parts))

        details = user_profile.get("mentioned_personal_details")
        if details and isinstance(details, dict) and any(details.values()):
            detail_str_parts = []
            for key, value in details.items():
                if value:
                    display_key = key.replace("_", " ").capitalize()
                    if isinstance(value, list):
                        if value: # Ensure list is not empty
                            detail_str_parts.append(f"- {display_key}: {', '.join(map(str, value))}")
                    elif isinstance(value, dict):
                        dict_items = [f"{k_inner}: {v_inner}" for k_inner, v_inner in value.items() if v_inner]
                        if dict_items:
                            detail_str_parts.append(f"- {display_key}: {{{', '.join(dict_items)}}}")
                    else:
                        detail_str_parts.append(f"- {display_key}: {str(value)}")
            if detail_str_parts:
                profile_info_parts.append("User's mentioned personal details:\n" + "\n".join(detail_str_parts))
        
        if profile_info_parts:
            prompt_parts.append("\n--- User Context ---")
            prompt_parts.extend(profile_info_parts)
            prompt_parts.append("--- End User Context ---\n")

    if history:
        prompt_parts.append("Here is the recent conversation history:")
        for msg in history:
            prompt_parts.append(f"{msg.get('author', 'Unknown')}: {msg.get('content', '')}")
        prompt_parts.append("---")
    
    prompt_parts.append(f"The latest message you are responding to is: '{current_message_content}'.")
    prompt_parts.append("Please provide a response that is consistent with your persona and the conversation history. Use the User Context provided above to inform your response, and if the user asks what you know about them, use the User Context to answer directly.")

    return "\n".join(prompt_parts)

def generate_user_summary_prompt(user_id: str, username: str, conversation_history: List[Dict[str, str]], existing_summary: str | None) -> str:
    """
    Generates a prompt for the LLM to summarize user interactions.
    """
    prompt_lines = [
        "You are an AI assistant tasked with maintaining a user profile.",
        f"User ID: {user_id}",
        f"Username: {username}",
        "\nExisting Interaction Summary for this user (if any):",
        existing_summary if existing_summary else "No existing summary.",
        "---",
        "Recent Conversation Snippet involving this user:"
    ]

    if conversation_history:
        for msg in conversation_history:
            prompt_lines.append(f"{msg.get('author', 'Unknown')}: {msg.get('content', '')}")
    else:
        prompt_lines.append("No recent conversation history provided.")
    
    prompt_lines.extend([
        "---",
        "Based on the existing summary and the recent conversation, provide an updated, concise interaction summary for this user. Focus on their expressed interests, preferences, questions, or any significant personal details they shared. If the existing summary is good and no update is needed, you can state 'No significant update needed' or simply refine it slightly. The summary should be a few sentences at most.",
        "Updated Summary:"
    ])

    return "\n".join(prompt_lines)

def generate_details_and_preferences_prompt(user_message_content: str) -> str:
    """
    Generates a prompt for the LLM to extract personal details and preferences
    from a user's message and return them in a structured JSON format.
    """
    json_schema_description = """
Please extract any personal details and preferences mentioned by the user in the following message.
Format the output as a JSON object with two main keys: "personal_details" and "preferences".

The "personal_details" object can include fields such as:
- "name" (string): The user's name.
- "location" (string): The user's location (city, country, etc.).
- "occupation" (string): The user's job or profession.
- "age" (integer): The user's age.
- "family_members" (list of strings): e.g., ["wife", "2 kids"]
- "pets" (list of strings): e.g., ["dog named Max", "cat"]
- "hobbies" (list of strings): e.g., ["hiking", "reading sci-fi"]
- "contact_info" (object): e.g., {{"email": "...", "phone": "..."}}
- "education" (string): e.g., "degree in Computer Science"
- "mood_or_feeling" (string): Current mood or feeling expressed.
- "goals_or_aspirations" (list of strings): e.g., ["learn to code", "travel more"]
- "recent_activities" (list of strings): e.g., ["went to a concert", "finished a project"]
- "other_facts" (list of strings): Any other specific facts mentioned by the user.

The "preferences" object can include fields such as:
- "likes" (list of strings): Things the user explicitly likes.
- "dislikes" (list of strings): Things the user explicitly dislikes.
- "communication_style" (string): Preferred way of communication (e.g., "formal", "casual", "prefers short messages").
- "topic_interest" (list of strings): Topics the user is interested in discussing.
- "preferred_activities" (list of strings): Activities the user enjoys.
- "preferred_interaction_types" (list of strings): e.g., ["getting advice", "casual chat", "learning new things"]
- "content_preferences" (object): e.g., {{"format": "text", "length": "concise", "tone": "friendly"}}
- "privacy_preferences" (string): e.g., "prefers not to share personal information"

Only include fields for which information is explicitly or strongly implicitly present in the user's message.
If no relevant information for a field is found, omit the field. If no personal details are found, the "personal_details" object can be empty or omitted.
Similarly, if no preferences are found, the "preferences" object can be empty or omitted.
If no information is found for either category, return an empty JSON object: {{}}.

User's message:
---
{user_message_content}
---
JSON Output:
"""
    return json_schema_description.format(user_message_content=user_message_content)


# --- LLM Interface Logic (from llmi.py) ---

async def get_llm_response(prompt: str) -> dict:
    """
    Communicates with the Google Gemini API to get a response from the LLM.
    Uses the model specified by the GEMINI_MODEL_NAME environment variable.
    The API key is configured globally from the GOOGLE_API_KEY environment variable.
    """
    if not GOOGLE_API_KEY: # Check if the global key was successfully configured
        logger.error("Error: GOOGLE_API_KEY not configured or failed to load.")
        return {"status": "error", "text": "Error: GOOGLE_API_KEY not configured.", "error_code": "API_KEY_MISSING"}

    try:
        model_to_use = gemini_model_name
        
        logger.info(f"LLMI: Attempting to initialize model: {model_to_use}")
        configured_model = genai.GenerativeModel(model_to_use)
        logger.info(f"LLMI: Successfully initialized model: {model_to_use}")
        
        if not configured_model:
            logger.error("LLMI: configured_model is not initialized.")
            return {"status": "error", "text": "LLM model could not be initialized.", "error_code": "MODEL_INITIALIZATION_FAILED"}

        logger.debug(f"Attempting to generate content with prompt: {prompt[:100]}...")
        api_response = await asyncio.to_thread(configured_model.generate_content, prompt)
        logger.debug(f"LLMI: Raw API response: {api_response}")

        if api_response.candidates:
            if api_response.candidates[0].finish_reason == 'SAFETY':
                logger.warning(f"Warning: Gemini API response was blocked due to safety settings. Prompt: '{prompt[:100]}...'")
                return {"status": "error", "text": "I'm sorry, but I cannot provide a response to that due to safety guidelines.", "error_code": "SAFETY_BLOCKED"}

            response_text = None
            if hasattr(api_response.candidates[0].content, 'parts') and api_response.candidates[0].content.parts:
                response_text = "".join(part.text for part in api_response.candidates[0].content.parts if hasattr(part, 'text')).strip()
            elif hasattr(api_response, 'text') and api_response.text:
                 response_text = api_response.text.strip()

            if response_text:
                return {"status": "success", "text": response_text}
            else:
                logger.error(f"Error: Gemini API response yielded empty text or unexpected format. Response: {api_response}")
                return {"status": "error", "text": "Sorry, I couldn't generate a response. The model returned an empty answer.", "error_code": "GENERATION_EMPTY"}
        else:
            if hasattr(api_response, 'prompt_feedback') and api_response.prompt_feedback.block_reason:
                reason_message = api_response.prompt_feedback.block_reason_message or str(api_response.prompt_feedback.block_reason)
                logger.error(f"Error: Prompt blocked by Gemini API. Reason: {api_response.prompt_feedback.block_reason}")
                return {"status": "error", "text": f"I'm sorry, but your request could not be processed. Reason: {reason_message}", "error_code": "PROMPT_BLOCKED"}
            
            logger.error(f"Error: No candidates found in Gemini API response. Response: {api_response}")
            return {"status": "error", "text": "Sorry, I couldn't generate a response from the AI (no candidates found).", "error_code": "NO_CANDIDATES_FOUND"}

    except DeadlineExceeded as e:
        logger.error(f"Google API DeadlineExceeded: {e}", exc_info=True)
        return {"status": "error", "text": "The request to the LLM timed out. Please try again.", "error_code": "GOOGLE_API_DEADLINE_EXCEEDED"}
    except ServiceUnavailable as e:
        logger.error(f"Google API ServiceUnavailable: {e}", exc_info=True)
        return {"status": "error", "text": "The LLM service is temporarily unavailable. Please try again later.", "error_code": "GOOGLE_API_SERVICE_UNAVAILABLE"}
    except InvalidArgument as e:
        logger.error(f"Google API InvalidArgument: {e}", exc_info=True)
        return {"status": "error", "text": f"There was an issue with the request format to the LLM: {e}", "error_code": "GOOGLE_API_INVALID_ARGUMENT"}
    except PermissionDenied as e:
        logger.error(f"Google API PermissionDenied: {e}", exc_info=True)
        return {"status": "error", "text": "Permission denied when trying to access the LLM. Check API key and permissions.", "error_code": "GOOGLE_API_PERMISSION_DENIED"}
    except Unauthenticated as e:
        logger.error(f"Google API Unauthenticated: {e}", exc_info=True)
        return {"status": "error", "text": "Authentication failed with the LLM. Check API key.", "error_code": "GOOGLE_API_UNAUTHENTICATED"}
    except GoogleAPICallError as e:
        logger.error(f"Google API Call Error: {e}", exc_info=True)
        return {"status": "error", "text": f"A Google API error occurred: {e}", "error_code": "GOOGLE_API_CALL_ERROR"}
    except genai.types.generation_types.BlockedPromptException as e:
        logger.error(f"Google Gemini API BlockedPromptException: {e}", exc_info=True)
        return {"status": "error", "text": "I'm sorry, but your request was blocked by the content filter.", "error_code": "BLOCKED_PROMPT_EXCEPTION"}
    except genai.types.generation_types.StopCandidateException as e:
        logger.error(f"Google Gemini API StopCandidateException: {e}", exc_info=True)
        return {"status": "error", "text": "I'm sorry, an issue occurred while generating the response, and it was stopped.", "error_code": "STOP_CANDIDATE_EXCEPTION"}
    except google.api_core.exceptions.ResourceExhausted as e:
        logger.error(f"Google API ResourceExhausted: {e}", exc_info=True)
        return {"status": "error", "text": f"API limit reached or resource exhausted: {str(e)}", "error_code": "API_RESOURCE_EXHAUSTED"}
    except Exception as e:
        logger.error(f"Unexpected error in get_llm_response: {e}", exc_info=True)
        return {"status": "error", "text": f"An unexpected error occurred while trying to get a response from the LLM: {str(e)}", "error_code": "UNEXPECTED_GENERATION_EXCEPTION"}


# --- Short-Term Memory Manager (from stmm.py) ---

class ShortTermMemoryManager:
    """
    Manages short-term conversation history for different channels.
    """
    def __init__(self, max_history_length: int = 10):
        self.histories = collections.defaultdict(lambda: collections.deque(maxlen=max_history_length))
        self.max_history_length = max_history_length

    def add_message(self, channel_id: int, author_name: str, content: str):
        message_record = {'author': author_name, 'content': content}
        self.histories[channel_id].append(message_record)

    def get_recent_history(self, channel_id: int, limit: int = 5) -> list:
        channel_history = list(self.histories[channel_id])
        return channel_history[-limit:]


# --- Long-Term Memory Store (from ltms.py) ---

class LongTermMemoryStore:
    def __init__(self, supabase_url: str, supabase_key: str):
        try:
            key_display = supabase_key[:5] + "..." if supabase_key and len(supabase_key) > 8 else "INVALID_OR_SHORT_KEY"
            logger.info(f"Initializing Supabase client with URL: {supabase_url}, Key (partial): {key_display}")
            self.supabase: Client = create_client(supabase_url, supabase_key)
            logger.info(f"Successfully connected to Supabase at {supabase_url}.")
            logger.info("Supabase client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise

    async def get_user_profile(self, user_id: str) -> dict | None:
        try:
            _query = self.supabase.table("user_profiles").select("*").eq("user_id", user_id).maybe_single()
            response = await asyncio.to_thread(_query.execute)
            if hasattr(response, 'error') and response.error:
                logger.error(f"Supabase API error in get_user_profile for {user_id}: {response.error.message} (Code: {response.error.code if hasattr(response.error, 'code') else 'N/A'})")
                return None
            if response.data:
                logger.debug(f"Retrieved profile for user_id: {user_id}")
                return response.data
            else:
                logger.debug(f"No profile found for user_id: {user_id}")
                return None
        except Exception as e:
            logger.error(f"Exception retrieving profile for user_id {user_id}: {e}")
            return None

    async def get_or_create_user_profile(self, user_id: str, username: str, guild_id: int = None, guild_name: str = None) -> dict | None:
        try:
            existing_profile = await self.get_user_profile(user_id)
            if existing_profile:
                logger.info(f"Profile already exists for user_id: {user_id}. Skipping creation.")
                return existing_profile

            now_iso = datetime.now(timezone.utc).isoformat()
            profile_data = {
                "user_id": user_id,
                "username_history": [username],
                "created_at": now_iso,
                "last_updated_at": now_iso,
                "last_seen_at": now_iso,
                "message_count": 0,
                "preferences": {},
                "mentioned_personal_details": {},
                "interaction_summary": None,
            }
            
            _query = self.supabase.table("user_profiles").insert(profile_data)
            response = await asyncio.to_thread(_query.execute)
            
            if hasattr(response, 'error') and response.error:
                logger.error(f"Supabase API error during insert in get_or_create_user_profile for {user_id}: {response.error.message} (Code: {response.error.code if hasattr(response.error, 'code') else 'N/A'})")
                return None

            if response.data:
                logger.info(f"Successfully created profile for user_id: {user_id}, username: {username}")
                return response.data[0]
            else:
                logger.error(f"Failed to create profile for user_id: {user_id} (no data returned after insert)")
                return None
        except Exception as e:
            logger.error(f"Exception creating profile for user_id {user_id}: {e}")
            return None

    async def update_user_profile(self, user_id: str, updates: dict) -> bool:
        try:
            if not updates:
                logger.warning(f"No updates provided for user_id: {user_id}. Skipping update.")
                return False

            updates_for_db = updates.copy()
            updates_for_db["last_updated_at"] = datetime.now(timezone.utc).isoformat()

            _query = self.supabase.table("user_profiles").update(updates_for_db).eq("user_id", user_id)
            response = await asyncio.to_thread(_query.execute)

            if hasattr(response, 'error') and response.error:
                logger.error(f"Supabase API error in update_user_profile for {user_id}: {response.error.message} (Code: {response.error.code if hasattr(response.error, 'code') else 'N/A'})")
                return False
            
            if response.data:
                 logger.info(f"Successfully updated profile for user_id: {user_id}. Data: {response.data}")
                 return True
            else:
                logger.info(f"Profile update call for user_id: {user_id} executed. No data returned in response, but no explicit API error. Assuming success or no change needed.")
                return True
        except Exception as e:
            logger.error(f"Exception updating profile for user_id {user_id}: {e}")
            return False

    async def increment_user_message_count(self, user_id: str) -> bool:
        logger.debug(f"LTMS: Attempting to increment message count for user_id: {user_id}.")
        logger.warning(f"LTMS: Potential race condition in increment_user_message_count for user_id: {user_id}. This is a read-modify-write operation.")
        try:
            profile = await self.get_user_profile(user_id)
            if not profile:
                logger.warning(f"LTMS: No profile found to increment message count for user_id: {user_id}.")
                return False

            original_message_count = profile.get("message_count", 0)
            new_message_count = original_message_count + 1
            logger.debug(f"LTMS: User {user_id} - Original message_count: {original_message_count}, New message_count: {new_message_count}")
            now_iso = datetime.now(timezone.utc).isoformat()
            
            updates = {
                "message_count": new_message_count,
                "last_seen_at": now_iso,
                "last_updated_at": now_iso
            }
            
            _query = self.supabase.table("user_profiles").update(updates).eq("user_id", user_id)
            response = await asyncio.to_thread(_query.execute)

            if hasattr(response, 'error') and response.error:
                logger.error(f"Supabase API error in increment_user_message_count for {user_id}: {response.error.message} (Code: {response.error.code if hasattr(response.error, 'code') else 'N/A'})")
                return False
            if response.data:
                logger.info(f"LTMS: Successfully incremented message count for user_id: {user_id} to {new_message_count}.")
                return True
            else:
                logger.info(f"LTMS: Message count increment call for {user_id} (to {new_message_count}) executed. No data returned, but no explicit API error. Assuming success.")
                return True
        except Exception as e:
            logger.error(f"Exception incrementing message count for user_id {user_id}: {e}")
            return False

    async def update_user_summary(self, user_id: str, summary: str) -> bool:
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            updates = {
                "interaction_summary": summary,
                "last_updated_at": now_iso
            }
            _query = self.supabase.table("user_profiles").update(updates).eq("user_id", user_id)
            response = await asyncio.to_thread(_query.execute)

            if hasattr(response, 'error') and response.error:
                logger.error(f"Supabase API error in update_user_summary for {user_id}: {response.error.message} (Code: {response.error.code if hasattr(response.error, 'code') else 'N/A'})")
                return False
            
            logger.info(f"Successfully updated interaction summary for user {user_id}.")
            return True
        except Exception as e:
            logger.error(f"Exception updating summary for user_id {user_id}: {e}")
            return False

    async def add_username_to_history(self, user_id: str, username: str) -> bool:
        logger.debug(f"LTMS: Attempting to add username '{username}' to history for user_id: {user_id}.")
        logger.warning(f"LTMS: Potential race condition in add_username_to_history for user_id: {user_id}. This is a read-modify-write operation.")
        try:
            profile = await self.get_user_profile(user_id)
            if not profile:
                logger.warning(f"LTMS: No profile found for user_id {user_id} to add username '{username}' to history.")
                return False

            original_username_history = list(profile.get("username_history", []))
            current_username_history = profile.get("username_history", [])

            if not current_username_history or current_username_history[-1] != username:
                logger.debug(f"LTMS: User {user_id} - Original username_history: {original_username_history}. Adding '{username}'.")
                current_username_history.append(username)
                now_iso = datetime.now(timezone.utc).isoformat()
                updates = {
                    "username_history": current_username_history,
                    "last_updated_at": now_iso
                }
                _query = self.supabase.table("user_profiles").update(updates).eq("user_id", user_id)
                response = await asyncio.to_thread(_query.execute)
                if hasattr(response, 'error') and response.error:
                    logger.error(f"LTMS: Supabase API error in add_username_to_history for {user_id} while adding '{username}': {response.error.message} (Code: {response.error.code if hasattr(response.error, 'code') else 'N/A'})")
                    return False
                logger.info(f"LTMS: Successfully added username '{username}' to history for user {user_id}. New history: {current_username_history}")
            else:
                logger.debug(f"LTMS: Username '{username}' is already the latest in history for user {user_id}. No update needed. History: {current_username_history}")
            
            return True
        except Exception as e:
            logger.error(f"Exception adding username to history for user_id {user_id}: {e}")
            return False


# --- Discord Bot Logic (from original bot.py) ---

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True

client = discord.Client(intents=intents)

stmm = ShortTermMemoryManager(max_history_length=CHANNEL_STM_MAX_LEN)

ltms_instance = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        ltms_instance = LongTermMemoryStore(supabase_url=SUPABASE_URL, supabase_key=SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize LongTermMemoryStore with Supabase: {e}")
else:
    logger.warning("SUPABASE_URL or SUPABASE_KEY is not set. LTMS features will be disabled.")


@client.event
async def on_ready():
    logger.info(f'{client.user} has connected to Discord!')
    logger.info(f'Successfully connected to: {", ".join([guild.name for guild in client.guilds])}')
    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY is not set. The bot will not be able to respond to mentions.")
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("SUPABASE_URL or SUPABASE_KEY is not set in .env file. Long-term memory features will be disabled.")
    elif ltms_instance is None:
        logger.error("Supabase URL/Key are set, but LTMS failed to initialize. Long-term memory features will be disabled.")

async def ensure_user_profile_exists(ltms: LongTermMemoryStore, user_id: str, username: str, logger_param: logging.Logger): # Renamed logger to logger_param to avoid conflict
    if not ltms:
        logger_param.info("LTMS not initialized, skipping profile check.")
        return None
    profile = None
    try:
        profile = await ltms.get_or_create_user_profile(user_id, username)
        if profile:
            logger_param.info(f"LTMS: Profile ensured (retrieved or created) for {user_id} ({username}).")
            current_usernames = profile.get("username_history", [])
            if not current_usernames or username != current_usernames[-1]:
                logger_param.info(f"LTMS: Username for {user_id} ('{username}') differs from latest in history ('{current_usernames[-1] if current_usernames else 'None'}'). Updating profile.")
                await ltms.add_username_to_history(user_id, username)
                profile_after_history_update = await ltms.get_user_profile(user_id)
                if profile_after_history_update:
                    profile = profile_after_history_update
                    logger_param.info(f"LTMS: Re-fetched profile for {user_id} after username history update.")
                else:
                    profile = None
                    logger_param.warning(f"LTMS: Failed to re-fetch profile for {user_id} after username history update; profile is now None.")
        else:
            logger_param.warning(f"LTMS: Failed to get or create profile for {user_id} ({username}).")
        return profile
    except Exception as e:
        logger_param.error(f"LTMS Error: Could not ensure profile for {user_id} ({username}): {e}")
        return None

async def _handle_llm_response(message: discord.Message, context: dict, stm_param: ShortTermMemoryManager, ltms_param: LongTermMemoryStore, google_api_key_param: str, logger_param: logging.Logger, user_profile: Optional[dict] = None) -> bool:
    _first_llm_call_successful_and_has_text = False
    recent_history = stm_param.get_recent_history(channel_id=message.channel.id, limit=LLM_RESPONSE_CONTEXT_HISTORY_LIMIT)
    current_message_content_for_prompt = context.get('message_content', '')
    replied_to_bot_message_content = context.get('replied_to_bot_message_content')

    if replied_to_bot_message_content:
        bot_message_context = {
            "author": client.user.name,
            "content": replied_to_bot_message_content
        }
        recent_history.append(bot_message_context)
        logger_param.debug(f"Appended replied-to bot message to history for prompt generation.")

    if user_profile:
        logger_param.debug(f"User profile data for prompt generation: {user_profile}")
    else:
        logger_param.debug(f"No user profile data provided for prompt generation for user {message.author.name}.")


    prompt = generate_prompt(
        current_message_content=current_message_content_for_prompt,
        history=recent_history,
        user_profile=user_profile
    )
    logger_param.debug(f"Generated prompt for LLM (with history, user profile, potentially including replied-to message): {prompt}")
    
    llm_response_data = await get_llm_response(prompt)
    
    if llm_response_data:
        if llm_response_data.get('status') == "success":
            llm_text_from_dict = llm_response_data['text']
            if len(llm_text_from_dict) > MAX_RESPONSE_CHAR_LIMIT:
                logger_param.info(f"LLM response exceeded {MAX_RESPONSE_CHAR_LIMIT} characters. Truncating.")
                llm_text_from_dict = llm_text_from_dict[:MAX_RESPONSE_CHAR_LIMIT-3] + "..."
            
            if "Sorry," in llm_text_from_dict:
                logger_param.info(f"LLM response (status: success) contained 'Sorry,': \"{llm_text_from_dict}\"")
            else:
                logger_param.info(f"Sending LLM response to #{message.channel}: {llm_text_from_dict}")
            
            await message.channel.send(llm_text_from_dict)
            stm_param.add_message(
                channel_id=message.channel.id,
                author_name=client.user.name,
                content=llm_text_from_dict # Store potentially truncated message
            )
            _first_llm_call_successful_and_has_text = True
        elif llm_response_data.get('status') == "error":
            error_message = llm_response_data.get('text', "An unspecified error occurred with the AI.")
            logger_param.error(f"LLM call (prompt) failed. Status: error. Message: {error_message}")
            await message.channel.send(error_message)
        else:
            status_val = llm_response_data.get('status', 'Unknown')
            text_val = llm_response_data.get('text', 'No text provided')
            logger_param.error(f"LLM call (prompt) returned non-standard response. Status: '{status_val}'. Text: '{text_val}'")
            await message.channel.send(f"Sorry, I received an unusual response from the AI (Status: {status_val}).")
    else:
        logger_param.error("LLM Error: Did not receive a valid data object from get_llm_response for prompt.")
        await message.channel.send("Sorry, I couldn't retrieve a response from the AI at the moment.")
    return _first_llm_call_successful_and_has_text

async def _handle_user_summary_update(message: discord.Message, context: dict, stm_param: ShortTermMemoryManager, ltms_param: LongTermMemoryStore, google_api_key_param: str, user_id_str: str, username: str, profile: dict, logger_param: logging.Logger):
    if not ltms_param:
        logger_param.info("LTMS not initialized, skipping user summary update.")
        return
    if not profile:
        logger_param.warning(f"LTMS: No profile object provided to _handle_user_summary_update for user {user_id_str} ({username}). Skipping summary update.")
        return

    logger_param.info(f"LTMS: Attempting to update interaction summary for user {user_id_str} ({username}).")
    try:
        existing_summary = profile.get("interaction_summary") if profile else None
        summary_conversation_history = stm_param.get_recent_history(channel_id=message.channel.id, limit=LLM_SUMMARY_CONTEXT_HISTORY_LIMIT)
        summary_prompt = generate_user_summary_prompt(
            user_id=user_id_str,
            username=username,
            conversation_history=summary_conversation_history,
            existing_summary=existing_summary
        )
        logger_param.debug(f"LTMS: Generated summary prompt for {user_id_str}: {summary_prompt[:200]}...")
        summary_response_data = await get_llm_response(summary_prompt)
        if summary_response_data:
            if summary_response_data.get('status') == "success":
                summary_text_from_dict = summary_response_data['text']
                if summary_text_from_dict and \
                   "no significant update needed" not in summary_text_from_dict.lower() and \
                   len(summary_text_from_dict.strip()) > 5:
                    actual_summary_text = summary_text_from_dict.strip()
                    success = await ltms_param.update_user_summary(user_id_str, actual_summary_text)
                    if success:
                        logger_param.info(f"LTMS: Successfully updated interaction summary for user {user_id_str}.")
                    else:
                        logger_param.error(f"LTMS: Failed to update interaction summary for user {user_id_str} in database.")
                elif summary_text_from_dict:
                    logger_param.info(f"LTMS: LLM indicated no significant summary update needed for {user_id_str} or summary too short. Response: '{summary_text_from_dict}'")
                else:
                    logger_param.warning(f"LTMS: LLM summary call for {user_id_str} status success but no text or empty text.")
            elif summary_response_data.get('status') == "error":
                error_message = summary_response_data.get('text', "Unspecified error during summary generation.")
                logger_param.error(f"LTMS: LLM summary call for {user_id_str} failed. Status: error. Message: {error_message}")
            else:
                status_val = summary_response_data.get('status', 'Unknown')
                text_val = summary_response_data.get('text', 'No text provided')
                logger_param.error(f"LTMS: LLM summary call for {user_id_str} returned non-standard response. Status: '{status_val}'. Text: '{text_val}'")
        else:
            logger_param.warning(f"LTMS: LLM did not return a valid data object for summary for user {user_id_str}.")
    except Exception as e:
        logger_param.error(f"LTMS Error: Could not update interaction summary for {user_id_str}: {e}")

async def _handle_details_and_preferences_update(discord_client_param: discord.Client, message: discord.Message, ltms_param: LongTermMemoryStore, logger_param: logging.Logger):
    if not ltms_param:
        logger_param.info("LTMS not initialized, skipping details and preferences update.")
        return
    if message.author.bot:
        return

    user_id_str = str(message.author.id)
    username = message.author.name
    message_content = message.content

    logger_param.info(f"LTMS: Attempting to extract details/preferences for user {user_id_str} ({username}) from message: \"{message_content[:100]}...\"")
    try:
        extraction_prompt = generate_details_and_preferences_prompt(message_content)
        logger_param.debug(f"LTMS: Generated details/preferences extraction prompt for {user_id_str}: {extraction_prompt[:200]}...")
        llm_response_data = await get_llm_response(extraction_prompt)

        if not llm_response_data or llm_response_data.get('status') != "success":
            error_text = llm_response_data.get('text', 'No text in response') if llm_response_data else 'No response data'
            logger_param.warning(f"LTMS: LLM call for details/preferences for {user_id_str} failed or did not return success. Status: {llm_response_data.get('status', 'N/A')}, Text: {error_text}")
            return

        llm_json_str = llm_response_data['text']
        if not llm_json_str.strip():
            logger_param.info(f"LTMS: LLM returned empty string for details/preferences for user {user_id_str}.")
            return
        
        logger_param.debug(f"LTMS: LLM response for details/preferences for {user_id_str}: {llm_json_str}")
        extracted_data = None
        try:
            json_start = llm_json_str.find('{')
            json_end = llm_json_str.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                llm_json_str_cleaned = llm_json_str[json_start:json_end]
                extracted_data = json.loads(llm_json_str_cleaned)
            else:
                logger_param.warning(f"LTMS: No JSON object found in LLM response for {user_id_str}. Response: {llm_json_str}")
                return
        except json.JSONDecodeError as e:
            logger_param.error(f"LTMS: Failed to parse JSON response for details/preferences for {user_id_str}. Error: {e}. Response: {llm_json_str}")
            return

        if not extracted_data or (not extracted_data.get("personal_details") and not extracted_data.get("preferences")):
            logger_param.info(f"LTMS: LLM extracted no new details or preferences for user {user_id_str}.")
            return

        new_personal_details = extracted_data.get("personal_details", {})
        new_preferences = extracted_data.get("preferences", {})
        
        profile = await ensure_user_profile_exists(ltms_param, user_id_str, username, logger_param) # Use ltms_param
        if not profile:
            logger_param.warning(f"LTMS: Could not get or create profile for {user_id_str} ({username}) before details/preferences update. Skipping.")
            return
            
        existing_personal_details = profile.get("mentioned_personal_details", {})
        existing_preferences = profile.get("preferences", {})
        merged_personal_details = existing_personal_details.copy()
        merged_preferences = existing_preferences.copy()

        if new_personal_details:
            logger_param.info(f"LTMS: Merging new personal details for {user_id_str}: {new_personal_details}")
            for key, value in new_personal_details.items():
                if value is None or (isinstance(value, (str, list, dict)) and not value): continue
                if key in ["name", "location", "occupation", "age", "education", "mood_or_feeling", "contact_info"]:
                    merged_personal_details[key] = value
                elif key in ["family_members", "pets", "hobbies", "goals_or_aspirations", "recent_activities", "other_facts"]:
                    current_list = merged_personal_details.get(key, [])
                    if isinstance(value, list): [current_list.append(item) for item in value if item not in current_list]
                    elif value not in current_list : current_list.append(value)
                    merged_personal_details[key] = current_list
                else: merged_personal_details[key] = value
        
        if new_preferences:
            logger_param.info(f"LTMS: Merging new preferences for {user_id_str}: {new_preferences}")
            for key, value in new_preferences.items():
                if value is None or (isinstance(value, (str, list, dict)) and not value): continue
                if key in ["communication_style", "privacy_preferences", "content_preferences"]:
                    merged_preferences[key] = value
                elif key in ["likes", "dislikes", "topic_interest", "preferred_activities", "preferred_interaction_types"]:
                    current_list = merged_preferences.get(key, [])
                    if isinstance(value, list): [current_list.append(item) for item in value if item not in current_list]
                    elif value not in current_list: current_list.append(value)
                    merged_preferences[key] = current_list
                else: merged_preferences[key] = value
        
        made_changes_details = merged_personal_details != existing_personal_details
        made_changes_prefs = merged_preferences != existing_preferences

        if made_changes_details or made_changes_prefs:
            update_payload = {}
            if made_changes_details: update_payload["mentioned_personal_details"] = merged_personal_details
            if made_changes_prefs: update_payload["preferences"] = merged_preferences
            
            success = await ltms_param.update_user_profile(user_id_str, update_payload) # Use ltms_param
            if success: logger_param.info(f"LTMS: Successfully updated details/preferences for user {user_id_str}. Details changed: {made_changes_details}, Prefs changed: {made_changes_prefs}")
            else: logger_param.error(f"LTMS: Failed to update details/preferences for user {user_id_str} in database.")
        else:
            logger_param.info(f"LTMS: No changes to details/preferences after merging for user {user_id_str}.")
    except Exception as e:
        logger_param.error(f"LTMS Error: Unexpected error in _handle_details_and_preferences_update for {user_id_str}: {e}", exc_info=True)

@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return

    user_id_str = None
    profile = None

    if message.author and not message.author.bot:
        user_id_str = str(message.author.id)
        # Use global logger instance for ensure_user_profile_exists
        profile = await ensure_user_profile_exists(ltms_instance, user_id_str, message.author.name, logger)

        if ltms_instance:
            try:
                success = await ltms_instance.increment_user_message_count(user_id_str)
                if success: logger.info(f"LTMS: Incremented message count for user {user_id_str}.")
                else: logger.warning(f"LTMS: Failed to increment message count for user {user_id_str} (profile might not exist or other error).")
            except Exception as e:
                logger.error(f"LTMS Error: Could not increment message count for {user_id_str}: {e}")
            
            if message.content:
                if profile:
                     # Use global client, ltms_instance, and logger for _handle_details_and_preferences_update
                     asyncio.create_task(_handle_details_and_preferences_update(client, message, ltms_instance, logger))
                else:
                    logger.warning(f"LTMS: Profile was None for user {user_id_str} before attempting details/preferences update task. Task not started.")

    stmm.add_message(
        channel_id=message.channel.id,
        author_name=message.author.name,
        content=message.content
    )

    should_respond = False
    replied_to_bot_message_content = None
    trigger_reason = ""

    if client.user.mentioned_in(message):
        should_respond = True
        trigger_reason = "direct mention"

    if not should_respond:
        content_lower = message.content.lower()
        if 'aibro' in content_lower or 'ai bro' in content_lower:
            should_respond = True
            trigger_reason = "keyword 'aibro' or 'ai bro'"

    if message.reference and message.reference.resolved and isinstance(message.reference.resolved, discord.Message):
        if message.reference.resolved.author == client.user:
            replied_to_bot_message_content = message.reference.resolved.content
            if not should_respond:
                should_respond = True
                trigger_reason = "reply to bot message"

    if should_respond:
        log_message_content = message.content[:100] + "..." if len(message.content) > 100 else message.content
        replied_log = f" Bot's original: \"{replied_to_bot_message_content[:100]}...\"" if replied_to_bot_message_content else ""
        logger.info(f"Response triggered for {message.author.name} ({message.author.id}) due to {trigger_reason}. User's message: \"{log_message_content}\"{replied_log}")
        
        current_context = {
            'message_content': message.content,
            'replied_to_bot_message_content': replied_to_bot_message_content
        }
        
        try:
            async with message.channel.typing():
                # Use global stmm, ltms_instance, GOOGLE_API_KEY, and logger for _handle_llm_response
                _first_llm_call_successful_and_has_text = await _handle_llm_response(
                    message, current_context, stmm, ltms_instance, GOOGLE_API_KEY, logger, profile
                )

                if _first_llm_call_successful_and_has_text and ltms_instance and message.author and not message.author.bot:
                    if user_id_str and profile:
                         # Use global stmm, ltms_instance, GOOGLE_API_KEY, and logger for _handle_user_summary_update
                         await _handle_user_summary_update(
                             message, current_context, stmm, ltms_instance, GOOGLE_API_KEY,
                             user_id_str, message.author.name, profile, logger
                         )
                    else:
                        logger.warning(f"Skipping user summary update for {message.author.name} because user_id_str or profile was not available (user_id_str: {user_id_str is not None}, profile: {profile is not None}).")
        except discord.Forbidden as e:
            logger.error(f"Discord Forbidden error: {e}. Check bot permissions for channel {message.channel.id}.")
        except discord.HTTPException as e:
            logger.error(f"Discord HTTP error: {e}")
            try:
                await message.channel.send("I ran into an issue communicating with Discord. Please try again later.")
            except discord.HTTPException:
                logger.error("Failed to send Discord HTTP error message to channel.")
        except Exception as e:
            logger.error(f"An unexpected error occurred in message processing for message ID {message.id}: {e}", exc_info=True)
            try:
                await message.channel.send("Sorry, I encountered an unexpected error while processing your message.")
            except discord.HTTPException:
                logger.error(f"Failed to send generic error message to channel for message ID {message.id}.")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Configure logging basic settings
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Re-check logger after basicConfig
    logger.info("Logging configured.")

    if not DISCORD_BOT_TOKEN:
        logger.critical("DISCORD_BOT_TOKEN not found. Make sure to set it in your .env file.")
    else:
        if not GOOGLE_API_KEY: # This check is already done above, but good for emphasis before run
            logger.warning("GOOGLE_API_KEY not found in .env file. LLM features will be disabled if not already caught.")
        try:
            logger.info("Attempting to run Discord client...")
            client.run(DISCORD_BOT_TOKEN)
        except discord.errors.LoginFailure:
            logger.critical("Failed to log in. Please check your DISCORD_BOT_TOKEN.")
        except Exception as e:
            logger.exception(f"An unexpected error occurred during client.run: {e}")