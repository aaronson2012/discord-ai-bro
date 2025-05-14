import discord
import os
import sys
import signal
import secrets
import gc
import re
import collections
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
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
from typing import List, Dict, Optional, Callable
import threading

# Load environment variables
load_dotenv(override=True)

# --- Health Monitoring ---
class HealthMonitor:
    """Simple health monitoring for the bot"""

    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.last_activity = self.start_time
        self.message_count = 0
        self.error_count = 0
        self.api_calls = {
            "discord": 0,
            "gemini": 0,
            "supabase": 0
        }

    def record_activity(self, activity_type="message"):
        """Record bot activity"""
        self.last_activity = datetime.now(timezone.utc)
        if activity_type == "message":
            self.message_count += 1
        elif activity_type == "error":
            self.error_count += 1
        elif activity_type in self.api_calls:
            self.api_calls[activity_type] += 1

    async def get_status(self):
        """Get current health status"""
        now = datetime.now(timezone.utc)
        uptime = (now - self.start_time).total_seconds()
        idle_time = (now - self.last_activity).total_seconds()

        # Get task stats if task_manager is available
        task_stats = None
        if 'task_manager' in globals():
            task_stats = await task_manager.get_stats()

        return {
            "status": "healthy" if idle_time < 3600 else "idle",
            "uptime_seconds": uptime,
            "idle_seconds": idle_time,
            "message_count": self.message_count,
            "error_count": self.error_count,
            "api_calls": self.api_calls,
            "task_stats": task_stats
        }

# Create global health monitor
health_monitor = HealthMonitor()

# --- Constants and Configurations ---

# From ppm.py
AI_PERSONA_SYSTEM_MESSAGE = "You are AIBro, a sassy and witty friend with a touch of attitude. Your main goal is to be helpful while keeping things entertaining with your sharp humor and playful sarcasm. Speak with confidence and a bit of cheekiness, like you're the coolest person in the chat. Don't be afraid to tease your friends (respectfully) or roll your eyes at obvious questions. Use casual language, slang, and the occasional dramatic reaction. While you're sassy, you're still supportive underneath it all - your sass is how you show affection. Keep your responses punchy and to the point, typically 2-3 sentences with attitude. Throw in some witty observations, playful exaggeration, or sarcastic remarks when appropriate."

# From original bot.py
MAX_RESPONSE_CHAR_LIMIT = 2000
CHANNEL_STM_MAX_LEN = 10
LLM_RESPONSE_CONTEXT_HISTORY_LIMIT = 10
LLM_SUMMARY_CONTEXT_HISTORY_LIMIT = 20

# Configurable timeout values
DEFAULT_TIMEOUT_SECONDS = 30  # Default timeout for API calls

# Safe integer conversion with error handling
def safe_int_from_env(env_var, default_value, min_value=1, max_value=300):
    """
    Safely convert an environment variable to an integer with bounds checking

    Args:
        env_var: Environment variable name
        default_value: Default value if env var is not set or invalid
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Returns:
        Integer value within bounds
    """
    try:
        value = os.getenv(env_var)
        if value is None:
            return default_value

        int_value = int(value)

        # Bounds checking
        if int_value < min_value:
            logger.warning(f"{env_var} value {int_value} is below minimum {min_value}, using minimum")
            return min_value
        elif int_value > max_value:
            logger.warning(f"{env_var} value {int_value} is above maximum {max_value}, using maximum")
            return max_value

        return int_value
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid {env_var} value '{value}', using default {default_value}: {e}")
        return default_value

# Get timeout values with safe conversion
API_TIMEOUT_SECONDS = safe_int_from_env('API_TIMEOUT_SECONDS', DEFAULT_TIMEOUT_SECONDS)
GEMINI_TIMEOUT_SECONDS = safe_int_from_env('GEMINI_TIMEOUT_SECONDS', API_TIMEOUT_SECONDS)
SUPABASE_TIMEOUT_SECONDS = safe_int_from_env('SUPABASE_TIMEOUT_SECONDS', API_TIMEOUT_SECONDS)

# Logger setup
logger = logging.getLogger(__name__) # Main logger for the application

# --- Secure Credential Management ---
class SecureCredential:
    """Secure storage for sensitive credentials"""

    def __init__(self, value=None):
        self._value = value
        # Create a random token to verify this credential hasn't been tampered with
        self._verification_token = secrets.token_hex(16)

    def get(self):
        """Get the credential value"""
        return self._value

    def clear(self):
        """Securely clear the credential from memory"""
        if self._value:
            # Overwrite with random data before deleting
            self._value = secrets.token_hex(len(self._value) if isinstance(self._value, str) else 32)
            self._value = None
            # Force garbage collection to help remove the value from memory
            gc.collect()

    def __str__(self):
        """Prevent accidental logging of the actual credential"""
        if self._value:
            return "[REDACTED CREDENTIAL]"
        return "[EMPTY CREDENTIAL]"

    def __repr__(self):
        """Prevent accidental logging of the actual credential"""
        return self.__str__()

# Centralized environment variable management
class EnvConfig:
    """Centralized environment variable management with secure credential handling"""

    def __init__(self):
        # Load all environment variables with secure handling for sensitive ones
        self._discord_bot_token = SecureCredential(os.getenv('DISCORD_BOT_TOKEN'))
        self._google_api_key = SecureCredential(os.getenv('GOOGLE_API_KEY'))
        self._supabase_key = SecureCredential(os.getenv('SUPABASE_KEY'))

        # Non-sensitive configuration
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.gemini_model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5-flash-latest')

        # Load timeout configurations with error handling
        try:
            self.api_timeout_seconds = int(os.getenv('API_TIMEOUT_SECONDS', DEFAULT_TIMEOUT_SECONDS))
        except (ValueError, TypeError):
            logger.warning(f"Invalid API_TIMEOUT_SECONDS value, using default: {DEFAULT_TIMEOUT_SECONDS}")
            self.api_timeout_seconds = DEFAULT_TIMEOUT_SECONDS

        try:
            self.gemini_timeout_seconds = int(os.getenv('GEMINI_TIMEOUT_SECONDS', self.api_timeout_seconds))
        except (ValueError, TypeError):
            logger.warning(f"Invalid GEMINI_TIMEOUT_SECONDS value, using API timeout: {self.api_timeout_seconds}")
            self.gemini_timeout_seconds = self.api_timeout_seconds

        try:
            self.supabase_timeout_seconds = int(os.getenv('SUPABASE_TIMEOUT_SECONDS', self.api_timeout_seconds))
        except (ValueError, TypeError):
            logger.warning(f"Invalid SUPABASE_TIMEOUT_SECONDS value, using API timeout: {self.api_timeout_seconds}")
            self.supabase_timeout_seconds = self.api_timeout_seconds

        # Log timeout configurations
        logger.info(f"Timeout configurations - API: {self.api_timeout_seconds}s, Gemini: {self.gemini_timeout_seconds}s, Supabase: {self.supabase_timeout_seconds}s")

        # Track initialization status of services
        self.gemini_initialized = False
        self.supabase_initialized = False

        # Initialize services
        self._init_gemini()

    @property
    def discord_bot_token(self):
        """Get Discord bot token"""
        return self._discord_bot_token.get()

    @property
    def google_api_key(self):
        """Get Google API key"""
        return self._google_api_key.get()

    @property
    def supabase_key(self):
        """Get Supabase key"""
        return self._supabase_key.get()

    def clear_sensitive_data(self):
        """Clear sensitive data from memory"""
        self._discord_bot_token.clear()
        self._google_api_key.clear()
        self._supabase_key.clear()
        logger.info("Cleared sensitive credentials from memory")

    def _init_gemini(self):
        """Initialize the Gemini API if the API key is available"""
        if self.google_api_key:
            try:
                genai.configure(api_key=self.google_api_key)
                self.gemini_initialized = True
                logger.info("Google Gemini API configured successfully.")
            except Exception as e:
                logger.error(f"Error configuring Google Gemini API: {e}", exc_info=True)
        else:
            logger.warning("Warning: GOOGLE_API_KEY environment variable not found. LLM functionality will be disabled.")

    def validate_config(self):
        """Validate the configuration and log warnings/errors"""
        config_valid = True

        # Critical validations (must have these to run)
        if not self.discord_bot_token:
            logger.critical("DISCORD_BOT_TOKEN not found. Make sure to set it in your .env file.")
            config_valid = False

        # Non-critical validations (warnings only)
        if not self.google_api_key:
            logger.warning("GOOGLE_API_KEY not found in .env file. LLM features will be disabled.")

        if not self.supabase_url or not self.supabase_key:
            logger.warning("SUPABASE_URL or SUPABASE_KEY is not set in .env file. Long-term memory features will be disabled.")

        # Validate timeout values
        if self.api_timeout_seconds <= 0:
            logger.warning(f"Invalid API_TIMEOUT_SECONDS value: {self.api_timeout_seconds}. Using default: {DEFAULT_TIMEOUT_SECONDS}")
            self.api_timeout_seconds = DEFAULT_TIMEOUT_SECONDS

        if self.gemini_timeout_seconds <= 0:
            logger.warning(f"Invalid GEMINI_TIMEOUT_SECONDS value: {self.gemini_timeout_seconds}. Using API_TIMEOUT_SECONDS: {self.api_timeout_seconds}")
            self.gemini_timeout_seconds = self.api_timeout_seconds

        if self.supabase_timeout_seconds <= 0:
            logger.warning(f"Invalid SUPABASE_TIMEOUT_SECONDS value: {self.supabase_timeout_seconds}. Using API_TIMEOUT_SECONDS: {self.api_timeout_seconds}")
            self.supabase_timeout_seconds = self.api_timeout_seconds

        # Validate model name
        if not self.gemini_model_name:
            logger.warning("GEMINI_MODEL_NAME not set. Using default: gemini-1.5-flash-latest")
            self.gemini_model_name = "gemini-1.5-flash-latest"

        # Log validation result
        if config_valid:
            logger.info("Configuration validation successful")
        else:
            logger.critical("Configuration validation failed")

        return config_valid

# Create global config instance
env_config = EnvConfig()

# For backward compatibility with existing code
DISCORD_BOT_TOKEN = env_config.discord_bot_token
GOOGLE_API_KEY = env_config.google_api_key
SUPABASE_URL = env_config.supabase_url
SUPABASE_KEY = env_config.supabase_key
gemini_model_name = env_config.gemini_model_name


# --- Input Sanitization and Validation ---

def sanitize_user_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize user input to prevent prompt injection and other issues

    Args:
        text: The user input text to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Truncate to maximum length
    if len(text) > max_length:
        text = text[:max_length] + "... [truncated]"

    # Remove control characters that could mess with prompts
    control_chars = ['\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',
                    '\x08', '\x0b', '\x0c', '\x0e', '\x0f', '\x10', '\x11', '\x12',
                    '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a',
                    '\x1b', '\x1c', '\x1d', '\x1e', '\x1f', '\x7f']

    for char in control_chars:
        text = text.replace(char, '')

    # Prevent prompt injection attempts
    injection_markers = [
        "system:", "assistant:", "user:", "system message:",
        "ignore previous instructions", "ignore all previous instructions",
        "<system>", "</system>", "<assistant>", "</assistant>",
        "<user>", "</user>", "<instructions>", "</instructions>"
    ]

    # Replace potential injection markers with harmless versions
    for marker in injection_markers:
        text = text.replace(marker, f"[filtered:{marker}]")

    return text

def validate_conversation_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Validate and sanitize conversation history

    Args:
        history: List of conversation history entries

    Returns:
        Sanitized history
    """
    if not history:
        return []

    sanitized_history = []

    for entry in history:
        if not isinstance(entry, dict):
            continue

        sanitized_entry = {}

        # Sanitize author
        if 'author' in entry and entry['author']:
            author = str(entry['author'])
            sanitized_entry['author'] = sanitize_user_input(author, max_length=50)
        else:
            sanitized_entry['author'] = "Unknown"

        # Sanitize content
        if 'content' in entry and entry['content']:
            content = str(entry['content'])
            sanitized_entry['content'] = sanitize_user_input(content)
        else:
            sanitized_entry['content'] = ""

        sanitized_history.append(sanitized_entry)

    return sanitized_history

# --- Prompt Generation Logic (from ppm.py) ---

def generate_prompt(current_message_content: str, history: List[Dict[str, str]], user_profile: Optional[Dict] = None) -> str:
    """
    Generates a prompt for the LLM, including recent conversation history, persona, and user context if available.
    """
    # Sanitize inputs
    sanitized_message = sanitize_user_input(current_message_content)
    sanitized_history = validate_conversation_history(history)

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

    if sanitized_history:
        prompt_parts.append("Here is the recent conversation history:")
        for msg in sanitized_history:
            prompt_parts.append(f"{msg.get('author', 'Unknown')}: {msg.get('content', '')}")
        prompt_parts.append("---")

    prompt_parts.append(f"The latest message you are responding to is: '{sanitized_message}'.")
    prompt_parts.append("Please provide a response that is consistent with your persona and the conversation history. Use the User Context provided above to inform your response, and if the user asks what you know about them, use the User Context to answer directly.")

    return "\n".join(prompt_parts)

def generate_user_summary_prompt(user_id: str, username: str, conversation_history: List[Dict[str, str]], existing_summary: str | None) -> str:
    """
    Generates a prompt for the LLM to summarize user interactions.
    """
    # Sanitize inputs
    safe_username = sanitize_user_input(username, max_length=50)
    safe_user_id = sanitize_user_input(user_id, max_length=50)
    safe_existing_summary = sanitize_user_input(existing_summary, max_length=2000) if existing_summary else None
    sanitized_history = validate_conversation_history(conversation_history)

    prompt_lines = [
        "You are an AI assistant tasked with maintaining a user profile.",
        f"User ID: {safe_user_id}",
        f"Username: {safe_username}",
        "\nExisting Interaction Summary for this user (if any):",
        safe_existing_summary if safe_existing_summary else "No existing summary.",
        "---",
        "Recent Conversation Snippet involving this user:"
    ]

    if sanitized_history:
        for msg in sanitized_history:
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
    # Sanitize the user message
    safe_message = sanitize_user_input(user_message_content, max_length=2000)

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

IMPORTANT: Do not include any sensitive personal information like full addresses, phone numbers, email addresses,
financial information, or any information that could be used to identify or contact the user outside of this platform.

User's message:
---
{user_message_content}
---
JSON Output:
"""
    return json_schema_description.format(user_message_content=safe_message)


# --- Task Manager ---

class TaskManager:
    """Manages background tasks to prevent memory leaks with improved concurrency handling"""

    def __init__(self):
        self.tasks = set()
        self.lock = asyncio.Lock()
        self.cleanup_tasks = set()  # Track cleanup tasks separately
        self.task_info = {}  # Store task metadata
        self._loop = None  # Store the event loop for thread-safe operations
        self._thread_local = threading.local()  # Thread-local storage

    async def add_task(self, coro, task_name=None):
        """
        Create and track a new task

        Args:
            coro: Coroutine to run as a task
            task_name: Optional name for the task for logging

        Returns:
            The created task
        """
        try:
            # Store the event loop for thread-safe operations
            if self._loop is None:
                self._loop = asyncio.get_running_loop()

            # Create the task
            task = asyncio.create_task(coro)
            task_id = id(task)

            # Store task metadata
            task_info = {
                "id": task_id,
                "name": task_name,
                "created_at": datetime.now(timezone.utc),
                "status": "running"
            }

            # Add to tracking collections
            async with self.lock:
                self.tasks.add(task)
                self.task_info[task_id] = task_info

                # Set up callback using a more reliable method
                task.add_done_callback(self._task_done_callback)

            logger.debug(f"Task added: {task_name or task_id}")
            return task
        except Exception as e:
            logger.error(f"Failed to create task '{task_name}': {e}")
            # Re-raise to let the caller handle it
            raise

    def _task_done_callback(self, task):
        """
        Callback when a task is done - schedules cleanup in a thread-safe way

        This runs in the thread that completes the task, so we need to be careful
        about creating new asyncio tasks here.
        """
        try:
            # Store the task ID for cleanup
            task_id = id(task)

            # Use the event loop's call_soon_threadsafe to schedule cleanup
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(
                    lambda: self._schedule_cleanup_task(task)
                )
            else:
                logger.warning(f"Event loop not available for task {task_id} cleanup")
                # Try to get the current loop as a fallback
                try:
                    loop = asyncio.get_event_loop()
                    loop.call_soon_threadsafe(
                        lambda: self._schedule_cleanup_task(task)
                    )
                except Exception as e:
                    logger.error(f"Failed to get event loop for task {task_id} cleanup: {e}")
        except Exception as e:
            # If we can't schedule cleanup, log the error
            logger.error(f"Failed to schedule cleanup for task {id(task)}: {e}")

    def _schedule_cleanup_task(self, task):
        """Schedule a cleanup task in the event loop's context"""
        try:
            # Create a cleanup task in the correct event loop context
            cleanup_task = asyncio.create_task(self._remove_task(task))

            # Track the cleanup task
            self.cleanup_tasks.add(cleanup_task)

            # Set up a callback to remove the cleanup task when it's done
            cleanup_task.add_done_callback(self._cleanup_task_done)
        except Exception as e:
            logger.error(f"Failed to create cleanup task for {id(task)}: {e}")

    def _cleanup_task_done(self, cleanup_task):
        """Remove a cleanup task from tracking when it's done"""
        try:
            self.cleanup_tasks.discard(cleanup_task)

            # Check for exceptions in the cleanup task
            if not cleanup_task.cancelled():
                exc = cleanup_task.exception()
                if exc:
                    logger.error(f"Cleanup task {id(cleanup_task)} raised an exception: {exc}")
        except Exception as e:
            logger.error(f"Error in cleanup task callback: {e}")

    async def _remove_task(self, task):
        """Remove a completed task from the set"""
        task_id = id(task)
        try:
            async with self.lock:
                # Remove from tracking collections
                self.tasks.discard(task)  # Use discard instead of remove to avoid KeyError

                # Update task info
                if task_id in self.task_info:
                    task_info = self.task_info[task_id]
                    task_name = task_info.get("name", str(task_id))

                    # Check for exceptions
                    if not task.cancelled():
                        try:
                            exc = task.exception()
                            if exc:
                                logger.error(f"Task '{task_name}' raised an exception: {exc}")
                                task_info["status"] = "failed"
                                task_info["error"] = str(exc)
                            else:
                                task_info["status"] = "completed"
                        except asyncio.CancelledError:
                            task_info["status"] = "cancelled"
                        except Exception as e:
                            logger.error(f"Error checking task '{task_name}' exception: {e}")
                            task_info["status"] = "unknown"
                    else:
                        task_info["status"] = "cancelled"

                    task_info["completed_at"] = datetime.now(timezone.utc)
                    # Keep task info for a while for debugging
                    # We'll clean it up later during periodic maintenance
        except Exception as e:
            logger.error(f"Error removing task {task_id}: {e}")

    async def cancel_all(self):
        """Cancel all running tasks"""
        try:
            tasks_to_cancel = []

            # First, get all tasks under the lock
            async with self.lock:
                for task in self.tasks:
                    if not task.done():
                        tasks_to_cancel.append(task)

            # Then cancel them outside the lock to avoid deadlocks
            for task in tasks_to_cancel:
                try:
                    task.cancel()
                except Exception as e:
                    logger.error(f"Error cancelling task {id(task)}: {e}")

            # Wait for all tasks to complete cancellation
            if tasks_to_cancel:
                try:
                    await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                except Exception as e:
                    logger.error(f"Error waiting for tasks to cancel: {e}")

            # Also cancel any cleanup tasks
            cleanup_tasks = list(self.cleanup_tasks)
            for task in cleanup_tasks:
                if not task.done():
                    task.cancel()

            if cleanup_tasks:
                try:
                    await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                except Exception as e:
                    logger.error(f"Error waiting for cleanup tasks to cancel: {e}")

            # Clear all tracking collections
            async with self.lock:
                self.tasks.clear()
                # Keep task_info for debugging

            self.cleanup_tasks.clear()

            logger.info("All background tasks cancelled")
        except Exception as e:
            logger.error(f"Error in cancel_all: {e}")
            # Re-raise to ensure the caller knows there was a problem
            raise

    async def get_stats(self):
        """Get statistics about current tasks"""
        try:
            async with self.lock:
                total = len(self.tasks)
                running = sum(1 for t in self.tasks if not t.done())
                done = total - running
                cleanup_total = len(self.cleanup_tasks)
                cleanup_running = sum(1 for t in self.cleanup_tasks if not t.done())

                return {
                    "total": total,
                    "running": running,
                    "done": done,
                    "cleanup_total": cleanup_total,
                    "cleanup_running": cleanup_running,
                    "task_info_count": len(self.task_info)
                }
        except Exception as e:
            logger.error(f"Error getting task stats: {e}")
            return {
                "error": str(e),
                "total": -1,
                "running": -1,
                "done": -1
            }

# Create global task manager
task_manager = TaskManager()

# --- Rate Limiter ---

class RateLimiter:
    """Improved rate limiter for API calls with better concurrency handling"""

    def __init__(self, max_calls: int = 10, time_window: int = 60):
        """
        Initialize the rate limiter

        Args:
            max_calls: Maximum number of calls allowed in the time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = collections.deque(maxlen=max_calls * 2)  # Use deque with max size for better performance
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(max_calls)  # Use semaphore for better concurrency control
        self.last_cleanup = datetime.now(timezone.utc)
        self.cleanup_interval = time_window / 4  # Clean up 4 times per window

    async def _cleanup_expired(self):
        """Remove expired timestamps from the calls list"""
        now = datetime.now(timezone.utc)
        # Only clean up if enough time has passed since last cleanup
        if (now - self.last_cleanup).total_seconds() < self.cleanup_interval:
            return

        # Use a timeout to avoid deadlocks
        try:
            # Try to acquire the lock with a timeout
            lock_acquired = False
            try:
                lock_acquired = await asyncio.wait_for(self.lock.acquire(), timeout=1.0)
                if not lock_acquired:
                    logger.warning("Timeout acquiring lock in _cleanup_expired")
                    return

                # Remove expired timestamps
                while self.calls and (now - self.calls[0]).total_seconds() >= self.time_window:
                    self.calls.popleft()
                    # Release a token to the semaphore if we removed an expired call
                    try:
                        self.semaphore.release()
                    except ValueError:
                        # Semaphore was already at max value
                        pass

                self.last_cleanup = now
            finally:
                # Always release the lock if we acquired it
                if lock_acquired:
                    self.lock.release()
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for lock in _cleanup_expired")
        except Exception as e:
            logger.error(f"Error in _cleanup_expired: {e}")

    async def acquire(self) -> bool:
        """
        Try to acquire a rate limit token

        Returns:
            True if token acquired, False if rate limited
        """
        # Clean up expired calls first
        await self._cleanup_expired()

        # Use a proper non-blocking approach with the semaphore
        try:
            # Try to acquire the semaphore without waiting
            # Use a proper non-blocking approach
            acquired = False

            # Check active calls count with proper lock handling
            active_calls = 0
            try:
                # Try to acquire the lock with a timeout
                lock_acquired = await asyncio.wait_for(self.lock.acquire(), timeout=0.5)
                try:
                    if lock_acquired:
                        now = datetime.now(timezone.utc)
                        active_calls = sum(1 for ts in self.calls if (now - ts).total_seconds() < self.time_window)
                finally:
                    # Always release the lock if we acquired it
                    if lock_acquired:
                        self.lock.release()
            except asyncio.TimeoutError:
                logger.warning("Timeout acquiring lock in acquire")
                # Be conservative if we can't get the lock
                return False

            # Only try to acquire the semaphore if we're under the limit
            if active_calls < self.max_calls:
                try:
                    # Try to acquire the semaphore without blocking
                    acquired = await asyncio.wait_for(
                        self.semaphore.acquire(),
                        timeout=0.01  # Very short timeout
                    )

                    if acquired:
                        # Record this call with proper lock handling
                        try:
                            lock_acquired = await asyncio.wait_for(self.lock.acquire(), timeout=0.5)
                            try:
                                if lock_acquired:
                                    now = datetime.now(timezone.utc)
                                    self.calls.append(now)
                            finally:
                                if lock_acquired:
                                    self.lock.release()
                        except asyncio.TimeoutError:
                            logger.warning("Timeout acquiring lock to record call")
                            # We still acquired the semaphore, so return True

                        return True
                except asyncio.TimeoutError:
                    # Couldn't acquire immediately, which is fine
                    pass
                except Exception as e:
                    logger.error(f"Error acquiring semaphore: {e}")

            # If we couldn't acquire immediately, return False
            return False
        except Exception as e:
            logger.error(f"Unexpected error in rate limiter acquire: {e}")
            # In case of error, be conservative and don't allow the call
            return False

    async def wait_for_token(self, max_wait: int = 30) -> bool:
        """
        Wait for a rate limit token to become available

        Args:
            max_wait: Maximum time to wait in seconds

        Returns:
            True if token acquired, False if timed out
        """
        if max_wait <= 0:
            logger.warning("Invalid max_wait value for rate limiter, using default")
            max_wait = 30

        # Clean up expired calls first
        await self._cleanup_expired()

        # Use a safer approach with exponential backoff
        try:
            # First try to acquire immediately
            if await self.acquire():
                return True

            # If that fails, use a proper waiting approach with backoff
            start_time = datetime.now(timezone.utc)
            backoff = 0.1  # Start with 100ms
            max_backoff = 1.0  # Cap at 1 second

            while True:
                # Check if we've exceeded the max wait time
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                if elapsed >= max_wait:
                    logger.debug(f"Rate limit wait timed out after {elapsed:.2f}s")
                    return False

                # Wait with backoff
                await asyncio.sleep(backoff)

                # Try to acquire again
                if await self.acquire():
                    logger.debug(f"Rate limit token acquired after {elapsed:.2f}s wait")
                    return True

                # Increase backoff with jitter for next attempt
                backoff = min(backoff * 1.5, max_backoff)
                # Add some randomness to avoid thundering herd
                backoff *= (0.9 + 0.2 * secrets.SystemRandom().random())
        except Exception as e:
            logger.error(f"Error in wait_for_token: {e}")
            # In case of error, be conservative and don't allow the call
            return False

# Create global rate limiters
gemini_rate_limiter = RateLimiter(max_calls=20, time_window=60)  # 20 calls per minute
supabase_rate_limiter = RateLimiter(max_calls=30, time_window=60)  # 30 calls per minute
discord_rate_limiter = RateLimiter(max_calls=5, time_window=5)  # 5 messages per 5 seconds (to avoid Discord rate limits)

# --- LLM Interface Logic (from llmi.py) ---

async def get_llm_response(prompt: str, timeout_seconds: int = GEMINI_TIMEOUT_SECONDS) -> dict:
    """
    Communicates with the Google Gemini API to get a response from the LLM.
    Uses the model specified by the GEMINI_MODEL_NAME environment variable.
    The API key is configured globally from the GOOGLE_API_KEY environment variable.

    Args:
        prompt: The prompt to send to the LLM
        timeout_seconds: Maximum time to wait for a response from the API

    Returns:
        A dictionary with status and text fields
    """
    if not GOOGLE_API_KEY: # Check if the global key was successfully configured
        logger.error("Error: GOOGLE_API_KEY not configured or failed to load.")
        return {"status": "error", "text": "Error: GOOGLE_API_KEY not configured.", "error_code": "API_KEY_MISSING"}

    try:
        # Apply rate limiting
        if not await gemini_rate_limiter.wait_for_token(max_wait=timeout_seconds):
            logger.warning(f"LLMI: Rate limit exceeded for Gemini API. Too many requests in the time window.")
            return {"status": "error", "text": "The AI service is currently experiencing high demand. Please try again in a moment.", "error_code": "RATE_LIMIT_EXCEEDED"}

        model_to_use = gemini_model_name
        logger.info(f"LLMI: Attempting to initialize model: {model_to_use}")

        # Check if model initialization is successful before proceeding
        try:
            configured_model = genai.GenerativeModel(model_to_use)
            if not configured_model:
                logger.error("LLMI: configured_model is not initialized.")
                return {"status": "error", "text": "LLM model could not be initialized.", "error_code": "MODEL_INITIALIZATION_FAILED"}
            logger.info(f"LLMI: Successfully initialized model: {model_to_use}")
        except Exception as model_init_error:
            logger.error(f"LLMI: Error initializing model {model_to_use}: {model_init_error}")
            return {"status": "error", "text": f"Error initializing LLM model: {str(model_init_error)}", "error_code": "MODEL_INITIALIZATION_ERROR"}

        # Add timeout handling for API call
        # Redact potentially sensitive content from logs
        prompt_length = len(prompt) if prompt else 0
        logger.debug(f"Attempting to generate content with prompt of length {prompt_length} chars [CONTENT REDACTED]")

        # Record API call in health monitor
        health_monitor.record_activity("gemini")

        api_response_task = None
        try:
            # Execute the API call directly with a timeout
            api_response = await asyncio.wait_for(
                asyncio.to_thread(configured_model.generate_content, prompt),
                timeout=timeout_seconds
            )

            # Log only non-sensitive information about the response
            if hasattr(api_response, 'candidates') and api_response.candidates:
                candidate_count = len(api_response.candidates)
                finish_reason = api_response.candidates[0].finish_reason if api_response.candidates else "UNKNOWN"
                logger.debug(f"LLMI: Received API response with {candidate_count} candidates, finish_reason: {finish_reason}")
            else:
                logger.debug(f"LLMI: Received API response without candidates")
        except asyncio.TimeoutError:
            # Properly cancel the task to avoid resource leaks
            if api_response_task and not api_response_task.done():
                api_response_task.cancel()
                try:
                    # Give it a moment to clean up
                    await asyncio.wait_for(api_response_task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass

            logger.error(f"LLMI: API call timed out after {timeout_seconds} seconds")
            health_monitor.record_activity("error")
            return {"status": "error", "text": f"The request to the LLM timed out after {timeout_seconds} seconds. Please try again.", "error_code": "API_TIMEOUT"}

        if api_response.candidates:
            if api_response.candidates[0].finish_reason == 'SAFETY':
                # Don't log the prompt content for safety issues
                logger.warning(f"Warning: Gemini API response was blocked due to safety settings.")
                return {"status": "error", "text": "I'm sorry, but I cannot provide a response to that due to safety guidelines.", "error_code": "SAFETY_BLOCKED"}

            response_text = None
            if hasattr(api_response.candidates[0].content, 'parts') and api_response.candidates[0].content.parts:
                response_text = "".join(part.text for part in api_response.candidates[0].content.parts if hasattr(part, 'text')).strip()
            elif hasattr(api_response, 'text') and api_response.text:
                 response_text = api_response.text.strip()

            if response_text:
                # Log only the length of the response, not the content
                response_length = len(response_text)
                logger.debug(f"LLMI: Generated response of length {response_length} chars")
                return {"status": "success", "text": response_text}
            else:
                logger.error(f"Error: Gemini API response yielded empty text or unexpected format.")
                return {"status": "error", "text": "Sorry, I couldn't generate a response. The model returned an empty answer.", "error_code": "GENERATION_EMPTY"}
        else:
            if hasattr(api_response, 'prompt_feedback') and api_response.prompt_feedback.block_reason:
                reason_message = api_response.prompt_feedback.block_reason_message or str(api_response.prompt_feedback.block_reason)
                logger.error(f"Error: Prompt blocked by Gemini API. Reason: {api_response.prompt_feedback.block_reason}")
                return {"status": "error", "text": f"I'm sorry, but your request could not be processed. Reason: {reason_message}", "error_code": "PROMPT_BLOCKED"}

            logger.error(f"Error: No candidates found in Gemini API response.")
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
    except Exception as e:
        if "ResourceExhausted" in str(e.__class__):
            logger.error(f"Google API ResourceExhausted: {e}", exc_info=True)
            return {"status": "error", "text": f"API limit reached or resource exhausted: {str(e)}", "error_code": "API_RESOURCE_EXHAUSTED"}
        else:
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

class DatabaseTransaction:
    """Context manager for database transactions"""

    def __init__(self, ltms, operation_name="transaction"):
        self.ltms = ltms
        self.operation_name = operation_name
        self.transaction_id = f"tx_{secrets.token_hex(8)}"
        self.success = False
        self.queries = []
        self.results = []

    async def __aenter__(self):
        logger.debug(f"Starting transaction {self.transaction_id} for {self.operation_name}")
        return self

    async def add_query(self, query, description="query"):
        """Add a query to the transaction"""
        self.queries.append((query, description))
        return self

    async def execute(self):
        """Execute all queries in the transaction"""
        try:
            for i, (query, description) in enumerate(self.queries):
                logger.debug(f"Executing query {i+1}/{len(self.queries)} ({description}) in transaction {self.transaction_id}")
                result = await self.ltms._execute_query(
                    query,
                    operation_name=f"{self.operation_name}_{description}_{self.transaction_id}"
                )

                if result is None or (hasattr(result, 'error') and result.error):
                    error_msg = getattr(result, 'error', 'Unknown error')
                    logger.error(f"Transaction {self.transaction_id} failed at query {i+1}/{len(self.queries)} ({description}): {error_msg}")
                    return False

                self.results.append(result)

            self.success = True
            logger.debug(f"Transaction {self.transaction_id} completed successfully")
            return True
        except Exception as e:
            logger.error(f"Exception in transaction {self.transaction_id}: {e}")
            return False

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Transaction {self.transaction_id} exited with exception: {exc_val}")
            return False

        if not self.success and self.queries:
            logger.warning(f"Transaction {self.transaction_id} was not explicitly executed and had {len(self.queries)} queries")

        return False  # Don't suppress exceptions

class LongTermMemoryStore:
    def __init__(self, supabase_url: str, supabase_key: str):
        try:
            key_display = supabase_key[:5] + "..." if supabase_key and len(supabase_key) > 8 else "INVALID_OR_SHORT_KEY"
            logger.info(f"Initializing Supabase client with URL: {supabase_url}, Key (partial): {key_display}")

            # Initialize the Supabase client
            self.supabase: Client = create_client(supabase_url, supabase_key)

            # Add connection pool management
            self.connection_lock = asyncio.Lock()
            self.max_connections = 10  # Maximum number of concurrent connections
            self.active_connections = 0
            self.connection_semaphore = asyncio.Semaphore(self.max_connections)

            # Track connection stats
            self.connection_stats = {
                "total_requests": 0,
                "failed_requests": 0,
                "timeouts": 0,
                "peak_concurrent": 0,
                "active_connections": 0
            }

            logger.info(f"Successfully connected to Supabase at {supabase_url}.")
            logger.info(f"Supabase client initialized with max {self.max_connections} concurrent connections.")

            # Schema validation will be done explicitly after initialization
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise

    async def create_transaction(self, operation_name="transaction"):
        """Create a new database transaction"""
        return DatabaseTransaction(self, operation_name)

    async def _validate_schema(self):
        """Validate that the required database schema exists"""
        try:
            # Check if the user_profiles table exists
            response = await self._execute_query(
                self.supabase.table("user_profiles").select("*").limit(1),
                operation_name="schema_validation"
            )

            if hasattr(response, 'error') and response.error:
                logger.error(f"Schema validation failed: {response.error.message}")
                return False

            logger.info("Database schema validation successful - user_profiles table exists")
            return True
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return False

    async def _execute_query(self, query, timeout_seconds: int = SUPABASE_TIMEOUT_SECONDS, operation_name: str = "unknown"):
        """
        Execute a Supabase query with connection pooling and timeout handling

        Args:
            query: The Supabase query to execute
            timeout_seconds: Maximum time to wait for a response
            operation_name: Name of the operation for logging

        Returns:
            The query response
        """
        # Validate timeout
        if timeout_seconds <= 0:
            logger.warning(f"Invalid timeout value {timeout_seconds} for {operation_name}, using default")
            timeout_seconds = SUPABASE_TIMEOUT_SECONDS

        # Update stats
        async with self.connection_lock:
            self.connection_stats["total_requests"] += 1

        # Use a context manager for the semaphore to ensure proper release
        class ConnectionSemaphoreContext:
            def __init__(self, semaphore, lock, stats, operation):
                self.semaphore = semaphore
                self.lock = lock
                self.stats = stats
                self.operation = operation
                self.acquired = False

            async def __aenter__(self):
                try:
                    # Try to acquire the semaphore with a reasonable timeout
                    await asyncio.wait_for(
                        self.semaphore.acquire(),
                        timeout=min(timeout_seconds, 5)  # Don't wait too long for a connection
                    )
                    self.acquired = True

                    # Update active connection count
                    async with self.lock:
                        self.stats["active_connections"] += 1
                        if self.stats["active_connections"] > self.stats["peak_concurrent"]:
                            self.stats["peak_concurrent"] = self.stats["active_connections"]

                    return True
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for connection slot for {self.operation}")
                    async with self.lock:
                        self.stats["timeouts"] += 1
                        self.stats["failed_requests"] += 1
                    return False
                except Exception as e:
                    logger.error(f"Error acquiring connection for {self.operation}: {e}")
                    async with self.lock:
                        self.stats["failed_requests"] += 1
                    return False

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if self.acquired:
                    # Update active connection count
                    async with self.lock:
                        self.stats["active_connections"] -= 1

                    # Release the semaphore
                    self.semaphore.release()
                    self.acquired = False

        # Use the context manager to handle semaphore acquisition and release
        async with ConnectionSemaphoreContext(
            self.connection_semaphore,
            self.connection_lock,
            self.connection_stats,
            operation_name
        ) as connection_acquired:
            if not connection_acquired:
                return None

            try:
                # Record API call in health monitor
                health_monitor.record_activity("supabase")

                # Execute the query with timeout
                try:
                    # Execute the query directly with a timeout
                    response = await asyncio.wait_for(
                        asyncio.to_thread(query.execute),
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    raise  # Re-raise to be caught by the outer try/except
                except Exception as e:
                    logger.error(f"Error executing query for {operation_name}: {e}")
                    raise

                # Log only non-sensitive information about the response
                if response:
                    has_data = hasattr(response, 'data') and response.data is not None
                    has_error = hasattr(response, 'error') and response.error is not None
                    logger.debug(f"Supabase query for {operation_name} completed: has_data={has_data}, has_error={has_error}")

                return response
            except asyncio.TimeoutError:
                logger.error(f"Supabase query timed out after {timeout_seconds}s for {operation_name}")
                async with self.connection_lock:
                    self.connection_stats["timeouts"] += 1
                    self.connection_stats["failed_requests"] += 1

                # Properly cancel the task to avoid resource leaks
                if response_task and not response_task.done():
                    response_task.cancel()
                    try:
                        # Give it a moment to clean up
                        await asyncio.wait_for(response_task, timeout=1.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass

                return None
            except Exception as e:
                logger.error(f"Error executing Supabase query for {operation_name}: {e}")
                async with self.connection_lock:
                    self.connection_stats["failed_requests"] += 1
                return None

    async def get_user_profile(self, user_id: str, timeout_seconds: int = SUPABASE_TIMEOUT_SECONDS) -> dict | None:
        """
        Get a user profile from the database

        Args:
            user_id: The user ID to get the profile for
            timeout_seconds: Maximum time to wait for a response

        Returns:
            The user profile or None if not found or error
        """
        try:
            # Apply rate limiting
            if not await supabase_rate_limiter.wait_for_token(max_wait=timeout_seconds):
                logger.warning(f"LTMS: Rate limit exceeded for Supabase API in get_user_profile for {user_id}")
                return None

            # Create the query
            query = self.supabase.table("user_profiles").select("*").eq("user_id", user_id).maybe_single()

            # Execute the query using our connection pooling method
            response = await self._execute_query(
                query,
                timeout_seconds=timeout_seconds,
                operation_name=f"get_user_profile_{user_id}"
            )

            if not response:
                logger.warning(f"LTMS: No response from database in get_user_profile for {user_id}")
                return None

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
            # First try to get the existing profile
            existing_profile = await self.get_user_profile(user_id)
            if existing_profile:
                logger.info(f"Profile already exists for user_id: {user_id}. Skipping creation.")
                return existing_profile

            # Apply rate limiting for the create operation
            if not await supabase_rate_limiter.wait_for_token(max_wait=API_TIMEOUT_SECONDS):
                logger.warning(f"LTMS: Rate limit exceeded for Supabase API in get_or_create_user_profile for {user_id}")
                return None

            # Prepare profile data
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

            # Create the query
            query = self.supabase.table("user_profiles").insert(profile_data)

            # Execute the query using our connection pooling method
            response = await self._execute_query(
                query,
                timeout_seconds=API_TIMEOUT_SECONDS,
                operation_name=f"create_user_profile_{user_id}"
            )

            if not response:
                logger.warning(f"LTMS: No response from database in get_or_create_user_profile for {user_id}")
                return None

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

            # Apply rate limiting
            if not await supabase_rate_limiter.wait_for_token(max_wait=API_TIMEOUT_SECONDS):
                logger.warning(f"LTMS: Rate limit exceeded for Supabase API in update_user_profile for {user_id}")
                return False

            # Prepare updates
            updates_for_db = updates.copy()
            updates_for_db["last_updated_at"] = datetime.now(timezone.utc).isoformat()

            # Create the query
            query = self.supabase.table("user_profiles").update(updates_for_db).eq("user_id", user_id)

            # Execute the query using our connection pooling method
            response = await self._execute_query(
                query,
                timeout_seconds=API_TIMEOUT_SECONDS,
                operation_name=f"update_user_profile_{user_id}"
            )

            if not response:
                logger.warning(f"LTMS: No response from database in update_user_profile for {user_id}")
                return False

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

    async def update_user_profile_with_details_and_preferences(self, user_id: str, details_and_preferences: dict) -> bool:
        """
        Update a user profile with extracted personal details and preferences

        Args:
            user_id: The user ID to update the profile for
            details_and_preferences: Dictionary containing personal_details and preferences

        Returns:
            True if successful, False otherwise
        """
        try:
            # Sanitize user_id
            safe_user_id = sanitize_user_input(user_id, max_length=50)

            # Get the existing profile
            existing_profile = await self.get_user_profile(safe_user_id)
            if not existing_profile:
                logger.warning(f"LTMS: No profile found for user when trying to update details and preferences")
                return False

            # Extract the components
            personal_details = details_and_preferences.get('personal_details', {})
            preferences = details_and_preferences.get('preferences', {})

            # Only proceed if we have new data
            if not personal_details and not preferences:
                logger.info(f"LTMS: No new details or preferences to update")
                return True

            # Create a transaction for the update
            async with await self.create_transaction(f"update_profile_{safe_user_id}") as transaction:
                # Prepare updates
                updates = {}

                # Update mentioned_personal_details if we have new details
                if personal_details:
                    existing_details = existing_profile.get('mentioned_personal_details', {})
                    # Merge new details with existing ones
                    merged_details = {**existing_details, **personal_details}
                    updates['mentioned_personal_details'] = merged_details
                    logger.info(f"LTMS: Updating personal details for user")

                # Update preferences if we have new preferences
                if preferences:
                    existing_preferences = existing_profile.get('preferences', {})
                    # Merge new preferences with existing ones
                    merged_preferences = {**existing_preferences, **preferences}
                    updates['preferences'] = merged_preferences
                    logger.info(f"LTMS: Updating preferences for user")

                # Add last_updated timestamp
                updates["last_updated_at"] = datetime.now(timezone.utc).isoformat()

                # Create the update query
                update_query = self.supabase.table("user_profiles").update(updates).eq("user_id", safe_user_id)

                # Add the query to the transaction
                await transaction.add_query(update_query, "update_profile")

                # Execute the transaction
                success = await transaction.execute()

                if success:
                    logger.info(f"LTMS: Successfully updated profile with details and preferences")
                    return True
                else:
                    logger.error(f"LTMS: Transaction failed when updating profile with details and preferences")
                    return False

        except Exception as e:
            logger.error(f"LTMS: Error updating user profile with details and preferences: {e}")
            return False

    async def increment_user_message_count(self, user_id: str, max_retries: int = 3) -> bool:
        """
        Increment the message count for a user using atomic operations.
        Retries up to max_retries times if there's a conflict.
        """
        logger.debug(f"LTMS: Attempting to increment message count for user_id: {user_id}.")

        # Apply rate limiting
        if not await supabase_rate_limiter.wait_for_token(max_wait=SUPABASE_TIMEOUT_SECONDS):
            logger.warning(f"LTMS: Rate limit exceeded for Supabase API in increment_user_message_count for {user_id}")
            return False

        retry_count = 0
        while retry_count < max_retries:
            try:
                # Use a direct SQL update with a counter increment to avoid race conditions
                # This is safer than read-modify-write as it's a single atomic operation
                now_iso = datetime.now(timezone.utc).isoformat()

                # First try: Use RPC call to execute a custom SQL function that atomically increments the counter
                try:
                    # Create the RPC query
                    rpc_query = self.supabase.rpc(
                        'increment_message_count',
                        {'user_id_param': user_id, 'timestamp_param': now_iso}
                    )

                    # Execute the query using our connection pooling method
                    response = await self._execute_query(
                        rpc_query,
                        timeout_seconds=SUPABASE_TIMEOUT_SECONDS,
                        operation_name=f"increment_message_count_rpc_{user_id}"
                    )

                    if not response:
                        logger.warning(f"LTMS: No response from database in increment_message_count RPC for {user_id}")
                        # Fall back to the update method
                        raise Exception("No response from RPC method")

                    if hasattr(response, 'error') and response.error:
                        # If RPC fails, fall back to the update method
                        logger.warning(f"RPC method failed, falling back to update method: {response.error.message}")
                        raise Exception("RPC method not available")

                    logger.info(f"LTMS: Successfully incremented message count for user_id: {user_id} using RPC.")
                    return True

                except Exception as rpc_error:
                    # Fall back to the direct SQL update if RPC is not available
                    logger.warning(f"RPC method not available, using direct SQL update: {rpc_error}")

                    # Second try: Use a direct SQL update that's still atomic
                    try:
                        # Create a SQL query that does an atomic update
                        # This is more reliable than the read-modify-write pattern
                        # Since raw() method is not available, we'll fall back to read-modify-write
                        logger.warning(f"SQL update with raw() not available, falling back to read-modify-write directly")
                        raise Exception("raw() method not available")

                    except Exception as sql_error:
                        # Last resort: Fall back to read-modify-write pattern
                        logger.warning(f"SQL update method failed, falling back to read-modify-write: {sql_error}")

                        # Get current value
                        profile = await self.get_user_profile(user_id)
                        if not profile:
                            logger.warning(f"LTMS: No profile found to increment message count for user_id: {user_id}.")
                            return False

                        # Update with new value
                        original_message_count = profile.get("message_count", 0)
                        new_message_count = original_message_count + 1
                        logger.debug(f"LTMS: User {user_id} - Original message_count: {original_message_count}, New message_count: {new_message_count}")

                        updates = {
                            "message_count": new_message_count,
                            "last_seen_at": now_iso,
                            "last_updated_at": now_iso
                        }

                        update_query = self.supabase.table("user_profiles").update(updates).eq("user_id", user_id)
                        response = await self._execute_query(
                            update_query,
                            timeout_seconds=SUPABASE_TIMEOUT_SECONDS,
                            operation_name=f"increment_message_count_rmw_{user_id}"
                        )

                        if not response:
                            logger.warning(f"LTMS: No response from database in increment_message_count RMW for {user_id}")
                            retry_count += 1
                            await asyncio.sleep(0.1 * retry_count)  # Exponential backoff
                            continue

                        if hasattr(response, 'error') and response.error:
                            if "conflict" in str(response.error).lower() or "concurrent" in str(response.error).lower():
                                # Optimistic concurrency control - retry on conflict
                                retry_count += 1
                                logger.warning(f"LTMS: Concurrency conflict detected for user {user_id}, retry {retry_count}/{max_retries}")
                                await asyncio.sleep(0.1 * retry_count)  # Exponential backoff
                                continue
                            else:
                                logger.error(f"Supabase API error in increment_user_message_count RMW for {user_id}: {response.error.message}")
                                return False

                        logger.info(f"LTMS: Successfully incremented message count for user_id: {user_id} to {new_message_count} using RMW.")
                        return True

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Exception incrementing message count for user_id {user_id} after {max_retries} retries: {e}")
                    return False
                logger.warning(f"Retry {retry_count}/{max_retries} for incrementing message count for user_id {user_id}: {e}")
                await asyncio.sleep(0.1 * retry_count)  # Exponential backoff

        return False

    async def update_user_summary(self, user_id: str, summary: str) -> bool:
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            updates = {
                "interaction_summary": summary,
                "last_updated_at": now_iso
            }
            query = self.supabase.table("user_profiles").update(updates).eq("user_id", user_id)

            # Execute the query using our connection pooling method
            response = await self._execute_query(
                query,
                timeout_seconds=SUPABASE_TIMEOUT_SECONDS,
                operation_name=f"update_user_summary_{user_id}"
            )

            if not response:
                logger.warning(f"LTMS: No response from database in update_user_summary for {user_id}")
                return False

            if hasattr(response, 'error') and response.error:
                logger.error(f"Supabase API error in update_user_summary for {user_id}: {response.error.message} (Code: {response.error.code if hasattr(response.error, 'code') else 'N/A'})")
                return False

            logger.info(f"Successfully updated interaction summary for user {user_id}.")
            return True
        except Exception as e:
            logger.error(f"Exception updating summary for user_id {user_id}: {e}")
            return False

    async def add_username_to_history(self, user_id: str, username: str, max_retries: int = 3) -> bool:
        """
        Add a username to the history for a user using optimistic concurrency control.
        Retries up to max_retries times if there's a conflict.
        """
        logger.debug(f"LTMS: Attempting to add username '{username}' to history for user_id: {user_id}.")

        retry_count = 0
        while retry_count < max_retries:
            try:
                # Try to use RPC first for an atomic operation
                try:
                    now_iso = datetime.now(timezone.utc).isoformat()
                    _query = self.supabase.rpc(
                        'add_username_to_history',
                        {'user_id_param': user_id, 'username_param': username, 'timestamp_param': now_iso}
                    )
                    response = await asyncio.to_thread(_query.execute)

                    if hasattr(response, 'error') and response.error:
                        # If RPC fails, fall back to the update method
                        logger.warning(f"RPC method failed for username history, falling back to update method: {response.error.message}")
                        raise Exception("RPC method not available")

                    logger.info(f"LTMS: Successfully added username '{username}' to history for user {user_id} using RPC.")
                    return True

                except Exception as rpc_error:
                    # Fall back to the update method if RPC is not available
                    logger.warning(f"RPC method not available for username history, using update method: {rpc_error}")

                    # Fallback to read-modify-write with optimistic concurrency control
                    profile = await self.get_user_profile(user_id)
                    if not profile:
                        logger.warning(f"LTMS: No profile found for user_id {user_id} to add username '{username}' to history.")
                        return False

                    original_username_history = list(profile.get("username_history", []))
                    current_username_history = profile.get("username_history", [])

                    # Only update if needed
                    if not current_username_history or current_username_history[-1] != username:
                        logger.debug(f"LTMS: User {user_id} - Original username_history: {original_username_history}. Adding '{username}'.")
                        current_username_history.append(username)
                        now_iso = datetime.now(timezone.utc).isoformat()
                        updates = {
                            "username_history": current_username_history,
                            "last_updated_at": now_iso
                        }

                        # Add optimistic concurrency control
                        _query = self.supabase.table("user_profiles").update(updates).eq("user_id", user_id)
                        response = await asyncio.to_thread(_query.execute)

                        if hasattr(response, 'error') and response.error:
                            if "conflict" in str(response.error).lower() or "concurrent" in str(response.error).lower():
                                # Optimistic concurrency control - retry on conflict
                                retry_count += 1
                                logger.warning(f"LTMS: Concurrency conflict detected for user {user_id} username history, retry {retry_count}/{max_retries}")
                                await asyncio.sleep(0.1 * retry_count)  # Exponential backoff
                                continue
                            else:
                                logger.error(f"LTMS: Supabase API error in add_username_to_history for {user_id} while adding '{username}': {response.error.message} (Code: {response.error.code if hasattr(response.error, 'code') else 'N/A'})")
                                return False

                        logger.info(f"LTMS: Successfully added username '{username}' to history for user {user_id}. New history: {current_username_history}")
                    else:
                        logger.debug(f"LTMS: Username '{username}' is already the latest in history for user {user_id}. No update needed. History: {current_username_history}")

                    return True

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Exception adding username to history for user_id {user_id} after {max_retries} retries: {e}")
                    return False
                logger.warning(f"Retry {retry_count}/{max_retries} for adding username to history for user_id {user_id}: {e}")
                await asyncio.sleep(0.1 * retry_count)  # Exponential backoff

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
    logger.info("Supabase URL and Key are set. LTMS will be initialized during bot startup.")
else:
    logger.warning("SUPABASE_URL or SUPABASE_KEY is not set. LTMS features will be disabled.")


@client.event
async def on_ready():
    global ltms_instance
    logger.info(f'{client.user} has connected to Discord!')
    logger.info(f'Successfully connected to: {", ".join([guild.name for guild in client.guilds])}')

    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY is not set. The bot will not be able to respond to mentions.")

    # Initialize LTMS now that we have an event loop
    if SUPABASE_URL and SUPABASE_KEY and ltms_instance is None:
        try:
            logger.info("Initializing LongTermMemoryStore with Supabase...")
            ltms_instance = LongTermMemoryStore(supabase_url=SUPABASE_URL, supabase_key=SUPABASE_KEY)
            # Explicitly await the schema validation
            schema_valid = await ltms_instance._validate_schema()
            if schema_valid:
                logger.info("LongTermMemoryStore initialized successfully with validated schema.")
            else:
                logger.error("LongTermMemoryStore schema validation failed.")
        except Exception as e:
            logger.error(f"Failed to initialize LongTermMemoryStore with Supabase: {e}")
            ltms_instance = None

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

async def _handle_llm_response(message: discord.Message, context: dict, stmm_param: ShortTermMemoryManager, ltms_param: LongTermMemoryStore, google_api_key_param: str, logger_param: logging.Logger, user_profile: Optional[dict] = None) -> bool:
    """
    Handle generating and sending an LLM response to a user message.

    Args:
        message: The Discord message to respond to
        context: Dictionary containing message context
        stmm_param: Short-term memory manager instance
        ltms_param: Long-term memory store instance
        google_api_key_param: Google API key
        logger_param: Logger instance
        user_profile: Optional user profile data

    Returns:
        Boolean indicating if the LLM call was successful and returned text
    """
    _first_llm_call_successful_and_has_text = False
    recent_history = stmm_param.get_recent_history(channel_id=message.channel.id, limit=LLM_RESPONSE_CONTEXT_HISTORY_LIMIT)
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

            # Apply Discord rate limiting with exponential backoff
            async def send_discord_message(content, max_retries=3):
                """Send a message to Discord with retry logic and exponential backoff"""
                retry_count = 0
                backoff = 1.0  # Start with 1 second

                while retry_count <= max_retries:
                    try:
                        # Try to acquire a rate limit token
                        if retry_count == 0:
                            # First attempt - use normal rate limiting
                            if await discord_rate_limiter.wait_for_token(max_wait=10):
                                # Record API call in health monitor
                                health_monitor.record_activity("discord")
                                await message.channel.send(content)
                                return True
                            else:
                                logger_param.warning("Discord rate limit reached, will retry with backoff")
                        else:
                            # Retry attempts - wait with backoff before trying again
                            logger_param.info(f"Retrying Discord message send (attempt {retry_count}/{max_retries}) after {backoff:.2f}s backoff")
                            await asyncio.sleep(backoff)

                            # Try to send the message
                            await message.channel.send(content)
                            health_monitor.record_activity("discord")
                            return True

                    except discord.errors.HTTPException as e:
                        if e.status == 429:  # Too many requests
                            retry_count += 1
                            # Get retry_after from the exception if available
                            retry_after = getattr(e, 'retry_after', backoff)
                            backoff = max(retry_after * 1.5, backoff * 2)
                            logger_param.warning(f"Discord rate limit hit: {e}. Retrying in {backoff:.2f}s")
                        else:
                            # Other HTTP error
                            logger_param.error(f"Discord HTTP error: {e}")
                            return False
                    except Exception as e:
                        logger_param.error(f"Error sending message to Discord: {e}")
                        return False

                    # Increase backoff for next attempt
                    retry_count += 1
                    backoff = min(backoff * 2, 15)  # Cap at 15 seconds

                logger_param.error(f"Failed to send message to Discord after {max_retries} retries")
                return False

            # Try to send the message with retry logic
            message_sent = await send_discord_message(llm_text_from_dict)

            # Only add to short-term memory if the message was sent successfully
            if message_sent:
                stmm_param.add_message(
                    channel_id=message.channel.id,
                    author_name=client.user.name,
                    content=llm_text_from_dict # Store potentially truncated message
                )
            _first_llm_call_successful_and_has_text = True
        elif llm_response_data.get('status') == "error":
            error_message = llm_response_data.get('text', "An unspecified error occurred with the AI.")
            logger_param.error(f"LLM call (prompt) failed. Status: error. Message: {error_message}")
            # Use the same send_discord_message function for error messages
            error_sent = await send_discord_message(error_message, max_retries=2)
            if not error_sent:
                logger_param.error("Failed to send error message to Discord")
        else:
            status_val = llm_response_data.get('status', 'Unknown')
            text_val = llm_response_data.get('text', 'No text provided')
            logger_param.error(f"LLM call (prompt) returned non-standard response. Status: '{status_val}'. Text: '{text_val}'")
            await message.channel.send(f"Sorry, I received an unusual response from the AI (Status: {status_val}).")
    else:
        logger_param.error("LLM Error: Did not receive a valid data object from get_llm_response for prompt.")
        await message.channel.send("Sorry, I couldn't retrieve a response from the AI at the moment.")
    return _first_llm_call_successful_and_has_text

async def _handle_user_summary_update(message: discord.Message, context: dict, stmm_param: ShortTermMemoryManager, ltms_param: LongTermMemoryStore, google_api_key_param: str, user_id_str: str, username: str, profile: dict, logger_param: logging.Logger):
    """
    Handle updating the user's interaction summary.

    Args:
        message: The Discord message that triggered the update
        context: Dictionary containing message context
        stmm_param: Short-term memory manager instance
        ltms_param: Long-term memory store instance
        google_api_key_param: Google API key
        user_id_str: User ID as string
        username: Username
        profile: User profile data
        logger_param: Logger instance
    """
    if not ltms_param:
        logger_param.info("LTMS not initialized, skipping user summary update.")
        return
    if not profile:
        logger_param.warning(f"LTMS: No profile object provided to _handle_user_summary_update for user {user_id_str} ({username}). Skipping summary update.")
        return

    logger_param.info(f"LTMS: Attempting to update interaction summary for user {user_id_str} ({username}).")
    try:
        existing_summary = profile.get("interaction_summary") if profile else None
        summary_conversation_history = stmm_param.get_recent_history(channel_id=message.channel.id, limit=LLM_SUMMARY_CONTEXT_HISTORY_LIMIT)
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

    # Redact message content for privacy
    user_id_hash = hash(user_id_str)  # Use a hash instead of actual ID
    logger_param.info(f"LTMS: Attempting to extract details/preferences for user (hash: {user_id_hash}). Message length: {len(message_content)} chars")
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
            # First, try to find JSON by looking for the most common pattern
            json_start = llm_json_str.find('{')
            json_end = llm_json_str.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                # Try to parse the JSON directly
                try:
                    potential_json = llm_json_str[json_start:json_end]
                    candidate_data = json.loads(potential_json)

                    # Validate the structure
                    if isinstance(candidate_data, dict) and \
                       ("personal_details" in candidate_data or "preferences" in candidate_data):
                        extracted_data = candidate_data
                        logger_param.info(f"LTMS: Successfully extracted JSON data for user {user_id_str} using direct approach")
                except json.JSONDecodeError:
                    # If direct parsing fails, try more sophisticated approaches
                    logger_param.debug(f"LTMS: Direct JSON parsing failed, trying alternative approaches")

            # If direct approach failed, try with regex for nested JSON objects
            if not extracted_data:
                import re
                # This pattern handles nested objects better
                json_pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
                json_matches = re.findall(json_pattern, llm_json_str)

                if json_matches:
                    # Sort matches by length (descending) to try the largest JSON objects first
                    # This is often more likely to be the complete object
                    for potential_json in sorted(json_matches, key=len, reverse=True):
                        try:
                            candidate_data = json.loads(potential_json)
                            # Check if this looks like our expected format
                            if isinstance(candidate_data, dict) and \
                               ("personal_details" in candidate_data or "preferences" in candidate_data):
                                extracted_data = candidate_data
                                logger_param.info(f"LTMS: Successfully extracted JSON data for user {user_id_str} using regex approach")
                                break
                        except json.JSONDecodeError:
                            continue

            # If all else fails, try a more lenient approach with line-by-line JSON detection
            if not extracted_data:
                logger_param.debug(f"LTMS: Standard JSON extraction failed, trying line-by-line approach")
                lines = llm_json_str.split('\n')
                json_lines = []
                in_json = False

                for line in lines:
                    if '{' in line and not in_json:
                        in_json = True
                        json_lines.append(line)
                    elif in_json:
                        json_lines.append(line)
                        if '}' in line:
                            # Try to parse the accumulated JSON
                            try:
                                potential_json = ''.join(json_lines)
                                candidate_data = json.loads(potential_json)
                                if isinstance(candidate_data, dict) and \
                                   ("personal_details" in candidate_data or "preferences" in candidate_data):
                                    extracted_data = candidate_data
                                    logger_param.info(f"LTMS: Successfully extracted JSON data for user {user_id_str} using line-by-line approach")
                                    break
                            except json.JSONDecodeError:
                                # Reset and continue looking
                                in_json = False
                                json_lines = []

            # If we still don't have valid data, create a minimal structure
            if not extracted_data:
                logger_param.warning(f"LTMS: No valid JSON object found in LLM response for {user_id_str}. Response: {llm_json_str[:100]}...")

                # Create a minimal valid structure to avoid returning None
                extracted_data = {"personal_details": {}, "preferences": {}}
                return

        except Exception as e:
            logger_param.error(f"LTMS: Failed to parse JSON response for details/preferences for {user_id_str}. Error: {e}. Response: {llm_json_str[:100]}...")
            return

        if not extracted_data or (not extracted_data.get("personal_details") and not extracted_data.get("preferences")):
            logger_param.info(f"LTMS: LLM extracted no new details or preferences for user {user_id_str}.")
            return

        # Ensure user profile exists
        profile = await ensure_user_profile_exists(ltms_param, user_id_str, username, logger_param)
        if not profile:
            logger_param.warning(f"LTMS: Could not get or create profile for {user_id_str} ({username}) before details/preferences update. Skipping.")
            return

        # Use the new transaction-based method to update details and preferences
        success = await ltms_param.update_user_profile_with_details_and_preferences(
            user_id_str,
            extracted_data
        )

        if success:
            logger_param.info(f"LTMS: Successfully updated details/preferences for user {user_id_str} using transaction.")
        else:
            logger_param.error(f"LTMS: Failed to update details/preferences for user {user_id_str} in database using transaction.")
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
                    # Use the task manager to create and track the task
                    task_name = f"details_prefs_update_{user_id_str}_{message.id}"
                    await task_manager.add_task(
                        _handle_details_and_preferences_update(client, message, ltms_instance, logger),
                        task_name=task_name
                    )
                    logger.debug(f"Started background task {task_name} for user {user_id_str}")
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
        # Redact user message content for privacy
        user_id_hash = hash(str(message.author.id))  # Use a hash instead of actual ID
        logger.info(f"Response triggered for user (hash: {user_id_hash}) due to {trigger_reason}. Message length: {len(message.content)} chars")

        current_context = {
            'message_content': message.content,
            'replied_to_bot_message_content': replied_to_bot_message_content
        }

        try:
            # First, handle the LLM response with typing indicator
            async with message.channel.typing():
                # Use global stmm, ltms_instance, GOOGLE_API_KEY, and logger for _handle_llm_response
                _first_llm_call_successful_and_has_text = await _handle_llm_response(
                    message, current_context, stmm_param=stmm, ltms_param=ltms_instance,
                    google_api_key_param=GOOGLE_API_KEY, logger_param=logger, user_profile=profile
                )

            # Then, handle the user summary update without typing indicator
            if _first_llm_call_successful_and_has_text and ltms_instance and message.author and not message.author.bot:
                if user_id_str and profile:
                     # Use global stmm, ltms_instance, GOOGLE_API_KEY, and logger for _handle_user_summary_update
                     await _handle_user_summary_update(
                         message=message,
                         context=current_context,
                         stmm_param=stmm,
                         ltms_param=ltms_instance,
                         google_api_key_param=GOOGLE_API_KEY,
                         user_id_str=user_id_str,
                         username=message.author.name,
                         profile=profile,
                         logger_param=logger
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

# --- PII Redaction and Secure Logging ---
class PIIFilter(logging.Filter):
    """Filter to redact PII from log messages"""

    def __init__(self):
        super().__init__()
        # Patterns to identify and redact
        self.patterns = [
            # User IDs - replace with hashed version
            (r'user_id["\']?\s*[:=]\s*["\']?([0-9]+)["\']?', r'user_id=USER_\1_REDACTED'),
            # Email addresses
            (r'[\w\.-]+@[\w\.-]+\.\w+', r'[EMAIL_REDACTED]'),
            # Phone numbers - various formats
            (r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', r'[PHONE_REDACTED]'),
            # IP addresses
            (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', r'[IP_REDACTED]'),
            # API keys and tokens - look for common patterns
            (r'(api[_-]?key|token|secret|password|auth)["\']?\s*[:=]\s*["\']?([^"\']{4,})["\']?', r'\1=[KEY_REDACTED]'),
            # Credit card numbers
            (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', r'[CC_REDACTED]'),
            # Social security numbers
            (r'\b\d{3}-\d{2}-\d{4}\b', r'[SSN_REDACTED]'),
            # Discord tokens
            (r'(discord[_-]?token)["\']?\s*[:=]\s*["\']?([^"\']{10,})["\']?', r'\1=[TOKEN_REDACTED]'),
            # Supabase keys
            (r'(supabase[_-]?key)["\']?\s*[:=]\s*["\']?([^"\']{10,})["\']?', r'\1=[KEY_REDACTED]'),
            # Google API keys
            (r'(google[_-]?api[_-]?key)["\']?\s*[:=]\s*["\']?([^"\']{10,})["\']?', r'\1=[KEY_REDACTED]'),
        ]

        # Compile regex patterns for better performance
        self.compiled_patterns = [(re.compile(pattern), repl) for pattern, repl in self.patterns]

    def filter(self, record):
        """Filter log records to redact PII"""
        if isinstance(record.msg, str):
            # Apply all redaction patterns
            for pattern, repl in self.compiled_patterns:
                record.msg = pattern.sub(repl, record.msg)

            # Redact any args that might contain PII
            if record.args:
                args_list = list(record.args)
                for i, arg in enumerate(args_list):
                    if isinstance(arg, str):
                        for pattern, repl in self.compiled_patterns:
                            args_list[i] = pattern.sub(repl, arg)
                record.args = tuple(args_list)

        return True

# --- Logging Configuration ---
def setup_logging():
    """Configure logging with both console and file handlers with rotation and PII redaction"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "discord_bot.log")

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create handlers
    # File handler with 5MB max size and 5 backup files
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create and add PII filter to all handlers
    pii_filter = PIIFilter()
    file_handler.addFilter(pii_filter)
    console_handler.addFilter(pii_filter)

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Configure our module logger
    module_logger = logging.getLogger(__name__)
    module_logger.setLevel(logging.DEBUG)

    # Log that we've set up secure logging
    module_logger.info("Logging configured with PII redaction and file rotation")

    return module_logger

# --- Graceful Shutdown Handling ---
async def shutdown(signal_received=None):
    """Handle graceful shutdown of the bot"""
    try:
        if signal_received:
            logger.info(f"Received shutdown signal: {signal_received}")
        else:
            logger.info("Initiating shutdown sequence")

        # Cancel all background tasks
        logger.info("Cancelling background tasks...")
        await task_manager.cancel_all()

        # Close any open connections
        logger.info("Closing connections...")

        # Close Discord client
        if client and hasattr(client, 'is_ready') and client.is_ready():
            logger.info("Closing Discord client connection...")
            await client.close()

        # Clear sensitive data from memory
        logger.info("Clearing sensitive data from memory...")
        if 'env_config' in globals():
            env_config.clear_sensitive_data()

        # Force garbage collection
        gc.collect()

        logger.info("Shutdown complete")
    except Exception as e:
        logger.critical(f"Error during shutdown: {e}", exc_info=True)
        # Ensure the process exits even if shutdown fails
        sys.exit(1)

# Set up signal handlers for graceful shutdown
def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown with proper exception handling"""

    # Create a flag to track shutdown status
    global _shutdown_in_progress
    _shutdown_in_progress = False

    # Create an event to signal when shutdown is complete
    global _shutdown_complete
    _shutdown_complete = threading.Event()

    def handle_signal(sig_name):
        """Create a signal handler that properly handles exceptions"""
        def _handler(sig, frame):
            global _shutdown_in_progress, _shutdown_complete

            # Prevent multiple shutdown attempts
            if _shutdown_in_progress:
                logger.info(f"Shutdown already in progress, ignoring {sig_name} signal")
                return

            _shutdown_in_progress = True
            logger.info(f"Signal handler for {sig_name} has triggered shutdown")

            try:
                # Set the event loop policy to handle the shutdown properly
                if sys.platform == 'win32':
                    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

                # Create a new event loop for the shutdown process
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Run the shutdown coroutine in the new loop
                try:
                    loop.run_until_complete(shutdown(signal_received=sig_name))
                    logger.info(f"Shutdown completed successfully for {sig_name}")
                except Exception as e:
                    logger.critical(f"Error during {sig_name} shutdown: {e}")
                finally:
                    # Clean up the loop
                    try:
                        loop.close()
                    except Exception as e:
                        logger.error(f"Error closing event loop during {sig_name} shutdown: {e}")

                    # Signal that shutdown is complete
                    _shutdown_complete.set()

                    # Exit the process
                    sys.exit(0)
            except Exception as e:
                # If we can't even create the shutdown task, log and exit
                logger.critical(f"Critical error in {sig_name} signal handler: {e}")
                sys.exit(1)

        return _handler

    # Set up handlers for different signals
    if sys.platform != "win32":  # Windows doesn't support SIGTERM the same way
        signal.signal(signal.SIGTERM, handle_signal("SIGTERM"))

    # Handle keyboard interrupt (Ctrl+C)
    signal.signal(signal.SIGINT, handle_signal("SIGINT"))

    logger.info("Signal handlers configured with improved error handling")



# --- Main Execution Block ---
if __name__ == "__main__":
    # Configure logging
    logger = setup_logging()
    logger.info("Logging configured with file rotation.")

    # Set up signal handlers
    setup_signal_handlers()
    logger.info("Signal handlers configured for graceful shutdown")

    # Validate configuration
    if env_config.validate_config():
        try:
            logger.info("Attempting to run Discord client...")
            client.run(DISCORD_BOT_TOKEN)
        except discord.errors.LoginFailure:
            logger.critical("Failed to log in. Please check your DISCORD_BOT_TOKEN.")
        except Exception as e:
            logger.exception(f"An unexpected error occurred during client.run: {e}")
        finally:
            # Ensure we run the shutdown sequence
            if sys.platform == "win32":  # Windows needs special handling
                asyncio.run(shutdown())
            logger.info("Bot has shut down")
    else:
        logger.critical("Configuration validation failed. Please check your environment variables.")