import os
import logging
import base64
import io
import yaml  # Import the YAML library
import asyncio # Import asyncio for the lock
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from groq import Groq

# --- 1. Configuration and Setup ---

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

groq_client = Groq(api_key=GROQ_API_KEY)
LLM_MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct"
WHISPER_MODEL_ID = "whisper-large-v3-turbo"

# --- 2. Persistent User Data Handling ---

USER_DATA_FILE = 'user_data.yml'
# A lock to prevent race conditions when writing to the file
data_lock = asyncio.Lock()
# In-memory cache of user data, loaded at startup
user_data = {}

LANGUAGES = {"EN": "English", "ES": "Spanish", "CN": "Chinese", "RU": "Russian"}
MAX_FILE_SIZE_MB = 25

def load_user_data():
    """Loads user data from the YAML file at startup."""
    try:
        with open(USER_DATA_FILE, 'r') as f:
            data = yaml.safe_load(f)
            if data is None:
                return {}
            logger.info(f"Successfully loaded data for {len(data)} users from {USER_DATA_FILE}")
            return data
    except FileNotFoundError:
        logger.info(f"{USER_DATA_FILE} not found. Starting with empty user data.")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing {USER_DATA_FILE}: {e}. Starting with empty user data.")
        return {}

async def save_user_data():
    """Saves the current user data to the YAML file."""
    async with data_lock:
        try:
            with open(USER_DATA_FILE, 'w') as f:
                yaml.dump(user_data, f, allow_unicode=True)
            logger.info("User data saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save user data: {e}")

def get_user_language(chat_id: int) -> str:
    """Gets a user's language from the persistent data, defaulting to English."""
    return user_data.get(chat_id, {}).get('language', 'EN')

async def set_user_language(chat_id: int, lang_code: str):
    """Sets a user's language and saves it to the file."""
    if chat_id not in user_data:
        user_data[chat_id] = {}
    user_data[chat_id]['language'] = lang_code
    await save_user_data()

def create_language_keyboard(current_lang_code: str) -> InlineKeyboardMarkup:
    # This function remains the same
    buttons = [InlineKeyboardButton(text=name, callback_data=code) for code, name in LANGUAGES.items() if code != current_lang_code]
    return InlineKeyboardMarkup([buttons[i:i+3] for i in range(0, len(buttons), 3)])

# --- 3. Core Logic (Functions unchanged, but their callers will change) ---

async def get_llm_translation(text_to_translate: str, target_language: str) -> str:
    # ... (no changes in this function)
    prompt = f"Translate the following text to {LANGUAGES[target_language]} directly, omitting any annotations, romanizations, or transliterations."
    messages = [{"role": "user", "content": f"{prompt}\n\n--- TEXT ---\n{text_to_translate}"}]
    try:
        chat_completion = groq_client.chat.completions.create(messages=messages, model=LLM_MODEL_ID, max_tokens=2048)
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling Groq LLM API: {e}")
        return "Sorry, an error occurred during text translation."

async def get_audio_transcription(file_bytes: bytearray, filename: str) -> str:
    # ... (no changes in this function)
    try:
        audio_stream = io.BytesIO(file_bytes)
        transcription = groq_client.audio.transcriptions.create(file=(filename, audio_stream), model=WHISPER_MODEL_ID, response_format="text")
        return transcription
    except Exception as e:
        logger.error(f"Error calling Groq Whisper API: {e}")
        return "Sorry, an error occurred during audio transcription."

# --- 4. Telegram Handlers (Updated to use new data functions) ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    # Set default language if user is new
    if chat_id not in user_data:
        await set_user_language(chat_id, 'EN')
    
    current_lang = get_user_language(chat_id)
    logger.info(f"User {chat_id} started the bot. Language: {current_lang}.")
    await update.message.reply_text(
        "Welcome! I can translate text, images, and audio. Send a message to get started.",
        reply_markup=create_language_keyboard(current_lang),
    )

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (logic is similar, just uses the new functions)
    chat_id = update.effective_chat.id
    user_text = update.message.text
    logger.info(f"Received text message from {chat_id}.")
    context.user_data["last_text"] = user_text
    context.user_data.pop("last_photo_file_id", None)
    context.user_data.pop("last_transcription", None)
    target_language = get_user_language(chat_id)
    translation = await get_llm_translation(user_text, target_language)
    await update.message.reply_text(translation, reply_markup=create_language_keyboard(target_language))

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (logic is similar, just uses the new functions)
    chat_id = update.effective_chat.id
    logger.info(f"Received photo message from {chat_id}.")
    photo_file = await update.message.photo[-1].get_file()
    context.user_data["last_photo_file_id"] = photo_file.file_id
    context.user_data.pop("last_text", None)
    context.user_data.pop("last_transcription", None)
    image_bytes = await photo_file.download_as_bytearray()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    prompt = f"Describe this image in English."
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
    try:
        chat_completion = groq_client.chat.completions.create(messages=messages, model=LLM_MODEL_ID, max_tokens=2048)
        description = chat_completion.choices[0].message.content
        target_language = get_user_language(chat_id)
        translation = await get_llm_translation(description, target_language)
    except Exception as e:
        logger.error(f"Error calling Groq LLM API for image: {e}")
        translation = "Sorry, an error occurred while processing the image."
    await update.message.reply_text(translation, reply_markup=create_language_keyboard(get_user_language(chat_id)))

async def handle_audio_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (logic is similar, just uses the new functions)
    chat_id = update.effective_chat.id
    audio_obj = update.message.audio or update.message.voice
    if audio_obj.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        await update.message.reply_text(f"Sorry, the audio file is too large. Max size is {MAX_FILE_SIZE_MB}MB.")
        return
    file_handle = await audio_obj.get_file()
    file_bytes = await file_handle.download_as_bytearray()
    filename = audio_obj.file_name if hasattr(audio_obj, 'file_name') else "voice.ogg"
    transcribed_text = await get_audio_transcription(file_bytes, filename)
    if "error occurred" in transcribed_text:
        await update.message.reply_text(transcribed_text)
        return
    context.user_data["last_transcription"] = transcribed_text
    context.user_data.pop("last_text", None)
    context.user_data.pop("last_photo_file_id", None)
    target_language = get_user_language(chat_id)
    final_translation = await get_llm_translation(transcribed_text, target_language)
    await update.message.reply_text(final_translation, reply_markup=create_language_keyboard(target_language))

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    new_lang_code = query.data
    if new_lang_code not in LANGUAGES:
        return
    
    logger.info(f"User {chat_id} set language to {new_lang_code}. Saving preference.")
    await set_user_language(chat_id, new_lang_code)
    
    translation = "Could not find the original message to re-translate."
    if original_text := context.user_data.get("last_text"):
        translation = await get_llm_translation(original_text, new_lang_code)
    elif original_transcription := context.user_data.get("last_transcription"):
        translation = await get_llm_translation(original_transcription, new_lang_code)
    elif file_id := context.user_data.get("last_photo_file_id"):
        translation = f"Language set to {LANGUAGES[new_lang_code]}. Please send the image again to describe it in the new language."
    
    await query.edit_message_text(text=translation, reply_markup=create_language_keyboard(new_lang_code))

# --- 5. Main Execution Block ---

def main() -> None:
    if not TELEGRAM_BOT_TOKEN or not GROQ_API_KEY:
        logger.critical("FATAL: TELEGRAM_BOT_TOKEN and GROQ_API_KEY must be set.")
        return
    
    # Load user data into the global variable at startup
    global user_data
    user_data = load_user_data()

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    application.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, handle_audio_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    logger.info("Bot is starting up with persistent data...")
    application.run_polling()
    logger.info("Bot has been stopped.")

if __name__ == "__main__":
    main()