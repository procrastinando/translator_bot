# import os
# import logging
# import base64
# import io
# import yaml
# import asyncio
# from dotenv import load_dotenv

# from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
# from telegram.ext import (
#     Application,
#     CommandHandler,
#     MessageHandler,
#     CallbackQueryHandler,
#     ContextTypes,
#     filters,
# )
# from groq import Groq

# # --- 1. Configuration and Setup ---

# load_dotenv()
# TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# logging.basicConfig(
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
# )
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logger = logging.getLogger(__name__)

# groq_client = Groq(api_key=GROQ_API_KEY)
# LLM_MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct"
# WHISPER_MODEL_ID = "whisper-large-v3-turbo"

# # --- 2. Persistent User Data Handling ---

# USER_DATA_FILE = 'user_data.yml'
# data_lock = asyncio.Lock()
# user_data = {}

# LANGUAGES = {"EN": "English", "ES": "Spanish", "CN": "Chinese", "RU": "Russian"}
# MAX_FILE_SIZE_MB = 25

# def load_user_data():
#     try:
#         with open(USER_DATA_FILE, 'r') as f:
#             data = yaml.safe_load(f)
#             if data is None: return {}
#             logger.info(f"Successfully loaded data for {len(data)} users from {USER_DATA_FILE}")
#             return data
#     except FileNotFoundError:
#         logger.info(f"{USER_DATA_FILE} not found. Starting with empty user data.")
#         return {}
#     except Exception as e:
#         logger.error(f"Error loading {USER_DATA_FILE}: {e}. Starting empty.")
#         return {}

# async def save_user_data():
#     async with data_lock:
#         try:
#             with open(USER_DATA_FILE, 'w') as f:
#                 yaml.dump(user_data, f, allow_unicode=True)
#         except Exception as e:
#             logger.error(f"Failed to save user data: {e}")

# def get_user_language(chat_id: int) -> str:
#     return user_data.get(chat_id, {}).get('language', 'EN')

# async def set_user_language(chat_id: int, lang_code: str):
#     if chat_id not in user_data:
#         user_data[chat_id] = {}
#     user_data[chat_id]['language'] = lang_code
#     await save_user_data()

# def create_language_keyboard(current_lang_code: str) -> InlineKeyboardMarkup:
#     buttons = [InlineKeyboardButton(text=name, callback_data=code) for code, name in LANGUAGES.items() if code != current_lang_code]
#     return InlineKeyboardMarkup([buttons[i:i+3] for i in range(0, len(buttons), 3)])

# # --- 3. Core Logic ---

# async def get_llm_translation(text_to_translate: str, target_language: str) -> str:
#     prompt = f"Translate the following text to {LANGUAGES[target_language]} directly, omitting any annotations or transliterations."
#     messages = [{"role": "user", "content": f"{prompt}\n\n--- TEXT ---\n{text_to_translate}"}]
#     try:
#         chat_completion = groq_client.chat.completions.create(messages=messages, model=LLM_MODEL_ID, max_tokens=2048)
#         return chat_completion.choices[0].message.content
#     except Exception as e:
#         logger.error(f"Error calling Groq LLM API: {e}")
#         return "Sorry, an error occurred during text translation."

# async def get_audio_transcription(file_bytes: bytearray, filename: str) -> tuple[str, str] | tuple[None, None]:
#     """Transcribes audio, returning the text and detected language."""
#     try:
#         logger.info(f"Calling Whisper API to transcribe '{filename}'.")
#         audio_stream = io.BytesIO(file_bytes)
#         transcription = groq_client.audio.transcriptions.create(
#             file=(filename, audio_stream),
#             model=WHISPER_MODEL_ID,
#             response_format="verbose_json", # Request rich JSON to get the language
#         )
#         logger.info(f"Whisper detected language: {transcription.language}")
#         return transcription.text, transcription.language
#     except Exception as e:
#         logger.error(f"Error calling Groq Whisper API: {e}")
#         return None, None

# # --- 4. Telegram Handlers ---

# async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     chat_id = update.effective_chat.id
#     if chat_id not in user_data:
#         await set_user_language(chat_id, 'EN')
#     current_lang = get_user_language(chat_id)
#     logger.info(f"User {chat_id} started the bot. Language: {current_lang}.")
#     await update.message.reply_text(
#         "Welcome! I can translate text, images, and audio.",
#         reply_markup=create_language_keyboard(current_lang),
#     )

# async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     chat_id = update.effective_chat.id
#     user_text = update.message.text
#     context.user_data["last_text"] = user_text
#     context.user_data.pop("last_photo_file_id", None)
#     context.user_data.pop("last_transcription", None)
#     target_language = get_user_language(chat_id)
#     translation = await get_llm_translation(user_text, target_language)
#     await update.message.reply_text(translation, reply_markup=create_language_keyboard(target_language))

# async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     chat_id = update.effective_chat.id
#     photo_file = await update.message.photo[-1].get_file()
#     context.user_data["last_photo_file_id"] = photo_file.file_id
#     context.user_data.pop("last_text", None)
#     context.user_data.pop("last_transcription", None)
#     image_bytes = await photo_file.download_as_bytearray()
#     base64_image = base64.b64encode(image_bytes).decode("utf-8")
#     prompt = "Describe this image in English."
#     messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
#     try:
#         chat_completion = groq_client.chat.completions.create(messages=messages, model=LLM_MODEL_ID, max_tokens=2048)
#         description = chat_completion.choices[0].message.content
#         target_language = get_user_language(chat_id)
#         translation = await get_llm_translation(description, target_language)
#     except Exception as e:
#         logger.error(f"Error calling Groq LLM API for image: {e}")
#         translation = "Sorry, an error occurred while processing the image."
#     await update.message.reply_text(translation, reply_markup=create_language_keyboard(get_user_language(chat_id)))

# async def handle_audio_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     chat_id = update.effective_chat.id
#     audio_obj = update.message.audio or update.message.voice
    
#     if audio_obj.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
#         await update.message.reply_text(f"Sorry, audio is too large. Max size: {MAX_FILE_SIZE_MB}MB.")
#         return

#     file_handle = await audio_obj.get_file()
#     file_bytes = await file_handle.download_as_bytearray()
#     filename = audio_obj.file_name if hasattr(audio_obj, 'file_name') else "voice.ogg"
    
#     transcribed_text, detected_lang = await get_audio_transcription(file_bytes, filename)
    
#     if transcribed_text is None:
#         await update.message.reply_text("Sorry, an error occurred during audio transcription.")
#         return

#     context.user_data["last_transcription"] = transcribed_text
#     context.user_data.pop("last_text", None)
#     context.user_data.pop("last_photo_file_id", None)
    
#     target_language = get_user_language(chat_id)
#     final_output = ""

#     # Compare detected language (e.g., "en") with target language (e.g., "EN")
#     if detected_lang and detected_lang.lower()[:2] == target_language.lower():
#         logger.info(f"Source audio language ({detected_lang}) matches target ({target_language}). Sending transcription directly.")
#         final_output = transcribed_text
#     else:
#         logger.info(f"Source audio language ({detected_lang}) differs from target ({target_language}). Translating.")
#         final_output = await get_llm_translation(transcribed_text, target_language)
    
#     await update.message.reply_text(final_output, reply_markup=create_language_keyboard(target_language))

# async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     query = update.callback_query
#     await query.answer()
#     chat_id = query.message.chat_id
#     new_lang_code = query.data
    
#     if new_lang_code not in LANGUAGES:
#         return
    
#     logger.info(f"User {chat_id} set language to {new_lang_code}. Saving preference.")
#     await set_user_language(chat_id, new_lang_code)
    
#     translation = "Could not find original message to re-translate."
#     if original_text := context.user_data.get("last_text"):
#         translation = await get_llm_translation(original_text, new_lang_code)
#     elif original_transcription := context.user_data.get("last_transcription"):
#         translation = await get_llm_translation(original_transcription, new_lang_code)
#     elif file_id := context.user_data.get("last_photo_file_id"):
#         translation = f"Language set to {LANGUAGES[new_lang_code]}. Please send the image again to describe it in the new language."
    
#     await query.edit_message_text(text=translation, reply_markup=create_language_keyboard(new_lang_code))

# # --- 5. Main Execution Block ---

# def main() -> None:
#     if not TELEGRAM_BOT_TOKEN or not GROQ_API_KEY:
#         logger.critical("FATAL: TELEGRAM_BOT_TOKEN and GROQ_API_KEY must be set.")
#         return
    
#     global user_data
#     user_data = load_user_data()

#     application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
#     application.add_handler(CommandHandler("start", start))
#     application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
#     application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
#     application.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, handle_audio_message))
#     application.add_handler(CallbackQueryHandler(button_callback))
    
#     logger.info("Bot is starting up with persistent data and smart transcription...")
#     application.run_polling()
#     logger.info("Bot has been stopped.")

# if __name__ == "__main__":
#     main()

import os
import logging
import base64
import io
import yaml
import asyncio
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

# --- 2. Language and Data Configuration ---

USER_DATA_FILE = 'user_data.yml'
data_lock = asyncio.Lock()
user_data = {}

# Master list of all supported languages
ALL_SUPPORTED_LANGUAGES = {
    "EN": "English", "ES": "Spanish", "CN": "Chinese", "RU": "Russian", "DE": "German",
    "FR": "French", "JA": "Japanese", "IT": "Italian", "KO": "Korean", "PT": "Portuguese",
    "AR": "Arabic", "HI": "Hindi", "BN": "Bengali", "NL": "Dutch", "SV": "Swedish",
    "TR": "Turkish", "PL": "Polish", "VI": "Vietnamese", "TH": "Thai", "HE": "Hebrew"
}

# This will be populated at startup based on the environment variable
LANGUAGES = {}
MAX_FILE_SIZE_MB = 25

def initialize_languages():
    """Initializes the active LANGUAGES dict from an environment variable."""
    global LANGUAGES
    # Get the comma-separated string from env, with a default value
    lang_codes_str = os.getenv("TRANSLATOR_LANGUAGES", "EN,ES,CN,RU")
    
    user_lang_codes = [code.strip().upper() for code in lang_codes_str.split(',')]
    
    for code in user_lang_codes:
        if code in ALL_SUPPORTED_LANGUAGES:
            LANGUAGES[code] = ALL_SUPPORTED_LANGUAGES[code]
        else:
            logger.warning(f"Invalid language code '{code}' in TRANSLATOR_LANGUAGES. Skipping.")
    
    if not LANGUAGES:
        logger.critical("No valid languages were configured. Exiting.")
        exit(1)
        
    logger.info(f"Bot configured with the following languages: {list(LANGUAGES.keys())}")


def load_user_data():
    # ... (no changes here)
    try:
        with open(USER_DATA_FILE, 'r') as f:
            data = yaml.safe_load(f); return data if data else {}
    except FileNotFoundError: return {}
    except Exception as e: logger.error(f"Error loading {USER_DATA_FILE}: {e}"); return {}

async def save_user_data():
    # ... (no changes here)
    async with data_lock:
        with open(USER_DATA_FILE, 'w') as f: yaml.dump(user_data, f, allow_unicode=True)

def get_user_language(chat_id: int) -> str:
    return user_data.get(chat_id, {}).get('language', list(LANGUAGES.keys())[0])

async def set_user_language(chat_id: int, lang_code: str):
    if chat_id not in user_data: user_data[chat_id] = {}
    user_data[chat_id]['language'] = lang_code
    await save_user_data()

def create_language_keyboard(current_lang_code: str) -> InlineKeyboardMarkup:
    buttons = [InlineKeyboardButton(text=name, callback_data=code) for code, name in LANGUAGES.items() if code != current_lang_code]
    return InlineKeyboardMarkup([buttons[i:i+3] for i in range(0, len(buttons), 3)])

# ... The rest of the file (Core Logic, Handlers) remains exactly the same as the previous version ...
# I am omitting it for brevity, as no other changes are required in those sections.

async def get_llm_translation(text_to_translate: str, target_language: str) -> str:
    prompt = f"Translate the following text to {LANGUAGES[target_language]} directly, omitting any annotations or transliterations."
    messages = [{"role": "user", "content": f"{prompt}\n\n--- TEXT ---\n{text_to_translate}"}]
    try:
        chat_completion = groq_client.chat.completions.create(messages=messages, model=LLM_MODEL_ID, max_tokens=2048)
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling Groq LLM API: {e}"); return "Sorry, an error occurred during text translation."

async def get_audio_transcription(file_bytes: bytearray, filename: str) -> tuple[str, str] | tuple[None, None]:
    try:
        audio_stream = io.BytesIO(file_bytes)
        transcription = groq_client.audio.transcriptions.create(file=(filename, audio_stream), model=WHISPER_MODEL_ID, response_format="verbose_json")
        logger.info(f"Whisper detected language: {transcription.language}")
        return transcription.text, transcription.language
    except Exception as e:
        logger.error(f"Error calling Groq Whisper API: {e}"); return None, None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    default_lang = list(LANGUAGES.keys())[0]
    if chat_id not in user_data:
        await set_user_language(chat_id, default_lang)
    current_lang = get_user_language(chat_id)
    logger.info(f"User {chat_id} started the bot. Language: {current_lang}.")
    await update.message.reply_text("Welcome! I can translate text, images, and audio.", reply_markup=create_language_keyboard(current_lang))

# All other handlers (handle_text_message, handle_photo_message, handle_audio_message, button_callback)
# are unchanged. They will automatically use the dynamically created LANGUAGES dictionary.

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_text = update.message.text
    context.user_data["last_text"] = user_text
    context.user_data.pop("last_photo_file_id", None)
    context.user_data.pop("last_transcription", None)
    target_language = get_user_language(chat_id)
    translation = await get_llm_translation(user_text, target_language)
    await update.message.reply_text(translation, reply_markup=create_language_keyboard(target_language))

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    photo_file = await update.message.photo[-1].get_file()
    context.user_data["last_photo_file_id"] = photo_file.file_id
    context.user_data.pop("last_text", None)
    context.user_data.pop("last_transcription", None)
    image_bytes = await photo_file.download_as_bytearray()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    prompt = "Describe this image in English."
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
    try:
        chat_completion = groq_client.chat.completions.create(messages=messages, model=LLM_MODEL_ID, max_tokens=2048)
        description = chat_completion.choices[0].message.content
        target_language = get_user_language(chat_id)
        translation = await get_llm_translation(description, target_language)
    except Exception as e:
        logger.error(f"Error calling Groq LLM API for image: {e}"); translation = "Sorry, an error occurred while processing the image."
    await update.message.reply_text(translation, reply_markup=create_language_keyboard(get_user_language(chat_id)))

async def handle_audio_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    audio_obj = update.message.audio or update.message.voice
    if audio_obj.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        await update.message.reply_text(f"Sorry, audio is too large. Max size: {MAX_FILE_SIZE_MB}MB."); return
    file_handle = await audio_obj.get_file()
    file_bytes = await file_handle.download_as_bytearray()
    filename = audio_obj.file_name if hasattr(audio_obj, 'file_name') else "voice.ogg"
    transcribed_text, detected_lang = await get_audio_transcription(file_bytes, filename)
    if transcribed_text is None:
        await update.message.reply_text("Sorry, an error occurred during audio transcription."); return
    context.user_data["last_transcription"] = transcribed_text
    context.user_data.pop("last_text", None); context.user_data.pop("last_photo_file_id", None)
    target_language = get_user_language(chat_id)
    final_output = ""
    if detected_lang and detected_lang.lower()[:2] == target_language.lower():
        final_output = transcribed_text
    else:
        final_output = await get_llm_translation(transcribed_text, target_language)
    await update.message.reply_text(final_output, reply_markup=create_language_keyboard(target_language))

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    new_lang_code = query.data
    if new_lang_code not in LANGUAGES: return
    await set_user_language(chat_id, new_lang_code)
    translation = "Could not find original message to re-translate."
    if original_text := context.user_data.get("last_text"):
        translation = await get_llm_translation(original_text, new_lang_code)
    elif original_transcription := context.user_data.get("last_transcription"):
        translation = await get_llm_translation(original_transcription, new_lang_code)
    elif file_id := context.user_data.get("last_photo_file_id"):
        translation = f"Language set to {LANGUAGES[new_lang_code]}. Please send the image again to describe it in the new language."
    await query.edit_message_text(text=translation, reply_markup=create_language_keyboard(new_lang_code))


def main() -> None:
    # Initialize languages first
    initialize_languages()
    
    if not TELEGRAM_BOT_TOKEN or not GROQ_API_KEY:
        logger.critical("FATAL: TELEGRAM_BOT_TOKEN and GROQ_API_KEY must be set.")
        return
    
    global user_data
    user_data = load_user_data()

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    application.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, handle_audio_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    logger.info("Bot is starting up with dynamically configured languages...")
    application.run_polling()
    logger.info("Bot has been stopped.")

if __name__ == "__main__":
    main()