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
    ConversationHandler,
)
from telegram import BotCommand

# --- Constants for ConversationHandler ---
API_KEY, CHOOSING = range(2)

from groq import AsyncGroq, RateLimitError

# --- 1. Configuration and Setup ---

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "meta-llama/llama-4-maverick-17b-128e-instruct")
WHISPER_MODEL_ID = os.getenv("WHISPER_MODEL_ID", "whisper-large-v3-turbo")
PROMPT_TEXT = os.getenv("PROMPT_TEXT", "Translate the following text to <target_language> directly, omitting any annotations or transliterations.")
PROMPT_OCR = os.getenv("PROMPT_OCR", "Transcribe any text you find in this image and translate it to <target_language> directly, omitting any annotations or transliterations.")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- 2. Language and Data Configuration ---

USER_DATA_FILE = 'database.yml'
data_lock = asyncio.Lock()
user_data = {}

# Master list of all supported languages
ALL_SUPPORTED_LANGUAGES = {
    "AR": "Arabic", "BN": "Bengali", "CN": "Chinese", "DE": "German", "EN": "English",
    "ES": "Spanish", "FR": "French", "HE": "Hebrew", "HI": "Hindi", "ID": "Indonesian",
    "IT": "Italian", "JA": "Japanese", "KO": "Korean", "NL": "Dutch", "PL": "Polish",
    "PT": "Portuguese", "RU": "Russian", "SV": "Swedish", "TH": "Thai", "TR": "Turkish",
    "UR": "Urdu", "VI": "Vietnamese"
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
    try:
        with open(USER_DATA_FILE, 'r') as f:
            data = yaml.safe_load(f)
            return data if data else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.error(f"Error loading {USER_DATA_FILE}: {e}")
        return {}

async def save_user_data():
    async with data_lock:
        with open(USER_DATA_FILE, 'w') as f:
            yaml.dump(user_data, f, allow_unicode=True)

def get_user_language(chat_id: int) -> str:
    return user_data.get(chat_id, {}).get('language', list(LANGUAGES.keys())[0])

def get_user_api_key(chat_id: int) -> str | None:
    return user_data.get(chat_id, {}).get('groq_api_key')

async def set_user_language(chat_id: int, lang_code: str):
    if chat_id not in user_data:
        user_data[chat_id] = {}
    user_data[chat_id]['language'] = lang_code
    await save_user_data()

async def set_user_api_key(chat_id: int, api_key: str):
    if chat_id not in user_data:
        user_data[chat_id] = {}
    user_data[chat_id]['groq_api_key'] = api_key
    await save_user_data()


def create_language_keyboard(current_lang_code: str) -> InlineKeyboardMarkup:
    """Creates an inline keyboard with a dynamic number of columns."""
    buttons = [InlineKeyboardButton(text=name, callback_data=code) for code, name in LANGUAGES.items() if code != current_lang_code]
    
    # Change the number here to control the number of columns.
    columns = 5
    
    # Slices the buttons list into rows of 'columns' items each.
    keyboard_layout = [buttons[i:i + columns] for i in range(0, len(buttons), columns)]
    
    return InlineKeyboardMarkup(keyboard_layout)

# --- 3. Core Logic ---

async def validate_api_key(api_key: str) -> bool:
    """Checks if an API key is valid by making a simple test call."""
    try:
        client = AsyncGroq(api_key=api_key)
        await client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model=LLM_MODEL_ID,
            max_tokens=2
        )
        return True
    except Exception as e:
        logger.error(f"API key validation failed: {e}")
        return False

async def get_llm_response(messages: list, api_key: str) -> str | None:
    """Generic function to get a response from the LLM, now with rate limit handling."""
    try:
        client = AsyncGroq(api_key=api_key)
        chat_completion = await client.chat.completions.create(
            messages=messages,
            model=LLM_MODEL_ID,
            max_tokens=2048
        )
        return chat_completion.choices[0].message.content
    except RateLimitError as e:
        logger.warning(f"Rate limit exceeded for user. Details: {e}")
        retry_after = e.response.headers.get("retry-after")
        wait_time = f"{retry_after} seconds" if retry_after else "a few moments"
        return f"You have exceeded your request limit. Please wait {wait_time} before trying again."
    except Exception as e:
        logger.error(f"Error calling Groq LLM API: {e}")
        return "Sorry, an error occurred with the AI model. Please ensure your API key is valid."


async def get_translation(text: str, target_lang_code: str, api_key: str) -> str:
    """Translates text using the LLM."""
    prompt = PROMPT_TEXT.replace("<target_language>", LANGUAGES[target_lang_code])
    messages = [{"role": "user", "content": f"{prompt}\n\n--- TEXT ---\n{text}"}]
    return await get_llm_response(messages, api_key)

async def get_image_transcription_and_translation(base64_image: str, target_lang_code: str, api_key: str) -> str:
    """Transcribes and translates text from an image in one call."""
    prompt = (
        f"First, transcribe any text you find in this image. "
        f"Second, translate the transcribed text into {LANGUAGES[target_lang_code]}. "
        f"Present the translation clearly. If there is no text, describe the image briefly in {LANGUAGES[target_lang_code]}."
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    }]
    return await get_llm_response(messages, api_key)

async def get_audio_transcription(file_bytes: bytearray, filename: str, api_key: str) -> tuple[str | None, str | None]:
    """Transcribes audio using the Whisper API."""
    try:
        client = AsyncGroq(api_key=api_key)
        audio_stream = io.BytesIO(file_bytes)
        transcription = await client.audio.transcriptions.create(
            file=(filename, audio_stream),
            model=WHISPER_MODEL_ID,
            response_format="verbose_json"
        )
        logger.info(f"Whisper detected language: {transcription.language}")
        return transcription.text, transcription.language
    except Exception as e:
        logger.error(f"Error calling Groq Whisper API: {e}")
        return None, None


# --- 4. Telegram Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_chat.id
    api_key = get_user_api_key(chat_id)

    if not api_key:
        await update.message.reply_text(
            "Welcome! To get started, please provide your Groq API key.\n"
            "You can get your free API key from https://console.groq.com/keys"
        )
        return API_KEY
    else:
        current_lang = get_user_language(chat_id)
        await update.message.reply_text(
            "Welcome back! I'm ready to translate for you.",
            reply_markup=create_language_keyboard(current_lang)
        )
        return ConversationHandler.END

async def api_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Asks the user for their API key."""
    await update.message.reply_text(
        "Please send me your new Groq API key.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Cancel", callback_data="cancel")]])
    )
    return API_KEY

async def receive_api_key(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives and validates the API key."""
    chat_id = update.effective_chat.id
    api_key = update.message.text

    if await validate_api_key(api_key):
        await set_user_api_key(chat_id, api_key)
        await update.message.reply_text("Thank you! Your API key has been updated.")
        return ConversationHandler.END
    else:
        await update.message.reply_text(
            "The API key you provided is not valid. Please try again.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Cancel", callback_data="cancel")]])
        )
        return API_KEY

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels the API key conversation."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Operation cancelled.")
    return ConversationHandler.END



async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    api_key = get_user_api_key(chat_id)
    if not api_key:
        await update.message.reply_text("Please set your Groq API key first using /api.")
        return

    user_text = update.message.text
    target_language = get_user_language(chat_id)
    context.user_data["last_item"] = {"type": "text", "content": user_text}
    translation = await get_translation(user_text, target_language, api_key)
    await update.message.reply_text(translation, reply_markup=create_language_keyboard(target_language))

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    api_key = get_user_api_key(chat_id)
    if not api_key:
        await update.message.reply_text("Please set your Groq API key first using /api.")
        return

    target_language = get_user_language(chat_id)
    photo_file = await update.message.photo[-1].get_file()
    image_bytes = await photo_file.download_as_bytearray()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    context.user_data["last_item"] = {"type": "photo", "content": base64_image}
    translation = await get_image_transcription_and_translation(base64_image, target_language, api_key)
    await update.message.reply_text(translation, reply_markup=create_language_keyboard(target_language))

async def handle_audio_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    api_key = get_user_api_key(chat_id)
    if not api_key:
        await update.message.reply_text("Please set your Groq API key first using /api.")
        return
        
    target_language = get_user_language(chat_id)
    audio_obj = update.message.audio or update.message.voice

    if audio_obj.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        await update.message.reply_text(f"Sorry, the audio file is too large. The maximum size is {MAX_FILE_SIZE_MB}MB.")
        return

    file_handle = await audio_obj.get_file()
    file_bytes = await file_handle.download_as_bytearray()
    filename = audio_obj.file_name if hasattr(audio_obj, 'file_name') and audio_obj.file_name else "voice.ogg"
    
    transcribed_text, detected_lang = await get_audio_transcription(file_bytes, filename, api_key)

    if transcribed_text is None:
        await update.message.reply_text("Sorry, I couldn't transcribe the audio. Please try again.")
        return

    context.user_data["last_item"] = {"type": "audio", "content": transcribed_text}

    if detected_lang and detected_lang.lower().startswith(target_language.lower()):
        final_output = transcribed_text
    else:
        final_output = await get_translation(transcribed_text, target_language, api_key)
        
    await update.message.reply_text(final_output, reply_markup=create_language_keyboard(target_language))

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    new_lang_code = query.data

    if new_lang_code not in LANGUAGES:
        logger.warning(f"Received invalid language code in callback: {new_lang_code}")
        return

    await set_user_language(chat_id, new_lang_code)
    
    api_key = get_user_api_key(chat_id)
    if not api_key:
        await query.edit_message_text(
            text=f"Language set to {LANGUAGES[new_lang_code]}. Now, please set your API key with /set_api_key.",
            reply_markup=create_language_keyboard(new_lang_code)
        )
        return

    last_item = context.user_data.get("last_item")
    translation = f"Language set to {LANGUAGES[new_lang_code]}. Send me something to translate!"

    if last_item:
        item_type = last_item.get("type")
        item_content = last_item.get("content")
        
        if item_type == "text":
            translation = await get_translation(item_content, new_lang_code, api_key)
        elif item_type == "photo":
            translation = await get_image_transcription_and_translation(item_content, new_lang_code, api_key)
        elif item_type == "audio":
            translation = await get_translation(item_content, new_lang_code, api_key)
            
    await query.edit_message_text(
        text=translation,
        reply_markup=create_language_keyboard(new_lang_code)
    )


async def post_init(application: Application):
    await application.bot.set_my_commands([
        BotCommand("api", "Change my groq API")
    ])

def main() -> None:
    initialize_languages()
    
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("FATAL: TELEGRAM_BOT_TOKEN must be set.")
        return
    
    global user_data
    user_data = load_user_data()

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start), CommandHandler("api", api_command)],
        states={
            API_KEY: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_api_key)],
        },
        fallbacks=[CallbackQueryHandler(cancel, pattern="^cancel$")],
    )

    application.add_handler(conv_handler)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    application.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, handle_audio_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    logger.info("Bot is starting up...")
    application.run_polling()
    logger.info("Bot has been stopped.")


if __name__ == "__main__":
    main()
