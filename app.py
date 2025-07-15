import os
import logging
import base64
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

# --- 2. Bot State and UI ---

user_languages = {}
LANGUAGES = {
    "EN": "English", "ES": "Spanish", "CN": "Chinese", "RU": "Russian"
}
# Groq API limit for free tier
MAX_FILE_SIZE_MB = 25


def create_language_keyboard(current_lang_code: str) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(text=name, callback_data=code)
        for code, name in LANGUAGES.items()
        if code != current_lang_code
    ]
    keyboard = [buttons[i : i + 3] for i in range(0, len(buttons), 3)]
    return InlineKeyboardMarkup(keyboard)


# --- 3. Core Logic ---

async def get_llm_translation(text_to_translate: str, target_language: str) -> str:
    """Translates a given string of text using the Groq LLM."""
    prompt = (
        f"Translate the following text to {LANGUAGES[target_language]} directly, "
        "omitting any annotations, romanizations, or transliterations."
    )
    messages = [
        {"role": "user", "content": f"{prompt}\n\n--- TEXT ---\n{text_to_translate}"}
    ]
    try:
        logger.info(f"Calling LLM to translate to {target_language}.")
        chat_completion = groq_client.chat.completions.create(
            messages=messages, model=LLM_MODEL_ID, max_tokens=2048
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling Groq LLM API: {e}")
        return "Sorry, an error occurred during text translation."


async def get_audio_transcription(file_path_or_bytes, filename: str) -> str:
    """Transcribes audio using Groq's Whisper model."""
    try:
        logger.info(f"Calling Whisper API to transcribe '{filename}'.")
        transcription = groq_client.audio.transcriptions.create(
            file=(filename, file_path_or_bytes),
            model=WHISPER_MODEL_ID,
            response_format="text",
        )
        logger.info("Successfully received transcription from Whisper.")
        return transcription
    except Exception as e:
        logger.error(f"Error calling Groq Whisper API: {e}")
        return "Sorry, an error occurred during audio transcription."


# --- 4. Telegram Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_languages[chat_id] = "EN"
    logger.info(f"New user {chat_id} started the bot. Language set to EN.")
    await update.message.reply_text(
        "Welcome! I can translate text, images, and audio. "
        "Send me a message to get started.",
        reply_markup=create_language_keyboard("EN"),
    )


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_text = update.message.text
    logger.info(f"Received text message from {chat_id}.")

    context.user_data["last_text"] = user_text
    context.user_data.pop("last_photo_file_id", None)
    context.user_data.pop("last_transcription", None)

    target_language = user_languages.get(chat_id, "EN")
    translation = await get_llm_translation(user_text, target_language)
    await update.message.reply_text(
        translation, reply_markup=create_language_keyboard(target_language)
    )

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    logger.info(f"Received photo message from {chat_id}.")
    
    photo_file = await update.message.photo[-1].get_file()
    
    context.user_data["last_photo_file_id"] = photo_file.file_id
    context.user_data.pop("last_text", None)
    context.user_data.pop("last_transcription", None)

    image_bytes = await photo_file.download_as_bytearray()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    prompt = (
        f"Describe this image in {LANGUAGES.get('EN')} directly, "
        "omitting any annotations, romanizations, or transliterations."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        }
    ]
    try:
        logger.info(f"Calling LLM to describe image for {chat_id}.")
        chat_completion = groq_client.chat.completions.create(
            messages=messages, model=LLM_MODEL_ID, max_tokens=2048
        )
        description = chat_completion.choices[0].message.content
        target_language = user_languages.get(chat_id, "EN")
        translation = await get_llm_translation(description, target_language)

    except Exception as e:
        logger.error(f"Error calling Groq LLM API for image: {e}")
        translation = "Sorry, an error occurred while processing the image."
    
    await update.message.reply_text(
        translation, reply_markup=create_language_keyboard(user_languages.get(chat_id, "EN"))
    )

async def handle_audio_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles audio or voice messages for transcription and translation."""
    chat_id = update.effective_chat.id
    audio_obj = update.message.audio or update.message.voice
    logger.info(f"Received {('voice' if update.message.voice else 'audio')} message from {chat_id}.")

    # Check file size before downloading
    if audio_obj.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        logger.warning(f"Audio file from {chat_id} is too large: {audio_obj.file_size / 1024**2:.2f} MB")
        await update.message.reply_text(f"Sorry, the audio file is too large. The maximum size is {MAX_FILE_SIZE_MB}MB.")
        return

    # Download the file into memory
    file_handle = await audio_obj.get_file()
    file_bytes = await file_handle.download_as_bytearray()
    
    # Safely get the filename or provide a default for voice notes
    filename = audio_obj.file_name if hasattr(audio_obj, 'file_name') else "voice.ogg"
    
    # Transcribe the audio
    transcribed_text = await get_audio_transcription(file_bytes, filename)
    
    if "error occurred" in transcribed_text:
        await update.message.reply_text(transcribed_text)
        return

    # Store transcription for re-translation
    context.user_data["last_transcription"] = transcribed_text
    context.user_data.pop("last_text", None)
    context.user_data.pop("last_photo_file_id", None)
    
    # Translate the transcribed text
    target_language = user_languages.get(chat_id, "EN")
    final_translation = await get_llm_translation(transcribed_text, target_language)
    
    await update.message.reply_text(
        final_translation, reply_markup=create_language_keyboard(target_language)
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    chat_id = query.message.chat_id
    new_lang_code = query.data
    
    if new_lang_code not in LANGUAGES:
        return

    logger.info(f"User {chat_id} clicked button to change language to {new_lang_code}.")
    user_languages[chat_id] = new_lang_code
    
    translation = "Could not find the original message to re-translate."

    if original_text := context.user_data.get("last_text"):
        logger.info(f"Re-translating text for user {chat_id}.")
        translation = await get_llm_translation(original_text, new_lang_code)
    
    elif original_transcription := context.user_data.get("last_transcription"):
        logger.info(f"Re-translating audio transcription for user {chat_id}.")
        translation = await get_llm_translation(original_transcription, new_lang_code)

    elif file_id := context.user_data.get("last_photo_file_id"):
        logger.info(f"Re-translating image (file_id: {file_id}) for user {chat_id}.")
        # To avoid re-processing the image, which adds complexity, we simply inform the user.
        translation = f"Language set to {LANGUAGES[new_lang_code]}. Please send the image again to describe it in the new language."

    await query.edit_message_text(
        text=translation, reply_markup=create_language_keyboard(new_lang_code)
    )

# --- 5. Main Execution Block ---

def main() -> None:
    if not TELEGRAM_BOT_TOKEN or not GROQ_API_KEY:
        logger.critical("FATAL: TELEGRAM_BOT_TOKEN and GROQ_API_KEY must be set.")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    application.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, handle_audio_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    logger.info("Bot is starting up with audio features...")
    application.run_polling()
    logger.info("Bot has been stopped.")


if __name__ == "__main__":
    main()