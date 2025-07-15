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

# Load environment variables from a .env file (recommended)
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set up detailed logging to monitor the bot in the terminal
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
# Set the logging level for the HTTPX client to WARNING to prevent overly verbose logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)
MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct"

# --- 2. Bot State and UI ---

# Dictionary to store the target language for each user.
# Using context.user_data is generally better for persistence, but this is simple.
user_languages = {}
LANGUAGES = {
    "EN": "English",
    "ES": "Spanish",
    "CN": "Chinese",
    "RU": "Russian",
}

def create_language_keyboard(current_lang_code: str) -> InlineKeyboardMarkup:
    """Creates an inline keyboard with buttons for other available languages."""
    buttons = [
        InlineKeyboardButton(text=name, callback_data=code)
        for code, name in LANGUAGES.items()
        if code != current_lang_code
    ]
    # Arrange buttons in a flexible grid, max 3 per row
    keyboard = [buttons[i:i + 3] for i in range(0, len(buttons), 3)]
    return InlineKeyboardMarkup(keyboard)


# --- 3. Core Translation Logic ---

async def get_translation_from_groq(content_parts: list, target_language: str) -> str:
    """Calls the Groq API and returns the translation."""
    prompt = (
        f"Translate this image or text to {LANGUAGES[target_language]} directly, "
        "omitting any annotations, romanizations, or transliterations."
    )
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}, *content_parts]}
    ]

    try:
        logger.info(f"Calling Groq API for language: {target_language}")
        chat_completion = groq_client.chat.completions.create(
            messages=messages, model=MODEL_ID, max_tokens=2048
        )
        translation = chat_completion.choices[0].message.content
        logger.info("Successfully received translation from Groq.")
        return translation
    except Exception as e:
        logger.error(f"Error calling Groq API: {e}")
        return "Sorry, I encountered an error during translation. Please try again."


# --- 4. Telegram Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message for the /start command."""
    chat_id = update.effective_chat.id
    user_languages[chat_id] = "EN"  # Default to English
    logger.info(f"New user {chat_id} started the bot. Language set to EN.")
    await update.message.reply_text(
        "Welcome! I can translate text and images for you. "
        "Send me a message to get started.",
        reply_markup=create_language_keyboard("EN"),
    )

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles and translates regular text messages."""
    chat_id = update.effective_chat.id
    user_text = update.message.text
    logger.info(f"Received text message from {chat_id}.")
    
    # Store the original text in user_data for re-translation via buttons
    context.user_data["last_text"] = user_text
    context.user_data.pop("last_photo_file_id", None) # Clear old photo data

    target_language = user_languages.get(chat_id, "EN")
    content_parts = [{"type": "text", "text": user_text}]
    
    translation = await get_translation_from_groq(content_parts, target_language)
    
    await update.message.reply_text(
        translation, reply_markup=create_language_keyboard(target_language)
    )

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles and translates images."""
    chat_id = update.effective_chat.id
    logger.info(f"Received photo message from {chat_id}.")

    photo_file = await update.message.photo[-1].get_file()
    
    # Store the file_id for re-translation. file_id is a permanent reference.
    context.user_data["last_photo_file_id"] = photo_file.file_id
    context.user_data.pop("last_text", None) # Clear old text data

    image_bytes = await photo_file.download_as_bytearray()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    target_language = user_languages.get(chat_id, "EN")
    content_parts = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]
    
    translation = await get_translation_from_groq(content_parts, target_language)
    
    await update.message.reply_text(
        translation, reply_markup=create_language_keyboard(target_language)
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles inline button clicks for instant re-translation."""
    query = update.callback_query
    await query.answer()  # Acknowledge the button press

    chat_id = query.message.chat_id
    new_lang_code = query.data
    
    if new_lang_code not in LANGUAGES:
        return

    logger.info(f"User {chat_id} clicked button to change language to {new_lang_code}.")
    user_languages[chat_id] = new_lang_code
    
    content_parts = []
    translation = "Could not find the original message to re-translate."

    # Check if the last message was text
    if original_text := context.user_data.get("last_text"):
        logger.info(f"Re-translating text for user {chat_id}.")
        content_parts = [{"type": "text", "text": original_text}]
        translation = await get_translation_from_groq(content_parts, new_lang_code)

    # Check if the last message was a photo
    elif file_id := context.user_data.get("last_photo_file_id"):
        logger.info(f"Re-translating image (file_id: {file_id}) for user {chat_id}.")
        try:
            # Re-download the image using the stored file_id
            photo_file = await context.bot.get_file(file_id)
            image_bytes = await photo_file.download_as_bytearray()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            content_parts = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
            translation = await get_translation_from_groq(content_parts, new_lang_code)
        except Exception as e:
            logger.error(f"Failed to re-process image for user {chat_id}: {e}")
            translation = "Error: Could not re-process the original image."

    # Edit the message with the new translation and the updated keyboard
    await query.edit_message_text(
        text=translation, reply_markup=create_language_keyboard(new_lang_code)
    )

# --- 5. Main Execution Block ---

def main() -> None:
    """Sets up the application and runs the bot."""
    if not TELEGRAM_BOT_TOKEN or not GROQ_API_KEY:
        logger.critical("FATAL: TELEGRAM_BOT_TOKEN and GROQ_API_KEY must be set.")
        return

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Register all the handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    logger.info("Bot is starting up...")
    
    # Run the bot until the user presses Ctrl-C
    application.run_polling()
    
    logger.info("Bot has been stopped.")


if __name__ == "__main__":
    main()