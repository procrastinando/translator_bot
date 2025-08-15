import os
import logging
import base64
import io
import yaml
import asyncio
import math
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand, BotCommandScopeChat
from telegram.request import HTTPXRequest
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
    ConversationHandler,
)
from telegram.error import BadRequest
from groq import AsyncGroq, RateLimitError

# --- 1. Configuration and Setup ---

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

TELEGRAM_MSG_LIMIT = 4096
CHUNK_PREFIX_BUFFER = 20 

PROMPT_SYSTEM_DEFAULT = (
    "You are a direct translation engine. Your sole function is to translate the "
    "provided text into <target_language>. Do not add any commentary, explanations, "
    "annotations, or transliterations. If the source text is already in "
    "<target_language>, output the original text verbatim without any changes or notifications."
)
PROMPT_OCR_DEFAULT = (
    "You are an optical character recognition (OCR) engine. Your primary task is to meticulously "
    "transcribe all text from the provided image. Present the transcribed text exactly as it appears. "
    "If there is no text in the image, and only in that case, provide a brief, one-sentence description of the image content in English."
)

WHISPER_MODEL_ID = "whisper-large-v3"
MODELS = {
    "Kimi k2": "moonshotai/kimi-k2-instruct",
    "📷 Llama 4 Maverick": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "GPT-Oss 120B": "openai/gpt-oss-120b"
}
MULTIMODAL_MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct"

API_KEY, FALLBACK_API_KEY, SYSTEM_PROMPT, OCR_PROMPT = range(4)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- 2. Language and Data Configuration ---

USER_DATA_FILE = 'database.yml'
data_lock = asyncio.Lock()
user_data = {}

ALL_SUPPORTED_LANGUAGES = {
    "AR": "Arabic", "BN": "Bengali", "CN": "Chinese", "DE": "German", "EN": "English",
    "ES": "Spanish", "FR": "French", "HE": "Hebrew", "HI": "Hindi", "ID": "Indonesian",
    "IT": "Italian", "JA": "Japanese", "KO": "Korean", "NL": "Dutch", "PL": "Polish",
    "PT": "Portuguese", "RU": "Russian", "SV": "Swedish", "TH": "Thai", "TR": "Turkish",
    "UR": "Urdu", "VI": "Vietnamese"
}

LANGUAGES = {}
MAX_FILE_SIZE_MB = 25

def initialize_languages():
    global LANGUAGES
    lang_codes_str = os.getenv("TRANSLATOR_LANGUAGES", "EN,ES,CN,RU")
    user_lang_codes = [code.strip().upper() for code in lang_codes_str.split(',')]
    for code in user_lang_codes:
        if code in ALL_SUPPORTED_LANGUAGES:
            LANGUAGES[code] = ALL_SUPPORTED_LANGUAGES[code]
    if not LANGUAGES:
        logger.critical("No valid languages configured. Exiting.")
        exit(1)
    logger.info(f"Bot configured with languages: {list(LANGUAGES.keys())}")

def load_user_data():
    try:
        with open(USER_DATA_FILE, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.error(f"Error loading {USER_DATA_FILE}: {e}")
        return {}

async def save_user_data():
    async with data_lock:
        with open(USER_DATA_FILE, 'w') as f:
            yaml.dump(user_data, f, allow_unicode=True)

# --- User Data Accessors ---
def get_user_language(chat_id: int) -> str:
    return user_data.get(chat_id, {}).get('language', list(LANGUAGES.keys())[0])

def get_user_api_key(chat_id: int) -> str | None:
    return user_data.get(chat_id, {}).get('groq_api_key')

def get_user_fallback_api_key(chat_id: int) -> str | None:
    return user_data.get(chat_id, {}).get('fallback_api_key')

def get_user_model_config(chat_id: int) -> dict:
    default_model = {"name": MULTIMODAL_MODEL_ID}
    return user_data.get(chat_id, {}).get('model_config', default_model)

def get_user_system_prompt(chat_id: int) -> str:
    return user_data.get(chat_id, {}).get('system_prompt', PROMPT_SYSTEM_DEFAULT)

def get_user_ocr_prompt(chat_id: int) -> str:
    return user_data.get(chat_id, {}).get('ocr_prompt', PROMPT_OCR_DEFAULT)

async def set_user_data_value(chat_id: int, key: str, value: any):
    if chat_id not in user_data:
        user_data[chat_id] = {}
    user_data[chat_id][key] = value
    await save_user_data()

# --- Dynamic Command Menu & UI Helpers ---
async def update_user_commands(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    def mask_key(key: str | None) -> str:
        if not key or len(key) < 12:
            return "Not set"
        return f"{key[:7]}...{key[-4:]}"

    model_config = get_user_model_config(chat_id)
    model_name_id = model_config.get('name', MULTIMODAL_MODEL_ID)
    model_friendly_name = next((name for name, mid in MODELS.items() if mid == model_name_id), "Unknown Model")
    if 'reasoning_effort' in model_config:
        model_friendly_name += f" ({model_config['reasoning_effort']})"

    api_desc = mask_key(get_user_api_key(chat_id))
    fallback_api_desc = mask_key(get_user_fallback_api_key(chat_id))

    commands = [
        BotCommand("models", model_friendly_name),
        BotCommand("system_prompt", "Customize translation prompt"),
        BotCommand("ocr_prompt", "Customize image transcription prompt"),
        BotCommand("api", api_desc),
        BotCommand("fallback_api", fallback_api_desc),
    ]
    try:
        await context.bot.set_my_commands(commands, scope=BotCommandScopeChat(chat_id=chat_id))
    except Exception as e:
        logger.warning(f"Could not set commands for user {chat_id}: {e}")

async def send_ephemeral_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, duration: int = 3):
    try:
        message = await context.bot.send_message(chat_id=chat_id, text=text)
        await asyncio.sleep(duration)
        await message.delete()
    except Exception as e:
        logger.warning(f"Could not send/delete ephemeral message: {e}")

async def cleanup_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        try:
            await update.message.delete()
        except BadRequest: pass
    
    last_bot_msg_id = context.user_data.pop('last_bot_message_id', None)
    if last_bot_msg_id:
        try:
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=last_bot_msg_id)
        except BadRequest: pass

def create_language_keyboard(current_lang_code: str) -> InlineKeyboardMarkup:
    buttons = [InlineKeyboardButton(text=name, callback_data=f"lang_{code}") for code, name in LANGUAGES.items() if code != current_lang_code]
    columns = 5
    keyboard_layout = [buttons[i:i + columns] for i in range(0, len(buttons), columns)]
    return InlineKeyboardMarkup(keyboard_layout)


# --- 3. Core Logic ---

async def validate_api_key(api_key: str) -> bool:
    if not api_key: return False
    try:
        client = AsyncGroq(api_key=api_key)
        await client.chat.completions.create(messages=[{"role": "user", "content": "test"}], model=MULTIMODAL_MODEL_ID, max_tokens=2)
        return True
    except Exception:
        return False

async def get_llm_response(context: ContextTypes.DEFAULT_TYPE, chat_id: int, messages: list, model_config: dict) -> str | None:
    main_key = get_user_api_key(chat_id)
    fallback_key = get_user_fallback_api_key(chat_id)
    
    for attempt in range(2):
        current_key = main_key if attempt == 0 else fallback_key
        key_name = "primary" if attempt == 0 else "fallback"

        if not current_key:
            continue

        try:
            client = AsyncGroq(api_key=current_key)
            params = {"messages": messages, "model": model_config['name'], "max_tokens": 8192}
            if 'reasoning_effort' in model_config:
                params['reasoning_effort'] = model_config['reasoning_effort']
                
            chat_completion = await client.chat.completions.create(**params)
            return chat_completion.choices[0].message.content

        except RateLimitError as e:
            logger.warning(f"Rate limit hit for user {chat_id} on {key_name} key.")
            if attempt == 0 and fallback_key:
                await send_ephemeral_message(context, chat_id, f"Primary API key limit reached. Trying fallback...", 2)
                continue
            
            retry_after = e.response.headers.get("retry-after")
            wait_time = f"{retry_after} seconds" if retry_after else "a few moments"
            return f"API limit reached on all keys. Please wait {wait_time}."
        
        except Exception as e:
            logger.error(f"Error on {key_name} key for user {chat_id}: {e}")
            if attempt == 0 and fallback_key:
                continue
            return "An API error occurred. Please check your key and model selection."
            
    return "API key not set or invalid. Please use /api."


async def get_translation(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, target_lang_code: str) -> str:
    prompt_template = get_user_system_prompt(chat_id)
    prompt = prompt_template.replace("<target_language>", LANGUAGES[target_lang_code])
    messages = [{"role": "user", "content": f"{prompt}\n\n--- TEXT ---\n{text}"}]
    model_config = get_user_model_config(chat_id)
    return await get_llm_response(context, chat_id, messages, model_config)

async def get_audio_transcription(chat_id: int, file_bytes: bytearray, filename: str) -> tuple[str | None, str | None]:
    api_key = get_user_api_key(chat_id)
    if not api_key:
        return "API Key not set.", None
    try:
        client = AsyncGroq(api_key=api_key)
        audio_stream = io.BytesIO(file_bytes)
        transcription = await client.audio.transcriptions.create(
            file=(filename, audio_stream), model=WHISPER_MODEL_ID, response_format="verbose_json"
        )
        return transcription.text, transcription.language
    except Exception as e:
        logger.error(f"Error calling Groq Whisper API for user {chat_id}: {e}")
        return None, None

# --- 4. Telegram Handlers ---

async def send_translation_response(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str | None):
    chat_id = update.effective_chat.id
    target_language = get_user_language(chat_id)
    
    if not text:
        logger.warning(f"Final translation for user {chat_id} was empty. Sending fallback message.")
        text_to_send = "Sorry, I couldn't generate a response. Please try again."
        await update.message.reply_text(
            text_to_send, 
            reply_markup=create_language_keyboard(target_language)
        )
        return

    if len(text) > TELEGRAM_MSG_LIMIT:
        chunk_size = TELEGRAM_MSG_LIMIT - CHUNK_PREFIX_BUFFER
        total_parts = math.ceil(len(text) / chunk_size)

        for i, part in enumerate(range(0, len(text), chunk_size)):
            chunk = text[part:part + chunk_size]
            part_number = i + 1
            
            reply_markup = create_language_keyboard(target_language) if part_number == total_parts else None
            message_text = f"({part_number}/{total_parts})\n\n{chunk}"
            
            await update.message.reply_text(
                text=message_text,
                reply_markup=reply_markup
            )
    else:
        await update.message.reply_text(
            text=text, 
            reply_markup=create_language_keyboard(target_language)
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_chat.id
    await update_user_commands(chat_id, context)
    if not get_user_api_key(chat_id):
        await update.message.reply_text(
            "Welcome! To get started, please provide your Groq API key.\n"
            "You can get one from https://console.groq.com/keys"
        )
        context.user_data['state'] = API_KEY
        return API_KEY
    else:
        current_lang = get_user_language(chat_id)
        await update.message.reply_text(
            "Welcome back! I'm ready to translate.",
            reply_markup=create_language_keyboard(current_lang)
        )
        return ConversationHandler.END

async def generic_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE, state: int, prompt_text: str) -> int:
    await update.message.delete()
    msg = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=prompt_text,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Cancel", callback_data="cancel")]]),
        parse_mode='Markdown'
    )
    context.user_data['state'] = state
    context.user_data['last_bot_message_id'] = msg.message_id
    return state

async def api_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await generic_prompt_command(update, context, API_KEY, "Please send your primary Groq API key.")

async def fallback_api_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await generic_prompt_command(update, context, FALLBACK_API_KEY, "Please send your fallback Groq API key.")

async def system_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    prompt = get_user_system_prompt(update.effective_chat.id)
    return await generic_prompt_command(update, context, SYSTEM_PROMPT, f"Please send your new system prompt.\n\nCurrent:\n`{prompt}`")

async def ocr_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    prompt = get_user_ocr_prompt(update.effective_chat.id)
    return await generic_prompt_command(update, context, OCR_PROMPT, f"Please send your new OCR prompt.\n\nCurrent:\n`{prompt}`")

async def receive_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_chat.id
    state = context.user_data.get('state')
    user_input = update.message.text
    
    await cleanup_messages(update, context)

    success_msg = ""
    if state == API_KEY or state == FALLBACK_API_KEY:
        if await validate_api_key(user_input):
            key_type = 'groq_api_key' if state == API_KEY else 'fallback_api_key'
            await set_user_data_value(chat_id, key_type, user_input)
            success_msg = f"{'Primary' if state == API_KEY else 'Fallback'} API key updated."
        else:
            await send_ephemeral_message(context, chat_id, "The API key provided is not valid.")
            return ConversationHandler.END
    elif state == SYSTEM_PROMPT:
        await set_user_data_value(chat_id, 'system_prompt', user_input)
        success_msg = "System prompt updated."
    elif state == OCR_PROMPT:
        await set_user_data_value(chat_id, 'ocr_prompt', user_input)
        success_msg = "OCR prompt updated."

    if success_msg:
        await send_ephemeral_message(context, chat_id, success_msg)
    
    await update_user_commands(chat_id, context)
    context.user_data.clear()
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.message.delete()
    await send_ephemeral_message(context, query.message.chat_id, "Operation cancelled.")
    context.user_data.clear()
    return ConversationHandler.END

async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.delete()
    buttons = [
        InlineKeyboardButton(name, callback_data=f"model_{model_id}") 
        for name, model_id in MODELS.items()
    ]
    keyboard = [buttons]
    await context.bot.send_message(
        chat_id=update.effective_chat.id, 
        text="Choose a model (for text/audio):", 
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not get_user_api_key(chat_id):
        await update.message.reply_text("Please set your Groq API key first using /api.")
        return

    user_text = update.message.text
    context.user_data["last_item"] = {"type": "text", "content": user_text}
    
    translation = await get_translation(context, chat_id, user_text, get_user_language(chat_id))
    await send_translation_response(update, context, translation)

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    *** REWRITTEN LOGIC ***
    Handles photos based on the user's selected model, using the correct one-step or two-step process.
    """
    chat_id = update.effective_chat.id
    if not get_user_api_key(chat_id):
        await update.message.reply_text("Please set your Groq API key first using /api.")
        return

    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    
    target_lang_code = get_user_language(chat_id)
    model_config = get_user_model_config(chat_id)
    
    photo_file = await update.message.photo[-1].get_file()
    image_bytes = await photo_file.download_as_bytearray()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    context.user_data["last_item"] = {"type": "photo", "content": base64_image}

    final_translation = None

    # --- BRANCHING LOGIC: ONE-STEP vs. TWO-STEP ---
    if model_config.get('name') == MULTIMODAL_MODEL_ID:
        # ONE-STEP: User has the multimodal model selected. Combine prompts for a single, powerful call.
        logger.info(f"User {chat_id}: Performing one-step image translation.")
        ocr_prompt = get_user_ocr_prompt(chat_id)
        system_prompt = get_user_system_prompt(chat_id).replace("<target_language>", LANGUAGES[target_lang_code])
        
        combined_prompt = f"{ocr_prompt}\n\nAfter processing the image, apply the following instruction to the text you found:\n\n{system_prompt}"
        
        messages = [{"role": "user", "content": [{"type": "text", "text": combined_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
        final_translation = await get_llm_response(context, chat_id, messages, model_config)

    else:
        # TWO-STEP: User has a text-only model selected. Use the vision model for OCR, then the user's model for translation.
        logger.info(f"User {chat_id}: Performing two-step image translation with model {model_config.get('name')}.")
        
        # Step 1: Always use the multimodal model for OCR with the user's custom OCR prompt.
        ocr_prompt = get_user_ocr_prompt(chat_id)
        ocr_messages = [{"role": "user", "content": [{"type": "text", "text": ocr_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
        transcribed_text = await get_llm_response(context, chat_id, ocr_messages, {"name": MULTIMODAL_MODEL_ID})
        
        # Step 2: If OCR was successful, use the user's selected model for translation.
        if transcribed_text and not transcribed_text.startswith("API"):
            final_translation = await get_translation(context, chat_id, transcribed_text, target_lang_code)
        else:
            final_translation = transcribed_text or "Could not transcribe or describe the image."

    await send_translation_response(update, context, final_translation)


async def handle_audio_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not get_user_api_key(chat_id):
        await update.message.reply_text("Please set your Groq API key first using /api.")
        return
        
    audio_obj = update.message.audio or update.message.voice
    if audio_obj.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        await update.message.reply_text(f"Max file size is {MAX_FILE_SIZE_MB}MB.")
        return

    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    
    try:
        file_handle = await audio_obj.get_file()
        file_bytes = await file_handle.download_as_bytearray()
        
        if hasattr(audio_obj, 'file_name') and audio_obj.file_name:
            filename = audio_obj.file_name
        else:
            filename = "voice.ogg"

        transcribed_text, detected_lang = await get_audio_transcription(chat_id, file_bytes, filename)
        if transcribed_text is None:
            await update.message.reply_text("Sorry, couldn't transcribe the audio.")
            return

        context.user_data["last_item"] = {"type": "audio", "content": transcribed_text}
        
        final_output = None
        if detected_lang and detected_lang.lower().startswith(get_user_language(chat_id).lower()):
            final_output = transcribed_text
        else:
            final_output = await get_translation(context, chat_id, transcribed_text, get_user_language(chat_id))
            
        await send_translation_response(update, context, final_output)
    except Exception as e:
        logger.error(f"Failed to process audio for user {chat_id}: {e}", exc_info=True)
        await update.message.reply_text("An error occurred while processing your audio file.")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    data = query.data

    if data.startswith("lang_"):
        new_lang_code = data.split("_")[1]
        await set_user_data_value(chat_id, 'language', new_lang_code)
        
        last_item = context.user_data.get("last_item")
        translation = f"Language set to {LANGUAGES[new_lang_code]}. Send something to translate!"

        if last_item:
            item_type, item_content = last_item["type"], last_item["content"]
            if item_type in ["text", "audio"]:
                translation = await get_translation(context, chat_id, item_content, new_lang_code)
            elif item_type == "photo":
                # Re-run the same robust photo logic for re-translation
                model_config = get_user_model_config(chat_id)
                if model_config.get('name') == MULTIMODAL_MODEL_ID:
                    ocr_prompt = get_user_ocr_prompt(chat_id)
                    system_prompt = get_user_system_prompt(chat_id).replace("<target_language>", LANGUAGES[new_lang_code])
                    combined_prompt = f"{ocr_prompt}\n\nAfter processing the image, apply the following instruction to the text you found:\n\n{system_prompt}"
                    messages = [{"role": "user", "content": [{"type": "text", "text": combined_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{item_content}"}}]}]
                    translation = await get_llm_response(context, chat_id, messages, model_config)
                else:
                    ocr_prompt = get_user_ocr_prompt(chat_id)
                    ocr_messages = [{"role": "user", "content": [{"type": "text", "text": ocr_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{item_content}"}}]}]
                    transcribed_text = await get_llm_response(context, chat_id, ocr_messages, {"name": MULTIMODAL_MODEL_ID})
                    if transcribed_text and not transcribed_text.startswith("API"):
                        translation = await get_translation(context, chat_id, transcribed_text, new_lang_code)
                    else:
                        translation = transcribed_text or "Failed to re-translate image."
        
        if not translation:
             translation = "Sorry, I couldn't re-translate to the new language."

        try:
            await query.edit_message_text(text=translation, reply_markup=create_language_keyboard(new_lang_code))
        except BadRequest as e:
            if "Message is not modified" in str(e):
                pass
            else:
                logger.error(f"Error editing message for re-translation: {e}")

    elif data.startswith("model_"):
        model_id = data.replace("model_", "", 1)
        if model_id == MODELS["GPT-Oss 120B"]:
            keyboard = [[InlineKeyboardButton(e.title(), callback_data=f"reason_{e}_{model_id}") for e in ["low", "medium", "high"]]]
            await query.edit_message_text("Select reasoning effort:", reply_markup=InlineKeyboardMarkup(keyboard))
            return
        else:
            await set_user_data_value(chat_id, 'model_config', {"name": model_id})
            model_name = [k for k, v in MODELS.items() if v == model_id][0]
            await query.message.delete()
            await send_ephemeral_message(context, chat_id, f"Model set to {model_name}.")
            await update_user_commands(chat_id, context)

    elif data.startswith("reason_"):
        _, effort, model_id = data.split("_", 2)
        await set_user_data_value(chat_id, 'model_config', {"name": model_id, "reasoning_effort": effort})
        model_name = [k for k, v in MODELS.items() if v == model_id][0]
        await query.message.delete()
        await send_ephemeral_message(context, chat_id, f"Model set to {model_name} with {effort} reasoning.")
        await update_user_commands(chat_id, context)

    elif data == "cancel":
        await cancel(update, context)

async def post_init(application: Application):
    """Sets generic bot commands for the global scope (new users)."""
    commands = [
        BotCommand("models", "Choose an AI model"),
        BotCommand("system_prompt", "Customize translation prompt"),
        BotCommand("ocr_prompt", "Customize image transcription prompt"),
        BotCommand("api", "Set your primary Groq API key"),
        BotCommand("fallback_api", "Set your fallback Groq API key"),
    ]
    await application.bot.set_my_commands(commands)

def main() -> None:
    initialize_languages()
    global user_data
    user_data = load_user_data()

    httpx_request = HTTPXRequest(connect_timeout=60.0, read_timeout=60.0)
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .request(httpx_request)
        .build()
    )

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
            CommandHandler("api", api_command),
            CommandHandler("fallback_api", fallback_api_command),
            CommandHandler("system_prompt", system_prompt_command),
            CommandHandler("ocr_prompt", ocr_prompt_command),
        ],
        states={
            API_KEY: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_text_input)],
            FALLBACK_API_KEY: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_text_input)],
            SYSTEM_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_text_input)],
            OCR_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_text_input)],
        },
        fallbacks=[CallbackQueryHandler(cancel, pattern="^cancel$")],
    )

    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("models", models_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    application.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, handle_audio_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    logger.info("Bot is starting up...")
    application.run_polling()
    logger.info("Bot has been stopped.")

if __name__ == "__main__":
    main()
