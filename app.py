import os
import logging
import base64
import io
import yaml
import asyncio
import math
import subprocess
import wave
import random
import html
from dotenv import load_dotenv
from piper import PiperVoice

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
from telegram.error import BadRequest, TimedOut
from groq import AsyncGroq, RateLimitError, BadRequestError, APIConnectionError

# --- 1. Configuration & Constants ---

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WHISPER_MODEL_ID = os.getenv("WHISPER_MODEL_ID", "whisper-large-v3")
TRANSLATOR_LANGUAGES_STR = os.getenv("TRANSLATOR_LANGUAGES", "ZH,EN,ES,FR,PT,RU,JA,DE,IT")
TTS_ID_STR = os.getenv("TTS_ID", "")

# Parse permitted IDs for Piper TTS
PIPER_PERMITTED_IDS = set(TTS_ID_STR.split(',')) if TTS_ID_STR else set()

# Default Prompts
PROMPT_SYSTEM_DEFAULT = (
    "You are a direct translation engine. Your sole function is to translate the provided text into <target_language>. "
    "Do not add any commentary, explanations, annotations or transliterations. "
    "If the source text is already in <target_language>, output the original text verbatim without any changes."
)
PROMPT_OCR_DEFAULT = (
    "You are an optical character recognition (OCR) engine. Your primary task is to meticulously "
    "transcribe all text from the provided image. Present the transcribed text exactly as it appears. "
    "If there is no text in the image, provide a brief, one-sentence description."
)

# Models
MODELS = {
    "Kimi k2": "moonshotai/kimi-k2-instruct-0905",
    "📷 Llama 4 Maverick": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "GPT-Oss 120B": "openai/gpt-oss-120b"
}
MULTIMODAL_MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct"

# TTS Configuration
VOICES_DIR = "voices"
AUDIO_DIR = "audio_temp"

PLAYAI_MODELS = { "EN": "canopylabs/orpheus-v1-english", "AR": "canopylabs/orpheus-arabic-saudi" }
PLAYAI_VOICES = { "EN": "autumn", "AR": "fahad" }

PIPER_VOICES = {
    "CA": "ca_ES-upc_ona-medium", "ZH": "zh_CN-huayan-medium", "CS": "cs_CZ-jirka-medium",
    "CY": "cy_GB-bu_tts-medium", "DA": "da_DK-talesyntese-medium", "DE": "de_DE-thorsten-high",
    "EL": "el_GR-rapunzelina-low", "ES": "es_MX-claude-high", "FA": "fa_IR-amir-medium",
    "FI": "fi_FI-harri-medium", "FR": "fr_FR-siwis-medium", "HI": "hi_IN-pratham-medium",
    "HU": "hu_HU-anna-medium", "IS": "is_IS-salka-medium", "IT": "it_IT-paola-medium",
    "KA": "ka_GE-natia-medium", "KK": "kk_KZ-issai-high", "LB": "lb_LU-marylux-medium",
    "LV": "lv_LV-aivars-medium", "ML": "ml_IN-meera-medium", "NE": "ne_NP-google-medium",
    "NL": "nl_NL-mls-medium", "NO": "no_NO-talesyntese-medium", "PL": "pl_PL-mc_speech-medium",
    "PT": "pt_BR-cadu-medium", "RO": "ro_RO-mihai-medium", "RU": "ru_RU-irina-medium",
    "SK": "sk_SK-lili-medium", "SL": "sl_SI-artur-medium", "SR": "sr_RS-serbski_institut-medium",
    "SV": "sv_SE-nst-medium", "SW": "sw_CD-lanfrica-medium", "TR": "tr_TR-dfki-medium",
    "UK": "uk_UA-ukrainian_tts-medium", "VI": "vi_VN-vais1000-medium"
}
loaded_voices = {}

# Conversation States
SET_API_KEY, SET_FALLBACK_KEY, EDIT_SYSTEM_PROMPT, EDIT_OCR_PROMPT = range(4)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- 2. Database & Data Management ---

USER_DATA_FILE = 'database.yml'
data_lock = asyncio.Lock()
user_data = {}

ALL_SUPPORTED_LANGUAGES = {
    "AR": "Arabic", "BN": "Bengali", "CA": "Catalan", "CS": "Czech", "CY": "Welsh",
    "DA": "Danish", "DE": "German", "EL": "Greek", "EN": "English", "ES": "Spanish",
    "FA": "Farsi", "FI": "Finnish", "FR": "French", "HE": "Hebrew", "HI": "Hindi",
    "HU": "Hungarian", "ID": "Indonesian", "IS": "Icelandic", "IT": "Italian",
    "JA": "Japanese", "KA": "Georgian", "KK": "Kazakh", "KO": "Korean", "LB": "Luxembourgish",
    "LV": "Latvian", "ML": "Malayalam", "NE": "Nepali", "NL": "Dutch", "NO": "Norwegian",
    "PL": "Polish", "PT": "Portuguese", "RO": "Romanian", "RU": "Russian", "SK": "Slovak",
    "SL": "Slovenian", "SR": "Serbian", "SV": "Swedish", "SW": "Swahili", "TH": "Thai",
    "TR": "Turkish", "UK": "Ukrainian", "UR": "Urdu", "VI": "Vietnamese", "ZH": "Chinese"
}

LANGUAGES = {}

def initialize_languages():
    global LANGUAGES
    user_lang_codes = [code.strip().upper() for code in TRANSLATOR_LANGUAGES_STR.split(',')]
    for code in user_lang_codes:
        if code in ALL_SUPPORTED_LANGUAGES:
            LANGUAGES[code] = ALL_SUPPORTED_LANGUAGES[code]
    if not LANGUAGES:
        logger.critical("No valid languages configured. Exiting.")
        exit(1)
    logger.info(f"Bot configured with languages: {list(LANGUAGES.keys())}")

def initialize_piper_voices():
    os.makedirs(VOICES_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    for lang_code in LANGUAGES:
        if lang_code in PIPER_VOICES:
            download_piper_voice(PIPER_VOICES[lang_code])

def download_piper_voice(voice_name: str):
    onnx_path = os.path.join(VOICES_DIR, f"{voice_name}.onnx")
    if not os.path.exists(onnx_path):
        logger.info(f"Downloading Piper voice: {voice_name}...")
        try:
            subprocess.run(
                ["python3", "-m", "piper.download", "--voice", voice_name, "--output-dir", VOICES_DIR],
                check=True, capture_output=True, text=True
            )
        except Exception as e:
            logger.error(f"Failed to download Piper voice {voice_name}: {e}")

def load_user_data():
    try:
        with open(USER_DATA_FILE, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError: return {}

async def save_user_data():
    async with data_lock:
        with open(USER_DATA_FILE, 'w') as f:
            yaml.dump(user_data, f, allow_unicode=True)

# --- Accessors ---
def get_user_language(chat_id: int) -> str:
    return user_data.get(chat_id, {}).get('language', list(LANGUAGES.keys())[0])

def get_user_model_config(chat_id: int) -> dict:
    return user_data.get(chat_id, {}).get('model_config', {"name": MULTIMODAL_MODEL_ID})

def get_user_system_prompt(chat_id: int) -> str:
    return user_data.get(chat_id, {}).get('system_prompt', PROMPT_SYSTEM_DEFAULT)

def get_user_ocr_prompt(chat_id: int) -> str:
    return user_data.get(chat_id, {}).get('ocr_prompt', PROMPT_OCR_DEFAULT)

def get_user_listen_mode(chat_id: int) -> bool:
    return user_data.get(chat_id, {}).get('listen_mode', False)

async def set_user_data_value(chat_id: int, key: str, value: any):
    if chat_id not in user_data: user_data[chat_id] = {}
    user_data[chat_id][key] = value
    await save_user_data()

# --- 3. Core Logic: API & Validation ---

async def validate_api_key(api_key: str) -> bool:
    if not api_key or not api_key.startswith("gsk_"):
        return False
    try:
        client = AsyncGroq(api_key=api_key, timeout=20.0)
        await client.models.list()
        return True
    except Exception:
        return False

async def get_groq_client(chat_id: int) -> AsyncGroq | None:
    primary = user_data.get(chat_id, {}).get('groq_api_key')
    fallback = user_data.get(chat_id, {}).get('fallback_api_key')
    
    available_keys = [k for k in [primary, fallback] if k]
    if not available_keys: return None
    
    selected_key = random.choice(available_keys)
    return AsyncGroq(api_key=selected_key, timeout=120.0, max_retries=2)

async def get_llm_response(chat_id: int, messages: list, model_config: dict) -> str:
    primary = user_data.get(chat_id, {}).get('groq_api_key')
    fallback = user_data.get(chat_id, {}).get('fallback_api_key')
    
    keys = [primary, fallback] if primary and fallback else ([primary] if primary else [fallback])
    if len(keys) > 1: random.shuffle(keys)
        
    for i, api_key in enumerate(keys):
        if not api_key: continue
        try:
            client = AsyncGroq(api_key=api_key, timeout=60.0)
            params = {"messages": messages, "model": model_config['name'], "max_tokens": 4096}
            if 'reasoning_effort' in model_config: params['reasoning_effort'] = model_config['reasoning_effort']
            chat_completion = await client.chat.completions.create(**params)
            return chat_completion.choices[0].message.content
        except Exception as e:
            if i < len(keys) - 1: continue
            return f"❌ Error: {str(e)}"
    return "❌ No valid API key configured."

# --- 4. Core Logic: Audio & TTS ---

async def generate_tts_file(chat_id: int, text: str) -> str | None:
    """Generates audio and returns the path to the MP3 file."""
    if not get_user_listen_mode(chat_id):
        return None
    
    lang_code = get_user_language(chat_id)
    wav_path = os.path.join(AUDIO_DIR, f"{chat_id}.wav")
    mp3_path = os.path.join(AUDIO_DIR, f"{chat_id}.mp3")
    
    success = False

    if lang_code in PLAYAI_VOICES:
        client = await get_groq_client(chat_id)
        if client:
            try:
                response = await client.audio.speech.create(
                    model=PLAYAI_MODELS[lang_code],
                    voice=PLAYAI_VOICES[lang_code],
                    input=text[:900], 
                    response_format="wav"
                )
                await response.write_to_file(wav_path)
                success = True
            except Exception as e:
                logger.error(f"Groq TTS Error: {e}")
    
    elif lang_code in PIPER_VOICES:
        if str(chat_id) in PIPER_PERMITTED_IDS:
            voice_name = PIPER_VOICES[lang_code]
            try:
                if voice_name not in loaded_voices:
                    onnx_path = os.path.join(VOICES_DIR, f"{voice_name}.onnx")
                    if os.path.exists(onnx_path):
                        loaded_voices[voice_name] = PiperVoice.load(onnx_path)
                
                if voice_name in loaded_voices:
                    def write_wav():
                        with wave.open(wav_path, "wb") as f:
                            loaded_voices[voice_name].synthesize_wav(text[:900], f)
                    await asyncio.to_thread(write_wav)
                    success = True
            except Exception as e:
                logger.error(f"Piper TTS Error: {e}")

    if success:
        subprocess.run(["ffmpeg", "-i", wav_path, "-y", "-b:a", "64k", "-vn", mp3_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(wav_path): os.remove(wav_path)
        return mp3_path

    return None

# --- 5. UI Helpers ---

async def send_ephemeral_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, duration: int = 2):
    try:
        msg = await context.bot.send_message(chat_id=chat_id, text=text)
        await asyncio.sleep(duration)
        await msg.delete()
    except Exception: pass

async def update_user_commands(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    def mask(k): return f"{k[:5]}...{k[-4:]}" if k else "Not set"
    primary = user_data.get(chat_id, {}).get('groq_api_key')
    fallback = user_data.get(chat_id, {}).get('fallback_api_key')
    listen_status = "🔊 ON" if get_user_listen_mode(chat_id) else "🔇 OFF"
    
    commands = [
        BotCommand("prompt", "⚙️ Edit Prompts"),
        BotCommand("models", "🧠 Select Model"),
        BotCommand("listen", f"TTS: {listen_status}"),
        BotCommand("api", f"Main: {mask(primary)}"),
        BotCommand("fallback_api", f"Bkp: {mask(fallback)}"),
    ]
    try: await context.bot.set_my_commands(commands, scope=BotCommandScopeChat(chat_id=chat_id))
    except: pass

def create_language_keyboard(current_lang_code: str):
    buttons = [InlineKeyboardButton(text=name, callback_data=f"lang_{code}") 
               for code, name in LANGUAGES.items() if code != current_lang_code]
    columns = 4
    return InlineKeyboardMarkup([buttons[i:i + columns] for i in range(0, len(buttons), columns)])

# --- 6. Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_chat.id
    await update_user_commands(chat_id, context)
    
    client = await get_groq_client(chat_id)
    
    if not client:
        msg = await update.message.reply_text(
            "👋 **Welcome!**\n\nTo begin, please send your **Groq API Key**.\n"
            "You can get one from [console.groq.com](https://console.groq.com/keys)",
            parse_mode='Markdown'
        )
        context.user_data['menu_id'] = msg.message_id
        return SET_API_KEY
    
    await update.message.reply_text(
        "✅ **System Ready**\nSend text, photos, or audio.",
        reply_markup=create_language_keyboard(get_user_language(chat_id)),
        parse_mode='Markdown'
    )
    return ConversationHandler.END

# --- API Keys ---

async def cancel_api_setup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancels the API setup process."""
    query = update.callback_query
    await query.answer()
    await query.message.delete()
    await send_ephemeral_message(context, query.message.chat_id, "❌ Operation cancelled.")
    return ConversationHandler.END

async def api_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data="api_cancel")]])
    await update.message.reply_text("🔑 Please send your **Primary** Groq API Key:", reply_markup=kb, parse_mode='Markdown')
    return SET_API_KEY

async def fallback_api_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data="api_cancel")]])
    await update.message.reply_text("🔑 Please send your **Fallback** Groq API Key:", reply_markup=kb, parse_mode='Markdown')
    return SET_FALLBACK_KEY

async def receive_primary_key(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    key = update.message.text.strip()
    await update.message.delete()
    
    if await validate_api_key(key):
        await set_user_data_value(chat_id, 'groq_api_key', key)
        await send_ephemeral_message(context, chat_id, "✅ Primary Key Saved")
        await update_user_commands(chat_id, context)
        if not user_data.get(chat_id, {}).get('language'):
             await update.message.reply_text("Select target language:", reply_markup=create_language_keyboard("EN"))
        return ConversationHandler.END
    else:
        # Re-send with cancel button if invalid
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data="api_cancel")]])
        await update.message.reply_text("❌ Invalid Key. Try again:", reply_markup=kb)
        return SET_API_KEY

async def receive_fallback_key(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    key = update.message.text.strip()
    await update.message.delete()
    
    if await validate_api_key(key):
        await set_user_data_value(chat_id, 'fallback_api_key', key)
        await send_ephemeral_message(context, chat_id, "✅ Fallback Key Saved")
        await update_user_commands(chat_id, context)
        return ConversationHandler.END
    else:
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data="api_cancel")]])
        await update.message.reply_text("❌ Invalid Key. Try again:", reply_markup=kb)
        return SET_FALLBACK_KEY

# --- Prompts ---

async def prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.delete()
    keyboard = [
        [InlineKeyboardButton("🧠 System Prompt", callback_data="prompt_edit_sys")],
        [InlineKeyboardButton("🖼️ OCR/Image Prompt", callback_data="prompt_edit_ocr")],
        [InlineKeyboardButton("❌ Cancel", callback_data="prompt_cancel")]
    ]
    msg = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="⚙️ **Prompt Configuration**",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )
    context.user_data['menu_msg_id'] = msg.message_id
    return ConversationHandler.END

async def prompt_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    chat_id = query.message.chat_id
    data = query.data
    
    if data == "prompt_cancel":
        await query.message.delete()
        return ConversationHandler.END
        
    elif data == "prompt_edit_sys":
        current = get_user_system_prompt(chat_id)
        escaped_current = html.escape(current)
        text = f"📝 **Edit System Prompt**\n\nTap to copy:\n<code>{escaped_current}</code>\n\n👇 Send new prompt:"
        await query.edit_message_text(text, parse_mode='HTML', 
                                      reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Cancel", callback_data="prompt_cancel")]]))
        return EDIT_SYSTEM_PROMPT

    elif data == "prompt_edit_ocr":
        current = get_user_ocr_prompt(chat_id)
        escaped_current = html.escape(current)
        text = f"📝 **Edit OCR Prompt**\n\nTap to copy:\n<code>{escaped_current}</code>\n\n👇 Send new prompt:"
        await query.edit_message_text(text, parse_mode='HTML', 
                                      reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Cancel", callback_data="prompt_cancel")]]))
        return EDIT_OCR_PROMPT

async def save_system_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.delete()
    
    menu_id = context.user_data.get('menu_msg_id')
    if menu_id:
        try: await context.bot.delete_message(chat_id, menu_id)
        except: pass
        
    await set_user_data_value(chat_id, 'system_prompt', update.message.text)
    await send_ephemeral_message(context, chat_id, "✅ System Prompt Updated")
    return ConversationHandler.END

async def save_ocr_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.delete()
    
    menu_id = context.user_data.get('menu_msg_id')
    if menu_id:
        try: await context.bot.delete_message(chat_id, menu_id)
        except: pass
        
    await set_user_data_value(chat_id, 'ocr_prompt', update.message.text)
    await send_ephemeral_message(context, chat_id, "✅ OCR Prompt Updated")
    return ConversationHandler.END

# --- Listen & Models ---

async def listen_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.delete()
    chat_id = update.effective_chat.id
    current = get_user_listen_mode(chat_id)
    new_mode = not current
    await set_user_data_value(chat_id, 'listen_mode', new_mode)
    
    status = "ON 🔊" if new_mode else "OFF 🔇"
    await send_ephemeral_message(context, chat_id, f"Audio Generation: {status}")
    await update_user_commands(chat_id, context)

async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.delete()
    btns = [[InlineKeyboardButton(n, callback_data=f"model_{v}") for n, v in MODELS.items()]]
    await context.bot.send_message(
        chat_id=update.effective_chat.id, 
        text="🧠 **Select AI Model:**", 
        reply_markup=InlineKeyboardMarkup(btns),
        parse_mode='Markdown'
    )

# --- Message Handling ---

async def handle_response_delivery(update: Update, context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, target_lang: str):
    """Delivers the response. Tries to send Voice+Caption if TTS is on, else Text."""
    keyboard = create_language_keyboard(target_lang)
    mp3_path = await generate_tts_file(chat_id, text)
    
    if mp3_path:
        try:
            with open(mp3_path, 'rb') as voice_file:
                if len(text) <= 1024:
                    await context.bot.send_voice(chat_id, voice=voice_file, caption=text, reply_markup=keyboard)
                else:
                    await context.bot.send_message(chat_id, text, reply_markup=keyboard)
                    await context.bot.send_voice(chat_id, voice=voice_file, caption="🔊 Audio Translation")
        finally:
            if os.path.exists(mp3_path): os.remove(mp3_path)
    else:
        await context.bot.send_message(chat_id, text, reply_markup=keyboard)

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not await get_groq_client(chat_id): return
        
    user_text = update.message.text
    context.user_data["last_item"] = {"type": "text", "content": user_text}
    
    target_lang = get_user_language(chat_id)
    sys_prompt = get_user_system_prompt(chat_id).replace("<target_language>", LANGUAGES[target_lang])
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_text}]
    
    translation = await get_llm_response(chat_id, messages, get_user_model_config(chat_id))
    
    await handle_response_delivery(update, context, chat_id, translation, target_lang)

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not await get_groq_client(chat_id): return

    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    
    photo_file = await update.message.photo[-1].get_file()
    image_bytes = await photo_file.download_as_bytearray()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    context.user_data["last_item"] = {"type": "photo", "content": base64_image}

    target_lang = get_user_language(chat_id)
    ocr_p = get_user_ocr_prompt(chat_id)
    sys_p = get_user_system_prompt(chat_id).replace("<target_language>", LANGUAGES[target_lang])
    
    combined_prompt = f"{ocr_p}\n\nStrictly after describing/transcribing, apply the following instruction to the result:\n{sys_p}"
    
    messages = [{
        "role": "user", 
        "content": [
            {"type": "text", "text": combined_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    }]
    
    translation = await get_llm_response(chat_id, messages, {"name": MULTIMODAL_MODEL_ID})
    
    await handle_response_delivery(update, context, chat_id, translation, target_lang)

async def handle_audio_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Downloads voice/audio, transcribes via Groq Whisper, determines logic, and sends response."""
    chat_id = update.effective_chat.id
    client = await get_groq_client(chat_id)
    if not client: return

    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    
    try:
        audio_file = await (update.message.voice or update.message.audio).get_file()
        file_bytes = await audio_file.download_as_bytearray()
        filename = "voice.ogg" if update.message.voice else (update.message.audio.file_name or "audio.mp3")
        
        transcription = await client.audio.transcriptions.create(
            file=(filename, bytes(file_bytes)),
            model=WHISPER_MODEL_ID,
            response_format="verbose_json"
        )
        
        transcribed_text = transcription.text
        if not transcribed_text:
            await update.message.reply_text("Could not transcribe audio.")
            return

        context.user_data["last_item"] = {"type": "text", "content": transcribed_text}

        target_lang = get_user_language(chat_id)
        sys_prompt = get_user_system_prompt(chat_id).replace("<target_language>", LANGUAGES[target_lang])
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": transcribed_text}]
        
        final_response = await get_llm_response(chat_id, messages, get_user_model_config(chat_id))
        
        await handle_response_delivery(update, context, chat_id, final_response, target_lang)

    except (APIConnectionError, RateLimitError) as e:
        logger.error(f"Audio handling API error: {e}")
        await update.message.reply_text("⚠️ Connection error with Transcription service. Please try again.")
    except Exception as e:
        logger.error(f"Audio handling error: {e}")
        await update.message.reply_text("Error processing audio message.")


async def generic_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = query.message.chat_id
    
    if data.startswith("lang_"):
        new_lang = data.split("_")[1]
        await set_user_data_value(chat_id, 'language', new_lang)
        
        last = context.user_data.get("last_item")
        if last:
            await query.message.delete()
            target_lang = new_lang
            if last['type'] == 'text':
                sys_prompt = get_user_system_prompt(chat_id).replace("<target_language>", LANGUAGES[target_lang])
                msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": last['content']}]
                res = await get_llm_response(chat_id, msgs, get_user_model_config(chat_id))
            elif last['type'] == 'photo':
                ocr_p = get_user_ocr_prompt(chat_id)
                sys_p = get_user_system_prompt(chat_id).replace("<target_language>", LANGUAGES[target_lang])
                combined_prompt = f"{ocr_p}\n\nApply instruction:\n{sys_p}"
                msgs = [{"role": "user", "content": [{"type": "text", "text": combined_prompt}, 
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{last['content']}"}}]}]
                res = await get_llm_response(chat_id, msgs, {"name": MULTIMODAL_MODEL_ID})
            
            await handle_response_delivery(update, context, chat_id, res, target_lang)
        else:
            await query.edit_message_text(f"Language set to: {LANGUAGES[new_lang]}")
            
    elif data.startswith("model_"):
        mid = data.replace("model_", "")
        await set_user_data_value(chat_id, 'model_config', {"name": mid})
        await query.message.delete()
        await send_ephemeral_message(context, chat_id, "🧠 Model Updated")
        await update_user_commands(chat_id, context)

# --- Main Application Setup ---

def main():
    initialize_languages()
    initialize_piper_voices()
    global user_data
    user_data = load_user_data()
    
    request = HTTPXRequest(
        connection_pool_size=8, 
        read_timeout=60.0, 
        write_timeout=60.0, 
        connect_timeout=60.0
    )

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).build()

    api_conv = ConversationHandler(
        entry_points=[
            CommandHandler("api", api_command),
            CommandHandler("fallback_api", fallback_api_command),
            CommandHandler("start", start)
        ],
        states={
            SET_API_KEY: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_primary_key)],
            SET_FALLBACK_KEY: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_fallback_key)],
        },
        fallbacks=[
            CommandHandler("start", start),
            CallbackQueryHandler(cancel_api_setup, pattern="^api_cancel$")
        ]
    )

    prompt_conv = ConversationHandler(
        entry_points=[
            CommandHandler("prompt", prompt_command),
            CallbackQueryHandler(prompt_callback_handler, pattern="^prompt_")
        ],
        states={
            EDIT_SYSTEM_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, save_system_prompt)],
            EDIT_OCR_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, save_ocr_prompt)],
        },
        fallbacks=[CallbackQueryHandler(prompt_callback_handler, pattern="^prompt_cancel$")]
    )

    app.add_handler(api_conv)
    app.add_handler(prompt_conv)
    app.add_handler(CommandHandler("listen", listen_command))
    app.add_handler(CommandHandler("models", models_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio_message))
    app.add_handler(CallbackQueryHandler(generic_callback))
    
    logger.info("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
