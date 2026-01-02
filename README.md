# Telegram Translator Bot

A high-performance multimodal Telegram bot that translates text, images, and audio using the ultra-fast Groq API. It features a redundant API key system, smart audio processing, and hybrid Text-to-Speech capabilities.

[![Powered by Groq](https://img.shields.io/badge/Powered%20by-Groq-green?style=flat-square)](https://groq.com/)

## Features

*   **⚡ Ultra-Fast Translation:** Uses Groq's LPU inference engine for near-instant text translations.
*   **🔑 Dual API System:** Supports a **Primary** and a **Fallback** API key. The bot randomizes usage between them to balance load and ensures uptime if one key hits a rate limit.
*   **🖼️ Multimodal Vision:** Uses **Llama 4 Maverick** to OCR and describe images, then applies your custom translation instructions.
*   **🎙️ Smart Audio Transcription:**
    *   Transcribes voice notes using **Whisper-large-v3**.
    *   Intelligently decides whether to translate the audio or output the verbatim transcription based on the target language.
*   **🔊 Hybrid Text-to-Speech (TTS):**
    *   **English & Arabic:** High-quality AI generation via Groq/PlayAI (Available to everyone).
    *   **Other Languages:** Local generation via **Piper TTS** (Restricted to specific Admin User IDs).
*   **⚙️ Advanced Customization:**
    *   **/prompt:** Interactive menu to edit System and OCR prompts with "Tap-to-Copy" convenience.
    *   **Persistent Settings:** Remembers your model choice, language, and TTS preferences via a local database.
    *   **Ephemeral UI:** "Saved" notifications self-destruct to keep your chat history clean.

## Bot Commands

| Command | Description |
| :--- | :--- |
| `/start` | Initialize the bot and check system status. |
| `/api` | Set your **Primary** Groq API Key. |
| `/fallback_api` | Set your **Fallback** Groq API Key (Optional). |
| `/prompt` | Customize the System Prompt or OCR Prompt. |
| `/listen` | Toggle Audio (TTS) generation ON/OFF. |
| `/models` | Switch between available AI models (e.g., Llama, Mixtral). |

## How to Run

### Prerequisites

*   **Docker** installed on your server/machine.
*   A **Telegram Bot Token** (from [@BotFather](https://t.me/BotFather)).
*   At least one **Groq API Key** (from [console.groq.com](https://console.groq.com/keys)).

---

### Deployment

#### 1. OPTION A: Single `docker run` Command

Replace `YOUR_TELEGRAM_TOKEN` and configure your `TTS_ID` (your Telegram User ID) to enable multilingual audio generation.

```bash
docker build -t procrastinando/translator_bot:latest https://github.com/procrastinando/translator_bot.git#main && \
docker run -d \
  --name translator_bot \
  --restart always \
  -e TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_TOKEN" \
  -e TRANSLATOR_LANGUAGES="ZH,EN,ES,FR,PT,RU,JA,DE,IT" \
  -e TTS_ID="YOUR_TELEGRAM_USER_ID" \
  -v translator_bot_data:/app \
  procrastinando/translator_bot:latest
```

#### 2. OPTION B: Docker Compose (Recommended)

2.1. Create a `docker-compose.yml` file:

```yaml
services:
  translator_bot:
    build:
      context: https://github.com/procrastinando/translator_bot.git#main
    image: procrastinando/translator_bot:latest
    container_name: translator_bot
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - WHISPER_MODEL_ID=${WHISPER_MODEL_ID}
      - TRANSLATOR_LANGUAGES=${TRANSLATOR_LANGUAGES}
      - TTS_ID=${TTS_ID}
    volumes:
      - translator_bot_data:/app
    restart: always

volumes:
  translator_bot_data:
```

2.2. Create an `.env` file in the same directory:

```env
TELEGRAM_BOT_TOKEN="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
WHISPER_MODEL_ID="whisper-large-v3"
# Comma-separated list of language codes to display in the menu
TRANSLATOR_LANGUAGES="ZH,EN,ES,FR,PT,RU,JA,DE,IT"
# Comma-separated list of Telegram User IDs allowed to generate Piper TTS (Non-EN/AR audio)
TTS_ID="12345678,87654321"
```

2.3. Launch the bot:

```bash
docker-compose up --build -d
```

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `TELEGRAM_BOT_TOKEN` | Your Telegram Bot API Token. | **Required** |
| `WHISPER_MODEL_ID` | The model used for audio transcription. | `whisper-large-v3` |
| `TRANSLATOR_LANGUAGES`| List of languages available in the UI buttons. | `ZH,EN,ES,FR,PT,RU,JA,DE,IT` |
| `TTS_ID` | User IDs permitted to use Piper TTS (for languages other than EN/AR). | Empty (None) |

### Supported Language Codes

Use these codes in `TRANSLATOR_LANGUAGES`:

| Language | Code | Language | Code | Language | Code |
|:---|:----:|:---|:----:|:---|:----:|
| Arabic | `AR` | Hindi | `HI` | Romanian | `RO` |
| Bengali | `BN` | Hungarian | `HU` | Russian | `RU` |
| Catalan | `CA` | Icelandic | `IS` | Serbian | `SR` |
| Chinese | `ZH` | Indonesian | `ID` | Slovak | `SK` |
| Czech | `CS` | Italian | `IT` | Slovenian | `SL` |
| Danish | `DA` | Japanese | `JA` | Spanish | `ES` |
| Dutch | `NL` | Kazakh | `KK` | Swahili | `SW` |
| English | `EN` | Korean | `KO` | Swedish | `SV` |
| Farsi | `FA` | Latvian | `LV` | Thai | `TH` |
| Finnish | `FI` | Luxembourgish | `LB` | Turkish | `TR` |
| French | `FR` | Malayalam | `ML` | Ukrainian | `UK` |
| Georgian | `KA` | Nepali | `NE` | Urdu | `UR` |
| German | `DE` | Norwegian | `NO` | Vietnamese | `VI` |
| Greek | `EL` | Polish | `PL` | Welsh | `CY` |
| Hebrew | `HE` | Portuguese | `PT` | | |

## Screenshots

<img width="400" alt="Bot Menu" src="https://github.com/user-attachments/assets/e904872c-1dbe-4375-a53f-13cf8217f103" />
