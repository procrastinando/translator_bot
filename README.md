# Telegram Translator Bot

A multimodal Telegram bot that translates text, images, and audio using the ultra-fast Groq API.

[![Powered by Groq](https://img.shields.io/badge/Powered%20by-Groq-green?style=flat-square)](https://groq.com/)

## Features

*   **Fast Translations:** Translates text messages instantly using GROQ LPU.
*   **Image Understanding:** Describes images and translates the description.
*   **Audio & Voice Transcription:** Transcribes audio files and voice notes using Whisper.
*   **Smart Transcription:** If an audio message is already in the target language, the bot provides a direct transcription.
*   **Customizable Languages:** You can define which languages are available in the bot via the Docker configuration.
*   **Persistent User Preferences:** Remembers each user's chosen target language permanently.
*   **Docker Ready:** Designed for easy and reliable deployment with Docker Compose.

## How to Run

### Prerequisites

*   Docker
*   A Telegram Bot Token

### Deployment

### 1.  OPTION A: Single docker run Command (The Quick Way)

Replace the placeholder values for YOUR_TELEGRAM_TOKEN and LANGUAGES.

```
docker build -t procrastinando/translator_bot:latest https://github.com/procrastinando/translator_bot.git#main && docker run -d --name translator_bot --restart always -e TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN}" -e TRANSLATOR_LANGUAGES="ZH,EN,ES,FR,PT,RU,JA,DE,IT" -v translator_bot_data:/app procrastinando/translator_bot:latest
```
Use the following codes in the `TRANSLATOR_LANGUAGES` environment variable to configure your bot.

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

### 2.  OPTION B: **Create `docker-compose.yml`**

2.1.  Create a `docker-compose.yml` file on your server. This example configures the bot to offer English, Spanish, German, French, and Japanese. Modify the `TRANSLATOR_LANGUAGES` variable to fit your needs.

```yaml
services:
  translator_bot:
    build:
      context: https://github.com/procrastinando/translator_bot.git#main
    image: procrastinando/translator_bot:latest
    container_name: translator_bot
    environment:
      # Define your secrets in an .env file or directly here
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN} # required
      WHISPER_MODEL_ID: ${WHISPER_MODEL_ID} # optional, by default whisper-large-v3
      TRANSLATOR_LANGUAGES: ${TRANSLATOR_LANGUAGES} # optional, by default "ZH,EN,ES,FR,PT,RU,JA,DE,IT"
    volumes:
      - translator_bot_data:/app
    restart: always
volumes:
  translator_bot_data:
```

2.2.  **Create `.env` File (Recommended for Secrets)**

In the same directory, create a `.env` file for your secret keys:

```env
TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN_HERE"
WHISPER_MODEL_ID: whisper-large-v3
TRANSLATOR_LANGUAGES: "ZH,EN,ES,FR,PT,RU,JA,DE,IT"
```

2.3.  **Launch the Bot**

Run the following command to build the image and start the container:
```bash
docker-compose up --build -d
```

The bot is now running with your custom language set. To view logs, use `docker logs translator_bot`.
![translator](https://github.com/user-attachments/assets/1b3f35f8-05ff-4ced-a467-eaa6d9be2a0a)
