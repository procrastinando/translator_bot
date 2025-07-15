# Telegram Translator Bot

A multimodal Telegram bot that translates text, images, and audio using the ultra-fast Groq API.

[![Powered by Groq](https://img.shields.io/badge/Powered%20by-Groq-green?style=flat-square)](https://groq.com/)

## Features

*   **Fast Translations:** Translates text messages instantly using Llama 4.
*   **Image Understanding:** Describes images and translates the description.
*   **Audio & Voice Transcription:** Transcribes audio files and voice notes using Whisper.
*   **Smart Transcription:** If an audio message is already in the target language, the bot provides a direct transcription.
*   **Customizable Languages:** You can define which languages are available in the bot via the Docker configuration.
*   **Persistent User Preferences:** Remembers each user's chosen target language permanently.
*   **Docker Ready:** Designed for easy and reliable deployment with Docker Compose.

## How to Run

### Prerequisites

*   Docker and Docker Compose
*   A Groq API Key
*   A Telegram Bot Token

### Deployment

1.  **Create `docker-compose.yml`**

    Create a `docker-compose.yml` file on your server. This example configures the bot to offer English, Spanish, German, French, and Japanese. Modify the `TRANSLATOR_LANGUAGES` variable to fit your needs.

    ```yaml
    services:
      translator_bot:
        build:
          context: https://github.com/procrastinando/translator_bot.git#main
        image: procrastinando/translator_bot:latest
        container_name: translator_bot
        environment:
          # Define your secrets in an .env file or directly here
          TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}
          GROQ_API_KEY: ${GROQ_API_KEY}
          # Define the languages available in the bot.
          # Use a comma-separated list of codes from the 'Supported Languages' table below.
          TRANSLATOR_LANGUAGES: "EN,ES,DE,FR,JA"
        volumes:
          - translator_bot_data:/app
        restart: always

    volumes:
      translator_bot_data:
    ```

2.  **Create `.env` File (Recommended for Secrets)**

    In the same directory, create a `.env` file for your secret keys:
    ```env
    TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN_HERE"
    GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"
    ```

3.  **Launch the Bot**

    Run the following command to build the image and start the container:
    ```bash
    docker-compose up --build -d
    ```

The bot is now running with your custom language set. To view logs, use `docker logs translator_bot`.

## Supported Languages

Use the following codes in the `TRANSLATOR_LANGUAGES` environment variable to configure your bot.

| Language | Code | Language | Code | Language | Code |
|:---|:----:|:---|:----:|:---|:----:|
| Arabic | `AR` | Hebrew | `HE` | Portuguese | `PT` |
| Bengali | `BN` | Hindi | `HI` | Russian | `RU` |
| Chinese | `CN` | Indonesian | `ID` | Spanish | `ES` |
| Dutch | `NL` | Italian | `IT` | Swedish | `SV` |
| English | `EN` | Japanese | `JA` | Thai | `TH` |
| French | `FR` | Korean | `KO` | Turkish | `TR` |
| German | `DE` | Polish | `PL` | Urdu | `UR` |
| Vietnamese| `VI` | | | | |
![translator](https://github.com/user-attachments/assets/1b3f35f8-05ff-4ced-a467-eaa6d9be2a0a)
