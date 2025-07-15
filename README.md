# Telegram Translator Bot

A multimodal Telegram bot that translates text and image descriptions using the ultra-fast Groq API.

[![Powered by Groq](https://img.shields.io/badge/Powered%20by-Groq-green?style=flat-square)](https://groq.com/)

## Features

*   **Fast Translations:** Translates text messages instantly.
*   **Image Understanding:** Describes images and translates the description.
*   **Language Switching:** Use inline buttons to switch between English, Spanish, Chinese, and Russian.
*   **Seamless UX:** Re-translates the last message immediately when you select a new language.
*   **Docker Ready:** Designed for easy deployment with Docker Compose.

## How to Run

### Prerequisites

*   Docker and Docker Compose
*   A Groq API Key
*   A Telegram Bot Token

### Deployment

1.  **Create `docker-compose.yml`**

    Create a `docker-compose.yml` file on your server:
    ```yaml
    services:
      translator_bot:
        build:
          context: https://github.com/procrastinando/translator_bot.git#main
        image: procrastinando/translator_bot:latest
        container_name: translator_bot
        env_file: .env
        restart: always
    ```

2.  **Create `.env` File**

    In the same directory, create a `.env` file for your secrets:
    ```env
    # Replace with your actual credentials
    TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN_HERE"
    GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"
    ```

3.  **Launch the Bot**

    Run the following command to build the image and start the container:
    ```bash
    docker-compose up --build -d
    ```

The bot is now running. To view logs, use `docker logs translator_bot`. To stop it, run `docker-compose down`.