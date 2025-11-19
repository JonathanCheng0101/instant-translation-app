import os
import asyncio
import json
import logging
import struct
import io

import azure.cognitiveservices.speech as speechsdk
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import aiohttp

# ====================================================
# 初始化
# ====================================================
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ====================================================
# Whisper → Azure Top 40 語言 mapping
# ====================================================
WHISPER_TO_AZURE = {

    # 英文（全球）
    "en": "en-US",

    # 中文（含廣東）
    "zh": "zh-CN",        # 普通話
    "zh-tw": "zh-TW",     # 台灣
    "yue": "yue-CN",      # 粵語

    # 東亞
    "ja": "ja-JP",
    "ko": "ko-KR",

    # 東南亞
    "vi": "vi-VN",
    "th": "th-TH",
    "id": "id-ID",
    "ms": "ms-MY",

    # 印度語族（最常用）
    "hi": "hi-IN",
    "bn": "bn-IN",
    "ta": "ta-IN",

    # 蒙古語
    "mn": "mn-MN",

    # 西歐
    "fr": "fr-FR",
    "de": "de-DE",
    "es": "es-ES",
    "pt": "pt-BR",
    "it": "it-IT",
    "nl": "nl-NL",
    "sv": "sv-SE",
    "no": "nb-NO",
    "da": "da-DK",
    "fi": "fi-FI",

    # 東歐
    "pl": "pl-PL",
    "cs": "cs-CZ",
    "sk": "sk-SK",
    "hu": "hu-HU",
    "ro": "ro-RO",
    "bg": "bg-BG",

    # 俄羅斯系
    "ru": "ru-RU",
    "uk": "uk-UA",
    "kk": "kk-KZ",

    # 中東
    "ar": "ar-EG",
    "he": "he-IL",
    "tr": "tr-TR",
    "fa": "fa-IR",

    # 非洲（Azure 支援度最高）
    "sw": "sw-KE",
}

# ====================================================
# PCM → WAV（給 Whisper）
# ====================================================
def pcm_to_wav(pcm_bytes: bytes, sample_rate=16000, bits_per_sample=16, channels=1) -> bytes:
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm_bytes)
    riff_size = 36 + data_size

    wav = io.BytesIO()
    wav.write(b'RIFF')
    wav.write(struct.pack('<I', riff_size))
    wav.write(b'WAVE')
    wav.write(b'fmt ')
    wav.write(struct.pack('<I', 16))
    wav.write(struct.pack('<H', 1))
    wav.write(struct.pack('<H', channels))
    wav.write(struct.pack('<I', sample_rate))
    wav.write(struct.pack('<I', byte_rate))
    wav.write(struct.pack('<H', block_align))
    wav.write(struct.pack('<H', bits_per_sample))
    wav.write(b'data')
    wav.write(struct.pack('<I', data_size))
    wav.write(pcm_bytes)

    return wav.getvalue()

# ====================================================
# Whisper detect
# ====================================================
async def whisper_detect_language(wav_bytes: bytes) -> str:
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    form = aiohttp.FormData()
    form.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")
    form.add_field("model", "whisper-1")

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=form) as resp:
            data = await resp.json()
            if resp.status != 200:
                logger.error(f"Whisper error: {data}")
                return "unknown"
            return data.get("language", "unknown")

# ====================================================
# Azure translator detect
# ====================================================
async def detect_language(text: str) -> str:
    if not text:
        return "unknown"

    endpoint = "https://api.cognitive.microsofttranslator.com/detect"
    params = "?api-version=3.0"
    headers = {
        "Ocp-Apim-Subscription-Key": TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": TRANSLATOR_REGION,
        "Content-Type": "application/json",
    }
    body = [{"text": text}]
    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint + params, headers=headers, json=body) as resp:
            data = await resp.json()
            try:
                return data[0]["language"]
            except:
                return "unknown"

# ====================================================
# 翻譯方向
# ====================================================
def decide_to_lang(lang: str) -> str:
    lang = (lang or "").lower()
    if lang.startswith("en"):
        return "zh-Hant"
    elif lang.startswith("zh"):
        return "en"
    elif lang.startswith("ja"):
        return "zh-Hant"
    else:
        return "en"

# ====================================================
# 翻譯
# ====================================================
async def translate_text(text, to_lang):
    if not text:
        return ""
    endpoint = "https://api.cognitive.microsofttranslator.com/translate"
    params = f"?api-version=3.0&to={to_lang}"

    headers = {
        "Ocp-Apim-Subscription-Key": TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": TRANSLATOR_REGION,
        "Content-Type": "application/json",
    }

    body = [{"text": text}]
    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint + params, headers=headers, json=body) as resp:
            data = await resp.json()
            try:
                return data[0]["translations"][0]["text"]
            except:
                return ""

# ====================================================
# WebSocket 主邏輯
# ====================================================

@app.get("/")
def home():
    return {"message": "backend is running"}

@app.websocket("/ws_stream")
async def ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("🔌 WebSocket connected")

    loop = asyncio.get_running_loop()

    fmt = speechsdk.audio.AudioStreamFormat(16000, 16, 1)
    stream = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
    audio_config = speechsdk.audio.AudioConfig(stream=stream)

    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.enable_dictation()

    # 預設語言（第 4 slot 等 Whisper detect 來改）
    default_auto = ["zh-TW", "ko-KR", "es-ES", "es-US"]

    auto_lang = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=default_auto
    )

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
        auto_detect_source_language_config=auto_lang,
    )

    # 狀態
    audio_buffer = bytearray()
    partial_buffer = []
    whisper_lang = None
    first_partial_time = None
    websocket_active = True  # ⭐ 保護 flag

    # ====================================================
    # 更新 Azure 語言模型（動態）
    # ====================================================
    async def update_azure_language(lang_code):
        nonlocal recognizer, auto_lang, websocket_active

        if not websocket_active:
            return

        azure_lang = WHISPER_TO_AZURE.get(lang_code)
        if not azure_lang:
            logger.warning(f"⚠️ Whisper 語言 Azure 不支援：{lang_code}")
            return

        logger.info(f"🔄 Azure 切換語言 → {azure_lang}")

        auto_lang = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=["zh-TW", "ko-KR", "es-ES", azure_lang]
        )

        try:
            recognizer.stop_continuous_recognition_async().get()
        except Exception as e:
            logger.warning(f"stop recognizer error: {e}")

        if not websocket_active:
            return

        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
            auto_detect_source_language_config=auto_lang,
        )

        recognizer.recognizing.connect(on_recognizing)
        recognizer.recognized.connect(on_recognized)

        try:
            recognizer.start_continuous_recognition_async().get()
        except Exception as e:
            logger.warning(f"restart recognizer error: {e}")

    # ====================================================
    # 語言確定後 → 回頭修正
    # ====================================================
    async def correct_previous_partials(lang_code):
        nonlocal partial_buffer, websocket_active

        if not websocket_active:
            return
        if not partial_buffer:
            return

        combined = " ".join(partial_buffer)
        corrected = await translate_text(combined, decide_to_lang(lang_code))

        try:
            await websocket.send_text(json.dumps({
                "type": "correction",
                "text": combined,
                "lang": lang_code,
                "translation": corrected,
            }))
        except Exception as e:
            logger.warning(f"send correction failed: {e}")

        partial_buffer.clear()

    # ====================================================
    # Whisper detect trigger
    # ====================================================
    async def try_whisper():
        nonlocal whisper_lang, first_partial_time, audio_buffer, websocket_active

        if not websocket_active:
            return
        if whisper_lang is not None:
            return
        if first_partial_time is None:
            return

        now = loop.time()
        if now - first_partial_time < 2.0:
            return

        wav_bytes = pcm_to_wav(bytes(audio_buffer))
        lang = await whisper_detect_language(wav_bytes)
        whisper_lang = lang
        print(f"🌍 Whisper detect: {lang}")

        if lang != "unknown" and websocket_active:
            asyncio.run_coroutine_threadsafe(update_azure_language(lang), loop)
            asyncio.run_coroutine_threadsafe(correct_previous_partials(lang), loop)

    # ====================================================
    # partial
    # ====================================================
    async def send_partial(text):
        nonlocal partial_buffer, websocket_active

        if not websocket_active:
            return

        lang = whisper_lang or await detect_language(text)
        translated = await translate_text(text, decide_to_lang(lang))

        try:
            await websocket.send_text(json.dumps({
                "type": "partial",
                "text": text,
                "lang": lang,
                "translation": translated,
            }))
        except Exception as e:
            logger.warning(f"send partial failed: {e}")
            return

        partial_buffer.append(text)

    def on_recognizing(evt):
        nonlocal first_partial_time, websocket_active

        if not websocket_active:
            return

        if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech and evt.result.text:
            if first_partial_time is None:
                first_partial_time = loop.time()

            asyncio.run_coroutine_threadsafe(try_whisper(), loop)
            asyncio.run_coroutine_threadsafe(send_partial(evt.result.text), loop)

    # ====================================================
    # final
    # ====================================================
    async def send_final(text):
        nonlocal whisper_lang, partial_buffer, audio_buffer, first_partial_time, websocket_active

        if not websocket_active:
            return

        lang = whisper_lang or await detect_language(text)
        translated = await translate_text(text, decide_to_lang(lang))

        try:
            await websocket.send_text(json.dumps({
                "type": "final",
                "text": text,
                "lang": lang,
                "translation": translated,
            }))
        except Exception as e:
            logger.warning(f"send final failed: {e}")

        audio_buffer.clear()
        partial_buffer.clear()
        whisper_lang = None
        first_partial_time = None

    def on_recognized(evt):
        nonlocal websocket_active

        if not websocket_active:
            return

        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
            asyncio.run_coroutine_threadsafe(send_final(evt.result.text), loop)

    # 綁 Azure callback
    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.start_continuous_recognition_async().get()

    # ====================================================
    # WebSocket 接收聲音 → Azure + Whisper
    # ====================================================
    try:
        async for msg in websocket.iter_bytes():
            if not websocket_active:
                break
            audio_buffer.extend(msg)
            stream.write(msg)
    except Exception as e:
        logger.info(f"WebSocket closed with exception: {e}")
    finally:
        websocket_active = False
        try:
            stream.close()
        except Exception:
            pass

        try:
            recognizer.stop_continuous_recognition_async().get()
        except Exception:
            pass

        try:
            await websocket.close()
        except Exception:
            pass

        logger.info("🔌 WebSocket disconnected")
