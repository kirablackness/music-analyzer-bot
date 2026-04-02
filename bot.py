import os
import gc
import shutil
import tempfile
import logging
import time
import re
import asyncio
from typing import Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

import librosa
import numpy as np
import pyloudnorm as pyln
import yt_dlp


BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
SAMPLE_RATE = 11025
SAMPLE_DURATION = 15.0

COOLDOWN_SECONDS = 30
MAX_FILE_SIZE_MB = 50
MAX_DURATION_MINUTES = 15

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

user_cooldown = {}
search_cache = {}


KEYBOARDS = {
    "main": [
        [InlineKeyboardButton("📥 Скачать", callback_data="download")],
        [InlineKeyboardButton("ℹ️ Инфо", callback_data="info")],
        [InlineKeyboardButton("❓ Помощь", callback_data="help")],
    ],
    "back": [[InlineKeyboardButton("◀️ Назад", callback_data="menu")]],
    "menu": [[InlineKeyboardButton("🏠 Меню", callback_data="menu")]],
}

MESSAGES = {
    "welcome": "🎵 *Music Analyzer Bot*\n\nВыбери действие:",
    "download_help": "📥 *Отправь ссылку или название песни*\n\n• Ссылка с YouTube/TikTok/Instagram/SoundCloud\n• Или просто напиши название трека - я найду его",
    "help": (
        "ℹ️ *Что я умею:*\n\n"
        "• Скачиваю с YouTube, TikTok, Instagram, SoundCloud\n"
        "• Ищу музыку по названию\n\n"
        "Просто отправь ссылку или название песни!"
    ),
    "info": (
        "🎬 *Media Download Bot*\n\n"
        "📦 *Поддерживает:*\n"
        "🎬 YouTube (видео, shorts)\n"
        "📱 TikTok (все видео)\n"
        "📸 Instagram (reels, посты)\n"
        "🎵 SoundCloud (треки)\n\n"
        "Просто отправь ссылку или название песни!\n"
        "При поиске покажу список - выбери нужный трек.\n\n"
        "⚠️ *Ограничения:*\n"
        "• Максимум 15 минут\n"
        "• Размер до 50МБ\n"
        "• 30 сек между запросами\n\n"
        "📋 *Команды:*\n"
        "/start - начало\n"
        "/info - информация о боте\n"
        "/help - помощь\n"
        "/status - статус бота"
    ),
}

YDL_OPTS_ANALYZE = {
    "format": "worstaudio/worst",
    "quiet": True,
    "no_warnings": True,
    "nocheckcertificate": True,
}

YDL_OPTS_DOWNLOAD_AUDIO = {
    "format": "bestaudio/best",
    "quiet": True,
    "no_warnings": True,
    "nocheckcertificate": True,
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "mp3",
        "preferredquality": "192",
    }],
}

YDL_OPTS_DOWNLOAD_VIDEO = {
    "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
    "quiet": True,
    "no_warnings": True,
    "nocheckcertificate": True,
    "merge_output_format": "mp4",
}

ALLOWED_DOMAINS = {
    "youtube": ["youtube.com", "youtu.be"],
    "tiktok": ["tiktok.com"],
    "instagram": ["instagram.com"],
    "soundcloud": ["soundcloud.com"],
    "vimeo": ["vimeo.com"],
}


def analyze_track(file_path: str) -> Optional[dict]:
    try:
        logger.info(f"Analyzing: {file_path}")
        
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=SAMPLE_DURATION)
        
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(tempo) if np.isscalar(tempo) else int(tempo[0])
        
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        lufs = round(loudness, 1) if loudness > -70 else "Too quiet"
        
        duration_sec = int(len(y) / sr)
        duration = f"{duration_sec // 60}:{duration_sec % 60:02d} (15s sample)"
        
        del y
        gc.collect()
        
        return {"bpm": bpm, "lufs": lufs, "duration": duration}
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return None


def detect_platform(url: str) -> Optional[str]:
    for platform, domains in ALLOWED_DOMAINS.items():
        if any(domain in url for domain in domains):
            return platform
    return None


def parse_duration(duration_str: str) -> int:
    if not duration_str or duration_str == "?:??":
        return 0
    parts = duration_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return int(parts[0]) if parts[0].isdigit() else 0


def check_cooldown(user_id: int) -> Optional[int]:
    now = int(time.time())
    last = user_cooldown.get(user_id, 0)
    diff = now - last
    if diff < COOLDOWN_SECONDS:
        return COOLDOWN_SECONDS - diff
    user_cooldown[user_id] = now
    return None


def download_audio(url: str, for_analysis: bool = True, format_type: str = "audio") -> tuple:
    temp_dir = tempfile.mkdtemp()
    
    if for_analysis:
        opts = YDL_OPTS_ANALYZE.copy()
    else:
        opts = (YDL_OPTS_DOWNLOAD_AUDIO if format_type == "audio" else YDL_OPTS_DOWNLOAD_VIDEO).copy()
    
    opts["outtmpl"] = os.path.join(temp_dir, "%(title)s.%(ext)s")
    
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "Unknown")
            filename = ydl.prepare_filename(info)
        
        if not os.path.exists(filename):
            files = os.listdir(temp_dir)
            if files:
                filename = os.path.join(temp_dir, files[0])
        
        return filename, title, temp_dir
    
    except Exception as e:
        logger.error(f"Download error: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, None


def search_youtube(query: str, count: int = 5) -> list:
    temp_dir = tempfile.mkdtemp()
    opts = {
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "extract_flat": True,
    }
    
    results = []
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch{count}:{query}", download=False)
            
            for entry in info.get("entries", []):
                duration_str = entry.get("duration_string", "")
                results.append({
                    "id": entry.get("id", ""),
                    "title": entry.get("title", "Unknown"),
                    "duration": duration_str,
                    "duration_sec": parse_duration(duration_str),
                })
        
        return results[:count]
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def is_valid_url(url: str) -> bool:
    return detect_platform(url) is not None


def cleanup_file(file_path: str, temp_dir: str = None):
    if file_path and os.path.exists(file_path):
        os.unlink(file_path)
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("=== START command received ===")
    logger.info("Start command received")
    context.user_data["mode"] = "download"
    await update.message.reply_text(
        MESSAGES["welcome"],
        reply_markup=InlineKeyboardMarkup(KEYBOARDS["main"]),
        parse_mode="Markdown"
    )


async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        MESSAGES["info"],
        parse_mode="Markdown"
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import subprocess
    try:
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
        version = result.stdout.strip()
        await update.message.reply_text(
            f"✅ Бот работает\n"
            f"🔧 yt-dlp: {version}\n"
            f"⚙️ Лимиты: {MAX_DURATION_MINUTES} мин, {MAX_FILE_SIZE_MB}МБ",
            parse_mode="Markdown"
        )
    except:
        await update.message.reply_text("❌ yt-dlp не установлен")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    handlers = {
        "download": lambda: _show_download_menu(query, context),
        "info": lambda: _show_info_menu(query),
        "help": lambda: _show_help_menu(query),
        "menu": lambda: _show_main_menu(query, context),
    }
    
    if query.data.startswith("dl_"):
        await _handle_search_download(query, context)
        return
    
    if query.data.startswith("toolong_"):
        await query.answer(f"Видео длиннее {MAX_DURATION_MINUTES} минут. Выберите другое.", show_alert=True)
        return
    
    if query.data.startswith("cancel_"):
        cache_key = query.data.replace("cancel_", "")
        if cache_key in search_cache:
            del search_cache[cache_key]
        await query.edit_message_text("Поиск отменён.")
        return
    
    handler = handlers.get(query.data)
    if handler:
        try:
            await handler()
        except Exception as e:
            logger.error(f"Callback error: {e}")


async def _show_download_menu(query, context):
    context.user_data["mode"] = "download"
    await query.edit_message_text(
        MESSAGES["download_help"],
        reply_markup=InlineKeyboardMarkup(KEYBOARDS["back"]),
        parse_mode="Markdown"
    )


async def _show_info_menu(query):
    await query.edit_message_text(
        MESSAGES["info"],
        reply_markup=InlineKeyboardMarkup(KEYBOARDS["back"]),
        parse_mode="Markdown"
    )


async def _show_help_menu(query):
    await query.edit_message_text(
        MESSAGES["help"],
        reply_markup=InlineKeyboardMarkup(KEYBOARDS["back"]),
        parse_mode="Markdown"
    )


async def _show_main_menu(query, context):
    context.user_data["mode"] = None
    try:
        await query.edit_message_text(
            MESSAGES["welcome"],
            reply_markup=InlineKeyboardMarkup(KEYBOARDS["main"]),
            parse_mode="Markdown"
        )
    except:
        await query.message.reply_text(
            MESSAGES["welcome"],
            reply_markup=InlineKeyboardMarkup(KEYBOARDS["main"]),
            parse_mode="Markdown"
        )


async def _handle_search_download(query, context):
    data = query.data.split("_")
    if len(data) < 4:
        await query.answer("Ошибка данных", show_alert=True)
        return
    
    cache_key = f"{data[1]}_{data[2]}"
    index = int(data[2])
    format_type = data[3]
    
    if cache_key not in search_cache:
        await query.answer("Результаты устарели. Попробуйте поиск заново.", show_alert=True)
        return
    
    results = search_cache[cache_key]
    if index >= len(results):
        await query.answer("Ошибка выбора", show_alert=True)
        return
    
    selected = results[index]
    format_text = "MP3" if format_type == "audio" else "видео"
    
    await query.edit_message_text(f"Скачиваю {format_text}: {selected['title']}")
    
    url = f"https://www.youtube.com/watch?v={selected['id']}"
    await _download_and_send(query.message, context, url, format_type, selected['title'])


async def download_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Использование: /download <ссылка>")
        return
    
    url = context.args[0]
    platform = detect_platform(url)
    if not platform:
        await update.message.reply_text("Поддерживаются: YouTube, TikTok, Instagram, SoundCloud, Яндекс.Музыка")
        return
    
    await _download_and_send(update.message, context, url, "audio")


async def _download_and_send(message, context, url: str, format_type: str, title: str = None):
    logger.info(f"Downloading: {url}, format: {format_type}")
    filename, downloaded_title, temp_dir = download_audio(url, for_analysis=False, format_type=format_type)
    
    logger.info(f"Download result: filename={filename}, title={downloaded_title}")
    
    if filename and os.path.exists(filename):
        final_title = title or downloaded_title
        file_size_mb = os.path.getsize(filename) / 1024 / 1024
        logger.info(f"File exists: {filename}, size: {file_size_mb:.1f}MB")
        
        await message.reply_text(f"Отправляю: {final_title}")
        
        if file_size_mb > MAX_FILE_SIZE_MB:
            await message.reply_text(f"Файл слишком большой ({file_size_mb:.1f}МБ). Максимум: {MAX_FILE_SIZE_MB}МБ")
            cleanup_file(filename, temp_dir)
            return
        
        is_audio = format_type == "audio" or filename.endswith(".mp3")
        caption = f"{'🎵' if is_audio else '🎬'} {final_title}"
        
        try:
            with open(filename, "rb") as f:
                if is_audio:
                    logger.info(f"Sending audio: {final_title}")
                    await message.reply_audio(
                        audio=f,
                        caption=caption,
                        title=final_title,
                    )
                else:
                    logger.info(f"Sending video: {final_title}")
                    await message.reply_video(
                        video=f,
                        caption=caption,
                    )
            logger.info("File sent successfully")
        except Exception as e:
            logger.error(f"Error sending file: {e}")
            await message.reply_text(f"Ошибка отправки: {e}")
        
        await message.reply_text(
            MESSAGES["welcome"],
            reply_markup=InlineKeyboardMarkup(KEYBOARDS["main"]),
            parse_mode="Markdown"
        )
    else:
        logger.error(f"File not found: {filename}")
        await message.reply_text("Не удалось скачать файл")
    
    cleanup_file(filename, temp_dir)


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = await update.message.reply_text("⏳ Анализирую...")
    
    audio = update.message.audio or update.message.document
    if not audio:
        await status_msg.edit_text("❌ Не могу найти аудиофайл")
        return
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        file = await context.bot.get_file(audio.file_id)
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name
    
    result = analyze_track(tmp_path)
    cleanup_file(tmp_path)
    
    if result:
        await status_msg.edit_text(
            f"✅ Результат:\n\n"
            f"🔊 BPM: {result['bpm']}\n"
            f"📢 LUFS: {result['lufs']}\n"
            f"⏱ Duration: {result['duration']}",
            reply_markup=InlineKeyboardMarkup(KEYBOARDS["menu"])
        )
    else:
        await status_msg.edit_text("❌ Ошибка анализа")


async def handle_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("=== Message received ===", update.message.text)
    logger.info(f"Message received: {update.message.text}")
    
    user_id = update.message.from_user.id
    
    cooldown = check_cooldown(user_id)
    if cooldown:
        await update.message.reply_text(f"⏳ Подождите {cooldown} сек")
        return
    
    text = update.message.text.strip()
    
    url_match = re.search(r'(https?://[^\s]+)', text)
    
    if url_match:
        url = url_match.group(1)
        platform = detect_platform(url)
        
        if not platform:
            await update.message.reply_text("❌ Платформа не поддерживается")
            return
        
        format_type = "video" if platform in ["tiktok", "instagram"] else "audio"
        await _download_and_send(update.message, context, url, format_type)
    else:
        await handle_search(update, context, text)


async def handle_search(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str):
    user_id = update.message.from_user.id
    
    cooldown = check_cooldown(user_id)
    if cooldown:
        await update.message.reply_text(f"⏳ Подождите {cooldown} сек")
        return
    
    status_msg = await update.message.reply_text("🔍 Ищу на YouTube...")
    
    results = search_youtube(query)
    
    if not results:
        await status_msg.edit_text("❌ Ничего не найдено. Попробуйте другой запрос.")
        return
    
    cache_key = f"{user_id}_{int(time.time())}"
    search_cache[cache_key] = results
    
    import asyncio
    asyncio.get_event_loop().call_later(300, lambda: search_cache.pop(cache_key, None))
    
    keyboard = []
    for i, item in enumerate(results):
        duration_text = f" [{item['duration']}]" if item['duration'] else ""
        short_title = item['title'][:35] + "..." if len(item['title']) > 35 else item['title']
        
        if item['duration_sec'] > MAX_DURATION_MINUTES * 60:
            keyboard.append([
                InlineKeyboardButton(f"❌ {short_title}{duration_text}", callback_data=f"toolong_{i}")
            ])
        else:
            keyboard.append([
                InlineKeyboardButton(f"🎵 {short_title}{duration_text}", callback_data=f"dl_{cache_key}_{i}_audio"),
                InlineKeyboardButton("🎬 Видео", callback_data=f"dl_{cache_key}_{i}_video")
            ])
    
    keyboard.append([InlineKeyboardButton("❌ Отмена", callback_data=f"cancel_{cache_key}")])
    
    await status_msg.edit_text(
        f'🎵 Результаты поиска "{query}":\n\n🎵 - MP3 | 🎬 - Видео',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("info", info_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("download", download_command))
    app.add_handler(CallbackQueryHandler(button_callback))
    # app.add_handler(MessageHandler(filters.AUDIO | filters.Document.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_url))
    
    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
