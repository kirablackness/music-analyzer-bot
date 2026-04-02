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
        [
            InlineKeyboardButton("📥 Скачать аудио", callback_data="mode_audio"),
            InlineKeyboardButton("📥 Скачать видео", callback_data="mode_video")
        ],
        [InlineKeyboardButton("ℹ️ Инфо", callback_data="info")],
        [InlineKeyboardButton("❓ Помощь", callback_data="help")],
    ],
    "back": [[InlineKeyboardButton("◀️ Назад", callback_data="menu")]],
    "menu": [[InlineKeyboardButton("🏠 Меню", callback_data="menu")]],
}

MESSAGES = {
    "welcome": "🎵 *Music Analyzer Bot*\n\nВыбери действие:",
    "download_help": "📥 *Отправь ссылку или название песни*\n\n• Ссылка с YouTube/TikTok/Instagram/SoundCloud\n• Или просто напиши название трека - я найду его",
    "mode_audio": "🎵 *Режим: Скачать аудио*\n\nОтправь ссылку или название песни",
    "mode_video": "🎬 *Режим: Скачать видео*\n\nОтправь ссылку или название песни",
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
        "Просто отправь ссылку или название песни!\n\n"
        "⚠️ *Ограничения:*\n"
        "• Максимум 15 минут\n"
        "• Размер до 50МБ\n\n"
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
    import subprocess
    
    temp_dir = tempfile.mkdtemp()
    timestamp = int(time.time())
    base_path = os.path.join(temp_dir, f"download_{timestamp}")
    template = f"{base_path}.%(ext)s"
    
    try:
        # Add --no-check-certificates and extract-audio like bot.js
        if format_type == "audio":
            cmd = f'yt-dlp --no-check-certificates --no-playlist -x --audio-format mp3 --audio-quality 0 -o "{template}" "{url}"'
        else:
            cmd = f'yt-dlp --no-check-certificates --no-playlist -f "bestvideo[height<=720]+bestaudio/best[height<=720]/best" --merge-output-format mp4 -o "{template}" "{url}"'
        
        logger.info(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"yt-dlp error: {result.stderr}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None, None
        
        # Find downloaded file
        filename = None
        for file in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, file)
            if os.path.isfile(filepath):
                filename = filepath
                break
        
        if not filename:
            logger.error("No file found after download")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None, None
        
        # Get title and artist
        title_cmd = f'yt-dlp --no-check-certificates --print "%(artist)s|||%(title)s" --no-warnings "{url}"'
        title_result = subprocess.run(title_cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        parts = title_result.stdout.strip().split("|||")
        if len(parts) >= 2:
            artist = parts[0] if parts[0] and parts[0] != "NA" else ""
            title = parts[1] or "Unknown"
            full_title = f"{artist} - {title}" if artist else title
        else:
            full_title = parts[0] if parts[0] else "Unknown"
        
        return filename, full_title, temp_dir
    
    except Exception as e:
        logger.error(f"Download error: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, None


def search_youtube(query: str, count: int = 5) -> list:
    import subprocess
    
    try:
        encoded_query = query.replace('"', '\\"')
        cmd = f'yt-dlp "https://music.youtube.com/search?q={encoded_query}" --flat-playlist --print "%(id)s|||%(title)s|||%(duration_string)s|||%(artist)s" --no-warnings'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        results = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = line.split('|||')
            if len(parts) >= 4:
                id_, title, duration, artist = parts[0], parts[1], parts[2], parts[3]
                
                clean_artist = artist if artist and artist != "NA" else ""
                clean_title = title if title else "Без названия"
                display_title = f"{clean_artist} - {clean_title}" if clean_artist else clean_title
                
                duration_sec = parse_duration(duration) if duration and duration != "NA" else 0
                duration_str = duration if duration and duration != "NA" else ""
                
                if id_ and len(id_) == 11 and clean_title != "NA":
                    results.append({
                        "id": id_,
                        "title": display_title,
                        "duration": duration_str,
                        "duration_sec": duration_sec,
                    })
        
        return results[:count]
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []


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
        "mode_audio": lambda: _set_mode_audio(query, context),
        "mode_video": lambda: _set_mode_video(query, context),
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


async def _set_mode_audio(query, context):
    context.user_data["mode"] = "audio"
    await query.edit_message_text(
        MESSAGES["mode_audio"],
        reply_markup=InlineKeyboardMarkup(KEYBOARDS["back"]),
        parse_mode="Markdown"
    )


async def _set_mode_video(query, context):
    context.user_data["mode"] = "video"
    await query.edit_message_text(
        MESSAGES["mode_video"],
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
    print(f"=== _handle_search_download called: {query.data} ===")
    data = query.data.split("_")
    print(f"=== Split data: {data} ===")
    
    if len(data) < 5:
        await query.answer("Ошибка данных", show_alert=True)
        return
    
    cache_key = f"{data[1]}_{data[2]}"
    index = int(data[3])
    format_type = data[4]
    
    print(f"=== Cache key: {cache_key}, index: {index}, format: {format_type} ===")
    
    if cache_key not in search_cache:
        print(f"=== Cache key not found! Available: {list(search_cache.keys())} ===")
        await query.answer("Результаты устарели. Попробуйте поиск заново.", show_alert=True)
        return
    
    results = search_cache[cache_key]
    if index >= len(results):
        await query.answer("Ошибка выбора", show_alert=True)
        return
    
    selected = results[index]
    print(f"=== Selected: {selected} ===")
    
    await query.answer()  # Just close the loading state
    
    url = f"https://www.youtube.com/watch?v={selected['id']}"
    print(f"=== Calling _download_and_send with URL: {url} ===")
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
    
    # Send initial status message with title
    format_text = "MP3" if format_type == "audio" else "видео"
    if title:
        status_msg = await message.reply_text(f"⏳ Скачиваю {format_text}: {title}")
    else:
        status_msg = await message.reply_text(f"⏳ Скачиваю {format_text}...")
    
    filename, downloaded_title, temp_dir = download_audio(url, for_analysis=False, format_type=format_type)
    
    logger.info(f"Download result: filename={filename}, title={downloaded_title}")
    
    if filename and os.path.exists(filename):
        final_title = title or downloaded_title
        file_size_mb = os.path.getsize(filename) / 1024 / 1024
        logger.info(f"File exists: {filename}, size: {file_size_mb:.1f}MB")
        
        # Update status message
        await status_msg.edit_text(f"📤 Отправляю: {final_title} ({file_size_mb:.1f}МБ)")
        
        if file_size_mb > MAX_FILE_SIZE_MB:
            await status_msg.edit_text(f"❌ Файл слишком большой ({file_size_mb:.1f}МБ). Максимум: {MAX_FILE_SIZE_MB}МБ")
            cleanup_file(filename, temp_dir)
            return
        
        is_audio = format_type == "audio" or filename.endswith(".mp3")
        caption = f"{'🎵' if is_audio else '🎬'} {final_title}"
        
        try:
            with open(filename, "rb") as f:
                if is_audio:
                    logger.info(f"Sending audio: {final_title}")
                    # Split artist and title like in bot.js
                    performer = ""
                    audio_title = final_title
                    if " - " in final_title:
                        parts = final_title.split(" - ")
                        performer = parts[0].strip()
                        audio_title = " - ".join(parts[1:]).strip()
                    
                    await message.reply_audio(
                        audio=f,
                        caption=caption,
                        title=audio_title,
                        performer=performer,
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
            await status_msg.edit_text(f"❌ Ошибка отправки: {e}")
        
        # Delete status message after sending
        await status_msg.delete()
        
        await message.reply_text(
            MESSAGES["welcome"],
            reply_markup=InlineKeyboardMarkup(KEYBOARDS["main"]),
            parse_mode="Markdown"
        )
    else:
        logger.error(f"File not found: {filename}")
        await status_msg.edit_text("❌ Не удалось скачать файл")
    
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
    print(f"=== HANDLE_URL CALLED === message: {update.message}")
    if update.message:
        print(f"=== Message text: {update.message.text}")
    logger.info(f"Message received: {update.message.text if update.message else 'No message'}")
    
    if not update.message:
        print("=== No message in update ===")
        return
    
    user_id = update.message.from_user.id
    print(f"=== User ID: {user_id} ===")
    
    text = update.message.text.strip()
    print(f"=== Text: {text} ===")
    
    url_match = re.search(r'(https?://[^\s]+)', text)
    print(f"=== URL match: {url_match} ===")
    
    if url_match:
        url = url_match.group(1)
        print(f"=== URL found: {url} ===")
        platform = detect_platform(url)
        print(f"=== Platform: {platform} ===")
        
        if not platform:
            print("=== Platform not supported ===")
            await update.message.reply_text("❌ Платформа не поддерживается")
            return
        
        # Check if user selected mode via buttons
        user_mode = context.user_data.get("mode")
        print(f"=== User mode: {user_mode} ===")
        
        if user_mode == "video":
            format_type = "video"
        elif user_mode == "audio":
            format_type = "audio"
        else:
            # Default: YouTube/shorts -> video if shorts, audio otherwise
            # TikTok/Instagram -> video
            if platform == "youtube" and "shorts" in url:
                format_type = "video"
            elif platform in ["tiktok", "instagram"]:
                format_type = "video"
            else:
                format_type = "audio"
        
        print(f"=== Format: {format_type} ===")
        print(f"=== Calling _download_and_send ===")
        await _download_and_send(update.message, context, url, format_type)
    else:
        # Search - always audio by default or user mode
        user_mode = context.user_data.get("mode")
        print(f"=== No URL found, calling search. User mode: {user_mode} ===")
        await handle_search(update, context, text)


async def handle_search(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str):
    user_id = update.message.from_user.id
    
    status_msg = await update.message.reply_text("🔍 Ищу на YouTube...")
    
    results = search_youtube(query, count=5)
    
    if not results:
        await status_msg.edit_text("❌ Ничего не найдено. Попробуйте другой запрос.")
        return
    
    cache_key = f"{user_id}_{int(time.time())}"
    search_cache[cache_key] = results
    
    import asyncio
    asyncio.get_event_loop().call_later(300, lambda: search_cache.pop(cache_key, None))
    
    # Get user's preferred format or default to audio
    user_mode = context.user_data.get("mode", "audio")
    
    keyboard = []
    for i, item in enumerate(results):
        duration_text = f" [{item['duration']}]" if item['duration'] else ""
        short_title = item['title'][:40] + "..." if len(item['title']) > 40 else item['title']
        
        if item['duration_sec'] > MAX_DURATION_MINUTES * 60:
            keyboard.append([
                InlineKeyboardButton(f"❌ {short_title}{duration_text} (длинное)", callback_data=f"toolong_{i}")
            ])
        else:
            # Show only one button based on user mode
            if user_mode == "video":
                keyboard.append([
                    InlineKeyboardButton(f"🎬 {short_title}{duration_text}", callback_data=f"dl_{cache_key}_{i}_video")
                ])
            else:
                keyboard.append([
                    InlineKeyboardButton(f"🎵 {short_title}{duration_text}", callback_data=f"dl_{cache_key}_{i}_audio")
                ])
    
    keyboard.append([InlineKeyboardButton("❌ Отмена", callback_data=f"cancel_{cache_key}")])
    
    format_text = "MP3" if user_mode == "audio" else "Видео"
    await status_msg.edit_text(
        f'🎵 Результаты поиска "{query}" (формат: {format_text}):',
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
