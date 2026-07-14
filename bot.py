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
            InlineKeyboardButton("🎵 Скачать Аудио", callback_data="mode_audio"),
            InlineKeyboardButton("🎬 Скачать Видео", callback_data="mode_video")
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
        "Просто отправь ссылку или название песни!\n\n"
        "💬 *Есть проблемы?*\n"
        "Пиши: @kirablackness"
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
        "/status - статус бота\n\n"
        "💬 *Проблемы?*\n"
        "@kirablackness"
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


def clean_artist_name(name: str) -> str:
    """Strip '- Topic' suffix and clean up channel/artist name."""
    if not name or name == "NA":
        return ""
    return re.sub(r'\s*- Topic\s*$', '', name).strip()


def build_display_title(artist: str, title: str) -> str:
    """Build 'Artist - Title' avoiding duplication when title already starts with artist."""
    if not title or title == "NA":
        title = "Без названия"
    if artist and not title.lower().startswith(artist.lower()):
        return f"{artist} - {title}"
    return title


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
    
    AUDIO_EXTS = {".mp3", ".m4a", ".opus", ".webm", ".ogg", ".wav"}
    VIDEO_EXTS = {".mp4", ".webm", ".mkv"}
    THUMB_EXTS = {".jpg", ".jpeg", ".webp", ".png"}
    
    try:
        if format_type == "audio":
            cmd = (
                f'yt-dlp --no-check-certificates --no-playlist -x --audio-format mp3 '
                f'--audio-quality 0 --embed-metadata --embed-thumbnail --write-thumbnail '
                f'--convert-thumbnails jpg '
                f'--parse-metadata "artist:%(artist|channel)s" '
                f'-o "{template}" "{url}"'
            )
        else:
            cmd = (
                f'yt-dlp --no-check-certificates --no-playlist '
                f'-f "bestvideo[height<=720]+bestaudio/best[height<=720]/best" '
                f'--merge-output-format mp4 --embed-metadata --write-thumbnail '
                f'--convert-thumbnails jpg '
                f'-o "{template}" "{url}"'
            )
        
        logger.info(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"yt-dlp error: {result.stderr}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None, None, None
        
        # Find downloaded media file (skip thumbnail images)
        target_exts = AUDIO_EXTS if format_type == "audio" else VIDEO_EXTS
        filename = None
        thumb_path = None
        for file in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, file)
            if not os.path.isfile(filepath):
                continue
            ext = os.path.splitext(file)[1].lower()
            if ext in target_exts and not filename:
                filename = filepath
            elif ext in THUMB_EXTS and not thumb_path:
                thumb_path = filepath
        
        if not filename:
            logger.error("No media file found after download")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None, None, None
        
        # Get full metadata with fallbacks: artist → channel → uploader
        meta_cmd = (
            f'yt-dlp --no-check-certificates '
            f'--print "%(artist)s|||%(title)s|||%(channel)s|||%(uploader)s" '
            f'--no-warnings "{url}"'
        )
        meta_result = subprocess.run(meta_cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        parts = meta_result.stdout.strip().split("|||")
        artist_raw = parts[0] if len(parts) > 0 else ""
        title_raw = parts[1] if len(parts) > 1 else ""
        channel = parts[2] if len(parts) > 2 else ""
        uploader = parts[3] if len(parts) > 3 else ""
        
        # Determine artist with fallbacks
        artist = clean_artist_name(artist_raw)
        if not artist:
            artist = clean_artist_name(channel) or clean_artist_name(uploader)
        
        title = title_raw if title_raw and title_raw != "NA" else "Unknown"
        full_title = build_display_title(artist, title)
        
        # Rename media file to proper title (not just timestamp numbers)
        ext = os.path.splitext(filename)[1]
        safe_name = re.sub(r'[<>:"/\\|?*\n\r\t]', '', full_title)[:100].strip()
        if safe_name:
            new_path = os.path.join(temp_dir, f"{safe_name}{ext}")
            try:
                os.rename(filename, new_path)
                filename = new_path
            except Exception as e:
                logger.warning(f"Could not rename file: {e}")
        
        logger.info(f"Downloaded: {full_title}, file: {filename}, thumb: {thumb_path}")
        return filename, full_title, temp_dir, thumb_path
    
    except Exception as e:
        logger.error(f"Download error: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, None, None


def prepare_thumbnail(thumb_path: str, temp_dir: str) -> Optional[str]:
    """Convert/resize thumbnail to JPEG 320x320 max (Telegram requirement) via ffmpeg."""
    if not thumb_path or not os.path.exists(thumb_path):
        return None
    import subprocess
    out_path = os.path.join(temp_dir, "cover.jpg")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", thumb_path, "-vf", "scale='min(320,iw)':'min(320,ih)':force_original_aspect_ratio=decrease",
             "-q:v", "5", out_path],
            capture_output=True, timeout=15,
        )
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
    except Exception as e:
        logger.warning(f"Thumbnail convert error: {e}")
    return None


def format_duration(seconds: int) -> str:
    """Format seconds to m:ss or h:mm:ss."""
    if not seconds or seconds <= 0:
        return ""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def search_youtube(query: str, count: int = 5) -> list:
    import subprocess
    
    try:
        encoded_query = query.replace('"', '\\"')
        cmd = (
            f'yt-dlp "https://music.youtube.com/search?q={encoded_query}" '
            f'--flat-playlist '
            f'--print "%(id)s|||%(title)s|||%(duration_string)s|||%(duration)s|||%(channel)s|||%(uploader)s" '
            f'--no-warnings'
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        results = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = line.split('|||')
            if len(parts) < 3:
                continue
            
            id_ = parts[0]
            title = parts[1] if parts[1] and parts[1] != "NA" else "Без названия"
            duration_str = parts[2] if len(parts) > 2 and parts[2] != "NA" else ""
            duration_sec_raw = parts[3] if len(parts) > 3 and parts[3] != "NA" else ""
            channel = parts[4] if len(parts) > 4 else ""
            uploader = parts[5] if len(parts) > 5 else ""
            
            # Parse duration: prefer duration_string, fall back to duration (seconds)
            duration_sec = parse_duration(duration_str)
            if not duration_sec and duration_sec_raw:
                try:
                    duration_sec = int(float(duration_sec_raw))
                except (ValueError, TypeError):
                    duration_sec = 0
            
            # Format duration string from seconds if missing
            if duration_sec and not duration_str:
                duration_str = format_duration(duration_sec)
            
            # Artist: use channel or uploader as fallback, strip "- Topic"
            artist = clean_artist_name(channel) or clean_artist_name(uploader)
            display_title = build_display_title(artist, title)
            
            if id_ and len(id_) == 11:
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
    logger.info(f"Downloading: {url}, format: {format_type}, title from search: {title}")
    
    # Send initial status message with title
    format_text = "MP3" if format_type == "audio" else "видео"
    if title:
        status_msg = await message.reply_text(f"⏳ Скачиваю {format_text}: {title}")
    else:
        status_msg = await message.reply_text(f"⏳ Скачиваю {format_text}...")
    
    filename, downloaded_title, temp_dir, thumb_path = download_audio(url, for_analysis=False, format_type=format_type)
    
    logger.info(f"Download result: filename={filename}, title={downloaded_title}, thumb={thumb_path}")
    
    if filename and os.path.exists(filename):
        final_title = downloaded_title
        logger.info(f"Using downloaded_title as final_title: {final_title}")
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
        
        # Prepare cover for Telegram (resize to 320x320 JPEG, max 200KB)
        cover_path = prepare_thumbnail(thumb_path, temp_dir)
        
        try:
            with open(filename, "rb") as f:
                thumb_file = None
                if cover_path and os.path.exists(cover_path):
                    thumb_file = open(cover_path, "rb")
                
                if is_audio:
                    logger.info(f"Sending audio: {final_title}, with cover: {bool(thumb_file)}")
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
                        thumbnail=thumb_file,
                    )
                else:
                    logger.info(f"Sending video: {final_title}, with cover: {bool(thumb_file)}")
                    await message.reply_video(
                        video=f,
                        caption=caption,
                        thumbnail=thumb_file,
                    )
            logger.info("File sent successfully")
        except Exception as e:
            logger.error(f"Error sending file: {e}")
            await status_msg.edit_text(f"❌ Ошибка отправки: {e}")
        finally:
            if thumb_file:
                thumb_file.close()
        
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
    
    # Debug: log results
    logger.info(f"Search returned {len(results)} results")
    for i, r in enumerate(results[:3]):  # Log first 3 results
        logger.info(f"Result {i}: title={r.get('title')}, duration={r.get('duration')}, duration_sec={r.get('duration_sec')}")
    
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
        # Show full title (Artist - Title) without cutting too much
        short_title = item['title'][:55] + "..." if len(item['title']) > 55 else item['title']
        
        if item['duration_sec'] > MAX_DURATION_MINUTES * 60:
            keyboard.append([
                InlineKeyboardButton(f"❌ {short_title}{duration_text} (длинное)", callback_data=f"toolong_{i}")
            ])
        else:
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
