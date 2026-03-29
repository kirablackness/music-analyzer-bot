import os
import gc
import shutil
import tempfile
import logging
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


KEYBOARDS = {
    "main": [
        # [InlineKeyboardButton("🎵 Анализ трека", callback_data="analyze")],
        [InlineKeyboardButton("📥 Скачать с YouTube/SoundCloud", callback_data="download")],
        [InlineKeyboardButton("ℹ️ Помощь", callback_data="help")],
    ],
    "back": [[InlineKeyboardButton("◀️ Назад", callback_data="menu")]],
    "menu": [[InlineKeyboardButton("🏠 Меню", callback_data="menu")]],
}

MESSAGES = {
    "welcome": "🎵 *Music Analyzer Bot*\n\nВыбери действие:",
    # "analyze_help": (
    #     "📤 *Отправь мне:*\n\n"
    #     "• Аудиофайл (MP3, WAV, FLAC)\n"
    #     "• Или ссылку на YouTube/SoundCloud\n\n"
    #     "Я определю BPM, громкость и длительность."
    # ),
    "download_help": "📥 *Отправь ссылку на YouTube или SoundCloud*\n\nЯ скачаю и отправлю файл.",
    "help": (
        "ℹ️ *Что я умею:*\n\n"
        "• Скачиваю с YouTube/SoundCloud\n\n"
        "Просто отправь ссылку!"
    ),
}

YDL_OPTS_ANALYZE = {
    "format": "worstaudio/worst",
    "quiet": True,
    "no_warnings": True,
    "nocheckcertificate": True,
}

YDL_OPTS_DOWNLOAD = {
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

ALLOWED_DOMAINS = ["youtube", "youtu.be", "soundcloud", "vimeo"]


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


def download_audio(url: str, for_analysis: bool = True) -> tuple:
    temp_dir = tempfile.mkdtemp()
    opts = YDL_OPTS_ANALYZE if for_analysis else YDL_OPTS_DOWNLOAD.copy()
    opts["outtmpl"] = os.path.join(temp_dir, "%(id)s.%(ext)s" if for_analysis else "%(title)s.%(ext)s")
    
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


def is_valid_url(url: str) -> bool:
    return any(domain in url for domain in ALLOWED_DOMAINS)


def cleanup_file(file_path: str, temp_dir: str = None):
    if file_path and os.path.exists(file_path):
        os.unlink(file_path)
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["mode"] = "download"
    await update.message.reply_text(
        MESSAGES["welcome"],
        reply_markup=InlineKeyboardMarkup(KEYBOARDS["main"]),
        parse_mode="Markdown"
    )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    handlers = {
        "analyze": lambda: _show_analyze_menu(query, context),
        "download": lambda: _show_download_menu(query, context),
        "help": lambda: _show_help_menu(query),
        "menu": lambda: _show_main_menu(query, context),
    }
    
    handler = handlers.get(query.data)
    if handler:
        try:
            await handler()
        except Exception as e:
            logger.error(f"Callback error: {e}")


async def _show_analyze_menu(query, context):
    context.user_data["mode"] = "analyze"
    await query.edit_message_text(
        MESSAGES["analyze_help"],
        reply_markup=InlineKeyboardMarkup(KEYBOARDS["back"]),
        parse_mode="Markdown"
    )


async def _show_download_menu(query, context):
    context.user_data["mode"] = "download"
    await query.edit_message_text(
        MESSAGES["download_help"],
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


async def download_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Использование: /download <ссылка>")
        return
    
    url = context.args[0]
    if not is_valid_url(url):
        await update.message.reply_text("Поддерживаются только YouTube, SoundCloud, Vimeo")
        return
    
    await update.message.reply_text("Скачиваю...")
    
    filename, title, temp_dir = download_audio(url, for_analysis=False)
    
    if filename and os.path.exists(filename):
        await update.message.reply_text(f"Отправляю: {title}")
        
        with open(filename, "rb") as f:
            await update.message.reply_document(
                document=f,
                filename=f"{title}.mp3",
                caption=f"🎵 {title}"
            )
        
        await update.message.reply_text(
            MESSAGES["welcome"],
            reply_markup=InlineKeyboardMarkup(KEYBOARDS["main"]),
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text("Не удалось скачать файл")
    
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
    url = update.message.text
    if not is_valid_url(url):
        return
    
    await _handle_download_url(update, url)


async def _handle_analyze_url(update: Update, url: str):
    await update.message.reply_text("Скачиваю и анализирую...")
    
    filename, title, temp_dir = download_audio(url)
    
    if not filename:
        await update.message.reply_text("Не удалось скачать")
        return
    
    result = analyze_track(filename)
    cleanup_file(filename, temp_dir)
    
    if result:
        await update.message.reply_text(
            f"🎵 {title}\n\n"
            f"BPM: {result['bpm']}\n"
            f"LUFS: {result['lufs']}\n"
            f"Duration: {result['duration']}",
            reply_markup=InlineKeyboardMarkup(KEYBOARDS["menu"])
        )
    else:
        await update.message.reply_text("Ошибка анализа")


async def _handle_download_url(update: Update, url: str):
    await update.message.reply_text("Скачиваю...")
    
    filename, title, temp_dir = download_audio(url, for_analysis=False)
    
    if filename and os.path.exists(filename):
        await update.message.reply_text(f"Отправляю: {title}")
        
        with open(filename, "rb") as f:
            await update.message.reply_document(
                document=f,
                filename=f"{title}.mp3",
                caption=f"🎵 {title}"
            )
        
        await update.message.reply_text(
            MESSAGES["welcome"],
            reply_markup=InlineKeyboardMarkup(KEYBOARDS["main"]),
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text("Не удалось скачать файл")
    
    cleanup_file(filename, temp_dir)


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("download", download_command))
    app.add_handler(CallbackQueryHandler(button_callback))
    # app.add_handler(MessageHandler(filters.AUDIO | filters.Document.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_url))
    
    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
