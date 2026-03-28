import os
import tempfile
import logging
import subprocess
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import librosa
import numpy as np
import pyloudnorm as pyln
import yt_dlp

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_wav(input_path):
    try:
        logger.info(f"Converting {input_path} to WAV...")
        output_path = input_path.rsplit('.', 1)[0] + '.wav'
        result = subprocess.run([
            'ffmpeg', '-i', input_path, 
            '-ar', '22050',
            '-ac', '1',
            '-y',
            output_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return input_path
        
        logger.info(f"Converted to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        return input_path

def analyze_track(file_path):
    try:
        if not file_path.endswith('.wav'):
            file_path = convert_to_wav(file_path)
        
        y, sr = librosa.load(file_path, sr=None)
        
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(tempo) if np.isscalar(tempo) else int(tempo[0])
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_idx = np.argmax(np.mean(chroma, axis=1))
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = keys[key_idx]
        
        minor_chroma = librosa.feature.chroma_stft(y=librosa.effects.harmonic(y), sr=sr)
        major_minor = np.sum(chroma) > np.sum(minor_chroma) * 0.5
        mode = "Major" if major_minor else "Minor"
        
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        lufs = round(loudness, 1) if loudness > -70 else "Too quiet"
        
        duration_sec = int(len(y) / sr)
        minutes = duration_sec // 60
        seconds = duration_sec % 60
        duration = f"{minutes}:{seconds:02d}"
        
        return {
            'bpm': bpm,
            'key': f"{key} {mode}",
            'lufs': lufs,
            'duration': duration
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return None

def download_from_url(url):
    try:
        temp_dir = tempfile.mkdtemp()
        ydl_opts = {
            'format': 'worstaudio/worst',
            'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'nocheckcertificate': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'Unknown')
            filename = ydl.prepare_filename(info)
        
        return filename, title
    except Exception as e:
        logger.error(f"Download error: {e}")
        return None, None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🎵 Анализ трека", callback_data="analyze")],
        [InlineKeyboardButton("📥 Скачать с YouTube/SoundCloud", callback_data="download")],
        [InlineKeyboardButton("ℹ️ Помощь", callback_data="help")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🎵 *Music Analyzer Bot*\n\n"
        "Выбери действие:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    try:
        if query.data == "analyze":
            context.user_data['mode'] = 'analyze'
            keyboard = [[InlineKeyboardButton("◀️ Назад", callback_data="menu")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "📤 *Отправь мне:*\n\n"
                "• Аудиофайл (MP3, WAV, FLAC)\n"
                "• Или ссылку на YouTube/SoundCloud\n\n"
                "Я определю BPM, тональность, громкость и длительность.",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        elif query.data == "download":
            context.user_data['mode'] = 'download'
            keyboard = [[InlineKeyboardButton("◀️ Назад", callback_data="menu")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "📥 *Отправь ссылку на YouTube или SoundCloud*\n\n"
                "Я скачаю и отправлю файл.",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        elif query.data == "help":
            keyboard = [[InlineKeyboardButton("◀️ Назад", callback_data="menu")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "ℹ️ *Что я умею:*\n\n"
                "• Определяю BPM\n"
                "• Определяю тональность (Major/Minor)\n"
                "• Измеряю громкость (LUFS)\n"
                "• Показываю длительность\n"
                "• Скачиваю с YouTube/SoundCloud\n\n"
                "Просто отправь файл или ссылку!",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        elif query.data == "menu":
            context.user_data['mode'] = None
            keyboard = [
                [InlineKeyboardButton("🎵 Анализ трека", callback_data="analyze")],
                [InlineKeyboardButton("📥 Скачать с YouTube/SoundCloud", callback_data="download")],
                [InlineKeyboardButton("ℹ️ Помощь", callback_data="help")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            try:
                await query.edit_message_text(
                    "🎵 *Music Analyzer Bot*\n\n"
                    "Выбери действие:",
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
            except:
                await query.message.reply_text(
                    "🎵 *Music Analyzer Bot*\n\n"
                    "Выбери действие:",
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
    except Exception as e:
        logger.error(f"Button callback error: {e}")

async def download_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Использование: /download <ссылка>")
        return
    
    url = context.args[0]
    
    if not any(x in url for x in ['youtube', 'youtu.be', 'soundcloud', 'vimeo']):
        await update.message.reply_text("Поддерживаются только YouTube, SoundCloud, Vimeo")
        return
    
    await update.message.reply_text("Скачиваю...")
    
    try:
        temp_dir = tempfile.mkdtemp()
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'nocheckcertificate': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'Unknown')
            filename = ydl.prepare_filename(info)
            ext = info.get('ext', 'mp3')
        
        if not os.path.exists(filename):
            files = os.listdir(temp_dir)
            if files:
                filename = os.path.join(temp_dir, files[0])
        
        final_filename = f"{title}.mp3"
        if os.path.exists(filename):
            await update.message.reply_text(f"Отправляю: {title}")
            
            with open(filename, 'rb') as audio_file:
                await update.message.reply_document(
                    document=audio_file,
                    filename=final_filename,
                    caption=f"🎵 {title}"
                )
            
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            await update.message.reply_text("Не удалось скачать файл")
            
    except Exception as e:
        logger.error(f"Download error: {e}")
        await update.message.reply_text("Произошла ошибка при скачивании")

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Анализирую...")
    
    try:
        audio = update.message.audio or update.message.document
        if not audio:
            await update.message.reply_text("Не могу найти аудиофайл")
            return
        
        file = await context.bot.get_file(audio.file_id)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            await file.download_to_drive(tmp.name)
            tmp_path = tmp.name
        
        result = analyze_track(tmp_path)
        
        os.unlink(tmp_path)
        
        if result:
            keyboard = [[InlineKeyboardButton("🏠 Меню", callback_data="menu")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            response = (
                f"🎵 Результат:\n\n"
                f"BPM: {result['bpm']}\n"
                f"Key: {result['key']}\n"
                f"LUFS: {result['lufs']}\n"
                f"Duration: {result['duration']}"
            )
        else:
            response = "Ошибка анализа"
            reply_markup = None
        
        await update.message.reply_text(response, reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("Произошла ошибка")

async def handle_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    url = update.message.text
    
    if not any(x in url for x in ['youtube', 'youtu.be', 'soundcloud', 'vimeo']):
        return
    
    mode = context.user_data.get('mode', 'analyze')
    
    if mode == 'download':
        await handle_download(update, context, url)
    else:
        await handle_analyze_url(update, context, url)

async def handle_analyze_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str):
    await update.message.reply_text("Скачиваю и анализирую...")
    
    try:
        file_path, title = download_from_url(url)
        
        if not file_path:
            await update.message.reply_text("Не удалось скачать")
            return
        
        result = analyze_track(file_path)
        
        if os.path.exists(file_path):
            os.unlink(file_path)
            parent_dir = os.path.dirname(file_path)
            if os.path.exists(parent_dir):
                import shutil
                shutil.rmtree(parent_dir, ignore_errors=True)
        
        if result:
            keyboard = [[InlineKeyboardButton("🏠 Меню", callback_data="menu")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            response = (
                f"🎵 {title}\n\n"
                f"BPM: {result['bpm']}\n"
                f"Key: {result['key']}\n"
                f"LUFS: {result['lufs']}\n"
                f"Duration: {result['duration']}"
            )
        else:
            response = "Ошибка анализа"
            reply_markup = None
        
        await update.message.reply_text(response, reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("Произошла ошибка")

async def handle_download(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str):
    await update.message.reply_text("Скачиваю...")
    
    try:
        temp_dir = tempfile.mkdtemp()
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'nocheckcertificate': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'Unknown')
            filename = ydl.prepare_filename(info)
        
        if not os.path.exists(filename):
            files = os.listdir(temp_dir)
            if files:
                filename = os.path.join(temp_dir, files[0])
        
        final_filename = f"{title}.mp3"
        if os.path.exists(filename):
            await update.message.reply_text(f"Отправляю: {title}")
            
            with open(filename, 'rb') as audio_file:
                await update.message.reply_document(
                    document=audio_file,
                    filename=final_filename,
                    caption=f"🎵 {title}"
                )
            
            keyboard = [[InlineKeyboardButton("🏠 Меню", callback_data="menu")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text("Готово!", reply_markup=reply_markup)
            
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            await update.message.reply_text("Не удалось скачать файл")
            
    except Exception as e:
        logger.error(f"Download error: {e}")
        await update.message.reply_text("Произошла ошибка при скачивании")

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("download", download_command))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.AUDIO | filters.Document.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_url))
    
    logger.info("Bot started")
    app.run_polling()

if __name__ == "__main__":
    main()
