"""
FB2 Reader with Russian Text-to-Speech

Desktop application for reading FB2 ebooks with integrated
Russian TTS (Silero v5). Features: FB2 parsing with formatting
preservation, adjustable reading speed, voice selection, text
search, reading position memory.

Tech stack: Python 3.11, PySide6, Silero TTS, lxml, PyTorch (CPU)
"""

# Standard library imports
import gc
import html
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import subprocess
import sys
import time

os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1"

import tempfile
import threading
from pathlib import Path

if sys.platform == 'win32':
    import winsound
    # Prepare Windows API handle for AppUserModelID call.
    try:
        from ctypes import windll
    except Exception:
        windll = None

# Third-party imports
import chardet
import numpy as np
import torch
from lxml import etree
from num2words import num2words
from scipy.io.wavfile import write as write_wav
from silero import silero_tts

# PySide6 imports
from PySide6.QtCore import (
    Qt, QThread, Signal, QMutex, QSize, QTimer, QByteArray
)
from PySide6.QtGui import (
    QPalette, QColor, QTextCursor, QFont, QTextDocument,
    QFontDatabase, QIcon, QPixmap, QPainter
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout,
    QHBoxLayout, QWidget, QLabel, QFileDialog, QMessageBox,
    QTextBrowser, QProgressBar, QSpinBox, QStyleFactory,
    QComboBox, QTextEdit, QDialog, QLineEdit, QListWidget,
    QListWidgetItem)

# Configure logging for production
if sys.platform == 'win32':
    # Windows: use AppData/Local
    log_dir = Path(os.getenv('LOCALAPPDATA', Path.home())) / 'RivlexReader'
elif sys.platform == 'darwin':
    # macOS: use Application Support
    log_dir = Path.home() / 'Library' / 'Application Support' / 'RivlexReader'
else:
    # Linux: use home directory
    log_dir = Path.home() / 'reader'

log_dir.mkdir(parents=True, exist_ok=True)
LOG_FILE = log_dir / 'reader.log'

file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding='utf-8'
)
file_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.INFO)  # Production: INFO level

handlers = [file_handler]
# Enable console logging for development
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(file_formatter)
# handlers.append(console_handler)

logging.basicConfig(level=logging.INFO, handlers=handlers)
logger = logging.getLogger(__name__)

# Application metadata
APP_NAME = "Rivlex Reader"
VERSION = "1.1.0"
AUTHOR = "OtlichnoBy"
GITHUB_URL = "https://github.com/OtlichnoBy/rivlex-reader"
LICENSE_INFO = "CC BY-NC-SA 4.0"

# UI Constants
CONTROL_HEIGHT = 40  # Standard height for buttons, spinboxes, comboboxes
BUTTON_SIZE = 50  # Width/height for square icon buttons
ICON_SIZE_SMALL = 20  # Size for small icons in labels
ICON_SIZE_LARGE = 32  # Size for large icons in buttons
BASE_FONT_SIZE = 16  # Base application font size

# Audio Constants
SAMPLE_RATE = 48000  # Audio sample rate in Hz
FADE_OUT_MS = 10  # Fade-out duration in milliseconds
FADE_OUT_SAMPLES = 480  # Fade-out length in samples (10ms at 48kHz)


def get_icon_path():
    """Returns the absolute path to the application icon."""
    if getattr(sys, 'frozen', False):
        # Path for the bundled executable
        base_path = sys._MEIPASS
    else:
        # Path for the source script execution
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Use .ico for Windows and .png for other platforms
    extension = "ico" if sys.platform == "win32" else "png"
    return os.path.join(base_path, f"icon.{extension}")


def load_custom_font():
    """Loads the custom Atkinson Hyperlegible Next font.

    Returns:
        str: Font family name or 'Arial' as fallback
    """
    font_dir = Path(__file__).parent / 'fonts'
    font_family = 'Atkinson Hyperlegible Next'

    if not font_dir.exists():
        logger.warning(f"Папка со шрифтами не найдена: {font_dir}")
        return 'Arial'

    try:
        font_files = [
            'AtkinsonHyperlegibleNext-Regular.otf',
            'AtkinsonHyperlegibleNext-Bold.otf',
            'AtkinsonHyperlegibleNext-RegularItalic.otf'
        ]

        loaded_count = 0
        for font_file in font_files:
            font_path = font_dir / font_file
            if font_path.exists():
                font_id = QFontDatabase.addApplicationFont(str(font_path))
                if font_id != -1:
                    loaded_count += 1
                    logger.info(f"Загружен шрифт: {font_file}")
                else:
                    logger.warning(f"Не удалось загрузить: {font_file}")

        if loaded_count > 0:
            logger.info(
                f"Шрифт {font_family} успешно загружен "
                f"({loaded_count}/{len(font_files)} файлов)"
            )
            return font_family
        else:
            logger.warning("Не удалось загрузить ни одного шрифта")
            return 'Arial'

    except Exception as e:
        logger.error(f"Ошибка при загрузке шрифта: {e}")
        return 'Arial'


def get_icon(icon_name):
    """Loads SVG icon from icons/ folder and makes it white.

    Args:
        icon_name (str): Icon name without extension

    Returns:
        QIcon: White icon or empty icon on error
    """
    icon_dir = Path(__file__).parent / 'icons'
    icon_path = icon_dir / f"{icon_name}.svg"

    if icon_path.exists():
        # Read SVG and replace color with white
        with open(icon_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
            # Replace stroke with white color
            svg_content = svg_content.replace(
                'stroke="currentColor"',
                'stroke="#FFFFFF"'
            )

        # Create temporary QPixmap from modified SVG
        renderer = QSvgRenderer(QByteArray(svg_content.encode('utf-8')))
        pixmap = QPixmap(renderer.defaultSize())
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()

        return QIcon(pixmap)
    else:
        logger.warning(f"Иконка не найдена: {icon_path}")
        return QIcon()


def get_settings_file():
    """Returns path to settings file"""
    home_dir = Path.home()
    reader_dir = home_dir / 'reader'
    # Create folder if it doesn't exist
    reader_dir.mkdir(exist_ok=True)
    return reader_dir / 'settings.json'


def load_settings():
    """Loads settings from file"""
    settings_file = get_settings_file()
    if settings_file.exists():
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.exception("Ошибка загрузки настроек: %s", e)
            return {}
    return {}


def save_settings(settings):
    """Saves settings to file"""
    settings_file = get_settings_file()
    try:
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.exception("Ошибка сохранения настроек: %s", e)


class CustomTextBrowser(QTextBrowser):
    doubleClicked = Signal(int)  # Signal with position for playback

    def mousePressEvent(self, event):
        """Handle single click - do NOT remove selection"""
        if event.button() == Qt.LeftButton:
            # Single click - just ignore, keep selection
            event.accept()
            return
        else:
            # Other mouse buttons - handle normally
            super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Handle double click - jump to new position"""
        cursor = self.cursorForPosition(event.position().toPoint())
        pos = cursor.position()

        # Get all text
        text = self.toPlainText()

        # Find sentence start (backwards to period/exclamation/question)
        start_pos = pos
        while start_pos > 0 and text[start_pos - 1] not in '.!?\n':
            start_pos -= 1

        # Find sentence end (forward to period/exclamation/question)
        end_pos = pos
        while end_pos < len(text) and text[end_pos] not in '.!?\n':
            end_pos += 1

        if end_pos < len(text):
            end_pos += 1  # include punctuation mark

        # Highlight sentence
        new_cursor = self.textCursor()
        new_cursor.setPosition(start_pos)
        new_cursor.movePosition(
            QTextCursor.Right,
            QTextCursor.KeepAnchor,
            end_pos - start_pos
        )
        self.setTextCursor(new_cursor)

        # Center selection
        self.ensureCursorVisible()

        # Emit signal with sentence start position
        self.doubleClicked.emit(start_pos)
        event.accept()

    def keyPressEvent(self, event):
        """Handle key presses - don't remove selection"""
        # Navigation keys
        if event.key() in (
                Qt.Key_PageUp, Qt.Key_PageDown,
                Qt.Key_Home, Qt.Key_End,
                Qt.Key_Up, Qt.Key_Down):
            # Pass these keys to parent class for scrolling
            super().keyPressEvent(event)
        # Application hotkeys - pass to parent
        elif (event.key() in (
                Qt.Key_Space, Qt.Key_F, Qt.Key_Plus,
                Qt.Key_Equal, Qt.Key_Minus) or
              (event.key() == Qt.Key_F and
              event.modifiers() == Qt.ControlModifier)):
            # Pass event to parent window for global
            # hotkeys handling
            event.ignore()
        else:
            # Ignore other keys to keep selection
            event.accept()


class FB2Parser:
    def __init__(self):
        self.text_content = ""
        self.text_plain = ""  # Plain text without HTML for correct length
        self.title = ""
        self.author = ""
        self.toc = []  # Table of contents: [(title, position), ...]

    def parse_book(self, file_path):
        """Parses FB2 file with auto-detection of encoding.

        Args:
            file_path (str): Path to FB2 file

        Returns:
            tuple: (success: bool, error: str or None)
        """
        self.text_content = ""
        self.title = ""
        self.author = ""
        try:
            # Read raw bytes for encoding detection
            with open(file_path, 'rb') as f:
                raw_data = f.read()

            # Try parsing raw bytes first
            try:
                # Use HTMLParser to handle HTML entities in XML
                parser = etree.XMLParser(recover=True, resolve_entities=True)
                tree = etree.fromstring(raw_data, parser=parser)
            except etree.XMLSyntaxError:
                # Fallback: auto-detect encoding and decode manually
                detection = chardet.detect(raw_data)
                encoding = detection['encoding'] or 'utf-8'
                logger.info(
                    f"Определена кодировка: {encoding} "
                    f"(уверенность: {detection['confidence']:.0%})"
                )
                xml_content = raw_data.decode(encoding, errors='replace')
                # Remove XML declaration to avoid error
                xml_content = re.sub(r'<\?xml[^>]+\?>', '', xml_content, count=1)
                parser = etree.XMLParser(recover=True, resolve_entities=True)
                tree = etree.fromstring(xml_content.encode('utf-8'), parser=parser)

            namespaces = {'fb': 'http://www.gribuser.ru/xml/fictionbook/2.0'}
            title_node = tree.find('.//fb:book-title', namespaces=namespaces)
            if title_node is not None:
                self.title = title_node.text
            author_first_name = tree.find(
                './/fb:first-name', namespaces=namespaces)
            author_last_name = tree.find(
                './/fb:last-name', namespaces=namespaces)
            if (author_first_name is not None and
                    author_last_name is not None):
                self.author = f"{author_first_name.text} {author_last_name.text}"
            
            # Get all body elements in document order
            body_elements = tree.xpath('//fb:body/*', namespaces=namespaces)
            
            html_paragraphs = []
            plain_paragraphs = []
            self.toc = []
            current_position = 0
            
            def process_element(elem, level=0, in_epigraph=False):
                nonlocal current_position
                tag = elem.tag.split('}')[-1]
                
                # Section
                if tag == 'section':
                    # Section title - process each paragraph separately
                    title_elem = elem.find('fb:title', namespaces=namespaces)
                    if title_elem is not None:
                        title_paras = title_elem.xpath(
                            './/fb:p', namespaces=namespaces
                        )
                        if title_paras:
                            # Add first title to TOC
                            first_title = (
                                self._element_to_plain_text(
                                    title_paras[0]
                                ).strip()
                            )
                            if first_title:
                                self.toc.append(
                                    (first_title, current_position)
                                )
                            
                            # Process each title paragraph separately
                            for p in title_paras:
                                title_text = self._element_to_plain_text(p).strip()
                                if title_text:
                                    html_paragraphs.append(
                                        f'<p style="font-size: 36px; font-weight: bold; '
                                        f'color: #CFECE0; margin-top: 40px; '
                                        f'margin-bottom: 20px;">{title_text}</p>'
                                    )
                                    plain_paragraphs.append(title_text)
                                    current_position += len(title_text) + 1
                    
                    # Process section children
                    for child in elem:
                        child_tag = child.tag.split('}')[-1]
                        if child_tag != 'title':
                            process_element(child, level + 1, in_epigraph)
                
                # Title (not in section - body title)
                elif tag == 'title':
                    title_paras = elem.xpath('.//fb:p', namespaces=namespaces)
                    for p in title_paras:
                        title_text = self._element_to_plain_text(p).strip()
                        if title_text:
                            html_paragraphs.append(
                                f'<p style="font-size: 36px; '
                                f'font-weight: bold; '
                                f'color: #CFECE0; '
                                f'margin-top: 40px; '
                                f'margin-bottom: 20px;"'
                                f'>{title_text}</p>'
                            )
                            plain_paragraphs.append(title_text)
                            current_position += len(title_text) + 1
                
                # Epigraph - process children with italic flag
                elif tag == 'epigraph':
                    for child in elem:
                        process_element(child, level, True)
                
                # Stanza (poetry)
                elif tag == 'stanza':
                    verses = elem.xpath('.//fb:v', namespaces=namespaces)
                    if verses:
                        verse_lines = []
                        verse_plain = []
                        for v in verses:
                            v_html = self._element_to_html(v)
                            v_text = self._element_to_plain_text(v)
                            if v_html.strip():
                                verse_lines.append(v_html)
                                verse_plain.append(v_text)
                        
                        if verse_lines:
                            html_paragraphs.append(
                                f'<p style="margin-left: 20px;">'
                                f'{"<br>".join(verse_lines)}</p>'
                            )
                            plain_text = '\n'.join(verse_plain)
                            plain_paragraphs.append(plain_text)
                            current_position += len(plain_text) + 1
                
                # Text elements: p, subtitle, text-author, cite, etc
                elif tag in ('p', 'subtitle', 'text-author', 'v'):
                    html_text = self._element_to_html(elem)
                    plain_text = self._element_to_plain_text(elem)
                    if html_text.strip():
                        if in_epigraph:
                            html_paragraphs.append(f"<p><i>{html_text}</i></p>")
                        else:
                            html_paragraphs.append(f"<p>{html_text}</p>")
                        plain_paragraphs.append(plain_text)
                        current_position += len(plain_text) + 1
                
                # Container elements - process children
                elif tag in ('poem', 'cite', 'annotation'):
                    for child in elem:
                        process_element(child, level, in_epigraph)
                
                # Empty line, image - skip
                elif tag in ('empty-line', 'image', 'a'):
                    pass
                
                # Unknown element - process children to not lose content
                else:
                    for child in elem:
                        process_element(child, level, in_epigraph)
            
            # Process all body elements
            for elem in body_elements:
                process_element(elem)
            
            self.text_content = "".join(html_paragraphs)
            self.text_plain = " ".join(plain_paragraphs)
            return True, None
        except Exception as e:
            logger.error(f"Ошибка парсинга FB2: {str(e)}")
            return False, str(e)

    def _element_to_html(self, element):
        """Recursively converts FB2 element to HTML"""
        result = []
        # Add text before first child element
        if element.text:
            result.append(element.text)

        # Process child elements
        for child in element:
            tag_name = child.tag.split('}')[-1]  # Remove namespace

            # Convert FB2 tags to HTML
            if tag_name == 'emphasis':
                child_html = self._element_to_html(child)
                result.append(f"<i>{child_html}</i>")
            elif tag_name == 'strong':
                child_html = self._element_to_html(child)
                result.append(f"<b>{child_html}</b>")
            elif tag_name == 'strikethrough':
                child_html = self._element_to_html(child)
                result.append(f"<s>{child_html}</s>")
            elif tag_name == 'sup':
                child_html = self._element_to_html(child)
                result.append(f"<sup>{child_html}</sup>")
            elif tag_name == 'sub':
                child_html = self._element_to_html(child)
                result.append(f"<sub>{child_html}</sub>")
            else:
                # For unknown tags just extract text
                child_html = self._element_to_html(child)
                result.append(child_html)

            # Add text after child element
            if child.tail:
                result.append(child.tail)

        return ''.join(result)
    
    def _element_to_plain_text(self, element):
        """Extract plain text without HTML tags"""
        return ''.join(element.itertext())

    def get_text(self):
        return self.text_content

    def get_title(self):
        return self.title

    def get_author(self):
        return self.author


class SileroTTSThread(QThread):
    finished = Signal()
    progress_changed = Signal(float)
    error = Signal(str)
    text_highlighted = Signal(int, int)

    def __init__(self, text, speaker=1, speed=100, parent=None):
        super().__init__(parent)
        self.text = text
        self.speaker = speaker
        self.speed = speed  # Speed percent (90-140)
        self.position_offset = 0
        self.is_running = True
        self.is_paused = False
        self.lock = QMutex()
        self.sample_rate = SAMPLE_RATE
        self.model = None
        self.next_audio = None
        self.next_start_pos = 0
        self.next_length = 0
        self._current_process = None  # Track external player process

    def update_speed(self, speed: int):
        """Update playback speed without recreating the thread."""
        self.lock.lock()
        try:
            self.speed = speed
        finally:
            self.lock.unlock()

    def update_speaker(self, speaker_index: int):
        """Update speaker voice on-the-fly."""
        self.lock.lock()
        try:
            self.speaker = speaker_index
            speakers = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene']
            self.speaker_name = speakers[speaker_index] if (
                speaker_index < len(speakers)) else 'baya'
        finally:
            self.lock.unlock()

    def speed_to_rate(self) -> str:
        """Map percent to SSML rate category."""
        # Direct mapping: 80->slow, 100->medium, 120->fast, 140->x-fast
        if self.speed < 90:
            return "slow"
        elif self.speed < 110:
            return "medium"
        elif self.speed < 130:
            return "fast"
        else:
            return "x-fast"

    def convert_numbers_to_text(self, text):
        """Converts numbers and Roman numerals to text in Russian"""
        def replace_number(match):
            num = match.group(0)
            try:
                return num2words(int(num), lang='ru')
            except (ValueError, TypeError):
                return num
        
        def replace_roman(match):
            """Converts Roman numerals to Russian words"""
            roman = match.group(0)
            roman_values = {
                'I': 1, 'V': 5, 'X': 10, 'L': 50,
                'C': 100, 'D': 500, 'M': 1000
            }
            try:
                result = 0
                prev_value = 0
                for char in reversed(roman):
                    value = roman_values.get(char, 0)
                    if value < prev_value:
                        result -= value
                    else:
                        result += value
                    prev_value = value
                if result > 0:
                    return num2words(result, lang='ru')
            except Exception:
                pass
            return roman

        # Replace Roman numerals first (I-XXX range)
        text = re.sub(r'\b([IVX]{1,5}|XX{0,2})\b', replace_roman, text)
        # Replace Arabic numbers
        return re.sub(r'\b\d+\b', replace_number, text)
    
    def filter_english_text(self, text):
        """Keep only Russian letters, digits, and basic punctuation"""
        # Fix Cyrillic initials - add pause after each
        text = re.sub(r'([А-ЯЁ])\.', r'\1, ', text)
        
        # Keep ONLY: Russian letters, digits, spaces, basic punctuation
        text = re.sub(r'[^А-Яа-яЁё0-9\s.,!?;:\-—()«»"]', ' ', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def split_text(self, text, max_length=1200):
        """Splits text into sentences preserving their positions"""
        sentences = []
        current_pos = 0
        
        # First split by newlines to respect paragraph structure
        lines = text.split('\n')
        
        for line in lines:
            line_stripped = line.strip()
            # Find where stripped content starts in original line
            leading_spaces = len(line) - len(line.lstrip())
            
            if not line_stripped:
                current_pos += len(line) + 1  # +1 for newline
                continue
            
            # Split each line by sentence boundaries
            pattern = r'(?<=[.!?])\s+'
            parts = re.split(pattern, line_stripped)
            
            line_pos = current_pos + leading_spaces
            
            for i, part in enumerate(parts):
                if not part.strip():
                    continue
                
                # If sentence too long, split by commas
                if len(part) > max_length:
                    sub_parts = re.split(r'(?<=[,;:])\s+', part)
                    sub_pos = line_pos
                    for sub_part in sub_parts:
                        if sub_part.strip():
                            # Store exact position in original text
                            sentences.append(
                                (sub_part.strip(), sub_pos, len(sub_part))
                            )
                            sub_pos += len(sub_part)
                    line_pos = sub_pos
                else:
                    # Store exact position and length
                    sentences.append(
                        (part.strip(), line_pos, len(part))
                    )
                    line_pos += len(part)
                    
                    # Add separator length if not last part
                    if i < len(parts) - 1:
                        # Find separator after this part
                        search_start = (
                            line_pos - current_pos - leading_spaces
                        )
                        match = re.search(
                            pattern, line_stripped[search_start:]
                        )
                        if match:
                            line_pos += len(match.group(0))
            
            # Move to next line (add newline character)
            current_pos += len(line) + 1

        # Filter out ONLY pure punctuation (keep short phrases like "А!")
        punctuation_only = [
            '.', '!', '?', ',', ';', ':', '-', '—', '...', ''
        ]
        sentences = [
            (part, pos, length) for part, pos, length in sentences
            if part not in punctuation_only
        ]
        return sentences

    def play_audio(self, audio_np, output_file, start_pos, length):
        """Plays audio in background and returns process"""
        try:
            if not isinstance(audio_np, np.ndarray):
                logger.error(
                    "Ошибка: audio_np имеет тип %s, ожидался numpy.ndarray",
                    type(audio_np))
                self.error.emit(
                    f"Ошибка: audio_np имеет тип {type(audio_np)}, "
                    "ожидался numpy.ndarray"
                )
                return None
            if audio_np.size == 0:
                logger.error("Ошибка: аудиоданные пусты")
                self.error.emit("Ошибка: аудиоданные пусты")
                return None
            logger.debug(
                "Форма audio_np: %s, тип: %s",
                getattr(audio_np, 'shape', None),
                getattr(audio_np, 'dtype', None)
            )
            if audio_np.ndim != 1:
                logger.warning(
                    "Предупреждение: audio_np имеет форму %s, "
                    "применяем squeeze",
                    getattr(audio_np, 'shape', None)
                )
                audio_np = audio_np.squeeze()
                if audio_np.ndim != 1:
                    logger.error(
                        "Ошибка: не удалось привести аудиоданные "
                        "к одномерному массиву"
                    )
                    self.error.emit(
                        "Ошибка: не удалось привести аудиоданные "
                        "к одномерному массиву"
                    )
                    return None

            # Add short fade-out to avoid click at the end
            fade_out_len = FADE_OUT_SAMPLES
            if len(audio_np) > fade_out_len:
                fade_out = np.linspace(1.0, 0.0, fade_out_len)
                audio_np[-fade_out_len:] *= fade_out

            audio_int16 = (audio_np * 32767).astype(np.int16)
            logger.debug(
                "Сохраняем аудио: длина %d сэмплов, файл: %s",
                len(audio_int16), output_file
            )
            write_wav(output_file, self.sample_rate, audio_int16)

            # Add offset to position
            adjusted_start_pos = start_pos + getattr(
                self, 'position_offset', 0)
            self.text_highlighted.emit(adjusted_start_pos, length)

            # Windows: winsound only (external players unavailable)
            if sys.platform == 'win32':
                try:
                    # Use SND_ASYNC for non-blocking playback
                    winsound.PlaySound(
                        output_file, 
                        winsound.SND_FILENAME | winsound.SND_ASYNC)
                    # Store filename and start time for checking completion
                    self._current_process = {
                        'file': output_file,
                        'start_time': time.time(),
                        'duration': len(audio_int16) / self.sample_rate
                    }
                    logger.info(
                        "Запущено воспроизведение через winsound: %s (%.2f сек)",
                        output_file, self._current_process['duration'])
                    return self._current_process
                except Exception:
                    logger.exception("Ошибка при запуске winsound")

            # *nix: external players (tempo already in model)
            players = ["ffplay", "mpv", "aplay", "paplay", "play"]
            for player in players:
                try:
                    if player == "ffplay":
                        cmd = [
                            player, "-nodisp", "-autoexit", "-loglevel",
                            "quiet", output_file
                        ]
                    elif player == "mpv":
                        cmd = [
                            player, "--no-video", "--really-quiet",
                            output_file
                        ]
                    else:
                        cmd = [player, output_file]

                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        start_new_session=True)
                    self._current_process = process
                    logger.info(
                        "Запущено воспроизведение через %s: %s",
                        player, output_file)
                    return process
                except FileNotFoundError:
                    logger.warning("Плеер %s не найден", player)
                except Exception:
                    logger.exception("Ошибка при запуске плеера %s", player)

            logger.error(
                "Не удалось воспроизвести аудио: ни один плеер не сработал")
            self.error.emit(
                "Не удалось воспроизвести аудио: ни один плеер не сработал")
            return None
        except Exception:
            logger.exception("Ошибка при сохранении аудио")
            self.error.emit("Ошибка при сохранении аудио")
            return None

    def run(self):
        try:
            logger.info("Silero TTS v5 с омографами запущен...")
            self.model, _ = silero_tts(language='ru', speaker='v5_ru')
            speakers = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene']
            self.speaker_name = speakers[self.speaker] if self.speaker < len(
                speakers) else 'baya'
            logger.info("Используем голос v5: %s", self.speaker_name)

            if not self.text.strip():
                logger.error("Ошибка: текст для синтеза пуст")
                self.error.emit("Ошибка: текст для синтеза пуст")
                return

            text_parts = self.split_text(self.text)
            total_parts = len(text_parts)
            logger.debug("Текст разбит на %d частей", total_parts)
            parts_processed = 0
            for i, (part, start_pos, length) in enumerate(text_parts):
                self.lock.lock()
                try:
                    if not self.is_running:
                        break
                    while self.is_paused:
                        self.lock.unlock()
                        self.msleep(10)
                        self.lock.lock()
                    if not self.is_running:
                        break
                finally:
                    self.lock.unlock()
                
                # Always generate audio (no prefetch)
                part = part.strip()
                if not part:
                    logger.debug("Пропускаем пустую часть %d", i+1)
                    parts_processed += 1
                    self.progress_changed.emit(
                        parts_processed / total_parts
                    )
                    continue

                logger.debug(
                    "Генерируем часть %d/%d: %s...",
                    i+1, total_parts, part[:50]
                )
                # Convert numbers to text before synthesis
                part_for_tts = self.convert_numbers_to_text(part)
                # Filter English words that cause TTS errors
                part_for_tts = self.filter_english_text(part_for_tts)
                # Show what exactly is passed to TTS
                logger.debug("Текст для TTS: '%s'...", part_for_tts[:100])

                # Extended check for valid TTS text
                if not part_for_tts.strip():
                    logger.debug(
                        "Пропускаем пустую часть после обработки"
                    )
                    parts_processed += 1
                    self.progress_changed.emit(
                        parts_processed / total_parts
                    )
                    continue

                # Skip ONLY punctuation-only fragments
                if part_for_tts.strip() in [
                        '.', '!', '?', ',', ';', ':', '-', '—', '...']:
                    logger.debug(
                        "Пропускаем знак препинания: '%s'",
                        part_for_tts
                    )
                    parts_processed += 1
                    self.progress_changed.emit(
                        parts_processed / total_parts
                    )
                    continue

                try:
                    rate_category = self.speed_to_rate()
                    ssml_text = (
                        f"<speak><prosody rate=\"{rate_category}\">"
                        f"{part_for_tts}</prosody></speak>"
                    )
                    audio = self.model.apply_tts(
                        ssml_text=ssml_text,
                        speaker=self.speaker_name,
                        sample_rate=self.sample_rate,
                        put_accent=True,
                        put_yo=True,
                        put_stress_homo=True,
                        put_yo_homo=True
                    )

                    audio_np = audio.detach().cpu().numpy()
                    if audio_np.ndim > 1:
                        audio_np = audio_np.squeeze()
                except (ValueError, Exception) as e:
                    logger.warning(
                        "Пропускаем проблемный фрагмент (позиция %d): %s",
                        start_pos, str(e)
                    )
                    parts_processed += 1
                    self.progress_changed.emit(
                        parts_processed / total_parts
                    )
                    continue

                if i < len(text_parts) - 1:
                    silence = np.zeros(int(self.sample_rate * 0.1))
                    audio_np = np.concatenate([audio_np, silence])

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    output_file = temp_file.name
                process = self.play_audio(
                    audio_np, output_file, start_pos, length)
                if not process:
                    try:
                        os.unlink(output_file)
                    except OSError:
                        pass
                    parts_processed += 1
                    self.progress_changed.emit(
                        parts_processed / total_parts
                    )
                    continue
                
                # Wait for playback to complete before continuing
                try:
                    if sys.platform == 'win32':
                        # Windows - poll completion (winsound async)
                        start_time = process['start_time']
                        duration = process['duration']
                        # Poll every 0.1s until audio should be finished
                        while time.time() - start_time < duration:
                            if not self.is_running:
                                break
                            time.sleep(0.1)
                        stdout, stderr = b'', b''
                    else:
                        # Linux/Mac - wait for subprocess
                        stdout, stderr = process.communicate(timeout=60)
                    if hasattr(process, 'returncode') and process.returncode != 0:
                        logger.error(
                            "Ошибка воспроизведения: %s",
                            stderr.decode()
                        )
                    try:
                        os.unlink(output_file)
                    except OSError:
                        pass
                    self._current_process = None
                    parts_processed += 1

                    # Track progress
                    read_chars = self.position_offset + \
                        sum(length for _, _, length in text_parts[:parts_processed])
                    total_chars = self.position_offset + \
                        sum(length for _, _, length in text_parts)
                    progress = round((read_chars / total_chars * 100), 2)

                    self.progress_changed.emit(progress)
                    logger.debug("Прогресс: %s%%", progress)
                except subprocess.TimeoutExpired:
                    logger.warning("Таймаут воспроизведения")
                    process.kill()
                    try:
                        os.unlink(output_file)
                    except OSError:
                        pass
                    self._current_process = None
                    parts_processed += 1
                    self.progress_changed.emit(
                        parts_processed / total_parts
                    )
                    continue
            if not self.is_running:
                return
            logger.info("Воспроизведение завершено")
            self.finished.emit()
        except Exception:
            logger.exception("Ошибка в Silero TTS")
            self.error.emit(
                "Ошибка Silero TTS: "
                "произошла непредвиденная ошибка"
            )
        finally:
            self.is_running = False
            if self.model:
                self.model = None
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def play_pause(self):
        self.lock.lock()
        try:
            self.is_paused = not self.is_paused
            logger.info("Пауза: %s", self.is_paused)
        finally:
            self.lock.unlock()

    def stop_playback(self):
        self.lock.lock()
        try:
            self.is_running = False
            # Try to kill external player subprocess if it's running
            if self._current_process:
                try:
                    if isinstance(self._current_process, dict):
                        # Windows - winsound dict, use SND_PURGE to stop
                        logger.info("Остановка winsound воспроизведения...")
                        if sys.platform == 'win32':
                            try:
                                import winsound
                                winsound.PlaySound(None, winsound.SND_PURGE)
                            except Exception:
                                logger.exception("Ошибка при остановке winsound")
                        self._current_process = None
                    else:
                        # Linux/Mac - subprocess
                        if self._current_process.poll() is not None:
                            # Process already finished
                            self._current_process = None
                            return
                        logger.info("Завершение внешнего плеера...")
                        self._current_process.terminate()
                        # Give it a moment to terminate gracefully
                        try:
                            self._current_process.wait(timeout=0.5)
                        except subprocess.TimeoutExpired:
                            # Force kill if didn't terminate
                            self._current_process.kill()
                        self._current_process = None
                except Exception:
                    logger.exception("Ошибка при завершении плеера")
        finally:
            self.lock.unlock()


class BookReaderWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} fb2 v{VERSION}")
        self.resize(1200, 900)
        icon_path = get_icon_path()
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.tts_thread = None
        self.parser = FB2Parser()
        self.DARK_GREEN = "#004d00"
        self._jumping = False  # Flag to prevent multiple clicks

        # Load custom font
        self.font_family = load_custom_font()

        # Load settings
        self.settings = load_settings()
        self.init_ui()

        # Install event filter to catch all keyboard events
        QApplication.instance().installEventFilter(self)

        # Restore state after UI creation
        self.restore_state()

    def format_reading_time(self, seconds):
        """Formats reading time in compact view"""
        if seconds <= 0:
            return "0 мин"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        if hours > 0:
            return f"{hours} ч {minutes} мин"
        else:
            return f"{minutes} мин"

    def format_progress_line(
            self, read_chars, total_chars, progress, time_str):
        """Formats progress bar string with vertical separators"""
        return f"{read_chars:,}/{total_chars:,} | {progress:.2f}% | {time_str}"

    def init_ui(self):
        # Set base font for application
        app_font = QFont(self.font_family, BASE_FONT_SIZE)
        QApplication.setFont(app_font)

        dark_style = f"""
        QPushButton#controlButton {{
            background-color: {self.DARK_GREEN};
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            padding: 8px;
            font-weight: bold;
            font-size: 36px;
            min-height: 40px;
            min-width: 50px;
            max-width: 50px;
        }}

        QPushButton#controlButton:hover {{
            background-color: #006600;
        }}

        QPushButton#controlButton:pressed {{
            background-color: #003200;
        }}

        QPushButton#controlButton:disabled {{
            background-color: #666666;
            color: #FFFFFF;
        }}

        QPushButton {{
            font-size: 14px;
            padding: 6px 12px;
        }}

        QLabel {{
            font-size: 20px;
            color: #FFFFFF;
            font-weight: bold;
        }}
        QLabel#aboutTitle {{
            font-size: 27px;
            font-weight: bold;
        }}
        QLabel#searchResultLabel {{
            font-size: 14px;
            font-weight: normal;
        }}
        QTextBrowser {{
            border: 3px solid #555555;
            border-radius: 10px;
            padding: 12px;
            background-color: #292610;
            color: #F0EBD8;
            selection-background-color: #003200;
            selection-color: #FFFFFF;
        }}
        QProgressBar {{
            border: 3px solid #555555;
            border-radius: 10px;
            text-align: center;
            background-color: #232015;
            color: #FFFFFF;
            font-size: 24px;
            min-height: 30px;
        }}
        QProgressBar::chunk {{
            background-color: {self.DARK_GREEN};
            border-radius: 7px;
        }}

        QSpinBox {{
            border: 2px solid #555555;
            border-radius: 6px;
            background-color: #2F2C18;
            color: #FFFFFF;
            font-size: 20px;
            height: 40px;
            padding: 0px 5px;
        }}

        QSpinBox::up-arrow, QSpinBox::down-arrow {{
            width: 8px;
            height: 8px;
            background-color: #FFFFFF;
            border-radius: 4px;
            margin: 5px;
        }}

        QSpinBox::up-button {{
            subcontrol-origin: border;
            subcontrol-position: top right;
            width: 20px;
            background: #444;
            border-radius: 3px;
        }}

        QSpinBox::down-button {{
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            width: 20px;
            background: #444;
            border-radius: 3px;
        }}

        QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
            background: #666;
        }}

        QComboBox {{
            border: 2px solid #555555;
            border-radius: 6px;
            background-color: #2F2C18;
            color: #FFFFFF;
            font-size: 20px;
            height: 40px;
            padding: 0px 5px;
        }}
        QComboBox::drop-down {{
            border: none;
            width: 0px;
            background-color: transparent;
            margin: 0px;
            padding: 0px;
        }}
        QComboBox::down-arrow {{
            image: none;
            width: 0px;
            height: 0px;
        }}

        QToolTip {{
            background-color: #232015;
            color: #FFFFFF;
            border: 2px solid #555555;
            border-radius: 6px;
            padding: 8px;
            font-size: 16px;
        }}

        QScrollBar:vertical {{
            background-color: #1F1C08;
            border: 1px solid #555555;
            width: 16px;
            margin: 0px;
        }}

        QScrollBar::handle:vertical {{
            background-color: #4F4C38;
            border: 1px solid #777777;
            border-radius: 3px;
            min-height: 20px;
        }}

        QScrollBar::handle:vertical:hover {{
            background-color: #6F6C58;
        }}

        QScrollBar::handle:vertical:pressed {{
            background-color: #8F8C78;
        }}

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
            width: 0px;
        }}

        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: transparent;
        }}

        QLineEdit {{
            border: 2px solid #555555;
            border-radius: 6px;
            background-color: #2F2C18;
            color: #FFFFFF;
            font-size: 20px;
            padding: 8px;
            min-height: 34px;
        }}

        QLineEdit:focus {{
            border: 2px solid #006600;
        }}

        QLabel#separator {{
            background-color: #555555;
            max-height: 2px;
        }}

        QTextBrowser#aboutInfo {{
            border: none;
            background-color: transparent;
            color: #FFFFFF;
        }}

        QTextBrowser#aboutInfo a {{
            color: #FFFFFF;
            text-decoration: underline;
        }}

        QPushButton#dialogButton {{
            font-size: 20px;
            padding: 8px 16px;
            min-height: 37px;
        }}

        QListWidget {{
            border: 2px solid #555555;
            border-radius: 6px;
            background-color: #2F2C18;
            color: #FFFFFF;
            font-size: 18px;
            padding: 5px;
        }}

        QListWidget::item {{
            padding: 8px;
            border-bottom: 1px solid #444444;
        }}

        QListWidget::item:hover {{
            background-color: #3F3C28;
        }}

        QListWidget::item:selected {{
            background-color: {self.DARK_GREEN};
        }}

        """
        self.setStyleSheet(dark_style)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout(main_widget)

        control_panel = QHBoxLayout()

        self.load_book_btn = QPushButton()
        self.load_book_btn.setIcon(get_icon('book-open'))
        self.load_book_btn.setIconSize(QSize(ICON_SIZE_LARGE, ICON_SIZE_LARGE))
        self.load_book_btn.setToolTip("Открыть книгу")
        self.load_book_btn.setFixedSize(BUTTON_SIZE, CONTROL_HEIGHT)
        self.load_book_btn.setFocusPolicy(Qt.NoFocus)
        self.load_book_btn.clicked.connect(self.load_book)

        self.search_btn = QPushButton()
        self.search_btn.setIcon(get_icon('search'))
        self.search_btn.setIconSize(QSize(ICON_SIZE_LARGE, ICON_SIZE_LARGE))
        self.search_btn.setToolTip("Поиск в книге")
        self.search_btn.setFixedSize(BUTTON_SIZE, CONTROL_HEIGHT)
        self.search_btn.setFocusPolicy(Qt.NoFocus)
        self.search_btn.clicked.connect(self.show_search_dialog)
        self.search_btn.setEnabled(False)

        self.toc_btn = QPushButton()
        self.toc_btn.setIcon(get_icon('list'))
        self.toc_btn.setIconSize(QSize(ICON_SIZE_LARGE, ICON_SIZE_LARGE))
        self.toc_btn.setToolTip("Оглавление")
        self.toc_btn.setFixedSize(BUTTON_SIZE, CONTROL_HEIGHT)
        self.toc_btn.setFocusPolicy(Qt.NoFocus)
        self.toc_btn.clicked.connect(self.show_toc_dialog)
        self.toc_btn.setEnabled(False)

        self.play_pause_btn = QPushButton()
        self.play_pause_btn.setIcon(get_icon('play'))
        self.play_pause_btn.setIconSize(QSize(ICON_SIZE_LARGE, ICON_SIZE_LARGE))
        self.play_pause_btn.setToolTip("Воспроизвести")
        self.play_pause_btn.setFixedSize(BUTTON_SIZE, CONTROL_HEIGHT)
        self.play_pause_btn.setFocusPolicy(Qt.NoFocus)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.play_pause_btn.setEnabled(False)

        font_size_layout = QHBoxLayout()
        font_size_label = QLabel()
        font_size_label.setPixmap(get_icon('type').pixmap(ICON_SIZE_SMALL, ICON_SIZE_SMALL))
        font_size_label.setAlignment(Qt.AlignVCenter)
        font_size_text = QLabel(" Шрифт:")
        self.font_size_spinbox = QSpinBox()
        self.font_size_spinbox.setRange(12, 48)
        self.font_size_spinbox.setValue(24)
        self.font_size_spinbox.setFixedHeight(CONTROL_HEIGHT)
        self.font_size_spinbox.setFixedWidth(58)
        self.font_size_spinbox.setFocusPolicy(Qt.ClickFocus)
        self.font_size_spinbox.valueChanged.connect(self.change_font_size)

        font_size_layout.addWidget(font_size_label, 0, Qt.AlignVCenter)
        font_size_layout.addWidget(font_size_text, 0, Qt.AlignVCenter)
        font_size_layout.addWidget(self.font_size_spinbox)

        voice_layout = QHBoxLayout()
        voice_icon_label = QLabel()
        voice_icon_label.setPixmap(get_icon('mic').pixmap(ICON_SIZE_SMALL, ICON_SIZE_SMALL))
        voice_icon_label.setAlignment(Qt.AlignVCenter)
        voice_label = QLabel(" Голос:")
        self.voice_combo = QComboBox()
        self.voice_combo.addItems([
            "Aidar (мужской)", "Baya (женский)",
            "Kseniya (женский)", "Xenia (женский)",
            "Eugene (мужской)"
        ])
        self.voice_combo.setCurrentIndex(1)
        self.voice_combo.setFixedWidth(200)
        self.voice_combo.setFixedHeight(CONTROL_HEIGHT)
        self.voice_combo.setFocusPolicy(Qt.ClickFocus)
        self.voice_combo.currentIndexChanged.connect(self.on_voice_changed)
        voice_layout.addWidget(voice_icon_label, 0, Qt.AlignVCenter)
        voice_layout.addWidget(voice_label, 0, Qt.AlignVCenter)
        voice_layout.addWidget(self.voice_combo)

        self.status_label = QLabel("Готово")
        self.status_label.setFixedSize(QSize(270, CONTROL_HEIGHT))

        control_panel.addWidget(self.load_book_btn, 0, Qt.AlignVCenter)
        control_panel.addWidget(self.toc_btn, 0, Qt.AlignVCenter)
        control_panel.addWidget(self.search_btn, 0, Qt.AlignVCenter)
        control_panel.addWidget(self.play_pause_btn, 0, Qt.AlignVCenter)
        control_panel.addWidget(self.status_label, 0, Qt.AlignVCenter)
        control_panel.addStretch()
        control_panel.addLayout(font_size_layout)
        control_panel.addLayout(voice_layout)

        # Playback speed (spinbox snapped to real SSML steps)
        speed_icon_label = QLabel()
        speed_icon_label.setPixmap(
            get_icon('gauge').pixmap(ICON_SIZE_SMALL, ICON_SIZE_SMALL))
        speed_icon_label.setAlignment(Qt.AlignVCenter)
        speed_label = QLabel(" Скорость:")
        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setRange(80, 140)
        self.speed_spinbox.setValue(100)
        self.speed_spinbox.setSuffix("%")
        self.speed_spinbox.setSingleStep(20)
        self.speed_spinbox.setFixedHeight(CONTROL_HEIGHT)
        self.speed_spinbox.setFixedWidth(88)
        self.speed_spinbox.setFocusPolicy(Qt.ClickFocus)
        self.speed_spinbox.setToolTip(
            "Скорость воспроизведения\n"
            "Значения: 80, 100, 120, 140%"
        )
        self.speed_spinbox.valueChanged.connect(self.on_speed_changed)
        control_panel.addWidget(speed_icon_label, 0, Qt.AlignVCenter)
        control_panel.addWidget(speed_label, 0, Qt.AlignVCenter)
        control_panel.addWidget(self.speed_spinbox)

        # About button at the very end
        self.about_btn = QPushButton()
        self.about_btn.setIcon(get_icon('info'))
        self.about_btn.setIconSize(QSize(ICON_SIZE_LARGE, ICON_SIZE_LARGE))
        self.about_btn.setToolTip("О программе")
        self.about_btn.setFixedSize(BUTTON_SIZE, CONTROL_HEIGHT)
        self.about_btn.setFocusPolicy(Qt.NoFocus)
        self.about_btn.clicked.connect(self.show_about_dialog)
        control_panel.addWidget(self.about_btn, 0, Qt.AlignVCenter)

        self.book_text_area = CustomTextBrowser()
        self.book_text_area.setReadOnly(True)
        self.book_text_area.setLineWrapMode(QTextEdit.WidgetWidth)

        doc = self.book_text_area.document()
        doc.setUseDesignMetrics(True)
        doc.setDefaultFont(self.book_text_area.font())

        self.book_text_area.doubleClicked.connect(self.on_double_click_jump)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(10000)

        # Initialize progress bar at zero (text not loaded yet)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('0.00%')

        self.progress_bar.setFixedHeight(30)

        main_layout.addLayout(control_panel)
        main_layout.addWidget(self.book_text_area)
        main_layout.addWidget(self.progress_bar)

        # Align all elements vertically
        control_panel.setAlignment(self.load_book_btn, Qt.AlignVCenter)
        control_panel.setAlignment(self.play_pause_btn, Qt.AlignVCenter)
        control_panel.setAlignment(self.voice_combo, Qt.AlignVCenter)
        control_panel.setAlignment(self.font_size_spinbox, Qt.AlignVCenter)
        control_panel.setAlignment(self.status_label, Qt.AlignVCenter)

    def change_font_size(self, size):
        # Use setFont() instead of setStyleSheet()
        font = self.book_text_area.font()
        font.setPixelSize(int(size * 1.33))
        self.book_text_area.setFont(font)

        self.book_text_area.document().setDefaultFont(font)

        # Auto-save settings
        if hasattr(self, 'settings'):
            self.settings['font_size'] = size
            save_settings(self.settings)

        # Restore position after font change
        self.restore_position_after_font_change()

    def restore_position_after_font_change(self):
        """Restores reading position after font change"""
        if (hasattr(self, 'settings') and
                'last_read_position' in self.settings and
                self.book_text_area.toPlainText()):

            position = self.settings['last_read_position']
            text = self.book_text_area.toPlainText()

            if position > 0 and len(text) > 0:
                def restore_position_delayed():
                    try:
                        cursor = self.book_text_area.textCursor()
                        cursor.setPosition(min(position, len(text) - 1))

                        # Find sentence end for highlighting
                        text_after_pos = text[position:]
                        match = re.search(r'[.!?](\s|$)', text_after_pos)
                        if match:
                            end_pos = position + match.end()
                        else:
                            end_pos = min(position + 50, len(text))

                        length = end_pos - position
                        if length <= 0:
                            length = 1

                        cursor.movePosition(
                            QTextCursor.Right, QTextCursor.KeepAnchor, length)
                        self.book_text_area.setTextCursor(cursor)

                        # Center using the same logic
                        scroll_bar = self.book_text_area.verticalScrollBar()
                        cursor_rect = self.book_text_area.cursorRect(cursor)
                        viewport_height = self.book_text_area.viewport().height()

                        scroll_bar.setValue(
                            scroll_bar.value() + cursor_rect.y() - viewport_height // 2)
                        self.book_text_area.ensureCursorVisible()

                        logger.info(
                            "Позиция восстановлена после изменения "
                            "шрифта: %s", position
                        )

                    except Exception:
                        logger.exception(
                            "Ошибка восстановления позиции при смене шрифта"
                        )

                # Execute restoration with delay for new font to apply
                QTimer.singleShot(100, restore_position_delayed)

    def load_book(self):
        # Stop any running playback BEFORE opening file dialog
        if self.tts_thread and self.tts_thread.isRunning():
            self.stop_playback()
            QApplication.processEvents()
        
        # Use last opened folder as initial directory when available
        initial_dir = self.settings.get('last_open_folder', str(Path.home()))
        
        # Use native file dialog without parent
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Открыть книгу", initial_dir, "FB2 files (*.fb2)"
        )
            
        if file_path:
            # Show wait cursor during loading
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                # Additional safety check - stop again if somehow started
                if self.tts_thread and self.tts_thread.isRunning():
                    self.stop_playback()
                QApplication.processEvents()  # Process pending UI events
                
                success, error = self.parser.parse_book(file_path)
                if success:
                    self.current_book_path = file_path  # Save book path
                    text = self.parser.get_text()
                    self.book_text_area.setHtml(text)
                    doc = self.book_text_area.document()
                    doc.setUseDesignMetrics(True)
                    doc.setDefaultFont(self.book_text_area.font())
                    self.setWindowTitle(
                        f"{APP_NAME} fb2 v{VERSION} - "
                        f"{self.parser.get_title()} by "
                        f"{self.parser.get_author()}")
                    self.status_label.setText("Книга загружена")
                    self.play_pause_btn.setEnabled(True)
                    self.search_btn.setEnabled(True)
                    self.toc_btn.setEnabled(True)
                    self.play_pause_btn.setToolTip("Воспроизвести")
                    self.progress_bar.setValue(0)

                    # Save new book path, folder and reset position
                    self.settings['last_opened_book'] = file_path
                    self.settings['last_open_folder'] = str(Path(file_path).parent)
                    self.settings['last_read_position'] = 0
                    save_settings(self.settings)
                    
                    # Set focus to text area for hotkeys to work
                    self.book_text_area.setFocus()
                else:
                    QMessageBox.critical(
                        self,
                        "Ошибка загрузки",
                        f"Не удалось загрузить книгу: {error}")
                    self.book_text_area.clear()
                    self.status_label.setText("Ошибка")
                    self.play_pause_btn.setEnabled(False)
            finally:
                # Restore normal cursor
                QApplication.restoreOverrideCursor()

    def toggle_play_pause(self):
        if not self.tts_thread or not self.tts_thread.isRunning():
            # Start playback
            self.start_playback()
        else:
            # Simple pause/resume without automatic restart
            if self.tts_thread.is_paused:
                # Resume
                self.tts_thread.play_pause()
                self.play_pause_btn.setIcon(get_icon('pause'))
                self.play_pause_btn.setToolTip("Пауза")
                self.status_label.setText("Воспроизведение...")
            else:
                # Pause
                self.tts_thread.play_pause()
                self.play_pause_btn.setIcon(get_icon('play'))
                self.play_pause_btn.setToolTip("Продолжить")
                self.status_label.setText("Пауза")

    def start_playback(self):
        # Protect from multiple starts
        if self.tts_thread and self.tts_thread.isRunning():
            logger.warning("Поток уже запущен, игнорируем запрос")
            return

        # Get text without HTML tags for TTS
        full_text = self.book_text_area.toPlainText()

        if not full_text:
            self.status_label.setText("Нет текста для чтения.")
            return

        # Get saved position
        start_position = self.settings.get('last_read_position', 0)

        # Read text from saved position
        text_to_read = full_text[start_position:]

        if not text_to_read.strip():
            self.status_label.setText("Чтение завершено.")
            return

        logger.info("Начинаем чтение с позиции: %s", start_position)

        speaker_name = self.voice_combo.currentText()
        silero_speakers = [
            "Aidar (мужской)",
            "Baya (женский)",
            "Kseniya (женский)",
            "Xenia (женский)",
            "Eugene (мужской)"]
        speaker_id = silero_speakers.index(
            speaker_name) if speaker_name in silero_speakers else 1

        # Get speed percent from spinbox (only 80/100/120/140 allowed)
        speed = self.speed_spinbox.value()

        self.tts_thread = SileroTTSThread(
            text_to_read, speaker=speaker_id, speed=speed, parent=self)
        # Pass position offset to thread
        self.tts_thread.position_offset = start_position

        self.tts_thread.finished.connect(self.on_playback_finished)
        self.tts_thread.progress_changed.connect(self.update_progress)
        self.tts_thread.text_highlighted.connect(self.highlight_text)
        self.tts_thread.error.connect(
            lambda e: logger.error("TTS error: %s", e))

        self.tts_thread.start()

        self.play_pause_btn.setIcon(get_icon('pause'))
        self.play_pause_btn.setToolTip("Пауза")

        # Set initial progress accounting for read part
        full_text_len = len(full_text) if full_text else 1
        initial_progress = (start_position / full_text_len) * 100
        scaled_value = int(initial_progress * 100)
        self.progress_bar.setValue(scaled_value)

        # Form detailed text for initial state
        reading_speed_cps = 15
        elapsed_seconds = start_position / reading_speed_cps
        time_str = self.format_reading_time(elapsed_seconds)
        progress_text = self.format_progress_line(
            start_position, full_text_len, initial_progress, time_str)
        self.progress_bar.setFormat(progress_text)
        self.status_label.setText(f"Воспроизведение...")

    def update_progress(self, progress):
        """Updates progress with additional information in progress bar"""
        scaled_value = int(progress * 100)
        self.progress_bar.setValue(scaled_value)

        # Use plain text from parser if available for accurate counting
        if (hasattr(self, 'parser') and 
                hasattr(self.parser, 'text_plain') and 
                self.parser.text_plain):
            total_chars = len(self.parser.text_plain)
        else:
            full_text = self.book_text_area.toPlainText()
            total_chars = len(full_text) if full_text else 0
        
        if total_chars > 0:
            read_chars = int((progress / 100) * total_chars)

            # Calculate reading time (approximately 15 characters per second)
            reading_speed_cps = 15
            elapsed_seconds = read_chars / reading_speed_cps
            time_str = self.format_reading_time(elapsed_seconds)

            # Complex format: "12,345/67,890   23.45%   2ч15м"
            progress_text = self.format_progress_line(
                read_chars, total_chars, progress, time_str)
            self.progress_bar.setFormat(progress_text)
        else:
            self.progress_bar.setFormat(f'{progress:.2f}%')

        if progress < 100:
            # Don't change status - it's already set in toggle_play_pause
            pass
        else:
            self.status_label.setText("Завершено")

    def cleanup_tts_thread(self):
        """Safely disconnects all signals from TTS thread"""
        if not self.tts_thread:
            return

        try:
            self.tts_thread.finished.disconnect()
        except (RuntimeError, TypeError):
            pass
        try:
            self.tts_thread.progress_changed.disconnect()
        except (RuntimeError, TypeError):
            pass
        try:
            self.tts_thread.text_highlighted.disconnect()
        except (RuntimeError, TypeError):
            pass
        try:
            self.tts_thread.error.disconnect()
        except (RuntimeError, TypeError):
            pass

    def stop_playback(self):
        if self.tts_thread and self.tts_thread.isRunning():
            # Disconnect error signal first to prevent error dialogs during stop
            try:
                self.tts_thread.error.disconnect()
            except (RuntimeError, TypeError):
                pass
            
            self.tts_thread.stop_playback()
            # Wait with timeout - if thread doesn't stop, force terminate
            if not self.tts_thread.wait(2000):
                logger.warning(
                    "TTS thread не остановился, "
                    "принудительное завершение"
                )
                self.tts_thread.terminate()
                self.tts_thread.wait(1000)

            self.cleanup_tts_thread()
            self.tts_thread = None

        self.play_pause_btn.setIcon(get_icon('play'))
        self.play_pause_btn.setToolTip("Воспроизвести")
        self.play_pause_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Готово")

        # Clear text selection
        cursor = self.book_text_area.textCursor()
        cursor.clearSelection()
        self.book_text_area.setTextCursor(cursor)

    def on_playback_finished(self):
        self.play_pause_btn.setIcon(get_icon('play'))
        self.play_pause_btn.setToolTip("Воспроизвести")
        self.status_label.setText("Завершено")

        if self.tts_thread:
            self.cleanup_tts_thread()  # Instead of manual disconnection
            self.tts_thread = None

    def highlight_text(self, start, length):
        cursor = self.book_text_area.textCursor()
        text = self.book_text_area.toPlainText()

        # Check that start and length are within valid bounds
        if start < 0 or start >= len(text) or length <= 0:
            return

        if start + length > len(text):
            length = len(text) - start

        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, length)
        self.book_text_area.setTextCursor(cursor)

        # Update last_read_position only in memory (without writing to disk)
        if hasattr(self, 'settings'):
            self.settings['last_read_position'] = start

        # Center selected text
        scroll_bar = self.book_text_area.verticalScrollBar()
        cursor_rect = self.book_text_area.cursorRect(cursor).topLeft()
        viewport_height = self.book_text_area.viewport().height()
        scroll_bar.setValue(
            scroll_bar.value() +
            cursor_rect.y() -
            viewport_height //
            2)
        self.book_text_area.ensureCursorVisible()

    def closeEvent(self, event):
        self.save_current_state()

        # Force stop thread before closing
        if self.tts_thread and self.tts_thread.isRunning():
            # If thread is paused, unpause so it can correctly finish
            if self.tts_thread.is_paused:
                self.tts_thread.play_pause()

            self.stop_playback()

            if self.tts_thread:  # Check that thread still exists
                # Wait for thread completion with timeout
                self.tts_thread.wait(3000)
                if self.tts_thread.isRunning():
                    self.tts_thread.terminate()
                    self.tts_thread.wait()

        event.accept()  # Allow window closing

    def save_current_state(self):
        """Saves current application state"""
        self.settings['font_size'] = self.font_size_spinbox.value()
        self.settings['voice_index'] = self.voice_combo.currentIndex()
        self.settings['speed'] = self.speed_spinbox.value()

        # Save reading position only if there is text
        if self.book_text_area.toPlainText():
            cursor = self.book_text_area.textCursor()
            self.settings['last_read_position'] = cursor.position()

        # Save open book path
        if hasattr(self, 'current_book_path') and self.current_book_path:
            self.settings['last_opened_book'] = self.current_book_path

        save_settings(self.settings)

    def restore_state(self):
        """Restores saved state"""
        # Restore font size (default 24 if no settings)
        font_size = self.settings.get('font_size', 24)
        self.font_size_spinbox.setValue(font_size)

        # Set base font
        font = self.book_text_area.font()
        font.setPixelSize(int(font_size * 1.33))
        self.book_text_area.setFont(font)

        # Restore voice (default Baya - index 1)
        voice_index = self.settings.get('voice_index', 1)
        if 0 <= voice_index < self.voice_combo.count():
            self.voice_combo.setCurrentIndex(voice_index)

        # Restore speed (default 100%)
        speed_value = self.settings.get('speed', 100)
        if 80 <= speed_value <= 140:
            self.speed_spinbox.setValue(speed_value)

        # Restore last book
        if 'last_opened_book' in self.settings:
            book_path = self.settings['last_opened_book']
            if os.path.exists(book_path):
                self.load_specific_book(book_path)

    def load_specific_book(self, file_path):
        """Loads specific book (for state restoration)"""
        success, error = self.parser.parse_book(file_path)
        if success:
            self.current_book_path = file_path
            text = self.parser.get_text()
            self.book_text_area.setHtml(text)
            doc = self.book_text_area.document()
            doc.setUseDesignMetrics(True)
            doc.setDefaultFont(self.book_text_area.font())
            self.setWindowTitle(
                f"Rivlex Reader fb2 v.1.1.0 - {self.parser.get_title()} "
                f"by {self.parser.get_author()}"
            )
            self.status_label.setText("Книга загружена")
            self.play_pause_btn.setEnabled(True)
            self.search_btn.setEnabled(True)
            self.toc_btn.setEnabled(True)

            # Set progress accounting for reading position
            if ('last_read_position' in self.settings and
                    self.settings['last_read_position'] > 0):
                position = self.settings['last_read_position']
                initial_progress = (position / len(text)) * 100
                scaled_value = int(initial_progress * 100)
                self.progress_bar.setValue(scaled_value)

                # Form detailed text as in start_playback
                reading_speed_cps = 15
                elapsed_seconds = position / reading_speed_cps
                time_str = self.format_reading_time(elapsed_seconds)
                progress_text = self.format_progress_line(
                    position, len(text), initial_progress, time_str)
                self.progress_bar.setFormat(progress_text)
            else:
                self.progress_bar.setValue(0)
                # For initial position also show detailed format
                progress_text = self.format_progress_line(
                    0, len(text), 0.0, "0 мин")
                self.progress_bar.setFormat(progress_text)

            # Restore reading position with first sentence highlight
            if ('last_read_position' in self.settings and
                    self.settings['last_read_position'] > 0):
                position = self.settings['last_read_position']
                cursor = self.book_text_area.textCursor()
                cursor.setPosition(min(position, len(text) - 1))

                # Find end of first sentence after position
                text_after_pos = text[position:]
                match = re.search(r'[.!?](\s|$)', text_after_pos)
                if match:
                    end_pos = position + match.end()  # sentence end position
                else:
                    end_pos = len(text)  # if not found, select to end

                length = end_pos - position
                if length <= 0:
                    length = 1  # at least select 1 character

                cursor.movePosition(
                    QTextCursor.Right, QTextCursor.KeepAnchor, length)
                self.book_text_area.setTextCursor(cursor)

                # Center selected text
                def center_delayed():
                    scroll_bar = self.book_text_area.verticalScrollBar()
                    cursor_rect = self.book_text_area.cursorRect(cursor)
                    viewport_height = self.book_text_area.viewport().height()

                    # Use same coordinates as in highlight_text
                    scroll_bar.setValue(
                        scroll_bar.value() +
                        cursor_rect.y() -
                        viewport_height //
                        2)
                    self.book_text_area.ensureCursorVisible()

                    logger.debug(
                        "Центрирование выполнено: cursor_y=%s, "
                        "viewport_height=%s",
                        cursor_rect.y(), viewport_height
                    )

                # Execute centering 200ms after loading
                QTimer.singleShot(200, center_delayed)

                logger.info(
                    "Восстановлена позиция чтения с выделением "
                    "первого предложения: %s", position
                )

            # Save book path and folder
            self.settings['last_opened_book'] = file_path
            self.settings['last_open_folder'] = str(Path(file_path).parent)
            save_settings(self.settings)
            
            # Set focus to text area for hotkeys to work
            self.book_text_area.setFocus()

    def on_voice_changed(self, index):
        """Handler for voice change. Applies on-the-fly."""
        if hasattr(self, 'settings'):
            self.settings['voice_index'] = index
            save_settings(self.settings)

        if self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.update_speaker(index)
            logger.info(
                "Голос изменён на индекс %s "
                "(применится на следующих фрагментах)",
                index)

    def on_speed_changed(self, value):
        """Handler for speed change."""
        self.save_current_state()

        if self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.update_speed(value)
            logger.info(
                "Скорость обновлена до %s%% "
                "(применится на следующих фрагментах)",
                value)

    def on_double_click_jump(self, position):
        """Handler for double click - jump to position and play"""
        logger.info("Переход к позиции: %s", position)

        # Prevent multiple clicks
        if hasattr(self, '_jumping') and self._jumping:
            logger.warning(
                "Переход уже выполняется, игнорируем клик"
            )
            return

        self._jumping = True

        try:
            # Force stop current playback using centralized method
            if self.tts_thread and self.tts_thread.isRunning():
                logger.info("Останавливаем текущий поток...")
                # Use BookReaderWindow.stop_playback() to safely
                # stop and cleanup the TTS thread (avoids races)
                try:
                    self.stop_playback()
                except Exception:
                    logger.exception("Ошибка при остановке потока")

            # Update position in settings
            self.settings['last_read_position'] = position
            save_settings(self.settings)

            # Update progress bar
            full_text = self.book_text_area.toPlainText()
            if full_text:
                progress = (position / len(full_text)) * 100
                scaled_value = int(progress * 100)
                self.progress_bar.setValue(scaled_value)
                self.progress_bar.setFormat(f'{progress:.2f}%')

            # Small delay before starting new thread
            QApplication.processEvents()

            # Automatically start playback from new position
            self.start_playback()

        except Exception:
            logger.exception("Ошибка при переходе")
            self.status_label.setText("Ошибка перехода")
        finally:
            self._jumping = False

    def eventFilter(self, obj, event):
        """Global event filter to handle hotkeys from anywhere"""
        if event.type() == event.Type.KeyPress:
            # Do not handle global hotkeys if focus is in QLineEdit
            focus_widget = QApplication.focusWidget()
            if isinstance(focus_widget, QLineEdit):
                return False  # Let QLineEdit handle input itself

            has_text = bool(self.book_text_area.toPlainText())
            key_text = event.text()
            
            # Search: F or а/А (Russian with Caps Lock)
            if event.key() == Qt.Key_F or key_text in ('а', 'А'):
                if has_text:
                    self.show_search_dialog()
                    return True
                return False
            # TOC: T or е/Е (Russian with Caps Lock)
            elif event.key() == Qt.Key_T or key_text in ('е', 'Е'):
                if has_text:
                    self.show_toc_dialog()
                    return True
                return False
            # Space for Play/Pause
            elif event.key() == Qt.Key_Space:
                # Only if focus is NOT in text edit dialogs
                if has_text and not isinstance(
                    obj, (QLineEdit, QTextEdit, QTextBrowser)
                ):
                    self.toggle_play_pause()
                    return True
                return False
            # Font size change
            elif event.key() in (Qt.Key_Plus, Qt.Key_Equal):
                val = self.font_size_spinbox.value()
                if val < self.font_size_spinbox.maximum():
                    self.font_size_spinbox.setValue(val + 1)
                    return True
                return False
            elif event.key() == Qt.Key_Minus:
                val = self.font_size_spinbox.value()
                if val > self.font_size_spinbox.minimum():
                    self.font_size_spinbox.setValue(val - 1)
                    return True
                return False
        
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        """Handles window resize preserving reading position"""
        super().resizeEvent(event)

        # Check that there is text and saved position
        if (hasattr(self, 'settings') and
                'last_read_position' in self.settings and
                self.book_text_area.toPlainText()):

            position = self.settings['last_read_position']
            text = self.book_text_area.toPlainText()

            if position > 0 and len(text) > 0:
                def restore_position_after_resize():
                    try:
                        cursor = self.book_text_area.textCursor()
                        cursor.setPosition(min(position, len(text) - 1))

                        # Find sentence end for highlighting
                        text_after_pos = text[position:]
                        match = re.search(r'[.!?](\s|$)', text_after_pos)
                        if match:
                            end_pos = position + match.end()
                        else:
                            # select next 50 characters
                            end_pos = min(position + 50, len(text))

                        length = end_pos - position
                        if length <= 0:
                            length = 1

                        cursor.movePosition(
                            QTextCursor.Right, QTextCursor.KeepAnchor, length)
                        self.book_text_area.setTextCursor(cursor)

                        # Center using the same logic
                        scroll_bar = self.book_text_area.verticalScrollBar()
                        cursor_rect = self.book_text_area.cursorRect(cursor)
                        viewport_height = self.book_text_area.viewport().height()

                        scroll_bar.setValue(
                            scroll_bar.value() + cursor_rect.y() - viewport_height // 2)
                        self.book_text_area.ensureCursorVisible()

                        logger.info(
                            "Позиция восстановлена после изменения "
                            "размера: %s", position
                        )

                    except Exception:
                        logger.exception(
                            "Ошибка восстановления позиции при "
                            "изменении размера"
                        )

                # Execute restoration with delay for resize completion
                
                QTimer.singleShot(100, restore_position_after_resize)

    def show_search_dialog(self):
        """Shows search dialog in book"""
        # Close old dialog if exists
        if hasattr(self, 'search_dialog') and self.search_dialog:
            self.search_dialog.close()

        dialog = QDialog(self)
        dialog.setWindowTitle("Поиск в книге")
        dialog.setWindowIcon(get_icon('search'))
        dialog.setMinimumSize(560, 250)
        dialog.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        dialog.setModal(False)

        layout = QVBoxLayout(dialog)

    def show_about_dialog(self):
        """Shows About program dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("О программе")
        dialog.setWindowIcon(get_icon('info'))
        dialog.setMinimumSize(900, 650)
        dialog.setWindowFlags(Qt.Window)
        dialog.setModal(False)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)

        # Header
        title = QLabel(f"{APP_NAME} FB2 v{VERSION}")
        title.setObjectName("aboutTitle")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 27px; font-weight: bold; font-family: 'Atkinson Hyperlegible Next', 'Atkinson Hyperlegible', sans-serif;")
        layout.addWidget(title)

        # Version
        version = QLabel("с русским TTS на базе Silero v5")
        version.setAlignment(Qt.AlignCenter)
        layout.addWidget(version)

        # Separator
        line = QLabel()
        line.setObjectName("separator")
        line.setFixedHeight(2)
        layout.addWidget(line)

        # Text field with information
        info_text = QTextBrowser()
        info_text.setObjectName("aboutInfo")
        info_text.setOpenExternalLinks(True)
        info_text.setStyleSheet("font-size: 22px; font-family: 'Atkinson Hyperlegible Next', 'Atkinson Hyperlegible', sans-serif; color: #FFFFFF; background: transparent; border: none;")


        info_html = f"""
        <style>
            a {{ color: #FFFFFF !important; }}
            h3 {{ font-size: 20px; margin-top: 15px; margin-bottom: 10px; }}
        </style>
        <h3>Основной функционал</h3>
        <ul>
            <li><b>Чтение FB2 книг</b> с сохранением форматирования</li>
            <li><b>Интерактивное оглавление</b> с быстрой навигацией по главам</li>
            <li><b>Озвучивание текста</b> с использованием нейросети <a href="https://github.com/snakers4/silero-models">Silero TTS v5</a></li>
            <li><b>5 голосов:</b> Aidar (м), Baya (ж), Kseniya (ж), Xenia (ж), Eugene (м)</li>
            <li><b>Регулировка скорости</b> от 80% до 140%</li>
            <li><b>Настраиваемый шрифт</b> 12-48 pt (<a href="https://brailleinstitute.org/freefont">Atkinson Hyperlegible Next</a>)</li>
            <li><b>Полнотекстовый поиск</b> с навигацией</li>
            <li><b>Автосохранение позиции</b> чтения</li>
            <li><b>Поддержка кодировок:</b> UTF-8, Windows-1251, KOI8-R</li>
        </ul>

        <h3>Горячие клавиши</h3>
        <table width="100%" border="0" cellspacing="5" cellpadding="5">
            <tr><td><b>Пробел</b></td><td>воспроизведение / пауза</td></tr>
            <tr><td><b>+</b> / <b>-</b></td><td>увеличить / уменьшить шрифт</td></tr>
            <tr><td><b>F</b></td><td>открыть поиск</td></tr>
            <tr><td><b>T</b></td><td>открыть оглавление</td></tr>
            <tr><td><b>Esc</b></td><td>закрыть поиск / оглавление</td></tr>
            <tr><td><b>PageUp / PageDown</b></td><td>прокрутка на страницу</td></tr>
            <tr><td><b>Home / End</b></td><td>в начало / конец текста</td></tr>
            <tr><td><b>Двойной клик</b></td><td>переход к озвучиванию предложения</td></tr>
        </table>

        <h3>Используемые библиотеки</h3>
        <table width="100%" border="0" cellspacing="5" cellpadding="5">
            <tr><td><b>PySide6</b></td><td>LGPL v3</td><td>GUI фреймворк</td></tr>
            <tr><td><b>PyTorch</b></td><td>BSD-3</td><td>Движок нейросети (CPU)</td></tr>
            <tr><td><b><a href="https://github.com/snakers4/silero-models">Silero TTS</a></b></td>
            <td>CC BY-NC-SA 4.0</td>
            <td>Синтез русской речи</td></tr>
            <tr><td><b>lxml</b></td><td>BSD-3</td><td>Парсинг FB2</td></tr>
            <tr><td><b>chardet</b></td><td>LGPL v2.1</td><td>Определение кодировки</td></tr>
            <tr><td><b>num2words</b></td><td>LGPL v3</td><td>Числа в слова</td></tr>
            <tr><td><b>scipy/numpy</b></td><td>BSD-3</td><td>Обработка аудио</td></tr>
            <tr><td><b><a href="https://brailleinstitute.org/freefont">Atkinson Font</a></b></td>
            <td>SIL OFL 1.1</td>
            <td>Улучшенная читаемость</td></tr>
            <tr><td><b><a href="https://lucide.dev/">Lucide Icons</a></b></td><td>MIT</td><td>SVG иконки</td></tr>
        </table>

        <h3>Особенности реализации</h3>
        <ul>
            <li><b>Fade-out:</b> плавное затухание 10 мс для устранения щелчков</li>
            <li><b>CPU-only:</b> работает без GPU на любом процессоре</li>
            <li><b>Оптимизация:</b> использование продвинутых алгоритмов буферизации</li>
            <li><b>Темная тема:</b> зеленые акценты на темно-коричневом</li>
        </ul>

        <h3><b>Лицензия:</b> {LICENSE_INFO}</h3>
        <div>
        Не для коммерческого использования (согласно лицензии моделей Silero)<br><br>
        <b>GitHub:</b> <a href="{GITHUB_URL}">{GITHUB_URL}</a>
        </div>
        """

        info_text.setHtml(info_html)
        layout.addWidget(info_text)

        # Close button
        close_btn = QPushButton("Закрыть")
        close_btn.setObjectName("dialogButton")
        close_btn.clicked.connect(dialog.close)
        close_btn.setFixedHeight(45)
        layout.addWidget(close_btn)

        dialog.show()

    def show_search_dialog(self):
        """Shows search dialog in book"""
        # Close old dialog if exists
        if hasattr(self, 'search_dialog') and self.search_dialog:
            self.search_dialog.close()

        dialog = QDialog(self)
        dialog.setWindowTitle("Поиск в книге")
        dialog.setWindowIcon(get_icon('search'))
        dialog.setMinimumSize(560, 250)
        dialog.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        dialog.setModal(False)

        layout = QVBoxLayout(dialog)

        # Search query input field
        search_label = QLabel("Введите текст для поиска:")
        layout.addWidget(search_label)

        search_input = QLineEdit()
        search_input.setPlaceholderText("Поисковый запрос...")
        search_input.setMinimumHeight(50)
        layout.addWidget(search_input)

        # Full-width Find button
        find_btn = QPushButton(" Найти")
        find_btn.setObjectName("dialogButton")
        find_btn.setIcon(get_icon('search'))
        find_btn.setIconSize(QSize(20, 20))
        find_btn.setDefault(True)
        layout.addWidget(find_btn)

        # Navigation buttons in separate row
        nav_layout = QHBoxLayout()

        find_prev_btn = QPushButton(" Предыдущее")
        find_prev_btn.setObjectName("dialogButton")
        find_prev_btn.setIcon(get_icon('chevron-left'))
        find_prev_btn.setIconSize(QSize(20, 20))

        find_next_btn = QPushButton(" Следующее")
        find_next_btn.setObjectName("dialogButton")
        find_next_btn.setIcon(get_icon('chevron-right'))
        find_next_btn.setIconSize(QSize(20, 20))

        nav_layout.addWidget(find_prev_btn)
        nav_layout.addWidget(find_next_btn)

        layout.addLayout(nav_layout)

        # Results label
        result_label = QLabel("")
        result_label.setObjectName("searchResultLabel")
        layout.addWidget(result_label)

        def do_search():
            query = search_input.text()
            if not query:
                result_label.setText("⚠️ Введите текст для поиска")
                return

            # Search from document beginning
            cursor = self.book_text_area.textCursor()
            cursor.movePosition(QTextCursor.Start)
            self.book_text_area.setTextCursor(cursor)

            found = self.book_text_area.find(query)
            if found:
                result_label.setText(f"✅ Найдено: \"{query}\"")
                self.book_text_area.setFocus()
            else:
                result_label.setText(f"❌ Не найдено: \"{query}\"")

        def find_next():
            query = search_input.text()
            if not query:
                result_label.setText("⚠️ Введите текст для поиска")
                return

            found = self.book_text_area.find(query)
            if found:
                result_label.setText(f"✅ Найдено: \"{query}\"")
                self.book_text_area.setFocus()
            else:
                result_label.setText(f"❌ Больше не найдено: \"{query}\"")

        def find_prev():
            query = search_input.text()
            if not query:
                result_label.setText("⚠️ Введите текст для поиска")
                return

            found = self.book_text_area.find(query, QTextDocument.FindBackward)
            if found:
                result_label.setText(f"✅ Найдено: \"{query}\"")
                self.book_text_area.setFocus()
            else:
                result_label.setText(f"❌ Больше не найдено: \"{query}\"")

        find_btn.clicked.connect(do_search)
        find_next_btn.clicked.connect(find_next)
        find_prev_btn.clicked.connect(find_prev)
        search_input.returnPressed.connect(do_search)

        self.search_dialog = dialog
        dialog.show()

    def show_toc_dialog(self):
        """Shows table of contents dialog"""
        if not hasattr(self.parser, 'toc') or not self.parser.toc:
            QMessageBox.information(
                self, "Оглавление", 
                "Оглавление пусто или не найдено")
            return
        
        # Close old dialog if exists
        if hasattr(self, 'toc_dialog') and self.toc_dialog:
            self.toc_dialog.close()

        dialog = QDialog(self)
        dialog.setWindowTitle("Оглавление")
        dialog.setWindowIcon(get_icon('list'))
        dialog.setMinimumSize(700, 500)
        dialog.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        dialog.setModal(False)

        layout = QVBoxLayout(dialog)

        # Title
        title_label = QLabel("Содержание книги:")
        layout.addWidget(title_label)

        # List widget for TOC
        toc_list = QListWidget()

        # Add TOC items
        for title, position in self.parser.toc:
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, position)
            toc_list.addItem(item)

        # Select the chapter that contains the last saved position
        try:
            last_pos = int(self.settings.get('last_read_position', 0))
            selected_row = 0
            for i, (_t, pos) in enumerate(self.parser.toc):
                if pos <= last_pos:
                    selected_row = i
                else:
                    break
            # Ensure selection and visibility
            if toc_list.count() > 0:
                toc_list.setCurrentRow(selected_row)
                item = toc_list.item(selected_row)
                if item:
                    toc_list.scrollToItem(item)
        except Exception:
            # Fall back to default (first item) on any error
            pass

        # Handle click - jump to chapter
        def on_toc_click(item):
            position = item.data(Qt.UserRole)
            if position is not None:
                # Stop playback if running
                if self.tts_thread and self.tts_thread.isRunning():
                    self.stop_playback()
                
                # Jump to position
                full_text = self.book_text_area.toPlainText()
                if position < len(full_text):
                    cursor = self.book_text_area.textCursor()
                    cursor.setPosition(position)
                    
                    # Find sentence end
                    text_after = full_text[position:]
                    match = re.search(r'[.!?](\s|$)', text_after)
                    if match:
                        end_pos = position + match.end()
                    else:
                        end_pos = min(position + 100, len(full_text))
                    
                    length = end_pos - position
                    if length > 0:
                        cursor.movePosition(
                            QTextCursor.Right, 
                            QTextCursor.KeepAnchor, 
                            length
                        )
                    
                    self.book_text_area.setTextCursor(cursor)
                    self.book_text_area.ensureCursorVisible()
                    
                    # Center
                    scroll_bar = self.book_text_area.verticalScrollBar()
                    cursor_rect = self.book_text_area.cursorRect(cursor)
                    viewport_height = self.book_text_area.viewport().height()
                    scroll_bar.setValue(
                        scroll_bar.value() + cursor_rect.y() - 
                        viewport_height // 2
                    )
                    
                    # Update position
                    self.settings['last_read_position'] = position
                    save_settings(self.settings)
                    
                    # Give focus back to main window
                    self.book_text_area.setFocus()

        toc_list.itemDoubleClicked.connect(on_toc_click)
        layout.addWidget(toc_list)

        # Close button
        close_btn = QPushButton("Закрыть")
        close_btn.setObjectName("dialogButton")
        close_btn.clicked.connect(dialog.close)
        close_btn.setFixedHeight(45)
        layout.addWidget(close_btn)

        self.toc_dialog = dialog
        dialog.show()


def setup_dark_theme(app):
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(27, 25, 10))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(41, 38, 16))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(35, 32, 21))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(0, 50, 0))
    dark_palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(dark_palette)
    app.setStyle(QStyleFactory.create("Fusion"))


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_DisableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_Use96Dpi, True)

    # Set AppUserModelID on Windows to stabilize the taskbar icon.
    if sys.platform == 'win32' and windll is not None:
        try:
            windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "com.rivlex.rivlexreader"
            )
        except Exception:
            pass

    app = QApplication(sys.argv)
    # Set application icon for taskbar
    icon_path = get_icon_path()
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    setup_dark_theme(app)
    main_window = BookReaderWindow()
    main_window.show()
    sys.exit(app.exec())
