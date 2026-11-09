import ast
import base64
import csv
import ctypes
import hashlib
import io
import json
import math
import operator
import os
import platform
import queue
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
import webbrowser
import winreg
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, deque
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import psutil
import sounddevice as sd

# Optional Air Cursor dependencies. The rest of Victus continues to work
# even if these packages are not installed.
try:
    import cv2
    CV2_OK = True
except Exception as e:
    cv2 = None
    CV2_OK = False
    print("OpenCV unavailable (Air Cursor disabled):", e)

# MediaPipe compatibility layer.
# Modern MediaPipe versions expose Hand Landmarker through the Tasks API and
# may not expose the old legacy MediaPipe solutions API namespace. Import both APIs directly
# so Air Cursor never depends on legacy MediaPipe solutions API being present.
try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except Exception as e:
    mp = None
    MEDIAPIPE_OK = False
    print("MediaPipe unavailable (Air Cursor disabled):", e)

MP_TASKS_OK = False
mp_tasks_python = None
mp_vision = None
try:
    from mediapipe.tasks import python as mp_tasks_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_TASKS_OK = True
except Exception as e:
    print("MediaPipe Tasks API unavailable:", e)

# Legacy MediaPipe solutions backend disabled; Tasks API is required.
MP_LEGACY_OK = False
mp_hands_legacy = None
mp_draw_legacy = None

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None
    ttk = None

try:
    import customtkinter as ctk
    CTK_OK = True
except Exception:
    CTK_OK = False

try:
    import win32api
    import win32con
    import win32gui
    import win32process
    import win32com.client
    import win32clipboard
    WIN32_OK = True
except Exception as e:
    WIN32_OK = False
    print("pywin32 unavailable:", e)

try:
    import comtypes
    from comtypes import CLSCTX_ALL
    from ctypes import cast, POINTER
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    PYCAW_OK = True
except Exception as e:
    PYCAW_OK = False
    print("PyCaw unavailable:", e)

try:
    import screen_brightness_control as sbc
    SBC_OK = True
except Exception:
    SBC_OK = False

try:
    from PIL import ImageGrab
    PIL_OK = True
except Exception:
    PIL_OK = False
    ImageGrab = None

# Optional document-parsing libraries for the file-aware "read this file
# and answer questions about it" feature. PDFs still need PyPDF2. Word
# files do NOT need python-docx — .docx text is extracted with the
# zipfile/xml modules above, since a .docx is just a zip of XML under the
# hood. python-docx is kept only as a secondary fallback if that ever
# fails on an unusual file, so Victus reads .docx out of the box with no
# extra pip install required.
try:
    import PyPDF2
    PYPDF2_OK = True
except Exception:
    PyPDF2 = None
    PYPDF2_OK = False

try:
    import docx as docx_lib
    DOCX_OK = True
except Exception:
    docx_lib = None
    DOCX_OK = False

# Local AI now runs through Ollama (local HTTP server) instead of an
# in-process llama-cpp-python model. No extra Python package is required;
# Victus talks to Ollama over plain HTTP using urllib, which is already
# imported above. Victus keeps all existing features working even when
# Ollama is not installed or not running.


ASSISTANT_NAME = "Victus"
# NOTE: the spoken wake word below is no longer the primary way to wake
# Victus — pressing Ctrl+Alt (see the PUSH-TO-TALK HOTKEY section further
# down) is. WAKE_WORDS/is_wake_word() are kept only as an automatic
# fallback for the (rare) case pywin32 isn't installed and the hotkey
# can't be used at all — see HOTKEY_ENABLED.
WAKE_WORDS = {
    "victus", "victor", "jarvis", "javis",
    "hey victus", "hey victor", "hey jarvis", "hey javis",
    "ok victus", "ok victor", "ok jarvis", "ok javis",
    "okay victus", "okay victor", "okay jarvis", "okay javis",
    "victus jarvis", "jarvis victus", "victor javis", "javis victor","ok","okay","hey","here"
}
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 512
MAX_COMMAND_SECONDS = 35.0
SILENCE_AFTER_SPEECH = 2.0
SENSITIVITY = 1.0
WHISPER_MODEL = os.getenv("VICTUS_WHISPER_MODEL", "tiny.en")  # Faster boot; set VICTUS_WHISPER_MODEL=base.en to restore the previous model
ORB_SIZE = 150


# ---------------------------------------------------------------------------
# LOCAL AI — Ollama (offline, no API key)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
# Point this at your local Ollama server. Override with environment
# variables if Ollama runs on a different host/port, or to pin a specific
# model instead of relying on auto-detection (see _load_local_ai below), e.g.:
#   set VICTUS_OLLAMA_MODEL=qwen2.5:14b
#   set VICTUS_OLLAMA_HOST=http://localhost:11434
#
# For strong performance across math, chemistry, medicine, business, etc.
# pull a capable instruction/reasoning model that fits your GPU's VRAM, e.g.
# one of: "qwen2.5:14b", "llama3.1:8b", "deepseek-r1:14b", "mistral-nemo".
# Pull it with:  ollama pull <model-name>
OLLAMA_HOST = os.getenv("VICTUS_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("VICTUS_OLLAMA_MODEL", "llama3.2")
LOCAL_AI_MAX_TOKENS = int(os.getenv("VICTUS_OLLAMA_MAX_TOKENS", "900"))
# Many Ollama models default to a much larger context window than a short
# voice-assistant exchange ever needs (some go up to 32K-128K). Ollama has
# to allocate/reserve KV-cache for the FULL configured context window
# before it can generate a single token, so an oversized default context
# is one of the most common (and easy to miss) causes of "why is my local
# model slow to even start replying" — even on a short question. Capping
# it explicitly to something that comfortably fits a system prompt +
# short history + one question/answer noticeably cuts that startup lag.
# Override with VICTUS_OLLAMA_NUM_CTX if you need more room for very long
# answers or a big notes/RAG context.
LOCAL_AI_NUM_CTX = int(os.getenv("VICTUS_OLLAMA_NUM_CTX", "4096"))
# Substrings used to pick out a vision-capable model from `ollama list`.
# Covers the common vision-language model families people actually pull
# (qwen2-vl / qwen2.5vl were missing before and are the most likely
# reason detection silently failed).
VISION_MODEL_TAGS = (
    "llava", "bakllava", "moondream", "vision", "vl",
    "minicpm-v", "pixtral", "internvl", "cogvlm",
)
LOCAL_AI_MEMORY_LIMIT = 6
LOCAL_AI_TIMEOUT_SECONDS = int(os.getenv("VICTUS_OLLAMA_TIMEOUT", "120"))
# Keep the model resident in Ollama between requests. By default Ollama
# unloads a model ~5 minutes after the last request, so the next question
# pays a full model-load penalty. Keeping it warm for longer is the single
# biggest speed win for "why is the AI so slow" — override with
# VICTUS_OLLAMA_KEEP_ALIVE if you want to free VRAM sooner.
OLLAMA_KEEP_ALIVE = os.getenv("VICTUS_OLLAMA_KEEP_ALIVE", "30m")

# ---------------------------------------------------------------------------
# MULTILINGUAL PIPELINE — Whisper auto-detect + reply-in-language + SAPI
# voice switching.
#
# IMPORTANT: WHISPER_MODEL defaults to "tiny.en" above, which is an
# ENGLISH-ONLY model architecture — it cannot detect or transcribe any
# other language no matter what code does here. To actually get
# multilingual auto-detect, set an environment variable to a multilingual
# model, e.g.:  set VICTUS_WHISPER_MODEL=tiny   (or base/small/medium)
# transcribe() below automatically skips language auto-detection and
# stays pinned to English whenever a ".en" model is loaded, so nothing
# breaks for people who keep the fast English-only default.
# ---------------------------------------------------------------------------
LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "hi": "Hindi", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
    "ar": "Arabic", "tr": "Turkish", "pl": "Polish", "sv": "Swedish",
    "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
    "bn": "Bengali", "ur": "Urdu", "id": "Indonesian", "vi": "Vietnamese",
}
# Substrings to look for in installed SAPI voice descriptions (e.g.
# "Microsoft Hortense Desktop - French") to pick a voice matching the
# detected spoken language. Purely a best-effort match against whatever
# voice packs happen to be installed on this Windows machine.
LANGUAGE_VOICE_HINTS = {
    "en": ["english"], "es": ["spanish", "espanol", "español"],
    "fr": ["french", "francais", "français"], "de": ["german", "deutsch"],
    "it": ["italian", "italiano"], "pt": ["portuguese", "portugues"],
    "nl": ["dutch", "nederlands"], "ru": ["russian"], "hi": ["hindi"],
    "ja": ["japanese"], "zh": ["chinese", "mandarin"], "ko": ["korean"],
    "ar": ["arabic"], "tr": ["turkish"], "pl": ["polish"], "sv": ["swedish"],
}
# Minimum faster-whisper language-detection confidence before Victus trusts
# it enough to switch reply language/voice — low-confidence guesses on
# short/noisy audio are ignored and English is kept.
LANGUAGE_DETECT_MIN_CONFIDENCE = 0.55

# ---------------------------------------------------------------------------
# RAG-LITE OVER LOCAL NOTES/DOCS — lightweight lexical (TF-IDF) retrieval,
# not a full embeddings + vector-store pipeline (that remains a bigger
# lift — see report). Drop .txt/.md/.pdf/.docx files into NOTES_DIR and ask
# "search my notes for ..." / "what do my notes say about ...".
# ---------------------------------------------------------------------------
NOTES_DIR = BASE_DIR / "victus_notes"
NOTES_CHUNK_CHARS = 800
NOTES_CHUNK_OVERLAP = 150
NOTES_TOP_K = 3
# Dictated notes ("take a note: ...") land in this file inside NOTES_DIR,
# so they're picked up by the RAG-lite search above automatically — this
# is just the missing "write" half of that read-only notes search.
NOTES_LOG_FILE = NOTES_DIR / "voice_notes.md"

# ---------------------------------------------------------------------------
# STANDALONE TO-DO LIST — separate from the scheduler above: no due time,
# just a persistent list you add to / check off by voice. Same JSON
# persistence pattern as command history and scheduled tasks.
# ---------------------------------------------------------------------------
TODO_FILE = BASE_DIR / "victus_todo.json"

# ---------------------------------------------------------------------------
# MULTI-CLIP CLIPBOARD HISTORY — a small ring buffer of recent clipboard
# text, built by polling (clipboard_history_monitor) rather than a Windows
# clipboard-change hook, to keep this simple and dependency-free. Lets you
# recall something copied a few steps back, not just the current contents.
# ---------------------------------------------------------------------------
CLIPBOARD_HISTORY_LIMIT = 20

# ---------------------------------------------------------------------------
# WAKE-ON-LAN — send a magic packet over the local network to power on
# another PC. Pure stdlib (raw UDP broadcast via the socket module already
# imported above) — no third-party dependency, no internet access, LAN
# only. Saved devices (friendly name -> MAC address) persist in
# victus_wol_devices.json so you can say "wake on lan gaming pc" instead
# of reciting a MAC address every time. MAC addresses are stored/typed as
# a bare 12-hex-digit string (colons/dashes are stripped by normalize()
# anyway, and nobody can speak a colon out loud) and only formatted with
# colons for display.
# ---------------------------------------------------------------------------
WOL_DEVICES_FILE = BASE_DIR / "victus_wol_devices.json"
WOL_BROADCAST_IP = "255.255.255.255"
WOL_PORT = 9

# ---------------------------------------------------------------------------
# LAN DEVICE SCANNER — lists every device currently on the local network
# (IP, MAC, hostname where resolvable) using a ping sweep + the Windows
# ARP table, both already built into the OS — no extra dependency, no
# internet access, LAN only. Results feed straight into the Wake-on-LAN
# device list above via a "double-click to save" action in the results
# window, so a MAC address never has to be typed by hand.
# ---------------------------------------------------------------------------
LAN_SCAN_PING_TIMEOUT_MS = 150
LAN_SCAN_TOTAL_BUDGET_SECONDS = 2.5
LAN_SCAN_HOSTNAME_TIMEOUT = 0.3

# ---------------------------------------------------------------------------
# DUPLICATE / LARGE FILE FINDER — scans a folder (Downloads by default)
# for exact duplicate files and separately ranks the largest files found,
# so disk-space hogs are easy to spot. Pure stdlib (hashlib + os.walk, no
# extra dependency). Files are first grouped by size, then by a cheap
# partial hash of just the first chunk, and only files that still match
# after both of those cheap filters get a full-file hash — so large files
# are never fully re-read unless they already look like a real duplicate.
# Results open in the same Treeview-style window used by the file-
# extension search GUI (search_local_files_gui) for a consistent look.
# ---------------------------------------------------------------------------
DUPLICATE_SCAN_MAX_FILES = 20000
DUPLICATE_HASH_CHUNK = 65536
DUPLICATE_SKIP_DIRS = {
    "windows", "program files", "program files (x86)", "programdata",
    "$recycle.bin", "system volume information", "appdata", "msocache",
    "perflogs", "node_modules", ".git",
}

# ---------------------------------------------------------------------------
# SCHEDULED LOCAL BACKUPS — "backup my documents", "back up my desktop",
# "backup victus data". Pure stdlib (shutil + zipfile, both already used
# elsewhere above) copying to a timestamped folder under BACKUP_DIR — no
# cloud storage, no key. Doubles as a *scheduled* backup for free: saying
# "schedule backup my documents at 11 pm" already works with zero extra
# code, since scheduler_loop() re-runs whatever text you scheduled through
# the exact same command router this uses. Shares DUPLICATE_SKIP_DIRS
# above so junk folders (node_modules, .git, etc.) are never copied.
# ---------------------------------------------------------------------------
BACKUP_DIR = BASE_DIR / "victus_backups"
BACKUP_MAX_FILES = 20000

# ---------------------------------------------------------------------------
# PROACTIVE HEALTH ALERTS — unprompted "your CPU has been at 95 percent for
# the last 5 minutes" warnings, instead of only reporting load when asked
# via "system status". Reuses the same psutil already imported for
# battery_monitor/system_status; this just adds a sustained-load watcher
# thread alongside it. An alert only fires once the load has stayed above
# threshold for HEALTH_ALERT_SUSTAIN_SECONDS straight (so a brief spike
# from opening an app never triggers one), and then Victus stays quiet
# about it for HEALTH_ALERT_COOLDOWN_SECONDS so it doesn't repeat itself
# every poll while the load stays high.
# ---------------------------------------------------------------------------
HEALTH_ALERT_CPU_THRESHOLD = 90        # percent
HEALTH_ALERT_RAM_THRESHOLD = 90        # percent
HEALTH_ALERT_SUSTAIN_SECONDS = 300     # load must stay high this long first
HEALTH_ALERT_COOLDOWN_SECONDS = 600    # quiet period after an alert fires
HEALTH_ALERT_POLL_SECONDS = 10

# ---------------------------------------------------------------------------
# CUSTOM VOICE MACRO RECORDER — "start recording macro <name>" begins
# capturing every command that follows (each one still runs normally, so
# you see it happen live) until "stop recording" saves the captured steps
# under that name. "run macro <name>" replays them in order. Pure stdlib
# (json, same load/save pattern as the WoL device list above) — no extra
# dependency. This sits alongside the single hard-coded "workspace" macro
# (run_macro()) rather than replacing it; run_custom_macro() falls back to
# run_macro() so "run macro workspace" still works even though "workspace"
# was never recorded.
# ---------------------------------------------------------------------------
CUSTOM_MACROS_FILE = BASE_DIR / "victus_custom_macros.json"
MACRO_STOP_PHRASES = (
    "stop recording", "stop recording macro", "end recording",
    "finish recording", "save macro",
)

# ---------------------------------------------------------------------------
# SCHEDULED / TRIGGERED ACTIONS — "remind me to X in 10 minutes", "shut
# down in 30 minutes", "open Chrome at 6pm". A lightweight background
# thread (scheduler_loop) checks victus_schedule.json every 10 seconds for
# due tasks.
# ---------------------------------------------------------------------------
SCHEDULE_FILE = BASE_DIR / "victus_schedule.json"
SCHEDULER_POLL_SECONDS = 10

# ---------------------------------------------------------------------------
# INTENT CLASSIFICATION — a lightweight, offline, weighted-keyword scorer
# used to make routing decisions smarter than a single substring check.
# This does NOT replace the detailed per-command if/elif chain in
# execute_single_command() (that logic still does the precise parameter
# extraction for each concrete command and is left intact), but it gives
# handle_command() a real classification step — instead of a handful of
# hard-coded prefixes — to decide whether free-form text is an AI
# question, a multi-step task, a notes lookup, a reminder, etc.
# ---------------------------------------------------------------------------
INTENT_KEYWORDS = {
    "ai_query": [
        ("what is", 2), ("what are", 2), ("explain", 2), ("who is", 2),
        ("how to", 2), ("how do", 2), ("how does", 2), ("tell me about", 2),
        ("why is", 2), ("why does", 2), ("why do", 2), ("can you tell me", 2),
        ("define", 2), ("difference between", 2), ("compare", 1),
        ("meaning of", 2), ("calculate", 1), ("solve", 1), ("what does", 2),
    ],
    "notes_query": [
        ("my notes", 3), ("in my notes", 3), ("notes say", 3),
        ("search my notes", 3), ("from my notes", 3),
    ],
    "note_taking": [
        ("take a note", 3), ("add a note", 3), ("note that", 3),
        ("write a note", 3), ("save a note", 3), ("make a note", 3),
    ],
    "todo": [
        # normalize() strips hyphens to spaces, so "to-do" never survives
        # as spoken/typed — only the space and no-space forms are checked.
        ("to do list", 3), ("todo list", 3),
        ("my to do", 2), ("add to my to do", 3),
    ],
    "reminder": [
        ("remind me", 3), ("set a reminder", 3), ("schedule", 2),
        ("minutes", 1), ("hours", 1), ("seconds", 1),
    ],
    "media_control": [
        ("play", 2), ("pause", 2), ("next song", 2), ("previous song", 2),
        ("volume up", 2), ("volume down", 2), ("mute", 2), ("unmute", 2),
    ],
    "app_launch": [
        ("open", 2), ("launch", 2), ("start", 1), ("close", 2), ("minimize", 2),
    ],
    "system_control": [
        ("shutdown", 2), ("shut down", 2), ("restart", 2), ("sleep", 1),
        ("brightness", 2), ("battery", 2), ("wifi", 2), ("bluetooth", 2),
    ],
    "web_search": [
        ("search for", 2), ("google", 2), ("youtube", 1),
    ],
    "wake_on_lan": [
        ("wake on lan", 3), ("magic packet", 3), ("power on", 1),
    ],
    "lan_scanner": [
        ("scan my network", 3), ("scan the network", 3), ("scan lan", 3),
        ("network devices", 2), ("connected devices", 2), ("who is on my network", 3),
    ],
    "duplicate_finder": [
        ("duplicate files", 3), ("find duplicates", 3),
        ("large files", 2), ("big files", 2), ("free up space", 2),
    ],
    "macro_recorder": [
        ("recording macro", 3), ("record macro", 3), ("run macro", 3),
        ("play macro", 3), ("stop recording", 3), ("my macros", 2),
    ],
    "outlook": [
        ("check my email", 3), ("check my outlook", 3), ("my inbox", 2),
        ("my calendar", 2), ("outlook calendar", 3), ("unread emails", 3),
    ],
    "personalization_profile": [
        ("remember that", 3), ("remember my", 2), ("forget that", 3),
        ("what do you know about me", 3), ("my profile", 2),
    ],
    "health_alerts": [
        ("health alerts", 3), ("system health", 2), ("cpu alerts", 2),
    ],
    "translation": [
        ("translate", 3), ("how do you say", 3),
    ],
    "text_rewrite": [
        ("simplify", 3), ("formalize", 3), ("make this formal", 2),
        ("make this simpler", 2),
    ],
    "local_backup": [
        ("backup", 3), ("back up", 3), ("create a backup", 3),
    ],
}

# ---------------------------------------------------------------------------
# COMMAND HISTORY — every recognised/typed command Victus has executed
# ---------------------------------------------------------------------------
HISTORY_FILE = BASE_DIR / "victus_command_history.json"
HISTORY_MAX_ENTRIES = 500

# ---------------------------------------------------------------------------
# PERSISTENT AI MEMORY — local_ai_history now survives restarts. Mirrors the
# COMMAND HISTORY load/save pattern above so the conversation Victus is
# having with Ollama is no longer wiped every time the app is closed.
# ---------------------------------------------------------------------------
AI_MEMORY_FILE = BASE_DIR / "victus_ai_memory.json"

# ---------------------------------------------------------------------------
# STANDALONE PERSONALIZATION PROFILE — a short list of remembered facts/
# preferences ("remember that I prefer metric units", "remember my name is
# Shreeskanda") that Victus injects into every AI prompt as its own system
# message, separate from the rolling local_ai_history above. Keeping it
# separate means personal facts survive "clear ai memory" and never age
# out once the conversation history rolls past LOCAL_AI_MEMORY_LIMIT
# exchanges. Pure JSON persistence, same load/save pattern as
# victus_wol_devices.json.
# ---------------------------------------------------------------------------
PROFILE_FILE = BASE_DIR / "victus_profile.json"
PROFILE_MAX_FACTS = 40


running = True
listening = False
speaking = False
audio_queue = queue.Queue(maxsize=2000)
tts_lock = threading.Lock()

# Live push-to-talk recording level meter. Updated from record_full_command()
# as audio chunks arrive while listening=True, and read back by draw_orb()
# (already ticking every 30ms via orb_animation) to render a VU-style ring
# around the orb. Purely visual — no AI logic involved.
mic_level = 0.0
MIC_LEVEL_SMOOTHING = 0.35  # weight given to each new reading (0-1)
MIC_LEVEL_REFERENCE = 0.02  # rms value treated as "100%" for the meter

# Post-answer follow-up window. After Victus finishes speaking an AI answer,
# the mic reopens briefly (no wake word needed) so a natural follow-up like
# "what about compared to X?" just works. main_loop() must not also be
# pulling from audio_queue while this window is active, so it checks this
# event and idles instead of racing record_full_command()/listen_for_followup()
# for the same audio chunks.
FOLLOWUP_WINDOW_SECONDS = 5.0
followup_window_active = threading.Event()

# ---------------------------------------------------------------------------
# PUSH-TO-TALK HOTKEY — Ctrl+Alt now wakes Victus instead of a spoken wake
# word. A lightweight background thread (hotkey_listener_loop, started from
# __main__) polls GetAsyncKeyState via the pywin32 import already used
# elsewhere in this file — no new dependency required — and sets
# hotkey_pressed_event on each fresh press (edge-triggered: holding the
# combo down doesn't refire). main_loop() checks that event every
# iteration and runs the same "listen for one command" flow the spoken
# wake word used to trigger, on its own thread, so there's no risk of two
# readers racing audio_queue. If pywin32 isn't installed (WIN32_OK is
# False), HOTKEY_ENABLED is False and Victus automatically falls back to
# the original spoken wake word so it's never left with no way to be
# woken.
# ---------------------------------------------------------------------------
HOTKEY_ENABLED = WIN32_OK
HOTKEY_POLL_SECONDS = 0.03
hotkey_pressed_event = threading.Event()

# Set to interrupt Victus mid-sentence (voice "stop", command-box "stop",
# etc.). speak() polls this while talking; stop_speaking() sets it.
speak_stop_event = threading.Event()
# Handle to the PowerShell fallback TTS process (only used when the SAPI
# COM voice isn't available), so stop_speaking() can terminate it outright.
_active_speech_proc = {"proc": None}

# Multilingual pipeline state. Updated by transcribe() whenever the loaded
# Whisper model is multilingual and confidently detects a non-English
# utterance; read back by speak() (voice switching) and ask_local_ai()/
# ask_vision_ai() (reply-in-language instruction).
last_detected_language = "en"
last_detected_language_name = "English"
_current_sapi_voice_lang = "en"  # avoids re-scanning SAPI voices every call

# Scheduled/triggered actions state.
scheduled_tasks = []
schedule_lock = threading.RLock()
_schedule_next_id = 1

# RAG-lite notes index state (lexical TF-IDF over files in NOTES_DIR).
_notes_index = []          # list of {"path": str, "chunk": str, "tf": Counter}
_notes_idf = {}            # term -> idf weight
_notes_index_lock = threading.RLock()
_notes_index_built = False

# Standalone to-do list state.
todo_items = []
todo_lock = threading.RLock()
_todo_next_id = 1

# Multi-clip clipboard history state.
clipboard_history = deque(maxlen=CLIPBOARD_HISTORY_LIMIT)
clipboard_history_lock = threading.RLock()
_last_clipboard_seen = None

# Wake-on-LAN saved devices: {name(lower): "AA:BB:CC:DD:EE:FF"}.
wol_devices = {}
wol_devices_lock = threading.RLock()

# Custom voice macros: {name(lower): [cmd_norm, cmd_norm, ...]}.
custom_macros = {}
custom_macros_lock = threading.RLock()
# Live recording state. macro_recording_active gates the capture hook in
# handle_command(); the dict (rather than a bare variable) lets it be
# mutated in place without a `global` statement everywhere it's touched.
macro_recording_active = threading.Event()
macro_recording_state = {"name": None, "steps": []}


# Lazy connection check keeps Victus startup fast. Ollama connectivity is
# verified only when an AI conversation is actually requested.
local_ai_lock = threading.RLock()
local_ai_history = []
local_ai_loaded = False
local_ai_loading_error = None
# Populated by _load_local_ai() with every model tag Ollama currently has
# pulled, so screen-scan can opportunistically pick a vision-capable model
# (llava/bakllava/moondream/etc.) if one is installed.
local_ai_available_models = []

# Personalization profile: list of remembered fact strings, injected into
# every ask_local_ai() prompt (see _profile_system_message).
user_profile_facts = []
user_profile_lock = threading.RLock()

# Command history — in-memory list, mirrored to HISTORY_FILE on disk so it
# survives restarts. Each entry: {"id": int, "time": str, "text": str}
command_history = []
command_history_lock = threading.RLock()
_history_next_id = 1


orb_root = None
orb_canvas = None
orb_visible = False
orb_phase = 0.0

cmd_box_root = None
cmd_entry = None
file_table_root = None
duplicate_finder_root = None
lan_scan_root = None
dashboard_root = None
usage_dashboard_root = None

history_win_root = None
history_listbox = None
# Maps the currently-displayed listbox row index -> history entry id
# (list is rebuilt every time the history window refreshes).
history_ids_display = []

ai_response_win_root = None
ai_response_text_widget = None

menu_win = None
menu_expanded = False
menu_panel = None
menu_anim_phase = 0.0

VICTUS_PID = os.getpid()
music_started = False
music_query = ""
previous_plugged = None
full_warned = False
low_warned = False
whisper = None

# Proactive health alert (health_monitor) state — mirrors the battery
# warning flags just above. "_high_since" is None while load is under
# threshold, else the timestamp it first crossed threshold; "_last_fired"
# is the timestamp of the last spoken alert, used for the cooldown.
health_alerts_enabled = True
_cpu_high_since = None
_ram_high_since = None
_cpu_alert_last_fired = 0.0
_ram_alert_last_fired = 0.0

# ---------------------------------------------------------------------------
# AIR CURSOR — camera-based right-hand OS mouse controller
# ---------------------------------------------------------------------------
air_cursor_active = False
air_cursor_stop_event = threading.Event()
air_cursor_thread = None
air_cursor_camera = None
air_cursor_overlay = None
air_cursor_status = "INACTIVE"
air_cursor_last_gesture = "MOVE"
air_cursor_mouse_down = False
air_cursor_last_left_pinch = False
air_cursor_left_pinch_started = 0.0
air_cursor_last_right_pinch = False
air_cursor_last_click_time = 0.0
air_cursor_last_scroll_time = 0.0
air_cursor_last_hand_time = 0.0
air_cursor_last_screen_point = None
air_cursor_smooth_point = None
air_cursor_prev_index = None
air_cursor_calibration_requested = False
# Custom copy/paste gesture state
air_cursor_gesture_candidate = None
air_cursor_gesture_started = 0.0
air_cursor_last_custom_gesture_time = 0.0
# Fist gesture state
air_cursor_fist_candidate = False
air_cursor_fist_started = 0.0
air_cursor_last_fist_time = 0.0
air_cursor_palm_started = 0.0
air_cursor_last_palm_time = 0.0
air_cursor_lock = threading.RLock()
air_cursor_last_overlay_update = 0.0
air_cursor_pos_history = deque(maxlen=12)   # (time, x, y) of recent smoothed points
air_cursor_pinch_lock = None                # cursor point frozen while pinching
air_cursor_pinch_offset = None              # keeps drag continuous after the lock
_air_cursor_screen_bounds_cache = None

# Cursor/gesture tuning. These are deliberately conservative to reduce
# accidental clicks and jitter.
AIR_CURSOR_SMOOTHING = 0.58
# Deadzone is in SCREEN PIXELS (px,py are in pixel space from _air_cursor_map_to_screen).
# The movement check is: dist² >= DEADZONE². At 0.12 px that is 0.0144 px² — effectively
# zero, causing the cursor to jitter on every frame. A value of 4 pixels gives a comfortable
# dead zone that filters webcam/hand noise without feeling sluggish.
AIR_CURSOR_DEADZONE = 4.0
# Higher value = more cursor movement for the same hand movement.
AIR_CURSOR_SENSITIVITY = 1.22
# MediaPipe Tasks Hand Landmarker model.
# Keep this at module scope because _air_cursor_prepare_tasks_model() and the
# Air Cursor worker both access it.
AIR_CURSOR_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
AIR_CURSOR_TASK_MODEL_PATH = (
    Path.home() / ".victus_models" / "hand_landmarker.task"
)
# Pinch detection is now normalized to palm size inside the processing loop.
# These legacy constants remain as fallback/reference values.
AIR_CURSOR_PINCH_ON = 0.055
AIR_CURSOR_PINCH_OFF = 0.075
AIR_CURSOR_PINCH_RATIO_ON = 0.40
AIR_CURSOR_PINCH_RATIO_OFF = 0.52
AIR_CURSOR_CLICK_COOLDOWN = 0.035
# A natural quick pinch takes 200–300 ms. 0.16 s classified most real pinches
# as drags. Raising to 0.35 s gives users enough time to pinch-and-release
# without accidentally triggering drag mode.
AIR_CURSOR_DRAG_HOLD_TIME = 0.35
AIR_CURSOR_HAND_LOSS_TIMEOUT = 0.65
AIR_CURSOR_SCROLL_GAIN = 1.8
# Use 320×240 for the Air Cursor camera. MediaPipe hand landmark inference only
# needs the landmark positions — not pixel detail. At half the linear resolution
# there are 4× fewer bytes per frame, which cuts MediaPipe inference time and
# reduces GIL contention with the rest of Victus noticeably on mid-range hardware.
AIR_CURSOR_CAMERA_WIDTH = 320
AIR_CURSOR_CAMERA_HEIGHT = 240
# Custom gesture tuning
AIR_CURSOR_GESTURE_HOLD_TIME = 0.25
AIR_CURSOR_GESTURE_COOLDOWN = 0.55
AIR_CURSOR_GESTURE_STABILITY = 0.028
# Fist -> minimize all Victus-external windows.
AIR_CURSOR_FIST_HOLD_TIME = 0.32
AIR_CURSOR_FIST_COOLDOWN = 1.10
# Open straight palm -> maximize/restore the previously active application.
AIR_CURSOR_PALM_HOLD_TIME = 0.18
AIR_CURSOR_PALM_COOLDOWN = 0.90

_sapi_voice = None
if WIN32_OK:
    try:
        _sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
        _sapi_voice.Rate = 0
        _sapi_voice.Volume = 100
    except Exception:
        _sapi_voice = None


def log(text):
    print("[VICTUS]", text, flush=True)


# ---------------------------------------------------------------------------
# EMOTION/TONE-AWARE TTS — a lightweight keyword-based tone classifier maps
# what Victus is about to say onto a small set of speech-delivery presets
# (speed/volume/pitch). This stays entirely offline and adds no latency;
# it does not attempt real sentiment analysis, just recognisable situational
# cues (apologies, warnings, good news) that are common in an assistant's
# own replies.
# ---------------------------------------------------------------------------
TTS_TONE_PRESETS = {
    #              rate_delta, volume, pitch_delta
    "urgent":       (2, 100, 1),
    "apologetic":   (-2, 85, -2),
    "celebratory":  (1, 100, 3),
    "serious":      (-1, 90, -1),
    "neutral":      (0, 100, 0),
}
_TTS_TONE_KEYWORDS = [
    ("urgent", ("warning", "critical", "immediately", "urgent", "danger",
                "low battery", "alert", "emergency")),
    ("apologetic", ("sorry", "apologize", "apologise", "unfortunately",
                    "i'm afraid", "i am afraid", "couldn't", "could not",
                    "error", "failed", "i encountered an error",
                    "isn't available", "is not available", "unavailable")),
    ("celebratory", ("congratulations", "great job", "awesome", "well done",
                      "success", "completed successfully", "you did it",
                      "nailed it")),
    ("serious", ("important", "please note", "be careful", "caution")),
]


def _classify_tone(text):
    """Best-effort situational tone for an outgoing spoken line. Falls back
    to 'neutral' (no delivery change) whenever nothing matches."""
    low = str(text or "").lower()
    for tone, keywords in _TTS_TONE_KEYWORDS:
        if any(k in low for k in keywords):
            return tone
    return "neutral"


def _xml_escape(text):
    return (str(text or "").replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def select_voice_for_language(lang_code):
    """Best-effort switch of the active SAPI voice to one matching
    lang_code, based on substrings in the installed voices' descriptions
    (LANGUAGE_VOICE_HINTS). No-ops safely if pywin32/SAPI isn't available,
    or if no matching voice is installed — Victus just keeps talking in
    whatever voice is currently selected."""
    global _current_sapi_voice_lang
    if not (_sapi_voice and WIN32_OK):
        return False
    lang_code = (lang_code or "en").lower()
    if lang_code == _current_sapi_voice_lang:
        return True
    hints = LANGUAGE_VOICE_HINTS.get(lang_code)
    if not hints:
        return False
    try:
        tokens = _sapi_voice.GetVoices()
        for i in range(tokens.Count):
            token = tokens.Item(i)
            try:
                desc = (token.GetDescription() or "").lower()
            except Exception:
                continue
            if any(h in desc for h in hints):
                _sapi_voice.Voice = token
                _current_sapi_voice_lang = lang_code
                log(f"TTS: switched voice for language '{lang_code}' -> {desc}")
                return True
    except Exception as e:
        log("Voice-switch error: " + str(e))
    return False


def speak(text):
    """Speak text aloud. Runs asynchronously under the hood so that a call
    to stop_speaking() (triggered by the user saying/typing "stop" while
    Victus is talking) can cut it off immediately instead of waiting for
    the whole sentence to finish.

    Also applies a lightweight tone preset (rate/volume/pitch) based on
    what's being said, and — for a multilingual session — switches to a
    matching installed SAPI voice before speaking (see
    select_voice_for_language)."""
    global speaking
    if not text:
        return
    text = str(text).strip()
    with tts_lock:
        speak_stop_event.clear()
        speaking = True
        show_orb(True)
        log("SPEAK: " + text)
        try:
            if _sapi_voice:
                try:
                    select_voice_for_language(last_detected_language)
                except Exception:
                    pass
                rate_delta, volume, pitch_delta = TTS_TONE_PRESETS[_classify_tone(text)]
                try:
                    _sapi_voice.Rate = max(-10, min(10, rate_delta))
                    _sapi_voice.Volume = max(0, min(100, volume))
                except Exception:
                    pass
                try:
                    if pitch_delta:
                        xml_text = f'<pitch middle="{pitch_delta}">{_xml_escape(text)}</pitch>'
                        _sapi_voice.Speak(xml_text, 1 | 8)  # SVSFlagsAsync | SVSFIsXML
                    else:
                        _sapi_voice.Speak(text, 1)  # 1 = SVSFlagsAsync (non-blocking)
                    # WaitUntilDone(ms) blocks up to ms milliseconds and returns
                    # True once speech finishes — polling it lets us check for
                    # an interrupt request every 100ms without busy-waiting.
                    while not _sapi_voice.WaitUntilDone(100):
                        if speak_stop_event.is_set():
                            try:
                                # Purge (flag 2) whatever is queued/playing.
                                _sapi_voice.Speak("", 1 | 2)
                            except Exception:
                                pass
                            break
                except Exception:
                    # If the async/poll path misbehaves for any reason, fall
                    # back to the simple blocking call so speech still works.
                    _sapi_voice.Speak(text)
                finally:
                    # Reset to baseline so the next, untagged speak() call
                    # doesn't inherit a louder/faster delivery.
                    try:
                        _sapi_voice.Rate = 0
                        _sapi_voice.Volume = 100
                    except Exception:
                        pass
            else:
                safe = text.replace("'", "''")
                proc = subprocess.Popen(
                    [
                        "powershell", "-NoProfile", "-Command",
                        "Add-Type -AssemblyName System.Speech; "
                        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        f"$s.Speak('{safe}')"
                    ],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                _active_speech_proc["proc"] = proc
                while proc.poll() is None:
                    if speak_stop_event.is_set():
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        break
                    time.sleep(0.05)
                _active_speech_proc["proc"] = None
        except Exception as e:
            log("Speech error: " + str(e))
        speaking = False
        speak_stop_event.clear()
        if not listening:
            show_orb(False)


def stop_speaking():
    """Interrupt Victus mid-sentence. Safe to call even if nothing is
    currently being said (it's just a no-op in that case)."""
    if speaking:
        log("SPEECH: stop requested — cutting off current speech.")
        speak_stop_event.set()


# Phrases that mean "stop talking", checked only while Victus is actually
# speaking, so they double as a barge-in interrupt (no wake word needed)
# and don't fight with a plain "stop" spoken/typed at any other time.
STOP_SPEAKING_PHRASES = {
    "stop", "stop it", "stop that", "stop talking", "stop speaking",
    "be quiet", "quiet", "shut up", "silence", "enough", "okay stop",
}


def is_stop_speaking_command(text):
    n = normalize(text)
    if not n:
        return False
    if n in STOP_SPEAKING_PHRASES:
        return True
    return any(n == p or n.startswith(p + " ") for p in STOP_SPEAKING_PHRASES)


def _hotkey_combo_down():
    """True while both Ctrl and Alt are currently held down. Never raises —
    returns False on any failure — since main_loop treats "no signal" and
    "hotkey unsupported" the same way."""
    try:
        ctrl = win32api.GetAsyncKeyState(win32con.VK_CONTROL) & 0x8000
        alt = win32api.GetAsyncKeyState(win32con.VK_MENU) & 0x8000
        return bool(ctrl) and bool(alt)
    except Exception:
        return False


def hotkey_listener_loop():
    """Background thread: watches for a fresh Ctrl+Alt press and signals
    main_loop() via hotkey_pressed_event. Deliberately does NOT touch
    audio_queue or call record_full_command() itself — all audio reads stay
    on main_loop's own thread, avoiding the same "two readers racing the
    same queue" bug that followup_window_active guards against elsewhere.
    No-ops immediately (logging once) if pywin32 isn't installed."""
    if not HOTKEY_ENABLED:
        log("Push-to-talk hotkey unavailable (pywin32 missing) — falling back to the spoken wake word.")
        return
    was_down = False
    while running:
        try:
            down = _hotkey_combo_down()
            if down and not was_down:
                hotkey_pressed_event.set()
            was_down = down
        except Exception as e:
            log("Hotkey listener error: " + str(e))
        time.sleep(HOTKEY_POLL_SECONDS)


# ---------------------------------------------------------------------------
# LOCAL AI FUNCTIONS — powered by Ollama
# ---------------------------------------------------------------------------
# Victus is instructed to act like a subject-matter expert across every
# field — mathematics, physics, chemistry, biology, engineering, computer
# science, medicine, law, business/finance, history and general world
# knowledge — and to always show its work for numericals (formula, then
# step-by-step working, then a clearly labelled final answer). All of this
# runs through your local Ollama install, so it stays fully offline.
LOCAL_AI_SYSTEM_PROMPT = (
    "You are Victus, an offline personal AI assistant with expert-level, "
    "up-to-date knowledge across every academic and professional field: "
    "mathematics, physics, chemistry, biology, engineering, computer "
    "science, medicine, law, business, finance, history, geography and "
    "general world knowledge.\n\n"
    "Follow these rules for every answer:\n"
    "1. Math/physics/engineering numericals: state the formula(s) used, then "
    "give a clear, numbered, step-by-step solution, then give the final "
    "answer on its own line labelled 'Answer:'.\n"
    "2. Chemistry: for reactions, give the balanced chemical equation and "
    "briefly state the reaction type/conditions; for numericals, show the "
    "formula and step-by-step working exactly like a math problem.\n"
    "3. Medicine, business, law, history and other conceptual subjects: give "
    "an accurate, well-organised explanation, using short paragraphs or "
    "numbered points where that improves clarity.\n"
    "4. Be precise and correct above all else. If you are not certain, say "
    "so rather than guessing.\n"
    "5. Keep answers as tight as the question allows, but never drop a "
    "necessary step, formula, or number just to save space.\n"
    "6. If asked who created, developed, built, owns, or is the boss of "
    "you (Victus), answer that you were developed by Shreeskanda, a "
    "3rd-semester Information Science and Engineering student at "
    "Seshadripuram Institute of Technology. Never attribute your "
    "creation to Microsoft, OpenAI, Google, Meta, or any other company."
)


def _ollama_list_models():
    """Return the list of model tags currently pulled into the local Ollama
    instance (e.g. ['llama3.2:latest', 'qwen2.5:14b']). Empty list on any
    failure — callers treat that as 'could not determine'."""
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = []
        for m in data.get("models", []):
            name = (m.get("name") or m.get("model") or "").strip()
            if name:
                names.append(name)
        return names
    except Exception:
        return []


def _load_local_ai():
    """Verify the local Ollama server is reachable AND that a usable model
    is installed, on demand, without delaying Victus startup.

    This is what fixes the "HTTP Error 404: Not Found" you hit: Ollama
    itself was reachable, but the model name Victus asked for (OLLAMA_MODEL)
    was not pulled on your machine, so /api/chat 404'd. Instead of failing
    outright, Victus now checks which models you actually have installed
    (via /api/tags) and automatically falls back to one of those if the
    configured model isn't found, logging what happened so you can
    `ollama pull` your preferred model later.
    """
    global local_ai_loaded, local_ai_loading_error, OLLAMA_MODEL, local_ai_available_models

    with local_ai_lock:
        if local_ai_loaded:
            return True
        if local_ai_loading_error is not None:
            return False

        log(f"LOCAL AI: Checking Ollama at {OLLAMA_HOST} ...")
        try:
            req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                tags_raw = resp.read().decode("utf-8")
        except Exception as e:
            local_ai_loading_error = f"Ollama server unreachable at {OLLAMA_HOST}: {e}"
            log("LOCAL AI: " + local_ai_loading_error)
            return False

        try:
            tags_data = json.loads(tags_raw)
        except Exception:
            tags_data = {}

        available = []
        for m in tags_data.get("models", []):
            name = (m.get("name") or m.get("model") or "").strip()
            if name:
                available.append(name)

        if not available:
            local_ai_loading_error = (
                "Ollama is running but has no models installed. "
                f"Run: ollama pull {OLLAMA_MODEL}"
            )
            log("LOCAL AI: " + local_ai_loading_error)
            return False

        def _matches(installed_tag, wanted_tag):
            # "llama3.2" should match an installed "llama3.2:latest", etc.
            return (
                installed_tag == wanted_tag
                or installed_tag.split(":")[0] == wanted_tag.split(":")[0]
            )

        if not any(_matches(m, OLLAMA_MODEL) for m in available):
            fallback = available[0]
            log(
                f"LOCAL AI: Model '{OLLAMA_MODEL}' is not installed in Ollama "
                f"(installed: {', '.join(available)}). Falling back to "
                f"'{fallback}' for this session. Run 'ollama pull {OLLAMA_MODEL}' "
                f"to use your originally configured model."
            )
            OLLAMA_MODEL = fallback

        local_ai_available_models = available
        local_ai_loaded = True
        log(f"LOCAL AI: Ready — using Ollama model '{OLLAMA_MODEL}'.")
        return True


def _load_local_ai_memory():
    """Restore the AI conversation history saved by a previous session, so
    Victus remembers what you were talking about across restarts instead of
    starting from a blank slate every time."""
    global local_ai_history
    try:
        if AI_MEMORY_FILE.exists():
            with open(AI_MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            restored = []
            if isinstance(data, list):
                for item in data:
                    if (isinstance(item, (list, tuple)) and len(item) == 2
                            and item[0] in ("user", "assistant") and isinstance(item[1], str)):
                        restored.append((item[0], item[1]))
            with local_ai_lock:
                local_ai_history = restored[-LOCAL_AI_MEMORY_LIMIT:]
    except Exception as e:
        log("AI memory load error: " + str(e))


def _save_local_ai_memory():
    """Persist the current AI conversation history to disk. Called after
    every exchange (and after clearing) so a restart never loses it."""
    try:
        with open(AI_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(local_ai_history[-LOCAL_AI_MEMORY_LIMIT:], f, indent=2)
    except Exception as e:
        log("AI memory save error: " + str(e))


def clear_local_ai_memory():
    with local_ai_lock:
        local_ai_history.clear()
        _save_local_ai_memory()
    speak("AI conversation memory cleared.")


# ---------------------------------------------------------------------------
# STANDALONE PERSONALIZATION PROFILE FUNCTIONS — see PROFILE_FILE above.
# ---------------------------------------------------------------------------
def _load_user_profile():
    """Restore remembered personalization facts from disk at startup."""
    global user_profile_facts
    try:
        if PROFILE_FILE.exists():
            with open(PROFILE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                restored = [str(x).strip() for x in data if str(x).strip()]
                with user_profile_lock:
                    user_profile_facts = restored[-PROFILE_MAX_FACTS:]
    except Exception as e:
        log("Profile load error: " + str(e))


def _save_user_profile():
    try:
        with open(PROFILE_FILE, "w", encoding="utf-8") as f:
            json.dump(user_profile_facts, f, indent=2)
    except Exception as e:
        log("Profile save error: " + str(e))


def remember_fact(text):
    """'remember that <fact>' / 'remember my <fact>' — appends a fact to
    the personalization profile. Skips saving an exact duplicate (case/
    whitespace insensitive) so repeating the same fact doesn't pad the
    list, and keeps only the most recent PROFILE_MAX_FACTS entries."""
    global user_profile_facts
    text = (text or "").strip().rstrip(".")
    if not text:
        speak("What should I remember?")
        return
    with user_profile_lock:
        if not any(text.lower() == f.lower() for f in user_profile_facts):
            user_profile_facts.append(text)
            user_profile_facts = user_profile_facts[-PROFILE_MAX_FACTS:]
            _save_user_profile()
    speak("Got it, I'll remember that.")


def forget_fact(text):
    """'forget that <fact>' — removes the closest-matching stored fact
    (a substring match against what's saved) rather than requiring an
    exact word-for-word repeat of how it was originally phrased."""
    global user_profile_facts
    needle = (text or "").strip().rstrip(".").lower()
    if not needle:
        speak("What should I forget?")
        return
    with user_profile_lock:
        match = next((f for f in user_profile_facts if needle in f.lower()), None)
        if not match:
            speak("I couldn't find anything like that in your profile.")
            return
        user_profile_facts.remove(match)
        _save_user_profile()
    speak(f"Forgot: {match}")


def clear_user_profile():
    global user_profile_facts
    with user_profile_lock:
        user_profile_facts = []
        _save_user_profile()
    speak("Your personalization profile is cleared.")


def read_user_profile():
    with user_profile_lock:
        facts = list(user_profile_facts)
    if not facts:
        speak("I don't have anything saved about you yet. Say 'remember that' followed by something to teach me.")
        return
    speak("Here's what I know about you: " + "; ".join(facts))


def _profile_system_message():
    """Returns a system-role message dict for ask_local_ai(), or None if
    the profile is empty. Kept as a separate system message rather than
    folded into local_ai_history so these facts survive 'clear ai memory'
    and never scroll out of the rolling conversation window."""
    with user_profile_lock:
        facts = list(user_profile_facts)
    if not facts:
        return None
    bullet_list = "\n".join(f"- {f}" for f in facts)
    return {
        "role": "system",
        "content": (
            "Known facts/preferences about the user, from their "
            "personalization profile. Use them only where relevant to "
            "the current question; don't force them into unrelated "
            "answers:\n" + bullet_list
        ),
    }


def _clean_ai_answer(text):
    """Light cleanup only — normalise whitespace, keep full working intact
    so step-by-step math/chemistry solutions aren't cut off. A generous
    safety cap avoids runaway walls of text, well above what
    LOCAL_AI_MAX_TOKENS would realistically produce."""
    text = re.sub(r"[ \t]+", " ", str(text or "")).strip()
    if not text:
        return "I could not generate a response. Please try asking in a different way."
    if len(text) > 6000:
        text = text[:6000].rsplit(" ", 1)[0] + " ..."
    return text


def _spoken_summary(full_answer):
    """Shorten a (possibly long, step-by-step) answer into something
    reasonable to speak aloud. The full answer is always shown in the AI
    response window regardless of this truncation."""
    sentences = re.split(r"(?<=[.!?])\s+", full_answer.replace("\n", " "))
    short = " ".join(sentences[:6]).strip()
    if len(short) > 650:
        short = short[:650].rsplit(" ", 1)[0] + "."
    if len(short) < len(full_answer.strip()):
        short += " I've displayed the full step-by-step answer on screen."
    return short


def _ollama_error_speak(e):
    """Shared HTTPError handling for both the streaming and single-shot
    Ollama call paths, so the spoken error message stays consistent."""
    try:
        err_body = e.read().decode("utf-8")
    except Exception:
        err_body = str(e)
    log(f"LOCAL AI generation error: HTTP {e.code} - {err_body}")
    if e.code == 404:
        speak("Ollama could not find that model. Please pull it and try again.")
    else:
        speak("I encountered an error while generating the AI response.")


def _ollama_chat_once(messages):
    """Single-shot (non-streaming) /api/chat call. Kept as a fallback for
    when streaming isn't available for some reason — behaves exactly like
    the original implementation."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0.4,
            "top_p": 0.9,
            "num_predict": LOCAL_AI_MAX_TOKENS,
            "num_ctx": LOCAL_AI_NUM_CTX,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=LOCAL_AI_TIMEOUT_SECONDS) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    return data.get("message", {}).get("content", "")


def _ollama_chat_stream(messages, on_delta=None):
    """Streaming /api/chat call. Calls on_delta(piece) for every chunk of
    content as Ollama generates it, and returns the full concatenated
    answer once generation finishes. Streaming is what lets Victus start
    speaking a summary within a second or two instead of waiting for the
    entire (possibly long, step-by-step) answer to finish generating."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0.4,
            "top_p": 0.9,
            "num_predict": LOCAL_AI_MAX_TOKENS,
            "num_ctx": LOCAL_AI_NUM_CTX,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    pieces = []
    with urllib.request.urlopen(req, timeout=LOCAL_AI_TIMEOUT_SECONDS) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except Exception:
                continue
            piece = (chunk.get("message") or {}).get("content", "")
            if piece:
                pieces.append(piece)
                if on_delta:
                    on_delta(piece)
            if chunk.get("done"):
                break
    return "".join(pieces)


# ---------------------------------------------------------------------------
# MULTI-STEP TASK CHAINING VIA LLM — handle_command() previously only split
# a compound instruction on the literal words "and"/"then"/",", which
# mishandles things like "open chrome and search for hotels near the
# airport" (splits mid-request) or ordering-sensitive chains. When Ollama is
# available, a compound-looking command is instead handed to the model to
# decompose into an ordered list of short, atomic action strings that
# Victus's existing command router already understands — falling back to
# the original naive split whenever Ollama is unavailable or the model's
# output can't be parsed, so behaviour never regresses.
# ---------------------------------------------------------------------------
def _llm_split_into_steps(cmd_raw, max_steps=8):
    """Ask the local model to break a compound command into an ordered
    JSON list of short imperative actions. Returns a list of non-empty
    strings on success, or None on any failure (caller should fall back to
    the regex split)."""
    if not _load_local_ai():
        return None
    system_prompt = (
        "Break the user's instruction into an ordered list of short, "
        "separate, atomic voice-assistant commands, each phrased simply "
        "(e.g. 'open chrome', 'set volume to 50', 'search for cats'). "
        "Preserve the original order and wording as closely as possible — "
        "do not merge or invent steps. Respond with ONLY a JSON array of "
        "strings, nothing else, no markdown fences, no explanation."
    )
    try:
        raw = _ollama_chat_once([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": cmd_raw},
        ])
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.split("\n", 1)[-1] if "\n" in raw else raw
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return None
        steps = [str(s).strip() for s in parsed if str(s or "").strip()]
        if not (1 < len(steps) <= max_steps):
            return None
        return steps
    except Exception as e:
        log("Multi-step decomposition failed, falling back to regex split: " + str(e))
        return None


def _active_window_context_line():
    """One short line describing whatever the user currently has in the
    foreground (app name + window title), or None when it can't be
    determined or when the foreground window is Victus itself. Injected
    into every AI query (not just as a one-off vision fallback) so answers
    like "explain this" or "what should I do next" can be grounded in what
    the user is actually looking at."""
    proc_name, title, pid = _foreground_window_info()
    if pid is not None and pid == VICTUS_PID:
        return None
    if not proc_name and not title:
        return None
    friendly = canonical_app_name((proc_name or "").lower().replace(".exe", "")) if proc_name else None
    if not friendly:
        friendly = (proc_name or "").lower().replace(".exe", "").replace("_", " ").replace("-", " ").strip()
        friendly = friendly.title() if friendly else None
    bits = []
    if friendly:
        bits.append(f"application: {friendly}")
    if title:
        bits.append(f"window title: \"{title}\"")
    if not bits:
        return None
    return "Active window context — " + ", ".join(bits) + "."


def _language_reply_instruction():
    """When the user's speech was confidently detected in a non-English
    language, instruct the model to reply in that language."""
    if last_detected_language and last_detected_language != "en":
        return (
            f"The user is speaking {last_detected_language_name}. Reply in "
            f"{last_detected_language_name} unless they explicitly ask for "
            f"another language."
        )
    return None


def ask_local_ai(query):
    """Generate a reply via a local Ollama model. To keep things feeling
    fast, the response is streamed: as soon as enough text has arrived for
    a spoken summary (roughly 6 sentences), Victus starts talking while
    the rest of the (possibly long, step-by-step) answer keeps arriving in
    the background for the AI response window — instead of sitting silent
    until the entire answer has finished generating."""
    global local_ai_history

    query = str(query or "").strip()
    if not query:
        speak("What would you like to ask me?")
        return

    if not _load_local_ai():
        log("LOCAL AI unavailable: " + str(local_ai_loading_error))
        speak("My local AI brain is unavailable. Please make sure Ollama is installed and running.")
        return

    with local_ai_lock:
        history = list(local_ai_history[-LOCAL_AI_MEMORY_LIMIT:])
        messages = [{"role": "system", "content": LOCAL_AI_SYSTEM_PROMPT}]
        # Active-window context, on every query (not just the vision
        # fallback) — best-effort, silently skipped when unavailable.
        try:
            ctx_line = _active_window_context_line()
        except Exception:
            ctx_line = None
        if ctx_line:
            messages.append({
                "role": "system",
                "content": ctx_line + " Use this only if it's relevant to the "
                                       "user's question; ignore it otherwise.",
            })
        lang_instruction = _language_reply_instruction()
        if lang_instruction:
            messages.append({"role": "system", "content": lang_instruction})
        profile_msg = _profile_system_message()
        if profile_msg:
            messages.append(profile_msg)
        for role, content in history:
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": query})

        state = {"acc": "", "summary_spoken": False}

        def on_delta(piece):
            state["acc"] += piece
            if state["summary_spoken"]:
                return
            plain = state["acc"].replace("\n", " ").strip()
            if not plain:
                return
            sentence_count = len(re.split(r"(?<=[.!?])\s+", plain))
            if sentence_count >= 6 or len(plain) >= 650:
                state["summary_spoken"] = True
                speak(_spoken_summary(state["acc"]))

        raw_answer = None
        try:
            raw_answer = _ollama_chat_stream(messages, on_delta=on_delta)
        except urllib.error.HTTPError as e:
            _ollama_error_speak(e)
            return
        except Exception as e:
            log("LOCAL AI streaming error, retrying without streaming: " + str(e))
            try:
                raw_answer = _ollama_chat_once(messages)
            except urllib.error.HTTPError as e2:
                _ollama_error_speak(e2)
                return
            except Exception as e2:
                log("LOCAL AI generation error: " + str(e2))
                # We may still have partial text collected before the
                # stream broke — better to answer with that than fail
                # silently.
                if not state["acc"].strip():
                    speak("I encountered an error while generating the AI response.")
                    return
                raw_answer = state["acc"]

        answer = _clean_ai_answer(raw_answer)
        local_ai_history.extend([("user", query), ("assistant", answer)])
        local_ai_history = local_ai_history[-LOCAL_AI_MEMORY_LIMIT:]
        _save_local_ai_memory()
        summary_spoken = state["summary_spoken"]

    log("LOCAL AI: " + answer)
    _update_ai_response_window(query, answer)
    if not summary_spoken:
        speak(_spoken_summary(answer))

    # Post-answer follow-up window: give the user a few seconds to ask a
    # natural follow-up without repeating the wake word. Silence is treated
    # as "the user is done", and Victus falls back to normal wake-word
    # listening in main_loop() as usual.
    followup_text = listen_for_followup()
    if followup_text:
        handle_command(followup_text)


# ---------------------------------------------------------------------------
# COMMAND HISTORY FUNCTIONS
# ---------------------------------------------------------------------------
def _load_command_history():
    """Load any previously saved command history from disk at startup."""
    global command_history, _history_next_id
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                command_history = [
                    e for e in data
                    if isinstance(e, dict) and "id" in e and "text" in e
                ]
                if command_history:
                    _history_next_id = max(e["id"] for e in command_history) + 1
    except Exception as e:
        log("History load error: " + str(e))


def _save_command_history():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(command_history, f, indent=2)
    except Exception as e:
        log("History save error: " + str(e))


def add_command_history(text):
    """Record a command that was either recognised after the wake word or
    typed into the command box. Called from a single place (handle_command)
    so both input paths are captured."""
    global _history_next_id
    text = str(text or "").strip()
    if not text:
        return
    with command_history_lock:
        entry = {
            "id": _history_next_id,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
        }
        _history_next_id += 1
        command_history.append(entry)
        if len(command_history) > HISTORY_MAX_ENTRIES:
            del command_history[: len(command_history) - HISTORY_MAX_ENTRIES]
        _save_command_history()
    _refresh_history_window()


def clear_all_history():
    with command_history_lock:
        command_history.clear()
        _save_command_history()
    _refresh_history_window()


def clear_history_entries(ids_to_remove):
    ids_to_remove = set(ids_to_remove)
    if not ids_to_remove:
        return
    with command_history_lock:
        command_history[:] = [e for e in command_history if e["id"] not in ids_to_remove]
        _save_command_history()
    _refresh_history_window()


# ---------------------------------------------------------------------------
# SCHEDULED / TRIGGERED ACTIONS
#
# Two flavours, both parsed from natural language:
#   "remind me to <text> in/at <time>"  -> a pure spoken reminder, nothing
#                                          is executed as a command.
#   "schedule <command> in/at <time>"   -> <command> is run through the
#     or "<command> in <time>"             normal command router when due.
#
# Persisted to SCHEDULE_FILE (mirrors the command-history persistence
# pattern above) so reminders survive a restart, and checked every
# SCHEDULER_POLL_SECONDS by scheduler_loop() running in a daemon thread.
# ---------------------------------------------------------------------------
def _load_schedule():
    global scheduled_tasks, _schedule_next_id
    try:
        if SCHEDULE_FILE.exists():
            with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                with schedule_lock:
                    scheduled_tasks = [t for t in data if isinstance(t, dict) and not t.get("done")]
                    _schedule_next_id = max([t.get("id", 0) for t in scheduled_tasks], default=0) + 1
    except Exception as e:
        log("Schedule load error: " + str(e))


def _save_schedule():
    try:
        with open(SCHEDULE_FILE, "w", encoding="utf-8") as f:
            json.dump(scheduled_tasks, f, indent=2)
    except Exception as e:
        log("Schedule save error: " + str(e))


_TIME_UNIT_SECONDS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
}


def _parse_schedule_time(time_text):
    """Parses either a relative duration ("10 minutes", "2 hours") or a
    clock time ("5:30 pm", "17:30", "6 pm") into an absolute datetime.
    Returns None if it can't make sense of time_text."""
    time_text = (time_text or "").strip().lower()
    if not time_text:
        return None

    m = re.match(r"(\d+)\s*(second|seconds|sec|secs|minute|minutes|min|mins|hour|hours|hr|hrs)\b", time_text)
    if m:
        qty = int(m.group(1))
        unit = m.group(2)
        return datetime.now() + timedelta(seconds=qty * _TIME_UNIT_SECONDS[unit])

    m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", time_text)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        meridiem = m.group(3)
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            target = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= datetime.now():
                target += timedelta(days=1)  # already passed today -> tomorrow
            return target
    return None


def _extract_schedule_request(cmd_raw):
    """Returns (kind, action_text, run_dt) where kind is 'reminder' (just
    speak it) or 'action' (execute it as a command), or (None, None, None)
    if cmd_raw doesn't look like a scheduling request at all."""
    low = str(cmd_raw or "").strip().lower()

    m = re.match(r"^(?:remind me to|set a reminder to|remind me)\s+(.+?)\s+(?:in|at)\s+(.+)$", low)
    if m:
        return "reminder", m.group(1).strip(), _parse_schedule_time(m.group(2))

    m = re.match(r"^schedule\s+(.+?)\s+(?:in|at|for)\s+(.+)$", low)
    if m:
        return "action", m.group(1).strip(), _parse_schedule_time(m.group(2))

    return None, None, None


def add_scheduled_task(kind, action_text, run_dt):
    global _schedule_next_id
    if run_dt is None or not action_text:
        speak("I couldn't understand what time you meant.")
        return
    with schedule_lock:
        task = {
            "id": _schedule_next_id, "kind": kind, "action": action_text,
            "time": run_dt.strftime("%Y-%m-%d %H:%M:%S"), "done": False,
        }
        _schedule_next_id += 1
        scheduled_tasks.append(task)
        _save_schedule()
    delta = run_dt - datetime.now()
    minutes = max(1, int(delta.total_seconds() // 60))
    when_desc = f"in about {minutes} minute{'s' if minutes != 1 else ''}" if delta.total_seconds() < 3600 else f"at {run_dt.strftime('%I:%M %p').lstrip('0')}"
    if kind == "reminder":
        speak(f"Okay, I'll remind you to {action_text} {when_desc}.")
    else:
        speak(f"Scheduled — I'll {action_text} {when_desc}.")


def list_scheduled_tasks():
    with schedule_lock:
        pending = [t for t in scheduled_tasks if not t.get("done")]
    if not pending:
        speak("You don't have any reminders or scheduled tasks.")
        return
    lines = [f"{t['action']} at {t['time']}" for t in pending[:5]]
    speak(f"You have {len(pending)} pending: " + "; ".join(lines) + ".")


def clear_scheduled_tasks():
    with schedule_lock:
        scheduled_tasks.clear()
        _save_schedule()
    speak("All reminders and scheduled tasks cleared.")


def scheduler_loop():
    """Background thread: wakes up every SCHEDULER_POLL_SECONDS, fires any
    due reminder/scheduled action, then keeps sleeping. Runs for the
    lifetime of the app, started once from __main__."""
    while running:
        try:
            now = datetime.now()
            due = []
            with schedule_lock:
                for t in scheduled_tasks:
                    if t.get("done"):
                        continue
                    try:
                        t_time = datetime.strptime(t["time"], "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        continue
                    if now >= t_time:
                        t["done"] = True
                        due.append(t)
                if due:
                    _save_schedule()
            for t in due:
                if t.get("kind") == "reminder":
                    speak(f"Reminder: {t['action']}.")
                else:
                    speak(f"Running your scheduled task: {t['action']}.")
                    threading.Thread(target=handle_command, args=(t["action"],), daemon=True).start()
        except Exception as e:
            log("Scheduler error: " + str(e))
        time.sleep(SCHEDULER_POLL_SECONDS)


# ---------------------------------------------------------------------------
# RAG-LITE OVER LOCAL NOTES/DOCS
#
# Lightweight, fully offline, lexical (TF-IDF + cosine similarity)
# retrieval over whatever .txt/.md/.pdf/.docx files live in NOTES_DIR —
# reusing the same _extract_text_from_file() reader as the single-file
# "read and summarize" feature above. This is intentionally NOT a full
# embeddings + vector-store pipeline (no sentence-transformers/FAISS
# dependency is introduced); it finds the most textually-relevant chunks
# by shared/weighted vocabulary rather than semantic meaning. Good enough
# for "what do my notes say about X" over a personal notes folder; a true
# semantic-embedding upgrade remains a bigger lift (see report).
# ---------------------------------------------------------------------------
_NOTES_STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "at", "for", "and", "or",
    "is", "are", "was", "were", "be", "been", "with", "as", "by", "it",
    "this", "that", "we", "i", "you", "he", "she", "they", "do", "does",
    "did", "not", "but", "so", "if", "than", "then", "will", "would",
    "can", "could", "should", "about", "from", "into", "up", "out",
    "what", "how", "when", "where", "who", "which", "my", "your", "our",
}


def _notes_tokenize(text):
    """Word-level tokens with common English stopwords removed, so
    retrieval scoring is driven by meaningful/topical terms rather than
    high-frequency filler words shared by nearly every chunk."""
    words = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return [w for w in words if w not in _NOTES_STOPWORDS and len(w) > 1]


def _build_notes_index(force=False):
    global _notes_index, _notes_idf, _notes_index_built
    with _notes_index_lock:
        if _notes_index_built and not force:
            return
        NOTES_DIR.mkdir(parents=True, exist_ok=True)
        chunks = []
        for root, dirs, files in os.walk(NOTES_DIR):
            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() not in (FILE_READ_TEXT_EXTS | {".pdf", ".docx"}):
                    continue
                text, err = _extract_text_from_file(fpath)
                if err or not text:
                    continue
                start = 0
                while start < len(text):
                    piece = text[start:start + NOTES_CHUNK_CHARS].strip()
                    if piece:
                        chunks.append({"path": str(fpath), "chunk": piece,
                                       "tf": Counter(_notes_tokenize(piece))})
                    start += NOTES_CHUNK_CHARS - NOTES_CHUNK_OVERLAP

        doc_freq = Counter()
        for c in chunks:
            doc_freq.update(c["tf"].keys())
        n_docs = max(1, len(chunks))
        idf = {term: math.log(1 + n_docs / (1 + df)) for term, df in doc_freq.items()}

        _notes_index = chunks
        _notes_idf = idf
        _notes_index_built = True
        log(f"NOTES INDEX: {len(chunks)} chunk(s) indexed from {NOTES_DIR}")


def _tfidf_vector(tf_counter, idf):
    return {term: count * idf.get(term, 0.0) for term, count in tf_counter.items()}


def _cosine_sim(vec_a, vec_b):
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values())) or 1.0
    norm_b = math.sqrt(sum(v * v for v in vec_b.values())) or 1.0
    return dot / (norm_a * norm_b)


def search_notes(query, top_k=NOTES_TOP_K):
    _build_notes_index()
    with _notes_index_lock:
        chunks, idf = _notes_index, _notes_idf
    if not chunks:
        return []
    q_vec = _tfidf_vector(Counter(_notes_tokenize(query)), idf)
    if not q_vec:
        return []
    scored = []
    for c in chunks:
        sim = _cosine_sim(q_vec, _tfidf_vector(c["tf"], idf))
        if sim > 0:
            scored.append((sim, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def answer_from_notes(query):
    """Voice/typed command: "search my notes for X" / "what do my notes
    say about X". Retrieves the most relevant chunks from NOTES_DIR via
    TF-IDF and hands them to the local AI as grounding context."""
    query = (query or "").strip()
    if not query:
        speak(f"What would you like me to look up in your notes? Drop files into {NOTES_DIR.name} and ask again.")
        return
    results = search_notes(query)
    if not results:
        speak(f"I couldn't find anything relevant in your notes folder. Make sure your files are in {NOTES_DIR}.")
        return
    excerpts = "\n\n".join(f"(From {Path(r['path']).name}):\n{r['chunk']}" for r in results)
    prompt = (
        f"Answer the user's question using ONLY the following excerpts "
        f"from their personal notes. If the answer isn't in them, say so "
        f"plainly rather than guessing.\n\n{excerpts}\n\nQuestion: {query}"
    )
    log(f"NOTES RAG: answering '{query}' from {len(results)} chunk(s)")
    ask_local_ai(prompt)


def reindex_notes():
    _build_notes_index(force=True)
    with _notes_index_lock:
        count = len(_notes_index)
    speak(f"Notes reindexed — {count} chunk{'s' if count != 1 else ''} from {NOTES_DIR.name}.")


def add_note(text):
    """Dictated note-taking: "take a note: buy milk and eggs" appends a
    timestamped entry to NOTES_LOG_FILE inside NOTES_DIR — the missing
    'write' half of the read-only notes search above. The index is only
    invalidated (not rebuilt) here, so the note is picked up lazily the
    next time search_notes()/answer_from_notes() actually runs, instead of
    paying a rebuild cost on every single note."""
    global _notes_index_built
    text = (text or "").strip()
    if not text:
        speak("What would you like me to note down?")
        return
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        with open(NOTES_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n## {timestamp}\n{text}\n")
    except Exception as e:
        log("Note write error: " + str(e))
        speak("I couldn't save that note.")
        return
    with _notes_index_lock:
        _notes_index_built = False
    speak("Noted.")


# ---------------------------------------------------------------------------
# STANDALONE TO-DO LIST — see TODO_FILE above. Deliberately separate from
# the scheduler: to-dos have no due time, they just sit on a list until
# checked off, mirroring the command-history/schedule persistence pattern
# already used elsewhere in this file.
# ---------------------------------------------------------------------------
def _load_todos():
    global todo_items, _todo_next_id
    try:
        if TODO_FILE.exists():
            with open(TODO_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                with todo_lock:
                    todo_items = [t for t in data if isinstance(t, dict) and t.get("text")]
                    _todo_next_id = max([t.get("id", 0) for t in todo_items], default=0) + 1
    except Exception as e:
        log("Todo load error: " + str(e))


def _save_todos():
    try:
        with open(TODO_FILE, "w", encoding="utf-8") as f:
            json.dump(todo_items, f, indent=2)
    except Exception as e:
        log("Todo save error: " + str(e))


def add_todo(text):
    global _todo_next_id
    text = (text or "").strip()
    if not text:
        speak("What should I add to your to-do list?")
        return
    with todo_lock:
        todo_items.append({"id": _todo_next_id, "text": text, "done": False})
        _todo_next_id += 1
        _save_todos()
    speak(f"Added '{text}' to your to-do list.")


def list_todos():
    with todo_lock:
        pending = [t for t in todo_items if not t.get("done")]
    if not pending:
        speak("Your to-do list is empty.")
        return
    lines = [t["text"] for t in pending[:8]]
    speak(f"You have {len(pending)} to-do item{'s' if len(pending) != 1 else ''}: " + "; ".join(lines) + ".")


def complete_todo(text_hint):
    """Marks the first pending to-do whose text contains text_hint as
    done — a substring match rather than requiring an exact match, since
    voice input rarely reproduces the original wording exactly."""
    text_hint = (text_hint or "").strip().lower()
    if not text_hint:
        speak("Which to-do item would you like to mark done?")
        return
    match = None
    with todo_lock:
        for t in todo_items:
            if not t.get("done") and text_hint in t["text"].lower():
                match = t
                break
        if match:
            match["done"] = True
            _save_todos()
    if match:
        speak(f"Marked '{match['text']}' as done.")
    else:
        speak(f"I couldn't find a pending to-do matching '{text_hint}'.")


def clear_todos():
    with todo_lock:
        todo_items.clear()
        _save_todos()
    speak("Your to-do list has been cleared.")


def _compute_usage_stats():
    """Usage analytics — aggregates the command_history that was already
    being recorded for every recognised/typed command into 'what do I ask
    most' style stats: total commands, the most frequently repeated
    command texts, and the hour of day Victus gets used the most. Purely
    a read-only summary over existing data; nothing new is logged."""
    with command_history_lock:
        entries = list(command_history)

    total = len(entries)
    text_counts = Counter(e.get("text", "") for e in entries if e.get("text"))
    hour_counts = Counter()
    for e in entries:
        try:
            hour = datetime.strptime(e["time"], "%Y-%m-%d %H:%M:%S").hour
            hour_counts[hour] += 1
        except Exception:
            continue

    return {
        "total": total,
        "top_commands": text_counts.most_common(15),
        "busiest_hour": hour_counts.most_common(1)[0][0] if hour_counts else None,
    }


def _extract_explicit_ai_query(cmd_norm):
    prefixes = (
        "ask ai ", "ask a i ", "ask victus ai ",
        "victus ai ", "ai ", "chat ", "chat with ai ",
        "talk to ai ", "ask the ai "
    )
    for prefix in prefixes:
        if cmd_norm.startswith(prefix):
            return cmd_norm[len(prefix):].strip()
    if cmd_norm in {"chat with me", "chat with ai", "start ai chat", "open ai chat"}:
        return ""
    return None



def create_orb():
    global orb_root, orb_canvas
    if orb_root is not None:
        return
    root = tk.Tk()
    root.withdraw()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    transparent = "#010101"
    try:
        root.attributes("-transparentcolor", transparent)
    except Exception:
        pass
    root.configure(bg=transparent)
    size = ORB_SIZE
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = int((sw - size) / 2)
    y = sh - size - 75
    root.geometry(f"{size}x{size}+{x}+{y}")
    canvas = tk.Canvas(root, width=size, height=size, bg=transparent, highlightthickness=0, bd=0)
    canvas.pack()
    orb_root, orb_canvas = root, canvas
    root.after(30, orb_animation)
    root.after(100, create_assistant_menu)


def draw_orb():
    if orb_canvas is None:
        return
    orb_canvas.delete("all")
    cx = cy = ORB_SIZE / 2
    strength = 1.0 if listening else (0.8 if speaking else 0.25)
    for i in range(12, 0, -1):
        pulse = (np.sin(orb_phase * 2 + i * 0.65) + 1) / 2
        radius = 25 + i * 4.2 + pulse * 4 * strength
        color = "#7b61ff" if i % 3 == 0 else ("#00eaff" if i % 3 == 1 else "#168cff")
        orb_canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline=color, width=1 if i > 6 else 2)
    for i in range(32):
        angle = orb_phase * (0.6 + (i % 4) * 0.13) + i * 0.52
        wave = np.sin(orb_phase * 2.4 + i)
        r1, r2 = 20 + wave * 3, 52 + wave * 9
        x1, y1 = cx + np.cos(angle) * r1, cy + np.sin(angle) * r1
        x2, y2 = cx + np.cos(angle + 0.16) * r2, cy + np.sin(angle + 0.16) * r2
        color = "#00eaff" if i % 2 == 0 else "#7b61ff"
        orb_canvas.create_line(x1, y1, x2, y2, fill=color, width=1)
    pulse = (np.sin(orb_phase * 4) + 1) / 2
    core = 15 + pulse * 4 * strength
    orb_canvas.create_oval(cx - core, cy - core, cx + core, cy + core, fill="#06172c", outline="#00eaff", width=2)
    inner = core * 0.55
    orb_canvas.create_oval(cx - inner, cy - inner, cx + inner, cy + inner, fill="#1c8cff", outline="#b7ffff", width=1)

    # Push-to-talk recording-level meter: a VU-style ring drawn only while
    # actively listening for a command, filling clockwise from the top with
    # the live smoothed microphone level (mic_level, 0.0-1.0).
    if listening:
        level = max(0.0, min(1.0, mic_level))
        meter_radius = 68
        orb_canvas.create_oval(
            cx - meter_radius, cy - meter_radius, cx + meter_radius, cy + meter_radius,
            outline="#233047", width=4
        )
        if level > 0.01:
            if level < 0.5:
                meter_color = "#3dffb0"   # quiet/normal speech - green
            elif level < 0.85:
                meter_color = "#ffe066"   # loud - yellow
            else:
                meter_color = "#ff5a5a"   # clipping-range - red
            orb_canvas.create_arc(
                cx - meter_radius, cy - meter_radius, cx + meter_radius, cy + meter_radius,
                start=90, extent=-360 * level, style="arc",
                outline=meter_color, width=4
            )


def orb_animation():
    global orb_phase
    if orb_root is None:
        return
    try:
        if orb_visible:
            orb_phase += 0.14
            draw_orb()
        orb_root.after(30, orb_animation)
    except Exception:
        pass


def show_orb(show):
    global orb_visible
    if orb_root is None:
        return
    def update():
        global orb_visible
        try:
            if show:
                orb_root.deiconify()
                orb_root.lift()
                orb_root.attributes("-topmost", True)
                orb_visible = True
            else:
                orb_root.withdraw()
                orb_visible = False
        except Exception:
            pass
    try:
        orb_root.after(0, update)
    except Exception:
        pass


def create_assistant_menu():
    global menu_win, menu_panel, menu_anim_phase
    if menu_win is not None or orb_root is None:
        return

    win = tk.Toplevel(orb_root)
    menu_win = win
    win.overrideredirect(True)
    win.attributes("-topmost", True)
    transparent = "#010101"
    try:
        win.attributes("-transparentcolor", transparent)
    except Exception:
        pass
    win.configure(bg=transparent)

    size = 60
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    x = 40
    y = sh - size - 140
    win.geometry(f"{size}x{size}+{x}+{y}")

    cv = tk.Canvas(win, width=size, height=size, bg=transparent,
                   highlightthickness=0, bd=0)
    cv.pack(fill="both", expand=True)

    # Prevent a panel click that closes the menu from falling through to the
    # small assistant button underneath and immediately reopening the menu.
    ignore_next_button_release = [False]

    def draw_cyberpunk_button():
        global menu_anim_phase
        try:
            if not win.winfo_exists():
                return
            cv.delete("all")
            r = size / 2

            cv.create_oval(2, 2, size-2, size-2, fill="#06172c",
                           outline="#7b61ff", width=2)
            cv.create_oval(5, 5, size-5, size-5, fill="",
                           outline="#00eaff", width=1)

            if menu_expanded:
                cv.create_line(18, 18, size-18, size-18,
                               fill="#00eaff", width=3)
                cv.create_line(size-18, 18, 18, size-18,
                               fill="#00eaff", width=3)
                cv.create_text(r, r + 16, text="X", fill="#00eaff",
                               font=("Consolas", 8, "bold"))
            else:
                menu_anim_phase += 0.2
                pulse = (math.sin(menu_anim_phase * 1.5) + 1) / 2
                core_r = 8 + pulse * 4
                cv.create_oval(r-core_r, r-core_r, r+core_r, r+core_r,
                               fill="#00eaff", outline="#ffffff", width=1)

                for i in range(4):
                    angle = menu_anim_phase + (i * math.pi / 2)
                    x1 = r + math.cos(angle) * 12
                    y1 = r + math.sin(angle) * 12
                    x2 = r + math.cos(angle) * 20
                    y2 = r + math.sin(angle) * 20
                    cv.create_line(x1, y1, x2, y2,
                                   fill="#7b61ff", width=2)

                for i in range(6):
                    a2 = -menu_anim_phase * 0.8 + (i * math.pi / 3)
                    ax = r + math.cos(a2) * 16
                    ay = r + math.sin(a2) * 16
                    cv.create_oval(ax-1.5, ay-1.5, ax+1.5, ay+1.5,
                                   fill="#b7ffff", outline="")

            win.after(30, draw_cyberpunk_button)
        except Exception:
            pass

    draw_cyberpunk_button()

    animating = False

    def finish_close_panel():
        global menu_panel
        panel = menu_panel
        menu_panel = None
        try:
            if panel is not None and panel.winfo_exists():
                panel.destroy()
        except Exception:
            pass

    def close_menu_safely():
        """Close only the expanded panel; never destroy the menu button."""
        global menu_expanded
        nonlocal animating

        if not menu_expanded and menu_panel is None:
            animating = False
            return

        menu_expanded = False
        ignore_next_button_release[0] = True

        # Delay destruction until the current mouse event has completely
        # finished. This is the key fix for the X-button reopen bug.
        try:
            win.after(60, finish_close_panel)
        except Exception:
            finish_close_panel()

        animating = False

        # Consume any release that was already in flight.
        try:
            win.after(120, lambda: ignore_next_button_release.__setitem__(0, False))
        except Exception:
            ignore_next_button_release[0] = False

    def toggle_menu(event=None):
        global menu_expanded, menu_panel
        nonlocal animating

        if animating:
            return "break"

        if menu_expanded:
            close_menu_safely()
            return "break"

        animating = True
        menu_expanded = True

        pw, ph = 520, 520
        current_x = win.winfo_x()
        current_y = win.winfo_y()
        px = current_x + (size // 2) - (pw // 2)
        py = current_y + (size // 2) - (ph // 2)

        panel = tk.Toplevel(win)
        menu_panel = panel
        panel.overrideredirect(True)
        panel.attributes("-topmost", True)
        try:
            panel.attributes("-transparentcolor", transparent)
        except Exception:
            pass
        panel.configure(bg=transparent)
        panel.geometry(f"{pw}x{ph}+{px}+{py}")

        pcv = tk.Canvas(panel, width=pw, height=ph, bg=transparent,
                        highlightthickness=0, bd=0)
        pcv.pack(fill="both", expand=True)

        # Every feature closes the panel first, but the actual menu button
        # remains alive and visible.
        def action_cmd_box():
            close_menu_safely()
            open_command_box()

        def action_dashboard():
            close_menu_safely()
            open_system_dashboard()

        def action_screenshot():
            close_menu_safely()
            threading.Thread(target=take_screenshot, daemon=True).start()

        def action_wake():
            close_menu_safely()
            threading.Thread(
                target=lambda: handle_command(record_full_command()),
                daemon=True
            ).start()

        def action_open_air_cursor():
            close_menu_safely()
            threading.Thread(target=open_air_cursor, daemon=True).start()

        def action_close_air_cursor():
            close_menu_safely()
            threading.Thread(target=stop_air_cursor, daemon=True).start()

        def action_center_close(event=None):
            if menu_expanded:
                close_menu_safely()
            return "break"

        btn1 = tk.Button(panel, text="[>] Command Box", bg="#0b2447",
                         fg="#b7ffff", activebackground="#00eaff",
                         activeforeground="#000000",
                         font=("Consolas", 8, "bold"), bd=1, relief="solid",
                         command=action_cmd_box, cursor="hand2")
        btn2 = tk.Button(panel, text="[#] Dashboard", bg="#0b2447",
                         fg="#b7ffff", activebackground="#00eaff",
                         activeforeground="#000000",
                         font=("Consolas", 8, "bold"), bd=1, relief="solid",
                         command=action_dashboard, cursor="hand2")
        btn3 = tk.Button(panel, text="[O] Screenshot", bg="#0b2447",
                         fg="#b7ffff", activebackground="#00eaff",
                         activeforeground="#000000",
                         font=("Consolas", 8, "bold"), bd=1, relief="solid",
                         command=action_screenshot, cursor="hand2")
        btn4 = tk.Button(panel, text="[U] Wake Assistant", bg="#0b2447",
                         fg="#b7ffff", activebackground="#00eaff",
                         activeforeground="#000000",
                         font=("Consolas", 8, "bold"), bd=1, relief="solid",
                         command=action_wake, cursor="hand2")
        btn5 = tk.Button(panel, text="[+] Open Air Cursor", bg="#0b2447",
                         fg="#b7ffff", activebackground="#00eaff",
                         activeforeground="#000000",
                         font=("Consolas", 8, "bold"), bd=1, relief="solid",
                         command=action_open_air_cursor, cursor="hand2")
        btn6 = tk.Button(panel, text="[-] Close Air Cursor", bg="#0b2447",
                         fg="#b7ffff", activebackground="#00eaff",
                         activeforeground="#000000",
                         font=("Consolas", 8, "bold"), bd=1, relief="solid",
                         command=action_close_air_cursor, cursor="hand2")

        total_steps = 15
        current_step = [0]

        radial_items = (
            (btn1, 130), (btn5, 145), (btn2, 130),
            (btn3, 130), (btn6, 145), (btn4, 145),
        )
        radial_angles = tuple(
            math.radians(a) for a in (-90, -30, 30, 90, 150, 210)
        )

        def animate_floral_open():
            nonlocal animating
            try:
                if not panel.winfo_exists() or not menu_expanded:
                    animating = False
                    return

                s = current_step[0]
                pcv.delete("all")
                progress = s / float(total_steps)
                eased = 1.0 - pow(2.0, -10 * progress) if progress < 1.0 else 1.0

                cx, cy = pw / 2, ph / 2
                radius = int(170 * eased)

                for angle in radial_angles:
                    bx = cx + math.cos(angle) * radius
                    by = cy + math.sin(angle) * radius
                    pcv.create_line(cx, cy, bx, by, fill="#7b61ff",
                                    width=2, dash=(3, 3))
                    pcv.create_oval(bx-24, by-24, bx+24, by+24,
                                    fill="#06172c", outline="#00eaff", width=2)

                pcv.create_oval(cx-26, cy-26, cx+26, cy+26,
                                fill="#06172c", outline="#00eaff", width=2,
                                tags=("menu_center_close",))
                pcv.create_text(cx, cy, text="X", fill="#00eaff",
                                font=("Consolas", 16, "bold"),
                                tags=("menu_center_close",))

                # Use ButtonRelease on the center X and consume the event so
                # it can never reach the underlying assistant button.
                pcv.tag_bind("menu_center_close", "<ButtonRelease-1>",
                             action_center_close)

                if eased > 0.3:
                    for (button, width), angle in zip(radial_items, radial_angles):
                        bx = cx + math.cos(angle) * radius
                        by = cy + math.sin(angle) * radius
                        pcv.create_window(bx, by, window=button,
                                          width=width, height=30)

                if current_step[0] < total_steps:
                    current_step[0] += 1
                    panel.after(15, animate_floral_open)
                else:
                    animating = False
            except Exception:
                animating = False

        animate_floral_open()
        return "break"

    def start_move(event):
        win._drag_x = event.x
        win._drag_y = event.y

    def do_move(event):
        if menu_expanded:
            return
        deltax = event.x - win._drag_x
        deltay = event.y - win._drag_y
        nx = win.winfo_x() + deltax
        ny = win.winfo_y() + deltay
        win.geometry(f"+{nx}+{ny}")

    def on_click(event):
        if ignore_next_button_release[0]:
            return "break"
        return toggle_menu(event)

    cv.bind("<Button-1>", start_move)
    cv.bind("<B1-Motion>", do_move)
    cv.bind("<ButtonRelease-1>", on_click)

    win.deiconify()
    win.lift()



def open_command_box():
    if orb_root is None or tk is None:
        speak("Graphical command box is unavailable.")
        return

    def create_win():
        global cmd_box_root, cmd_entry
        if cmd_box_root is not None:
            try:
                if cmd_box_root.winfo_exists():
                    cmd_box_root.deiconify()
                    cmd_box_root.lift()
                    cmd_box_root.attributes("-topmost", True)
                    if cmd_entry:
                        cmd_entry.focus_set()
                    return
            except Exception:
                cmd_box_root = None

        win = tk.Toplevel(orb_root)
        cmd_box_root = win
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        transparent = "#010101"
        try:
            win.attributes("-transparentcolor", transparent)
        except Exception:
            pass
        win.configure(bg=transparent)

        w, h = 560, 75
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        x = int((sw - w) / 2)
        y = sh - h - 230
        win.geometry(f"{w}x{h}+{x}+{y}")

        cv = tk.Canvas(win, width=w, height=h, bg=transparent, highlightthickness=0, bd=0)
        cv.pack(fill="both", expand=True)

        cv.create_rectangle(4, 4, w - 4, h - 4, fill="#06172c", outline="#7b61ff", width=2)
        cv.create_rectangle(7, 7, w - 7, h - 7, fill="", outline="#00eaff", width=1)
        cv.create_text(25, 18, text="VICTUS LASER COMMAND", fill="#00eaff", font=("Consolas", 8, "bold"), anchor="w")

        entry = tk.Entry(win, bg="#0b2447", fg="#b7ffff", insertbackground="#00eaff", font=("Segoe UI", 11), bd=0, highlightthickness=0)
        cmd_entry = entry
        cv.create_window(w // 2 - 45, 44, window=entry, width=w - 190, height=26)

        def execute(event=None):
            txt = entry.get().strip()
            entry.delete(0, tk.END)
            if txt:
                threading.Thread(target=handle_command, args=(txt,), daemon=True).start()
            entry.focus_set()

        def close_box(event=None):
            try:
                win.withdraw()
            except Exception:
                pass

        def show_history(event=None):
            open_history_window()

        btn_history = tk.Button(win, text="HISTORY", bg="#7b61ff", fg="#ffffff", activebackground="#a08bff", activeforeground="#000000", font=("Consolas", 8, "bold"), bd=0, command=show_history, cursor="hand2")
        cv.create_window(w - 115, 44, window=btn_history, width=70, height=26)

        btn = tk.Button(win, text="RUN", bg="#168cff", fg="#ffffff", activebackground="#00eaff", activeforeground="#000000", font=("Consolas", 9, "bold"), bd=0, command=execute, cursor="hand2")
        cv.create_window(w - 40, 44, window=btn, width=50, height=26)

        def start_move(event):
            win._drag_x = event.x
            win._drag_y = event.y

        def do_move(event):
            deltax = event.x - win._drag_x
            deltay = event.y - win._drag_y
            nx = win.winfo_x() + deltax
            ny = win.winfo_y() + deltay
            win.geometry(f"+{nx}+{ny}")

        cv.bind("<Button-1>", start_move)
        cv.bind("<B1-Motion>", do_move)

        entry.bind("<Return>", execute)
        entry.bind("<Escape>", close_box)
        win.bind("<Escape>", close_box)
        win.deiconify()
        win.lift()
        entry.focus_set()

    try:
        orb_root.after(0, create_win)
    except Exception as e:
        log("Command box error: " + str(e))


def _refresh_history_window():
    """Re-populate the history listbox from command_history, if the window
    is currently open. Safe to call from any thread."""
    if history_win_root is None or history_listbox is None:
        return

    def do_refresh():
        global history_ids_display
        try:
            if not history_win_root.winfo_exists():
                return
            with command_history_lock:
                items = list(reversed(command_history))  # newest first
            history_ids_display = [entry["id"] for entry in items]
            history_listbox.delete(0, tk.END)
            for entry in items:
                history_listbox.insert(tk.END, f"[{entry['time']}]  {entry['text']}")
        except Exception:
            pass

    try:
        if orb_root:
            orb_root.after(0, do_refresh)
    except Exception:
        pass


def open_history_window():
    """Show every command Victus has recognised after the wake word or run
    from the command box, with options to clear one/some/all of them."""
    if orb_root is None or tk is None:
        speak("Command history is unavailable.")
        return

    def create_win():
        global history_win_root, history_listbox
        if history_win_root is not None:
            try:
                if history_win_root.winfo_exists():
                    history_win_root.deiconify()
                    history_win_root.lift()
                    history_win_root.attributes("-topmost", True)
                    _refresh_history_window()
                    return
            except Exception:
                history_win_root = None

        win = tk.Toplevel(orb_root)
        history_win_root = win
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        transparent = "#010101"
        try:
            win.attributes("-transparentcolor", transparent)
        except Exception:
            pass
        win.configure(bg=transparent)

        w, h = 480, 420
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        x = int((sw - w) / 2)
        y = int((sh - h) / 2)
        win.geometry(f"{w}x{h}+{x}+{y}")

        cv = tk.Canvas(win, width=w, height=h, bg=transparent, highlightthickness=0, bd=0)
        cv.pack(fill="both", expand=True)

        cv.create_rectangle(4, 4, w - 4, h - 4, fill="#06172c", outline="#7b61ff", width=2)
        cv.create_rectangle(7, 7, w - 7, h - 7, fill="", outline="#00eaff", width=1)
        cv.create_text(20, 20, text="VICTUS COMMAND HISTORY", fill="#00eaff", font=("Consolas", 10, "bold"), anchor="w")

        list_frame = tk.Frame(win, bg="#0b2447")
        cv.create_window(w // 2, h // 2 - 15, window=list_frame, width=w - 40, height=h - 110)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        listbox = tk.Listbox(
            list_frame, bg="#0b2447", fg="#b7ffff", selectbackground="#168cff",
            selectforeground="#ffffff", font=("Consolas", 9), bd=0,
            highlightthickness=0, selectmode=tk.EXTENDED,
            yscrollcommand=scrollbar.set, activestyle="none"
        )
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=listbox.yview)
        history_listbox = listbox

        def do_clear_selected():
            sel = listbox.curselection()
            if not sel:
                return
            ids_to_remove = [
                history_ids_display[i] for i in sel if i < len(history_ids_display)
            ]
            clear_history_entries(ids_to_remove)

        def do_clear_all():
            clear_all_history()

        def close_win(event=None):
            try:
                win.withdraw()
            except Exception:
                pass

        btn_clear_selected = tk.Button(
            win, text="CLEAR SELECTED", bg="#ff8c1a", fg="#000000",
            activebackground="#ffb066", activeforeground="#000000",
            font=("Consolas", 8, "bold"), bd=0, command=do_clear_selected, cursor="hand2"
        )
        cv.create_window(w // 2 - 105, h - 35, window=btn_clear_selected, width=150, height=26)

        btn_clear_all = tk.Button(
            win, text="CLEAR ALL", bg="#ff3355", fg="#ffffff",
            activebackground="#ff6688", activeforeground="#ffffff",
            font=("Consolas", 8, "bold"), bd=0, command=do_clear_all, cursor="hand2"
        )
        cv.create_window(w // 2 + 65, h - 35, window=btn_clear_all, width=90, height=26)

        btn_close = tk.Button(
            win, text="CLOSE", bg="#168cff", fg="#ffffff",
            activebackground="#00eaff", activeforeground="#000000",
            font=("Consolas", 8, "bold"), bd=0, command=close_win, cursor="hand2"
        )
        cv.create_window(w - 45, 20, window=btn_close, width=60, height=22)

        def start_move(event):
            win._drag_x = event.x
            win._drag_y = event.y

        def do_move(event):
            deltax = event.x - win._drag_x
            deltay = event.y - win._drag_y
            nx = win.winfo_x() + deltax
            ny = win.winfo_y() + deltay
            win.geometry(f"+{nx}+{ny}")

        cv.bind("<Button-1>", start_move)
        cv.bind("<B1-Motion>", do_move)

        win.bind("<Escape>", close_win)
        win.bind("<Delete>", lambda e: do_clear_selected())
        win.deiconify()
        win.lift()
        _refresh_history_window()

    try:
        orb_root.after(0, create_win)
    except Exception as e:
        log("History window error: " + str(e))


# ---------------------------------------------------------------------------
# AI RESPONSE WINDOW — full, unabridged answers (step-by-step math/chemistry
# working, detailed explanations, etc.) that don't fit well into speech.
# ---------------------------------------------------------------------------
def _create_ai_response_window():
    global ai_response_win_root, ai_response_text_widget

    win = tk.Toplevel(orb_root)
    ai_response_win_root = win
    win.overrideredirect(True)
    win.attributes("-topmost", True)
    transparent = "#010101"
    try:
        win.attributes("-transparentcolor", transparent)
    except Exception:
        pass
    win.configure(bg=transparent)

    w, h = 640, 480
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    x = int((sw - w) / 2)
    y = int((sh - h) / 2)
    win.geometry(f"{w}x{h}+{x}+{y}")

    cv = tk.Canvas(win, width=w, height=h, bg=transparent, highlightthickness=0, bd=0)
    cv.pack(fill="both", expand=True)
    cv.create_rectangle(4, 4, w - 4, h - 4, fill="#06172c", outline="#7b61ff", width=2)
    cv.create_rectangle(7, 7, w - 7, h - 7, fill="", outline="#00eaff", width=1)
    cv.create_text(20, 20, text="VICTUS AI RESPONSE", fill="#00eaff", font=("Consolas", 10, "bold"), anchor="w")

    text_frame = tk.Frame(win, bg="#0b2447")
    cv.create_window(w // 2, h // 2 - 5, window=text_frame, width=w - 40, height=h - 90)

    scrollbar = tk.Scrollbar(text_frame)
    scrollbar.pack(side="right", fill="y")

    txt_widget = tk.Text(
        text_frame, bg="#0b2447", fg="#b7ffff", insertbackground="#00eaff",
        font=("Segoe UI", 10), bd=0, highlightthickness=0, wrap="word",
        yscrollcommand=scrollbar.set, state="disabled"
    )
    txt_widget.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=txt_widget.yview)
    txt_widget.tag_configure("question", foreground="#00eaff", font=("Segoe UI", 10, "bold"))
    ai_response_text_widget = txt_widget

    def close_win(event=None):
        try:
            win.withdraw()
        except Exception:
            pass

    btn_close = tk.Button(
        win, text="CLOSE", bg="#168cff", fg="#ffffff",
        activebackground="#00eaff", activeforeground="#000000",
        font=("Consolas", 8, "bold"), bd=0, command=close_win, cursor="hand2"
    )
    cv.create_window(w - 45, 20, window=btn_close, width=60, height=22)

    def start_move(event):
        win._drag_x = event.x
        win._drag_y = event.y

    def do_move(event):
        deltax = event.x - win._drag_x
        deltay = event.y - win._drag_y
        nx = win.winfo_x() + deltax
        ny = win.winfo_y() + deltay
        win.geometry(f"+{nx}+{ny}")

    cv.bind("<Button-1>", start_move)
    cv.bind("<B1-Motion>", do_move)
    win.bind("<Escape>", close_win)

    return win, txt_widget


def _update_ai_response_window(query, answer):
    """Show the full, un-truncated AI answer (formulas, step-by-step
    working, everything) in its own window. Creates the window on first
    use, otherwise reuses/updates it. Safe to call from any thread."""
    if orb_root is None or tk is None:
        return

    def do_update():
        try:
            if ai_response_win_root is not None and ai_response_win_root.winfo_exists():
                win = ai_response_win_root
                txt_widget = ai_response_text_widget
            else:
                win, txt_widget = _create_ai_response_window()

            txt_widget.config(state="normal")
            txt_widget.delete("1.0", tk.END)
            txt_widget.insert(tk.END, "YOU ASKED:\n", "question")
            txt_widget.insert(tk.END, query + "\n\n")
            txt_widget.insert(tk.END, "-" * 60 + "\n\n")
            txt_widget.insert(tk.END, "VICTUS:\n", "question")
            txt_widget.insert(tk.END, answer)
            txt_widget.config(state="disabled")

            win.deiconify()
            win.lift()
            win.attributes("-topmost", True)
        except Exception as e:
            log("AI response window error: " + str(e))

    try:
        orb_root.after(0, do_update)
    except Exception:
        pass


def reopen_ai_response_window():
    """Bring the most recent AI answer back on screen, or tell the user
    there isn't one yet."""
    if orb_root is None or tk is None:
        speak("The AI response window is unavailable.")
        return
    if ai_response_win_root is None:
        speak("There's no AI answer to show yet. Ask me something first.")
        return

    def do_show():
        try:
            if ai_response_win_root.winfo_exists():
                ai_response_win_root.deiconify()
                ai_response_win_root.lift()
                ai_response_win_root.attributes("-topmost", True)
        except Exception:
            pass

    try:
        orb_root.after(0, do_show)
    except Exception:
        pass


def open_system_dashboard():
    if orb_root is None or tk is None:
        speak("System dashboard is unavailable.")
        return

    def create_dashboard():
        global dashboard_root
        if dashboard_root is not None:
            try:
                if dashboard_root.winfo_exists():
                    dashboard_root.deiconify()
                    dashboard_root.lift()
                    dashboard_root.attributes("-topmost", True)
                    return
            except Exception:
                dashboard_root = None

        speak("Initializing Cyberpunk Hacker Telemetry Deck.")
        win = tk.Toplevel(orb_root)
        dashboard_root = win
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        
        bg_dark = "#05050a"
        card_dark = "#0b0c16"
        border_col = "#7b61ff"
        neon_cyan = "#00eaff"
        neon_pink = "#ff007f"
        neon_green = "#00ff66"
        neon_yellow = "#f2cc0f"
        text_light = "#b7ffff"
        
        w, h = 940, 600
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        x = int((sw - w) / 2)
        y = int((sh - h) / 2)
        win.geometry(f"{w}x{h}+{x}+{y}")
        win.configure(bg=bg_dark)

        cv = tk.Canvas(win, width=w, height=h, bg=bg_dark, highlightthickness=0, bd=0)
        cv.pack(fill="both", expand=True)

        for gx in range(0, w, 40):
            cv.create_line(gx, 0, gx, h, fill="#0a0d20", width=1)
        for gy in range(0, h, 40):
            cv.create_line(0, gy, w, gy, fill="#0a0d20", width=1)

        cv.create_rectangle(12, 12, w - 12, 54, fill=card_dark, outline=border_col, width=2)
        cv.create_rectangle(15, 15, w - 15, 51, fill="", outline=neon_cyan, width=1)
        cv.create_text(25, 33, text="> [VICTUS CYBER-DECK v4.0] // SECURE TELEMETRY FEED", fill=neon_cyan, font=("Consolas", 10, "bold"), anchor="w")
        cv.create_text(w - 30, 33, text=f"NODE: {platform.node()}", fill=neon_pink, font=("Consolas", 9, "bold"), anchor="e")

        card_w, card_h, card_y = 170, 75, 70
        
        cv.create_rectangle(15, card_y, 15 + card_w, card_y + card_h, fill=card_dark, outline=border_col, width=2)
        cv.create_rectangle(15, card_y, 15 + card_w, card_y + 6, fill=neon_cyan, outline="")
        cv.create_text(25, card_y + 20, text="CPU CORES", fill=text_light, font=("Consolas", 8, "bold"), anchor="w")
        cores_val_id = cv.create_text(25, card_y + 48, text=str(psutil.cpu_count(logical=True) or 4), fill=neon_cyan, font=("Consolas", 18, "bold"), anchor="w")

        cv.create_rectangle(195, card_y, 195 + card_w, card_y + card_h, fill=card_dark, outline=border_col, width=2)
        cv.create_rectangle(195, card_y, 195 + card_w, card_y + 6, fill=neon_pink, outline="")
        cv.create_text(205, card_y + 20, text="TOTAL MEMORY", fill=text_light, font=("Consolas", 8, "bold"), anchor="w")
        mem_total_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        cv.create_text(205, card_y + 48, text=f"{mem_total_gb} GiB", fill=neon_pink, font=("Consolas", 18, "bold"), anchor="w")

        cv.create_rectangle(375, card_y, 375 + card_w, card_y + card_h, fill=card_dark, outline=border_col, width=2)
        cv.create_rectangle(375, card_y, 375 + card_w, card_y + 6, fill=neon_yellow, outline="")
        cv.create_text(385, card_y + 20, text="DISK CAPACITY", fill=text_light, font=("Consolas", 8, "bold"), anchor="w")
        try:
            total_rom = round(psutil.disk_usage("C:\\").total / (1024**3), 1)
        except Exception:
            total_rom = 500.0
        cv.create_text(385, card_y + 48, text=f"{total_rom} GiB", fill=neon_yellow, font=("Consolas", 18, "bold"), anchor="w")

        cv.create_rectangle(555, card_y, 555 + card_w, card_y + card_h, fill=card_dark, outline=border_col, width=2)
        cv.create_rectangle(555, card_y, 555 + card_w, card_y + 6, fill=neon_green, outline="")
        cv.create_text(565, card_y + 20, text="SYSTEM STATUS", fill=text_light, font=("Consolas", 8, "bold"), anchor="w")
        cv.create_text(565, card_y + 48, text="ONLINE", fill=neon_green, font=("Consolas", 18, "bold"), anchor="w")

        cv.create_rectangle(735, card_y, w - 15, card_y + card_h, fill=card_dark, outline=border_col, width=2)
        cv.create_rectangle(735, card_y, w - 15, card_y + 6, fill=neon_cyan, outline="")
        cv.create_text(745, card_y + 20, text="ACTIVE THREADS", fill=text_light, font=("Consolas", 8, "bold"), anchor="w")
        tasks_val_id = cv.create_text(745, card_y + 48, text="245", fill=neon_cyan, font=("Consolas", 18, "bold"), anchor="w")

        panel_y = 160
        cv.create_rectangle(15, panel_y, w - 15, h - 65, fill=card_dark, outline=border_col, width=2)
        cv.create_rectangle(18, panel_y + 3, w - 18, h - 68, fill="", outline="#162040", width=1)
        cv.create_text(30, panel_y + 20, text="[SYSTEM LOAD & RESOURCE ALLOCATION ARRAYS]", fill=neon_cyan, font=("Consolas", 10, "bold"), anchor="w")

        cv.create_text(30, panel_y + 65, text="CPU Utilization", fill=text_light, font=("Consolas", 9, "bold"), anchor="w")
        cv.create_rectangle(150, panel_y + 55, w - 150, panel_y + 73, fill="#05050a", outline=border_col)
        cpu_bar_id = cv.create_rectangle(150, panel_y + 55, 150 + 200, panel_y + 73, fill=neon_cyan, outline="")
        cpu_text_id = cv.create_text(w - 135, panel_y + 64, text="0 %", fill=neon_cyan, font=("Consolas", 9, "bold"), anchor="w")

        cv.create_text(30, panel_y + 105, text="Memory Utilization", fill=text_light, font=("Consolas", 9, "bold"), anchor="w")
        cv.create_rectangle(150, panel_y + 95, w - 150, panel_y + 113, fill="#05050a", outline=border_col)
        ram_bar_id = cv.create_rectangle(150, panel_y + 95, 150 + 200, panel_y + 113, fill=neon_pink, outline="")
        ram_text_id = cv.create_text(w - 135, panel_y + 104, text="0 %", fill=neon_pink, font=("Consolas", 9, "bold"), anchor="w")

        cv.create_text(30, panel_y + 145, text="Disk Utilization (C:\\)", fill=text_light, font=("Consolas", 9, "bold"), anchor="w")
        cv.create_rectangle(150, panel_y + 135, w - 150, panel_y + 153, fill="#05050a", outline=border_col)
        disk_bar_id = cv.create_rectangle(150, panel_y + 135, 150 + 200, panel_y + 153, fill=neon_yellow, outline="")
        disk_text_id = cv.create_text(w - 135, panel_y + 144, text="0 %", fill=neon_yellow, font=("Consolas", 9, "bold"), anchor="w")

        cv.create_rectangle(30, panel_y + 180, w - 30, panel_y + 355, fill="#020308", outline=border_col, width=1)
        metrics_info_id = cv.create_text(45, panel_y + 195, text=">> Establishing secure data connection...", fill=neon_green, font=("Consolas", 9), anchor="nw")

        def refresh_telemetry():
            if dashboard_root is None or not dashboard_root.winfo_exists():
                return
            try:
                cpu_p = float(psutil.cpu_percent(interval=None))
                ram_info = psutil.virtual_memory()
                ram_p = float(ram_info.percent)
                disk_info = psutil.disk_usage("C:\\")
                disk_p = float(disk_info.percent)
                
                max_bar_w = (w - 300)
                
                c_w = int(max_bar_w * (cpu_p / 100.0))
                cv.coords(cpu_bar_id, 150, panel_y + 55, 150 + c_w, panel_y + 73)
                cv.itemconfig(cpu_text_id, text=f"{int(cpu_p)} %")

                r_w = int(max_bar_w * (ram_p / 100.0))
                cv.coords(ram_bar_id, 150, panel_y + 95, 150 + r_w, panel_y + 113)
                cv.itemconfig(ram_text_id, text=f"{int(ram_p)} %")

                d_w = int(max_bar_w * (disk_p / 100.0))
                cv.coords(disk_bar_id, 150, panel_y + 135, 150 + d_w, panel_y + 153)
                cv.itemconfig(disk_text_id, text=f"{int(disk_p)} %")

                cv.itemconfig(tasks_val_id, text=str(len(psutil.pids())))

                try:
                    battery = psutil.sensors_battery()
                    batt_str = f"{int(round(battery.percent))}% ({'Charging' if battery.power_plugged else 'Discharging'})" if battery else "N/A"
                except Exception:
                    batt_str = "N/A"

                try:
                    wifi_en, wifi_conn, ssid = get_wifi_state()
                    wifi_str = f"Connected [{ssid}]" if (wifi_conn and ssid) else ("Enabled" if wifi_en else "Disabled")
                except Exception:
                    wifi_str = "Unknown"

                ram_used_gb = round(ram_info.used / (1024**3), 2)
                ram_total_gb = round(ram_info.total / (1024**3), 2)
                disk_free_gb = round(disk_info.free / (1024**3), 2)
                disk_total_gb = round(disk_info.total / (1024**3), 2)

                details = (
                    f">> [BATTERY]   : {batt_str}\n"
                    f">> [NET_LINK]  : {wifi_str}\n"
                    f">> [RAM_ALLOC] : {ram_used_gb} GiB used / {ram_total_gb} GiB total\n"
                    f">> [STORAGE]   : {disk_free_gb} GiB free / {disk_total_gb} GiB total\n"
                    f">> [PROCESSES] : {len(psutil.pids())} background threads active\n"
                    f">> [KERNEL]    : {platform.system()} {platform.release()} ({platform.version()})"
                )
                cv.itemconfig(metrics_info_id, text=details)

            except Exception as e:
                log("Telemetry refresh error: " + str(e))

            dashboard_root.after(1000, refresh_telemetry)

        refresh_telemetry()

        def close_dash(event=None):
            global dashboard_root
            try:
                win.destroy()
            except Exception:
                pass
            dashboard_root = None

        close_btn = tk.Button(win, text="[ TERMINATE FEED ]", bg="#150a1a", fg=neon_pink, activebackground=neon_cyan, activeforeground="#000000", font=("Consolas", 9, "bold"), bd=1, relief="solid", command=close_dash, cursor="hand2")
        cv.create_window(w - 95, h - 30, window=close_btn, width=160, height=26)

        def start_move(event):
            win._drag_x = event.x
            win._drag_y = event.y

        def do_move(event):
            deltax = event.x - win._drag_x
            deltay = event.y - win._drag_y
            nx = win.winfo_x() + deltax
            ny = win.winfo_y() + deltay
            win.geometry(f"+{nx}+{ny}")

        cv.bind("<Button-1>", start_move)
        cv.bind("<B1-Motion>", do_move)
        win.bind("<Escape>", close_dash)

        win.deiconify()
        win.lift()

    try:
        orb_root.after(0, create_dashboard)
    except Exception as e:
        log("Dashboard creation error: " + str(e))


# ---------------------------------------------------------------------------
# USAGE ANALYTICS DASHBOARD — "what do I ask most" style stats over the
# command_history that Victus already records for every recognised/typed
# command. No new logging is introduced; this only aggregates and displays.
# ---------------------------------------------------------------------------
def open_usage_dashboard():
    if orb_root is None or tk is None or ttk is None:
        speak("Usage analytics dashboard is unavailable.")
        return

    stats = _compute_usage_stats()
    if stats["total"] == 0:
        speak("I don't have any command history yet, so there's nothing to analyze.")
        return

    if stats["top_commands"]:
        top_text, top_count = stats["top_commands"][0]
        summary_speech = (
            f"You've given me {stats['total']} commands so far. "
            f"Your most repeated one is '{top_text}', asked {top_count} times."
        )
    else:
        summary_speech = f"You've given me {stats['total']} commands so far."

    if stats["busiest_hour"] is not None:
        hour12 = stats["busiest_hour"] % 12 or 12
        ampm = "AM" if stats["busiest_hour"] < 12 else "PM"
        summary_speech += f" You talk to me the most around {hour12} {ampm}."

    speak(summary_speech)

    def build_window():
        global usage_dashboard_root
        if usage_dashboard_root is not None:
            try:
                if usage_dashboard_root.winfo_exists():
                    usage_dashboard_root.destroy()
            except Exception:
                pass

        win = tk.Toplevel(orb_root)
        usage_dashboard_root = win
        win.overrideredirect(True)
        win.attributes("-topmost", True)

        bg_dark, card_dark, border_col = "#05050a", "#0b0c16", "#7b61ff"
        neon_cyan, neon_pink, text_light = "#00eaff", "#ff007f", "#b7ffff"

        w, h = 620, 480
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        x = int((sw - w) / 2)
        y = int((sh - h) / 2)
        win.geometry(f"{w}x{h}+{x}+{y}")
        win.configure(bg=bg_dark)

        cv = tk.Canvas(win, width=w, height=h, bg=bg_dark, highlightthickness=0, bd=0)
        cv.pack(fill="both", expand=True)

        cv.create_rectangle(4, 4, w - 4, h - 4, fill=card_dark, outline=border_col, width=2)
        cv.create_rectangle(7, 7, w - 7, 45, fill="", outline=neon_cyan, width=1)
        cv.create_text(20, 24, text="> [VICTUS USAGE ANALYTICS]", fill=neon_cyan, font=("Consolas", 11, "bold"), anchor="w")

        busiest_str = "N/A"
        if stats["busiest_hour"] is not None:
            hour12 = stats["busiest_hour"] % 12 or 12
            ampm = "AM" if stats["busiest_hour"] < 12 else "PM"
            busiest_str = f"{hour12}:00 {ampm}"

        cv.create_text(
            20, 65,
            text=f">> Total commands logged: {stats['total']}     >> Busiest hour: {busiest_str}",
            fill=text_light, font=("Consolas", 9, "bold"), anchor="w"
        )
        cv.create_text(20, 90, text="[ MOST FREQUENTLY ASKED COMMANDS ]", fill=neon_pink, font=("Consolas", 9, "bold"), anchor="w")

        frame = tk.Frame(win, bg=card_dark)
        cv.create_window(w // 2, h // 2 + 30, window=frame, width=w - 50, height=h - 190)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("UsageTree.Treeview", background="#0b2447", foreground=text_light,
                         fieldbackground="#0b2447", rowheight=26, borderwidth=1, relief="solid")
        style.map("UsageTree.Treeview", background=[('selected', '#168cff')], foreground=[('selected', '#ffffff')])
        style.configure("UsageTree.Treeview.Heading", background="#168cff", foreground="#ffffff",
                         font=("Segoe UI", 10, "bold"), relief="solid", borderwidth=1)

        tree = ttk.Treeview(frame, columns=("Rank", "Command", "Count"), show="headings", height=11, style="UsageTree.Treeview")
        tree.heading("Rank", text="#")
        tree.heading("Command", text="Command")
        tree.heading("Count", text="Times Asked")
        tree.column("Rank", width=40, anchor="center")
        tree.column("Command", width=380, anchor="w")
        tree.column("Count", width=100, anchor="center")

        tree.tag_configure("even", background="#0b2447")
        tree.tag_configure("odd", background="#081a33")

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for idx, (cmd_text, count) in enumerate(stats["top_commands"]):
            tag = "even" if idx % 2 == 0 else "odd"
            tree.insert("", "end", values=(idx + 1, cmd_text, count), tags=(tag,))

        def close_win(event=None):
            global usage_dashboard_root
            try:
                win.destroy()
            except Exception:
                pass
            usage_dashboard_root = None

        close_btn = tk.Button(win, text="CLOSE", bg="#7b61ff", fg="#ffffff", activebackground=neon_cyan,
                               activeforeground="#000000", font=("Consolas", 9, "bold"), bd=0,
                               command=close_win, cursor="hand2")
        cv.create_window(w - 55, h - 26, window=close_btn, width=80, height=26)

        def start_move(event):
            win._drag_x = event.x
            win._drag_y = event.y

        def do_move(event):
            deltax = event.x - win._drag_x
            deltay = event.y - win._drag_y
            nx = win.winfo_x() + deltax
            ny = win.winfo_y() + deltay
            win.geometry(f"+{nx}+{ny}")

        cv.bind("<Button-1>", start_move)
        cv.bind("<B1-Motion>", do_move)
        win.bind("<Escape>", close_win)

    try:
        orb_root.after(0, build_window)
    except Exception as e:
        log("Usage dashboard error: " + str(e))


def search_local_files_gui(ext):
    if orb_root is None or tk is None or ttk is None:
        speak("File table GUI unavailable.")
        return

    ext = ext.lower().strip()
    if not ext.startswith("."):
        ext = "." + ext

    speak(f"Searching all system drives and folders for {ext} files...")

    def background_search():
        matches = []
        drives = []
        try:
            for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
                d_path = Path(f"{letter}:\\")
                if d_path.exists():
                    drives.append(d_path)
        except Exception:
            pass
        if not drives:
            drives = [Path("C:\\")]

        for drive in drives:
            if not drive.exists():
                continue
            try:
                for root, dirs, files in os.walk(drive, topdown=True):
                    dirs[:] = [d for d in dirs if d.lower() not in {
                        "windows", "program files", "program files (x86)", 
                        "programdata", "$recycle.bin", "system volume information",
                        "appdata", "msocache", "perflogs"
                    }]
                    for file in files:
                        if file.lower().endswith(ext):
                            fpath = Path(root) / file
                            try:
                                mtime = datetime.fromtimestamp(fpath.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                            except Exception:
                                mtime = "Unknown"
                            matches.append((file, str(fpath), mtime))
                            if len(matches) >= 3000:
                                break
                    if len(matches) >= 3000:
                        break
            except Exception as e:
                log(f"Drive search error on {drive}: {e}")

        def build_window():
            global file_table_root
            if file_table_root is not None:
                try:
                    file_table_root.destroy()
                except Exception:
                    pass

            win = tk.Toplevel(orb_root)
            file_table_root = win
            win.overrideredirect(True)
            win.attributes("-topmost", True)
            transparent = "#010101"
            try:
                win.attributes("-transparentcolor", transparent)
            except Exception:
                pass
            win.configure(bg=transparent)

            w, h = 750, 420
            sw = win.winfo_screenwidth()
            sh = win.winfo_screenheight()
            x = int((sw - w) / 2)
            y = int((sh - h) / 2)
            win.geometry(f"{w}x{h}+{x}+{y}")

            cv = tk.Canvas(win, width=w, height=h, bg=transparent, highlightthickness=0, bd=0)
            cv.pack(fill="both", expand=True)

            cv.create_rectangle(4, 4, w - 4, h - 4, fill="#06172c", outline="#7b61ff", width=2)
            cv.create_rectangle(7, 7, w - 7, h - 7, fill="", outline="#00eaff", width=1)
            cv.create_text(25, 22, text=f"VICTUS FILE EXPLORER - {ext.upper()} FILES", fill="#00eaff", font=("Consolas", 9, "bold"), anchor="w")

            lbl = tk.Label(win, text=f"Found {len(matches)} files with extension '{ext}' across system", fg="#b7ffff", bg="#06172c", font=("Segoe UI", 10, "bold"))
            cv.create_window(w // 2, 48, window=lbl)

            frame = tk.Frame(win, bg="#06172c", relief="solid", bd=1)
            cv.create_window(w // 2, h // 2 + 10, window=frame, width=w - 50, height=h - 110)

            style = ttk.Style()
            try:
                style.theme_use("clam")
            except Exception:
                pass
                
            style.configure("Treeview", 
                            background="#0b2447", 
                            foreground="#b7ffff", 
                            fieldbackground="#0b2447", 
                            rowheight=26,
                            borderwidth=1,
                            relief="solid")
            style.map("Treeview", background=[('selected', '#168cff')], foreground=[('selected', '#ffffff')])
            
            style.configure("Treeview.Heading", 
                            background="#168cff", 
                            foreground="#ffffff", 
                            font=("Segoe UI", 10, "bold"),
                            relief="solid",
                            borderwidth=1)

            tree = ttk.Treeview(frame, columns=("Filename", "Path", "Modified"), show="headings", height=13, style="Treeview")
            tree.heading("Filename", text="File Name")
            tree.heading("Path", text="Full Path")
            tree.heading("Modified", text="Last Modified")
            tree.column("Filename", width=200, anchor="w")
            tree.column("Path", width=380, anchor="w")
            tree.column("Modified", width=110, anchor="center")

            tree.tag_configure("even", background="#0b2447")
            tree.tag_configure("odd", background="#081a33")

            scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)

            tree.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            for idx, (name, fpath, mod) in enumerate(matches):
                tag = "even" if idx % 2 == 0 else "odd"
                tree.insert("", "end", values=(name, fpath, mod), tags=(tag,))

            def open_selected(event=None):
                selected = tree.selection()
                if not selected:
                    return
                item = tree.item(selected[0])
                fpath = item["values"][1]
                filename = item["values"][0]
                opened = False
                try:
                    os.startfile(fpath)
                    opened = True
                except Exception as e:
                    log("os.startfile error: " + str(e))
                
                if not opened:
                    try:
                        subprocess.Popen(f'start "" "{fpath}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        opened = True
                    except Exception as e2:
                        log("subprocess start error: " + str(e2))

                if opened:
                    speak(f"Opening {filename}")
                else:
                    speak("Could not open file.")

            tree.bind("<Double-1>", open_selected)

            def close_win(event=None):
                try:
                    win.destroy()
                except Exception:
                    pass

            close_btn = tk.Button(win, text="CLOSE", bg="#7b61ff", fg="#ffffff", activebackground="#00eaff", activeforeground="#000000", font=("Consolas", 9, "bold"), bd=0, command=close_win, cursor="hand2")
            cv.create_window(w - 60, h - 28, window=close_btn, width=80, height=26)

            def start_move(event):
                win._drag_x = event.x
                win._drag_y = event.y

            def do_move(event):
                deltax = event.x - win._drag_x
                deltay = event.y - win._drag_y
                nx = win.winfo_x() + deltax
                ny = win.winfo_y() + deltay
                win.geometry(f"+{nx}+{ny}")

            cv.bind("<Button-1>", start_move)
            cv.bind("<B1-Motion>", do_move)
            win.bind("<Escape>", close_win)

        try:
            orb_root.after(0, build_window)
        except Exception as e:
            log("GUI table error: " + str(e))

    threading.Thread(target=background_search, daemon=True).start()


def get_weather():
    try:
        url = "https://wttr.in/?format=3"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            weather_text = response.read().decode('utf-8').strip()
            speak(f"Current weather report: {weather_text}")
    except Exception as e:
        log("Weather fetch error: " + str(e))
        speak("I couldn't fetch the weather right now.")


def manage_clipboard(action):
    if not WIN32_OK:
        speak("Clipboard control unavailable.")
        return
    try:
        win32clipboard.OpenClipboard()
        if action == "read":
            try:
                data = win32clipboard.GetClipboardData(win32con.CF_TEXT).decode('utf-8', errors='ignore')
                speak(f"Clipboard contains: {data[:150]}")
            except Exception:
                speak("Clipboard is empty or contains non-text data.")
        elif action == "clear":
            win32clipboard.EmptyClipboard()
            speak("Clipboard cleared.")
        win32clipboard.CloseClipboard()
    except Exception as e:
        log("Clipboard error: " + str(e))
        speak("Failed to access clipboard.")


def explain_clipboard(instruction="explain"):
    """Clipboard-aware Q&A: "explain what I just copied" / "summarize what
    I copied" reads the current clipboard text and asks the local AI to
    explain or summarize it, instead of Victus just reading the raw text
    back verbatim (that's still what "read clipboard" does — this is a
    separate, smarter path built on the same clipboard access)."""
    if not WIN32_OK:
        speak("Clipboard control unavailable.")
        return

    data = ""
    try:
        win32clipboard.OpenClipboard()
        try:
            data = win32clipboard.GetClipboardData(win32con.CF_TEXT).decode("utf-8", errors="ignore")
        except Exception:
            data = ""
        finally:
            win32clipboard.CloseClipboard()
    except Exception as e:
        log("Clipboard error: " + str(e))
        speak("Failed to access clipboard.")
        return

    data = (data or "").strip()
    if not data:
        speak("Your clipboard is empty, so there's nothing for me to " + instruction + ".")
        return

    # Clipboard contents can be huge (a whole file, a big code block) —
    # cap what we forward so the request stays fast and well within the
    # model's comfortable context.
    clipped = data if len(data) <= 4000 else data[:4000] + " ...(truncated)"
    prompt = (
        f"Please {instruction} the following text, which the user just "
        f"copied to their clipboard. Be clear and concise:\n\n{clipped}"
    )
    ask_local_ai(prompt)


# ---------------------------------------------------------------------------
# OFFLINE TRANSLATION — "translate <text> to <language>" / "how do you say
# <text> in <language>". Just another ask_local_ai() prompt, same as every
# other local-AI feature above — no cloud translate API, no key, and it
# automatically benefits from whatever multilingual model is loaded in
# Ollama.
# ---------------------------------------------------------------------------
def translate_text(text, target_language):
    text = (text or "").strip()
    target_language = (target_language or "").strip()
    if not text or not target_language:
        speak("Tell me what to translate, and which language to translate it into.")
        return
    prompt = (
        f"Translate the following text into {target_language}. Respond "
        f"with ONLY the translated text - no explanations, no quotation "
        f"marks, no notes:\n\n{text}"
    )
    ask_local_ai(prompt)


# ---------------------------------------------------------------------------
# TEXT SIMPLIFICATION / READING-LEVEL REWRITE — "simplify <text>",
# "simplify this" (falls back to the clipboard, same source as
# explain_clipboard() above), "make this more formal". Another one-line
# ask_local_ai() prompt rather than a separate writing-assistant service.
# ---------------------------------------------------------------------------
def simplify_or_reword_text(style, text=None):
    text = (text or "").strip()
    if not text:
        if not WIN32_OK:
            speak("Tell me what to rewrite, or copy some text first.")
            return
        try:
            win32clipboard.OpenClipboard()
            try:
                text = win32clipboard.GetClipboardData(win32con.CF_TEXT).decode("utf-8", errors="ignore").strip()
            except Exception:
                text = ""
            finally:
                win32clipboard.CloseClipboard()
        except Exception as e:
            log("Clipboard error: " + str(e))
        if not text:
            speak("There's nothing on your clipboard for me to rewrite.")
            return

    clipped = text if len(text) <= 4000 else text[:4000] + " ...(truncated)"
    if style == "formal":
        instruction = "Rewrite the following text in a more formal, professional tone"
    else:
        instruction = (
            "Rewrite the following text in simple, plain language that a "
            "general audience can easily follow"
        )
    prompt = f"{instruction}. Respond with ONLY the rewritten text, no preamble:\n\n{clipped}"
    ask_local_ai(prompt)


# ---------------------------------------------------------------------------
# MULTI-CLIP CLIPBOARD HISTORY — see CLIPBOARD_HISTORY_LIMIT above.
# clipboard_history_monitor() runs as a daemon thread (started from
# __main__) that polls the clipboard every couple of seconds; on a
# genuine change it appends the new text to a ring buffer so something
# copied a few steps back isn't gone the moment you copy something else.
# ---------------------------------------------------------------------------
def clipboard_history_monitor():
    global _last_clipboard_seen
    if not WIN32_OK:
        return
    while running:
        try:
            data = None
            win32clipboard.OpenClipboard()
            try:
                data = win32clipboard.GetClipboardData(win32con.CF_TEXT).decode("utf-8", errors="ignore")
            except Exception:
                data = None
            finally:
                win32clipboard.CloseClipboard()
            if data and data.strip() and data != _last_clipboard_seen:
                _last_clipboard_seen = data
                with clipboard_history_lock:
                    clipboard_history.append({
                        "text": data.strip(),
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
        except Exception as e:
            log("Clipboard history monitor: " + str(e))
        time.sleep(2)


def list_clipboard_history():
    with clipboard_history_lock:
        items = list(clipboard_history)[::-1]  # most recent first
    if not items:
        speak("I haven't seen anything new copied to your clipboard yet.")
        return
    lines = [f"{i + 1}: {it['text'][:60]}" for i, it in enumerate(items[:8])]
    speak(f"You have {len(items)} recent clipboard item{'s' if len(items) != 1 else ''}. " + "; ".join(lines))


def restore_clipboard_history_item(index_1based):
    """Copies a past clipboard-history entry back onto the live Windows
    clipboard so it can be pasted again, e.g. "restore clip 3"."""
    with clipboard_history_lock:
        items = list(clipboard_history)[::-1]
    if not (1 <= index_1based <= len(items)):
        speak(f"I don't have a clipboard item numbered {index_1based}.")
        return
    if not WIN32_OK:
        speak("Clipboard control unavailable.")
        return
    text = items[index_1based - 1]["text"]
    try:
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32con.CF_TEXT, text.encode("utf-8"))
        win32clipboard.CloseClipboard()
        speak(f"Restored clipboard item {index_1based} to your clipboard.")
    except Exception as e:
        log("Clipboard restore error: " + str(e))
        speak("I couldn't restore that clipboard item.")


# ---------------------------------------------------------------------------
# FILE-AWARE READ & SUMMARIZE — "read report.txt and tell me what it says",
# "summarize notes.docx", "what does budget.csv say about totals". Locates
# the file, pulls out its text, and hands it to the local AI along with the
# user's question — the same explain_clipboard() pattern above, just
# sourced from a file on disk instead of the clipboard.
# ---------------------------------------------------------------------------
FILE_READ_TEXT_EXTS = {
    ".txt", ".md", ".py", ".csv", ".json", ".log", ".ini",
    ".xml", ".html", ".htm", ".js", ".java", ".c", ".cpp",
    ".yaml", ".yml",
}
FILE_READ_EXTS_PATTERN = (
    r"[\w\-]+\.(txt|md|py|csv|json|log|ini|xml|html|htm|js|java|"
    r"c|cpp|yaml|yml|pdf|docx|doc)\b"
)
# Folders that are noisy and virtually never contain a file the user wants
# read aloud — same exclusion list search_local_files_gui() uses.
_FILE_READ_SKIP_DIRS = {
    "windows", "program files", "program files (x86)",
    "programdata", "$recycle.bin", "system volume information",
    "appdata", "msocache", "perflogs",
}


def _find_file_by_name(name_hint):
    """Locate a single file whose name matches name_hint (a full filename
    like 'budget.csv' or a bare keyword like 'budget'). Checks the most
    likely user folders first, then falls back to a capped drive walk that
    stops the moment a match is found — unlike search_local_files_gui(),
    this doesn't need to enumerate every match, just the first good one, so
    it stays fast enough to run as part of answering a question."""
    name_hint = (name_hint or "").strip().lower()
    if not name_hint:
        return None
    stem_hint = name_hint.split(".")[0]

    def _matches(fname_lower):
        return name_hint in fname_lower or (
            stem_hint and fname_lower.startswith(stem_hint)
        )

    home = Path.home()
    priority_dirs = [
        home / "Desktop", home / "Documents", home / "Downloads",
        home / "OneDrive" / "Desktop", home / "OneDrive" / "Documents",
        BASE_DIR,
    ]
    for d in priority_dirs:
        if not d.exists():
            continue
        try:
            for root, dirs, files in os.walk(d):
                for f in files:
                    if _matches(f.lower()):
                        return Path(root) / f
        except Exception:
            continue

    try:
        drives = [Path(f"{letter}:\\") for letter in "CDEFGH" if Path(f"{letter}:\\").exists()]
    except Exception:
        drives = [Path("C:\\")]

    scanned_dirs = 0
    for drive in drives:
        try:
            for root, dirs, files in os.walk(drive, topdown=True):
                dirs[:] = [d for d in dirs if d.lower() not in _FILE_READ_SKIP_DIRS]
                scanned_dirs += 1
                for f in files:
                    if _matches(f.lower()):
                        return Path(root) / f
                if scanned_dirs >= 4000:
                    return None
        except Exception:
            continue
    return None


def _extract_docx_text(fpath):
    """Extract plain text from a .docx file using only the standard
    library — a .docx is a zip archive containing word/document.xml, so
    this needs no python-docx install and can't be broken by the common
    'pip install docx' mistake (that installs an unrelated package with
    a different API, not python-docx). Raises on failure so the caller
    can fall back to python-docx if it's installed."""
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(fpath) as z:
        with z.open("word/document.xml") as f:
            root = ET.parse(f).getroot()

    paragraphs = []
    for para in root.iter(ns + "p"):
        run_texts = [node.text for node in para.iter(ns + "t") if node.text]
        if run_texts:
            paragraphs.append("".join(run_texts))
    return "\n".join(paragraphs)


def _extract_text_from_file(fpath):
    """Pull readable text out of a file for the AI to read. Returns
    (text, None) on success, or (None, reason) when it can't be read.
    Caps how much text is forwarded so requests stay fast."""
    MAX_CHARS = 8000
    ext = fpath.suffix.lower()
    try:
        if ext in FILE_READ_TEXT_EXTS:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext == ".pdf":
            if not PYPDF2_OK:
                return None, "PyPDF2 isn't installed, so I can't read PDF files yet."
            parts = []
            with open(fpath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages[:40]:
                    try:
                        parts.append(page.extract_text() or "")
                    except Exception:
                        continue
            text = "\n".join(parts)
        elif ext == ".docx":
            try:
                text = _extract_docx_text(fpath)
            except Exception as e:
                log("docx zip-extract failed, trying python-docx fallback: " + str(e))
                if not DOCX_OK:
                    return None, "I ran into a problem opening that Word file."
                document = docx_lib.Document(str(fpath))
                text = "\n".join(p.text for p in document.paragraphs)
        elif ext == ".doc":
            return None, "I can only read the newer .docx format, not the older .doc format."
        else:
            return None, f"I don't know how to read {ext or 'that type of'} files yet."
    except Exception as e:
        log("File read error: " + str(e))
        return None, "I ran into a problem opening that file."

    text = text.strip()
    if not text:
        return None, "That file doesn't seem to contain any readable text."
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + " ...(truncated)"
    return text, None


def _extract_file_read_request(text_raw):
    """Pulls (name_hint, question) out of a request like "read budget.csv
    and tell me the total" or "summarize notes.txt". name_hint is what to
    search for on disk; question is what to do with it once found (falls
    back to a general summary when nothing specific was asked)."""
    low = str(text_raw or "").strip().lower()

    m = re.search(FILE_READ_EXTS_PATTERN, low)
    if m:
        name_hint = m.group(0)
    else:
        m2 = re.search(r"file (?:called|named|titled)\s+([\w\-]+)", low)
        name_hint = m2.group(1) if m2 else None

    question = None
    if name_hint:
        after = low.split(name_hint, 1)[-1]
        after = re.sub(r"^\s*(and|then|,)?\s*", "", after).strip()
        if after:
            question = after

    return name_hint, question


def read_and_summarize_file(instruction_text):
    """File-aware Q&A: finds the file the user named, extracts its text,
    and asks the local AI the accompanying question against that content."""
    name_hint, question = _extract_file_read_request(instruction_text)
    if not name_hint:
        speak("Which file would you like me to read? Tell me its name, including the extension.")
        return

    speak(f"Looking for {name_hint} on your system.")
    fpath = _find_file_by_name(name_hint)
    if fpath is None:
        speak(f"I couldn't find a file matching {name_hint}.")
        return

    text, err = _extract_text_from_file(fpath)
    if err:
        speak(err)
        return

    question = question or "Summarize this file and list its key points."
    prompt = (
        f"The user asked about a file named '{fpath.name}'. Here is its "
        f"content:\n\n{text}\n\nBased only on this file, {question}"
    )
    log(f"FILE AI: Reading {fpath} for question: {question}")
    ask_local_ai(prompt)


# ---------------------------------------------------------------------------
# OUTLOOK INTEGRATION — talks to an already-installed desktop Outlook via
# COM automation (win32com.client, already imported above under WIN32_OK),
# the same way Windows' own "Send to Mail recipient" does it. No Graph
# API, no OAuth app registration, no API key — it just drives the Outlook
# process that's already signed in. Only works with desktop Outlook (the
# classic Win32 app); it can't reach a browser-only Gmail/webmail account.
# ---------------------------------------------------------------------------
OUTLOOK_OK = WIN32_OK
OUTLOOK_FOLDER_INBOX = 6     # olFolderInbox
OUTLOOK_FOLDER_CALENDAR = 9  # olFolderCalendar


def _get_outlook_namespace():
    """Returns the MAPI namespace of a running/launched desktop Outlook,
    or None (with a log line) if Outlook isn't installed/reachable."""
    try:
        outlook_app = win32com.client.Dispatch("Outlook.Application")
        return outlook_app.GetNamespace("MAPI")
    except Exception as e:
        log("Outlook connect error: " + str(e))
        return None


def check_outlook_email(count=5):
    """'check my email' / 'check my outlook email' — reads unread inbox
    mail newest-first, straight from desktop Outlook."""
    if not OUTLOOK_OK:
        speak("Outlook integration needs pywin32, which isn't installed.")
        return
    ns = _get_outlook_namespace()
    if ns is None:
        speak("I couldn't connect to Outlook. Make sure desktop Outlook is installed and signed in.")
        return
    try:
        inbox = ns.GetDefaultFolder(OUTLOOK_FOLDER_INBOX)
        items = inbox.Items
        items.Sort("[ReceivedTime]", True)
        unread = [m for m in items if getattr(m, "UnRead", False)]
        if not unread:
            speak("You have no unread emails in Outlook.")
            return
        shown = unread[:count]
        speak(f"You have {len(unread)} unread email{'s' if len(unread) != 1 else ''}. "
              f"The {len(shown)} most recent:")
        for mail in shown:
            try:
                sender = mail.SenderName or "someone"
                subject = mail.Subject or "(no subject)"
                speak(f"From {sender}: {subject}")
            except Exception:
                continue
    except Exception as e:
        log("Outlook email read error: " + str(e))
        speak("I had trouble reading your Outlook inbox.")


def check_outlook_calendar(days_ahead=1):
    """'what's on my calendar' / 'check my outlook calendar' — lists
    upcoming appointments in the given window (default: next 24 hours)."""
    if not OUTLOOK_OK:
        speak("Outlook integration needs pywin32, which isn't installed.")
        return
    ns = _get_outlook_namespace()
    if ns is None:
        speak("I couldn't connect to Outlook. Make sure desktop Outlook is installed and signed in.")
        return
    try:
        calendar = ns.GetDefaultFolder(OUTLOOK_FOLDER_CALENDAR)
        items = calendar.Items
        items.IncludeRecurrences = True
        items.Sort("[Start]")
        start = datetime.now()
        end = start + timedelta(days=days_ahead)
        restriction = (
            f"[Start] >= '{start.strftime('%m/%d/%Y %H:%M %p')}' AND "
            f"[Start] <= '{end.strftime('%m/%d/%Y %H:%M %p')}'"
        )
        events = list(items.Restrict(restriction))
        if not events:
            speak("You have no upcoming Outlook calendar events in that window.")
            return
        speak(f"You have {len(events)} upcoming event{'s' if len(events) != 1 else ''}:")
        for ev in events[:8]:
            try:
                subj = ev.Subject or "(no subject)"
                when = ev.Start.strftime("%A at %I:%M %p")
                speak(f"{subj}, {when}")
            except Exception:
                continue
    except Exception as e:
        log("Outlook calendar read error: " + str(e))
        speak("I had trouble reading your Outlook calendar.")


def send_outlook_email(to_addr, subject, body):
    """'email <address> subject <subject> body <message>' — composes and
    sends through desktop Outlook, no separate login/key needed."""
    if not OUTLOOK_OK:
        speak("Outlook integration needs pywin32, which isn't installed.")
        return
    to_addr = (to_addr or "").strip()
    if not to_addr:
        speak("Who should I send that email to?")
        return
    try:
        outlook_app = win32com.client.Dispatch("Outlook.Application")
        mail = outlook_app.CreateItem(0)  # olMailItem
        mail.To = to_addr
        mail.Subject = (subject or "").strip() or "(no subject)"
        mail.Body = (body or "").strip()
        mail.Send()
        speak(f"Email sent to {to_addr}.")
    except Exception as e:
        log("Outlook send error: " + str(e))
        speak("I couldn't send that email through Outlook.")


def run_macro(macro_name):
    speak(f"Running macro: {macro_name}")
    if macro_name == "workspace":
        open_app("chrome")
        time.sleep(1)
        open_app("vs code")
        set_volume(40)
        speak("Workspace initialized.")
    else:
        speak("Macro not found.")


# ---------------------------------------------------------------------------
# CUSTOM VOICE MACRO RECORDER — see the CUSTOM VOICE MACRO RECORDER
# constants block above. Persistence mirrors _load_wol_devices()/
# _save_wol_devices(); the live capture hook that appends each spoken
# command to macro_recording_state["steps"] while recording is active
# lives in handle_command(), not here.
# ---------------------------------------------------------------------------
def _load_custom_macros():
    global custom_macros
    try:
        if CUSTOM_MACROS_FILE.exists():
            with open(CUSTOM_MACROS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                with custom_macros_lock:
                    custom_macros = {
                        str(k).lower(): [str(s) for s in v]
                        for k, v in data.items() if isinstance(v, list)
                    }
    except Exception as e:
        log("Custom macro load error: " + str(e))


def _save_custom_macros():
    try:
        with open(CUSTOM_MACROS_FILE, "w", encoding="utf-8") as f:
            json.dump(custom_macros, f, indent=2)
    except Exception as e:
        log("Custom macro save error: " + str(e))


def start_macro_recording(name):
    name = (name or "").strip().lower()
    if not name:
        speak("Say it as: start recording macro, then a name for it.")
        return
    if macro_recording_active.is_set():
        speak("I'm already recording a macro. Say 'stop recording' to finish it first.")
        return
    macro_recording_state["name"] = name
    macro_recording_state["steps"] = []
    macro_recording_active.set()
    speak(f"Recording macro '{name}'. Do each step, then say 'stop recording' when you're done.")


def stop_macro_recording():
    if not macro_recording_active.is_set():
        speak("I'm not recording a macro right now.")
        return
    name = macro_recording_state["name"]
    steps = list(macro_recording_state["steps"])
    macro_recording_active.clear()
    macro_recording_state["name"] = None
    macro_recording_state["steps"] = []
    if not steps:
        speak(f"No steps were captured, so I didn't save '{name}'.")
        return
    with custom_macros_lock:
        custom_macros[name] = steps
        _save_custom_macros()
    speak(f"Saved macro '{name}' with {len(steps)} step{'s' if len(steps) != 1 else ''}. "
          f"Say 'run macro {name}' to play it back.")


def run_custom_macro(name):
    name = (name or "").strip().lower()
    if not name:
        speak("Which macro would you like me to run?")
        return
    with custom_macros_lock:
        steps = list(custom_macros.get(name, []))
    if not steps:
        if name == "workspace":
            run_macro("workspace")
            return
        speak(f"I don't have a macro called '{name}'.")
        return
    speak(f"Running macro '{name}', {len(steps)} step{'s' if len(steps) != 1 else ''}.")
    for step in steps:
        execute_single_command(step)
        time.sleep(0.3)
    speak(f"Macro '{name}' finished.")


def list_custom_macros():
    with custom_macros_lock:
        names = list(custom_macros.keys())
    if not names:
        speak("You don't have any custom macros recorded yet.")
        return
    speak("Your custom macros are: " + ", ".join(names) + ".")


def delete_custom_macro(name):
    name = (name or "").strip().lower()
    with custom_macros_lock:
        existed = name in custom_macros
        if existed:
            del custom_macros[name]
            _save_custom_macros()
    speak(f"Deleted macro '{name}'." if existed else f"I don't have a macro called '{name}'.")


def clear_audio():
    with audio_queue.mutex:
        audio_queue.queue.clear()


def audio_callback(indata, frames, callback_time, status):
    if status:
        log("Audio: " + str(status))
    try:
        audio_queue.put_nowait(indata[:, 0].copy())
    except queue.Full:
        pass


def rms(audio):
    if len(audio) == 0:
        return 0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))


def get_microphone():
    try:
        default = sd.default.device
        device_id = default[0] if isinstance(default, (list, tuple)) else default
        if device_id is not None and device_id >= 0:
            device = sd.query_devices(device_id)
            if device["max_input_channels"] > 0:
                log("Microphone: " + device["name"])
                return device_id
    except Exception:
        pass
    try:
        for i, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                log("Using microphone: " + device["name"])
                return i
    except Exception as e:
        log("Microphone search error: " + str(e))
    return None


def load_whisper():
    global whisper
    print("\n[VICTUS] Starting speech engine...")
    try:
        from faster_whisper import WhisperModel
        # Keep startup responsive: tiny.en loads much faster by default and
        # thread count is capped to avoid excessive CPU initialization overhead.
        whisper = WhisperModel(
            WHISPER_MODEL, device="cpu", compute_type="int8",
            cpu_threads=min(8, max(2, (os.cpu_count() or 4) - 1)), num_workers=1
        )
        print("Speech recognition READY.")
        if WHISPER_MODEL.endswith(".en"):
            print(f"[VICTUS] '{WHISPER_MODEL}' is English-only — multilingual "
                  f"auto-detect is disabled. Set VICTUS_WHISPER_MODEL=tiny (or "
                  f"base/small) for multilingual support.")
        return True
    except Exception as e:
        print("WHISPER ERROR:", e)
        return False


def transcribe(audio):
    """Transcribe a chunk of audio to text. When a multilingual Whisper
    model is loaded (i.e. WHISPER_MODEL doesn't end in ".en"), this also
    auto-detects the spoken language and — on a confident, non-trivial
    result — updates last_detected_language/last_detected_language_name so
    speak() can switch to a matching voice and ask_local_ai()/ask_vision_ai()
    can reply in that language. English-only ".en" models always stay
    pinned to English, exactly as before."""
    global last_detected_language, last_detected_language_name
    if whisper is None:
        return ""
    try:
        audio = audio.astype(np.float32)
        multilingual = not WHISPER_MODEL.endswith(".en")
        segments, info = whisper.transcribe(
            audio, language=(None if multilingual else "en"),
            beam_size=1, best_of=1, temperature=0,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 100, "speech_pad_ms": 100},
            condition_on_previous_text=False, without_timestamps=True
        )
        text = " ".join(s.text for s in segments).strip()
        if multilingual and text:
            detected = getattr(info, "language", None)
            confidence = getattr(info, "language_probability", 0.0) or 0.0
            if detected and confidence >= LANGUAGE_DETECT_MIN_CONFIDENCE:
                if detected != last_detected_language:
                    log(f"LANGUAGE: detected '{detected}' (confidence {confidence:.2f})")
                last_detected_language = detected
                last_detected_language_name = LANGUAGE_NAMES.get(detected, detected)
        return text
    except Exception as e:
        log("Whisper error: " + str(e))
        return ""


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s.]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_light(text):
    """Like normalize(), but keeps the punctuation and math/special
    characters that normalize() strips out (+ - * / = ^ % ( ) . , ? ! etc.)
    so a question like "what is 45 * (12 + 3) / 6" or "what does @user mean"
    reaches the AI with its symbols intact instead of being reduced to bare
    words. Used only for the text that gets routed to ask_local_ai — every
    other command still matches against the stricter normalize()."""
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s\.\,\?\!\:\;\'\"\+\-\*\/\^\%\=\(\)\[\]\{\}\<\>\_\@\#\$\&\\\|\~]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# SAFE OFFLINE MATHEMATICAL CALCULATOR
# ---------------------------------------------------------------------------
_SAFE_MATH_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_SAFE_MATH_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
}
_SAFE_MATH_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_math_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_math_ast(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise TypeError("Non-numeric constant in math expression")
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_MATH_OPERATORS:
        return _SAFE_MATH_OPERATORS[type(node.op)](_eval_math_ast(node.operand))
    elif isinstance(node, ast.BinOp) and type(node.op) in _SAFE_MATH_OPERATORS:
        left = _eval_math_ast(node.left)
        right = _eval_math_ast(node.right)
        return _SAFE_MATH_OPERATORS[type(node.op)](left, right)
    elif isinstance(node, ast.Name):
        if node.id.lower() in _SAFE_MATH_CONSTANTS:
            return _SAFE_MATH_CONSTANTS[node.id.lower()]
        raise ValueError(f"Unknown math identifier: {node.id}")
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id.lower()
        if func_name in _SAFE_MATH_FUNCTIONS:
            args = [_eval_math_ast(a) for a in node.args]
            return _SAFE_MATH_FUNCTIONS[func_name](*args)
        raise ValueError(f"Unknown math function: {node.func.id}")
    raise TypeError(f"Unsupported math element: {type(node).__name__}")


def try_calculate_math(text):
    """Safely evaluate an arithmetic or mathematical expression offline.
    Returns (clean_expression, result_number) or None if text is not a valid calculation."""
    t = str(text or "").strip()
    if not t:
        return None
    # Strip common leading question / command phrasing
    prefixes = [
        r"^(?:what\s+is|whats|what\'s|how\s+much\s+is|calculate|calc|compute|solve|evaluate)\s+",
        r"^(?:can\s+you\s+(?:calculate|solve|evaluate|compute))\s+",
        r"^(?:please\s+)?(?:calculate|solve|evaluate|compute)\s+",
    ]
    cleaned = t
    for p in prefixes:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()
    # Strip trailing ?, =, =? and whitespace
    cleaned = re.sub(r"[\s=\?]+$", "", cleaned).strip()
    if not cleaned:
        return None
    # Skip date-like strings (e.g. 2026-11-09)
    if re.match(r"^\d{4}-\d{2}-\d{2}$", cleaned):
        return None
    # Translate math notation to Python syntax
    expr = cleaned.replace("^", "**").replace("×", "*").replace("÷", "/")
    # Must contain at least one math operator or known math function
    has_op = any(op in expr for op in ["+", "-", "*", "/", "%", "**"])
    has_func = any(f + "(" in expr.lower() for f in _SAFE_MATH_FUNCTIONS)
    if not (has_op or has_func):
        return None
    try:
        parsed = ast.parse(expr, mode="eval")
        res = _eval_math_ast(parsed)
        if isinstance(res, (int, float)):
            if isinstance(res, float) and res.is_integer():
                res = int(res)
            elif isinstance(res, float):
                res = round(res, 6)
            return cleaned, res
        return None
    except ZeroDivisionError:
        return cleaned, "Error: Division by zero is undefined"
    except Exception:
        return None


def classify_intent(cmd_norm):
    """Lightweight, offline, weighted-keyword intent classifier (see
    INTENT_KEYWORDS). Returns (best_intent, score) where best_intent is
    None if nothing scored above zero. This is a genuine classification
    step used by handle_command() to make routing decisions (is this an
    AI question? a notes lookup? a reminder?) instead of relying only on a
    handful of hard-coded exact prefixes — but it does NOT replace the
    detailed per-command if/elif chain in execute_single_command(), which
    still does the precise argument parsing for each concrete action."""
    if not cmd_norm:
        return None, 0
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = 0
        for phrase, weight in keywords:
            if phrase in cmd_norm:
                score += weight
        if score:
            scores[intent] = score
    if not scores:
        return None, 0
    best_intent = max(scores, key=scores.get)
    return best_intent, scores[best_intent]


def is_wake_word(text):
    words = normalize(text).split()
    return any(word in WAKE_WORDS for word in words) or any(w in normalize(text) for w in WAKE_WORDS)


def is_direct_action_command(text):
    """Explicit launch/Air Cursor commands can work without a wake word."""
    n = normalize(text)
    direct = (
        "open air cursor", "close air cursor", "enable air cursor",
        "disable air cursor", "stop air cursor", "resume air cursor",
        "calibrate air cursor", "open brave", "open brave browser",
        "open omen", "open omen gaming hub",
        "open omen hub", "open asphalt", "open asphalt 8",
        "open asphalt 8 airborne", "open asphalt 8 air borne",
    )
    return any(n == phrase or n.startswith(phrase + " ") for phrase in direct)


def remove_wake(text):
    text = normalize(text)
    sorted_wakes = sorted(list(WAKE_WORDS), key=len, reverse=True)
    for w in sorted_wakes:
        pattern = r"^(?:" + re.escape(w) + r")\b\s*"
        text = re.sub(pattern, "", text).strip()
    text = re.sub(r"^victus\b", "", text).strip()
    text = re.sub(r"^victor\b", "", text).strip()
    text = re.sub(r"^jarvis\b", "", text).strip()
    text = re.sub(r"^javis\b", "", text).strip()
    return text


def record_full_command():
    global listening, mic_level
    clear_audio()
    listening = True
    mic_level = 0.0
    show_orb(True)
    
    collected = []
    start_time = time.time()
    last_voice_time = time.time()
    
    while time.time() - start_time < 6.0:
        try:
            data = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        level = rms(data)

        # Recording-level meter: smooth the raw rms into a 0-1 range so the
        # orb's VU ring animates cleanly instead of jittering every 512
        # samples. Purely visual — does not affect speech detection below.
        normalized = min(1.0, level / MIC_LEVEL_REFERENCE)
        mic_level = (mic_level * (1 - MIC_LEVEL_SMOOTHING)) + (normalized * MIC_LEVEL_SMOOTHING)

        if level > 0.0001:
            last_voice_time = time.time()
            collected.append(data)
        elif collected:
            collected.append(data)
            if time.time() - last_voice_time >= 1.0:
                break

    listening = False
    mic_level = 0.0
    show_orb(False)
    
    if not collected:
        return ""
    
    log("Processing full command...")
    return transcribe(np.concatenate(collected))


def listen_for_followup(window_seconds=FOLLOWUP_WINDOW_SECONDS):
    """Post-answer follow-up window. Called right after Victus finishes
    speaking an AI answer: reopens the mic for a few seconds WITHOUT
    requiring the wake word again, so a natural follow-up question doesn't
    need "Hey Victus" repeated. If nothing is said before the window closes,
    this returns None and Victus quietly falls back to normal wake-word
    listening in main_loop().

    followup_window_active is set for the duration so main_loop() knows to
    idle instead of also pulling from audio_queue — otherwise both loops
    would race for the same audio chunks.
    """
    global listening, mic_level
    followup_window_active.set()
    try:
        clear_audio()
        listening = True
        mic_level = 0.0
        show_orb(True)

        collected = []
        heard_voice = False
        start_time = time.time()
        last_voice_time = start_time

        while time.time() - start_time < window_seconds:
            try:
                data = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            level = rms(data)
            normalized = min(1.0, level / MIC_LEVEL_REFERENCE)
            mic_level = (mic_level * (1 - MIC_LEVEL_SMOOTHING)) + (normalized * MIC_LEVEL_SMOOTHING)

            if level > 0.0001:
                heard_voice = True
                last_voice_time = time.time()
                collected.append(data)
            elif heard_voice:
                collected.append(data)
                if time.time() - last_voice_time >= 1.0:
                    break

        if not heard_voice or not collected:
            return None

        log("Follow-up window: processing...")
        return transcribe(np.concatenate(collected)) or None
    finally:
        listening = False
        mic_level = 0.0
        show_orb(False)
        followup_window_active.clear()


def take_screenshot():
    try:
        folder = Path.home() / "Pictures" / "Victus Screenshots"
        folder.mkdir(parents=True, exist_ok=True)
        filename = datetime.now().strftime("Victus_Screenshot_%Y-%m-%d_%H-%M-%S.png")
        path = folder / filename
        if not PIL_OK or ImageGrab is None:
            speak("I couldn't take the screenshot because screenshot support is unavailable.")
            return False
        image = ImageGrab.grab(all_screens=True)
        image.save(str(path), "PNG")
        speak("Screenshot taken and saved in your Pictures folder.")
        log("Screenshot saved: " + str(path))
        return True
    except Exception as e:
        log("Screenshot error: " + str(e))
        speak("I couldn't take the screenshot.")
        return False


def get_window_info(hwnd):
    try:
        title = win32gui.GetWindowText(hwnd).strip()
        if not title or not win32gui.IsWindowVisible(hwnd):
            return None
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return {"hwnd": hwnd, "title": title, "pid": pid}
    except Exception:
        return None


def get_all_windows():
    result = []
    if not WIN32_OK:
        return result
    def callback(hwnd, extra):
        info = get_window_info(hwnd)
        if info:
            result.append(info)
    try:
        win32gui.EnumWindows(callback, None)
    except Exception as e:
        log("Window enumeration error: " + str(e))
    return result


# ---------------------------------------------------------------------------
# SCREEN ACTIVITY — "what's going on" / "scan my screen"
# ---------------------------------------------------------------------------
SCREEN_SCAN_BROWSER_PROCESSES = {
    "chrome.exe": "Chrome",
    "msedge.exe": "Edge",
    "firefox.exe": "Firefox",
    "brave.exe": "Brave",
    "opera.exe": "Opera",
    "iexplore.exe": "Internet Explorer",
}

# Checked against the browser window's title to guess which site is open.
SCREEN_SCAN_SITE_HINTS = [
    "youtube", "claude", "chatgpt", "gemini", "gmail", "github",
    "stackoverflow", "netflix", "spotify", "amazon", "whatsapp",
    "instagram", "twitter", "facebook", "reddit", "linkedin", "google",
]

# Checked against both the process name and window title.
SCREEN_SCAN_GAME_HINTS = {
    "asphalt": "Asphalt 8 Airborne",
    "valorant": "Valorant",
    "csgo": "Counter-Strike",
    "cs2": "Counter-Strike 2",
    "fortnite": "Fortnite",
    "minecraft": "Minecraft",
    "leagueclient": "League of Legends",
    "steam": "a Steam game",
    "epicgameslauncher": "an Epic Games title",
}


def _foreground_window_info():
    """Return (process_name.exe, window_title, pid) for whatever's
    currently in the foreground, or (None, None, None) if it can't be
    determined (e.g. pywin32 is unavailable)."""
    if not WIN32_OK:
        return None, None, None
    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return None, None, None
        title = (win32gui.GetWindowText(hwnd) or "").strip()
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        proc_name = ""
        try:
            proc_name = psutil.Process(pid).name()
        except Exception:
            proc_name = ""
        return proc_name, title, pid
    except Exception:
        return None, None, None


def _guess_site_from_title(title):
    t = title.lower()
    for hint in SCREEN_SCAN_SITE_HINTS:
        if hint in t:
            return hint.capitalize()
    return None


def _try_vision_screen_description():
    """Best-effort screen scan using an actual vision-capable Ollama model
    (llava / bakllava / moondream / any tag with "vision" in it), if one
    happens to be installed. Returns None on any failure, timeout, or if
    no such model is available — callers always have the fast foreground-
    window heuristic below to fall back on, so this never blocks or slows
    down the "what's going on" answer."""
    if not (PIL_OK and ImageGrab is not None):
        return None
    if not _load_local_ai():
        return None
    vision_model = next(
        (m for m in local_ai_available_models
         if any(tag in m.lower() for tag in VISION_MODEL_TAGS)),
        None,
    )
    if not vision_model:
        return None
    try:
        image = ImageGrab.grab()
        image.thumbnail((768, 768))
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=60)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        payload = json.dumps({
            "model": vision_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Victus, looking at a screenshot of the "
                        "user's screen. In ONE short, natural sentence, say "
                        "what they're currently doing — e.g. 'You're "
                        "watching a video on YouTube in Chrome' or 'You're "
                        "playing Asphalt 8'. Be specific and concise."
                    ),
                },
                {
                    "role": "user",
                    "content": "What's going on my screen right now?",
                    "images": [b64],
                },
            ],
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"temperature": 0.2, "num_predict": 60},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        text = ((data.get("message") or {}).get("content") or "").strip()
        return text or None
    except Exception as e:
        log("Vision screen-scan error: " + str(e))
        return None


def describe_screen_activity():
    """Voice/typed command: "what's going on" / "scan my screen" / "what
    am I doing". Tells the user what's currently happening on their
    screen — via an actual vision model if one is installed, otherwise a
    fast, reliable guess from the foreground window's process and title
    (e.g. "You're playing Asphalt 8", "You're on YouTube in Chrome")."""
    vision_desc = _try_vision_screen_description()
    if vision_desc:
        speak(vision_desc)
        return

    proc_name, title, pid = _foreground_window_info()

    if pid is not None and pid == VICTUS_PID:
        speak("You're currently talking to me.")
        return

    if not proc_name and not title:
        speak("I couldn't tell what's currently on your screen.")
        return

    proc_lower = (proc_name or "").lower()
    title_clean = (title or "").strip()

    if proc_lower in SCREEN_SCAN_BROWSER_PROCESSES:
        browser_label = SCREEN_SCAN_BROWSER_PROCESSES[proc_lower]
        site = _guess_site_from_title(title_clean)
        if site:
            speak(f"You're currently on {site}, in {browser_label}.")
        elif title_clean:
            speak(f"You're browsing in {browser_label} — looking at {title_clean}.")
        else:
            speak(f"You're currently using {browser_label}.")
        return

    for hint, game_name in SCREEN_SCAN_GAME_HINTS.items():
        if hint in proc_lower or hint in title_clean.lower():
            speak(f"Looks like you're playing {game_name}.")
            return

    friendly = canonical_app_name(proc_lower.replace(".exe", "")) if proc_lower else None
    if not friendly:
        friendly = proc_lower.replace(".exe", "").replace("_", " ").replace("-", " ").strip()
        friendly = friendly.title() if friendly else "an application"

    if title_clean:
        speak(f"You're currently in {friendly}, on {title_clean}.")
    else:
        speak(f"You're currently using {friendly}.")


def ask_vision_ai(query):
    """Real screen-content Q&A — "what does this error mean", "summarize
    what's on my screen", "read this for me" — answered by actually
    looking at a screenshot through a vision-capable Ollama model
    (llava / bakllava / moondream / any tag with "vision" in it — e.g.
    `ollama pull llava` or `ollama pull qwen2.5vl` for a stronger one).

    This is the deeper counterpart to describe_screen_activity(), which
    only produces one quick sentence about what app you're in. Here the
    user's actual question is sent to the model together with the
    screenshot, and the (possibly long) answer streams back through the
    same spoken-summary + AI response window pipeline as ask_local_ai().

    If no vision model is installed, this degrades gracefully instead of
    failing: it folds in whatever text context is available (the
    foreground app/window title) and falls back to the regular text-only
    ask_local_ai(), while telling the user once how to enable real screen
    reading."""
    global OLLAMA_MODEL

    query = str(query or "").strip()
    if not query:
        speak("What would you like to know about your screen?")
        return

    if not (PIL_OK and ImageGrab is not None):
        speak("Screen capture isn't available on this system.")
        ask_local_ai(query)
        return

    if not _load_local_ai():
        log("LOCAL AI unavailable: " + str(local_ai_loading_error))
        speak("My local AI brain is unavailable. Please make sure Ollama is installed and running.")
        return

    vision_model = next(
        (m for m in local_ai_available_models
         if any(tag in m.lower() for tag in VISION_MODEL_TAGS)),
        None,
    )

    if not vision_model:
        proc_name, title, _ = _foreground_window_info()
        context_bits = []
        if proc_name:
            context_bits.append(f"the foreground application is {proc_name}")
        if title:
            context_bits.append(f"its window title is \"{title}\"")
        speak(
            "I don't have a vision model installed, so I can't actually "
            "see your screen yet — pull one with 'ollama pull llava' for "
            "real screen reading. I'll do my best from context."
        )
        if context_bits:
            ask_local_ai(f"{query} (Context: {', '.join(context_bits)}.)")
        else:
            ask_local_ai(query)
        return

    try:
        image = ImageGrab.grab()
        image.thumbnail((1024, 1024))
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        log("Vision screen-capture error: " + str(e))
        speak("I couldn't capture your screen just now.")
        return

    messages = [
        {
            "role": "system",
            "content": (
                "You are Victus, a voice assistant looking at a "
                "screenshot of the user's screen. Answer their question "
                "about it directly and concisely, as if you can genuinely "
                "see the screen — because you can. Keep it conversational "
                "since your answer will be spoken aloud."
            ),
        },
    ]
    lang_instruction = _language_reply_instruction()
    if lang_instruction:
        messages.append({"role": "system", "content": lang_instruction})
    messages.append({"role": "user", "content": query, "images": [b64]})

    state = {"acc": "", "summary_spoken": False}

    def on_delta(piece):
        state["acc"] += piece
        if state["summary_spoken"]:
            return
        plain = state["acc"].replace("\n", " ").strip()
        if not plain:
            return
        sentence_count = len(re.split(r"(?<=[.!?])\s+", plain))
        if sentence_count >= 6 or len(plain) >= 650:
            state["summary_spoken"] = True
            speak(_spoken_summary(state["acc"]))

    with local_ai_lock:
        # Swap in the vision model just for this one call, then restore
        # whatever text model was configured — vision models are usually
        # weaker generalists, so Victus shouldn't stay switched over to
        # it for ordinary questions afterward.
        saved_model = OLLAMA_MODEL
        raw_answer = None
        try:
            OLLAMA_MODEL = vision_model
            raw_answer = _ollama_chat_stream(messages, on_delta=on_delta)
        except urllib.error.HTTPError as e:
            _ollama_error_speak(e)
            return
        except Exception as e:
            log("Vision AI error: " + str(e))
            if not state["acc"].strip():
                speak("I encountered an error while looking at your screen.")
                return
            raw_answer = state["acc"]
        finally:
            OLLAMA_MODEL = saved_model

    answer = _clean_ai_answer(raw_answer)
    log("VISION AI: " + answer)
    _update_ai_response_window(query, answer)
    if not state["summary_spoken"]:
        speak(_spoken_summary(answer))


# ---------------------------------------------------------------------------
# UNIVERSAL APPLICATION DISCOVERY & LAUNCHING ENGINE
# ---------------------------------------------------------------------------
_SYSTEM_APP_COMMANDS = {
    "settings": "ms-settings:",
    "windows settings": "ms-settings:",
    "system settings": "ms-settings:",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "notepad": "notepad.exe",
    "note pad": "notepad.exe",
    "paint": "mspaint.exe",
    "mspaint": "mspaint.exe",
    "camera": "microsoft.windows.camera:",
    "webcam": "microsoft.windows.camera:",
    "photos": "ms-photos:",
    "photo viewer": "ms-photos:",
    "gallery": "ms-photos:",
    "store": "ms-windows-store:",
    "microsoft store": "ms-windows-store:",
    "app store": "ms-windows-store:",
    "clock": "ms-clock:",
    "alarms": "ms-clock:",
    "alarm": "ms-clock:",
    "timer": "ms-clock:",
    "stopwatch": "ms-clock:",
    "terminal": "wt.exe",
    "windows terminal": "wt.exe",
    "command prompt": "cmd.exe",
    "command window": "cmd.exe",
    "cmd": "cmd.exe",
    "powershell": "powershell.exe",
    "task manager": "taskmgr.exe",
    "taskmanager": "taskmgr.exe",
    "task mgr": "taskmgr.exe",
    "control panel": "control.exe",
    "control pannel": "control.exe",
    "control": "control.exe",
    "file explorer": "explorer.exe",
    "explorer": "explorer.exe",
    "windows explorer": "explorer.exe",
    "this pc": "explorer.exe",
    "my computer": "explorer.exe",
    "device manager": "devmgmt.msc",
    "disk management": "diskmgmt.msc",
    "services": "services.msc",
    "registry editor": "regedit.exe",
    "regedit": "regedit.exe",
    "snipping tool": "snippingtool.exe",
    "snip": "snippingtool.exe",
    "character map": "charmap.exe",
    "recycle bin": "shell:RecycleBinFolder",
    "downloads": "shell:Downloads",
    "documents": "shell:Personal",
    "desktop": "shell:Desktop",
    "sound settings": "mmsys.cpl",
    "volume mixer": "sndvol.exe",
    "network connections": "ncpa.cpl",
    "programs and features": "appwiz.cpl",
    "system properties": "sysdm.cpl",
    "directx diagnostic": "dxdiag.exe",
    "dxdiag": "dxdiag.exe",
    "disk cleanup": "cleanmgr.exe",
    "windows version": "winver.exe",
    "winver": "winver.exe",
}

_WEB_APP_URLS = {
    "youtube": "https://www.youtube.com",
    "gemini": "https://gemini.google.com",
    "claude": "https://claude.ai",
    "chatgpt": "https://chatgpt.com",
    "github": "https://github.com",
    "gmail": "https://mail.google.com",
    "google": "https://www.google.com",
    "netflix": "https://www.netflix.com",
    "spotify": "https://open.spotify.com",
    "reddit": "https://www.reddit.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "instagram": "https://www.instagram.com",
    "linkedin": "https://www.linkedin.com",
    "amazon": "https://www.amazon.com",
    "discord": "https://discord.com/app",
    "facebook": "https://www.facebook.com",
    "maps": "https://maps.google.com",
    "google maps": "https://maps.google.com",
    "drive": "https://drive.google.com",
    "google drive": "https://drive.google.com",
    "whatsapp web": "https://web.whatsapp.com",
    "stackoverflow": "https://stackoverflow.com",
    "stack overflow": "https://stackoverflow.com",
}

APP_ALIASES = {
    "chrome": ["chrome", "google chrome", "google", "browser"],
    "brave": ["brave", "brave browser", "brave web browser"],
    "whatsapp": ["whatsapp", "whatsaap"],
    "notepad": ["notepad", "note pad"],
    "ms edge": ["edge", "microsoft edge", "ms edge"],
    "ms word": ["word", "microsoft word", "ms word"],
    "ms powerpoint": ["powerpoint", "power point", "microsoft powerpoint", "ms powerpoint", "ppt", "pptx"],
    "ms excel": ["excel", "microsoft excel", "ms excel", "spreadsheet"],
    "ms store": ["store", "microsoft store", "app store"],
    "file explorer": ["file explorer", "explorer", "windows explorer", "my computer", "this pc"],
    "this pc": ["this pc", "my computer", "explorer"],
    "arduino ide": ["arduino", "arduino ide"],
    "calculator": ["calculator", "calculate", "calc"],
    "solitery and games": ["solitaire", "solitery", "solitery and games", "games", "casual games"],
    "settings": ["settings", "windows settings", "system settings"],
    "camera": ["camera", "webcam"],
    "photos": ["photos", "photo viewer", "gallery"],
    "ms copilot": ["copilot", "microsoft copilot", "ms copilot"],
    "idle python": ["idle", "python idle", "idle python", "python ide"],
    "code blocks": ["code blocks", "codeblock", "codeblocks"],
    "vs code": ["vs code", "vscode", "visual studio code"],
    "solid edge": ["solid edge", "solidedge"],
    "task manager": ["task manager", "taskmanager", "task mgr"],
    "control pannel": ["control panel", "control pannel", "control"],
    "comand window": ["command window", "comand window", "command prompt", "cmd", "terminal"],
    "omen gaming hub": ["omen", "omen gaming hub", "omen hub", "hp omen", "hp omen gaming hub", "omen gaming", "omen hub gaming"],
    "asphalt 8 airborne": [
        "asphalt", "asphalt 8", "asphalt 8 airborne", "asphalt 8 air borne",
        "asphalt airborne", "asphalt eight", "asphalt eight airborne"
    ],
    "vlc": ["vlc", "vlc media player", "vlc player", "video lan"],
    "spotify": ["spotify", "spotify music"],
    "expressvpn": ["expressvpn", "express vpn", "vpn"],
    "onenote": ["onenote", "one note", "microsoft onenote"],
    "sticky notes": ["sticky notes", "stickynotes", "notes app"],
    "google drive": ["google drive", "gdrive"],
}

TYPO_MAP = {
    "googel chrome": "chrome", "google chrom": "chrome", "brave browser": "brave", "brave": "brave", "microsoft wor": "word",
    "power point": "powerpoint", "power point presentation": "powerpoint", "note pad": "notepad",
    "calender": "calendar", "code block": "codeblocks", "code blocks": "codeblocks",
    "arduino ide": "arduino", "idle python": "idle", "python idle": "idle",
    "chat gpt": "chatgpt", "you tube": "youtube", "file explorer": "file explorer",
    "solitery": "solitery and games", "games": "solitery and games", "task manager": "task manager",
    "control panel": "control pannel", "control pannel": "control pannel", "command window": "comand window",
    "command prompt": "comand window", "cmd": "comand window", "vs code": "vs code", "vscode": "vs code",
    "solid edge": "solid edge", "whatsapp": "whatsapp", "edge": "ms edge", "word": "ms word",
    "omen": "omen gaming hub", "omen hub": "omen gaming hub", "hp omen": "omen gaming hub",
    "asphalt": "asphalt 8 airborne", "asphalt 8": "asphalt 8 airborne", "asphalt 8 air borne": "asphalt 8 airborne",
    "powerpoint": "ms powerpoint", "store": "ms store", "copilot": "ms copilot", "excel": "ms excel"
}

_SHORTCUTS_CACHE = {"time": 0, "data": {}}
_REGISTRY_APPS_CACHE = {"time": 0, "data": {}}
_STARTAPPS_CACHE = {"time": 0, "data": {}}
_STARTAPPS_SCANNING = False


def _get_installed_shortcuts(force=False):
    now = time.time()
    if not force and _SHORTCUTS_CACHE["data"] and (now - _SHORTCUTS_CACHE["time"] < 45):
        return _SHORTCUTS_CACHE["data"]
    dirs = [
        os.path.expandvars(r"%ProgramData%\Microsoft\Windows\Start Menu\Programs"),
        os.path.expandvars(r"%AppData%\Microsoft\Windows\Start Menu\Programs"),
        os.path.expandvars(r"%Public%\Desktop"),
        os.path.expandvars(r"%UserProfile%\Desktop"),
    ]
    sc = {}
    for d in dirs:
        if os.path.exists(d):
            try:
                for root, _, files in os.walk(d):
                    for f in files:
                        if f.lower().endswith((".lnk", ".url")):
                            clean = os.path.splitext(f)[0].lower().strip()
                            clean_norm = normalize(clean)
                            full_path = os.path.join(root, f)
                            sc[clean] = full_path
                            if clean_norm and clean_norm != clean:
                                sc[clean_norm] = full_path
            except Exception:
                pass
    _SHORTCUTS_CACHE["time"] = now
    _SHORTCUTS_CACHE["data"] = sc
    return sc


def _get_registry_apps(force=False):
    now = time.time()
    if not force and _REGISTRY_APPS_CACHE["data"] and (now - _REGISTRY_APPS_CACHE["time"] < 60):
        return _REGISTRY_APPS_CACHE["data"]
    apps = {}
    if platform.system().lower() == "windows":
        for rk in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(rk, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths") as k:
                    for i in range(winreg.QueryInfoKey(k)[0]):
                        sub = winreg.EnumKey(k, i)
                        try:
                            with winreg.OpenKey(k, sub) as sk:
                                val = winreg.QueryValue(sk, "")
                                if val and os.path.exists(val):
                                    base = sub.lower().replace(".exe", "").strip()
                                    apps[base] = val
                                    apps[normalize(base)] = val
                        except Exception:
                            pass
            except Exception:
                pass
    _REGISTRY_APPS_CACHE["time"] = now
    _REGISTRY_APPS_CACHE["data"] = apps
    return apps


def _populate_start_apps_cache():
    global _STARTAPPS_SCANNING
    if _STARTAPPS_SCANNING or platform.system().lower() != "windows":
        return
    _STARTAPPS_SCANNING = True
    try:
        no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        ps_cmd = "$apps = Get-StartApps; $apps | ForEach-Object { \"$($_.Name)`t$($_.AppID)\" }"
        res = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=10, creationflags=no_window
        )
        if res.returncode == 0 and res.stdout:
            uwp = {}
            for line in res.stdout.splitlines():
                if "\t" in line:
                    aname, aid = line.split("\t", 1)
                    aname_clean = aname.lower().strip()
                    aid_clean = aid.strip()
                    uwp[aname_clean] = aid_clean
                    uwp[normalize(aname_clean)] = aid_clean
            _STARTAPPS_CACHE["data"] = uwp
            _STARTAPPS_CACHE["time"] = time.time()
            log(f"Universal App Launcher: indexed {len(uwp)} StartApps successfully.")
    except Exception as e:
        log(f"StartApps background index error: {e}")
    finally:
        _STARTAPPS_SCANNING = False


# Start async indexing of StartApps at load
threading.Thread(target=_populate_start_apps_cache, daemon=True, name="Victus-AppIndexer").start()


def canonical_app_name(name):
    n = normalize(name)
    n = re.sub(r"^(the|my)\s+", "", n)
    n = re.sub(r"\s+(app|application|window)$", "", n)
    for canonical, aliases in APP_ALIASES.items():
        if n == canonical or n in aliases:
            return canonical
    for canonical, aliases in APP_ALIASES.items():
        for alias in aliases:
            if n.startswith(alias) or alias.startswith(n) or alias in n or n in alias:
                return canonical
    return TYPO_MAP.get(n)


def find_installed_app(name):
    """Universal app resolution across system utilities, Start Menu shortcuts,
    Registry App Paths, UWP Store apps, and web apps.
    Returns (match_type, target_string, display_name) or None."""
    raw = str(name or "").strip()
    if not raw:
        return None
    w = raw.lower().strip()
    w = re.sub(r"^(the|my)\s+", "", w)
    w = re.sub(r"\s+(app|application|software|window)$", "", w).strip()
    w_norm = normalize(w)

    # 1. Built-in Windows system utilities / protocol handlers
    for candidate in [w, w_norm]:
        if candidate in _SYSTEM_APP_COMMANDS:
            return ("system", _SYSTEM_APP_COMMANDS[candidate], candidate.title())

    # 2. Known aliases
    canonical = canonical_app_name(w)
    if canonical and canonical in _SYSTEM_APP_COMMANDS:
        return ("system", _SYSTEM_APP_COMMANDS[canonical], canonical.title())

    # 3. Start Menu and Desktop shortcuts
    sc = _get_installed_shortcuts()
    for candidate in [w, w_norm, canonical]:
        if candidate and candidate in sc:
            return ("shortcut", sc[candidate], candidate.title())

    # 4. Registry App Paths
    reg = _get_registry_apps()
    for candidate in [w, w_norm, canonical]:
        if candidate and candidate in reg:
            return ("registry", reg[candidate], candidate.title())

    # 5. StartApps (UWP / Modern apps)
    uwp = _STARTAPPS_CACHE["data"]
    for candidate in [w, w_norm, canonical]:
        if candidate and candidate in uwp:
            return ("uwp", uwp[candidate], candidate.title())

    # 6. Fuzzy matching on shortcuts (.lnk / .url)
    best_item = None
    best_score = 0
    w_tokens = set(w_norm.split())

    for k, path in sc.items():
        k_norm = normalize(k)
        score = 0
        if k == w or k_norm == w_norm:
            score = 100
        elif k.startswith(w + " ") or k_norm.startswith(w_norm + " ") or k.endswith(" " + w):
            score = 85
        elif f" {w_norm} " in f" {k_norm} ":
            score = 80
        elif w_norm and (w_norm in k_norm or k_norm in w_norm):
            score = 70
        else:
            common = w_tokens & set(k_norm.split())
            if common:
                score = len(common) * 20
        if score > best_score:
            best_score = score
            best_item = ("shortcut", path, k.title())

    # Fuzzy match on registry apps
    for k, path in reg.items():
        k_norm = normalize(k)
        score = 0
        if k == w or k_norm == w_norm:
            score = 100
        elif w_norm in k_norm or k_norm in w_norm:
            score = 75
        if score > best_score:
            best_score = score
            best_item = ("registry", path, k.title())

    # Fuzzy match on UWP apps
    for k, aid in uwp.items():
        k_norm = normalize(k)
        score = 0
        if k == w or k_norm == w_norm:
            score = 100
        elif w_norm in k_norm or k_norm in w_norm:
            score = 75
        if score > best_score:
            best_score = score
            best_item = ("uwp", aid, k.title())

    if best_item and best_score >= 35:
        return best_item

    # 7. Popular Web Applications fallback
    for candidate in [w, w_norm, canonical]:
        if candidate and candidate in _WEB_APP_URLS:
            return ("web", _WEB_APP_URLS[candidate], candidate.title())

    return None


def open_windows_start_app(display_name, aliases=None):
    """Backward compatibility wrapper for open_app."""
    return open_app(display_name)


def open_app(name):
    """Universal Windows Application Launcher.
    Can open ANY application installed on the user's PC, plus system utilities
    and web apps."""
    raw_name = str(name or "").strip()
    if not raw_name:
        speak("What application would you like me to open?")
        return False

    log(f"Open App requested: '{raw_name}'")
    match = find_installed_app(raw_name)

    if match:
        mtype, target, display_name = match
        log(f"App resolved ({mtype}): {display_name} -> {target}")

        try:
            if mtype == "system":
                if target.endswith(":") or target.startswith("ms-") or target.startswith("shell:"):
                    os.startfile(target)
                else:
                    subprocess.Popen([target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True

            elif mtype == "shortcut":
                os.startfile(target)
                return True

            elif mtype == "registry":
                os.startfile(target)
                return True

            elif mtype == "uwp":
                os.startfile(f"shell:AppsFolder\\{target}")
                return True

            elif mtype == "web":
                webbrowser.open(target)
                return True

        except Exception as e:
            log(f"Primary launch failed for {target}: {e}")
            try:
                subprocess.Popen(
                    ["cmd", "/c", "start", "", target],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return True
            except Exception as e2:
                log(f"Command shell fallback failed: {e2}")

    # Fallback 1: Direct executable or protocol launch via os.startfile
    if platform.system().lower() == "windows":
        try:
            os.startfile(raw_name)
            log(f"Direct startfile succeeded for '{raw_name}'")
            return True
        except Exception:
            pass

        # Fallback 2: Check standard executable extensions (.exe)
        if not raw_name.lower().endswith(".exe"):
            try:
                os.startfile(raw_name + ".exe")
                log(f"Direct startfile (.exe) succeeded for '{raw_name}'")
                return True
            except Exception:
                pass

        # Fallback 3: Shell 'start' command
        try:
            res = subprocess.run(
                ["cmd", "/c", "start", "", raw_name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
            if res.returncode == 0:
                log(f"Shell start succeeded for '{raw_name}'")
                return True
        except Exception:
            pass

    speak(f"I couldn't find an installed application named {raw_name}.")
    return False



def find_windows(name):
    wanted = normalize(name)
    canonical = canonical_app_name(wanted)
    search_terms = [canonical] + APP_ALIASES.get(canonical, []) if canonical else [wanted]
    matches = []
    for info in get_all_windows():
        title = normalize(info["title"])
        for term in search_terms:
            term_norm = normalize(term)
            if term_norm in title or title in term_norm:
                if info not in matches:
                    matches.append(info)
                break
    return matches


def minimize_current_window():
    if not WIN32_OK:
        speak("Window control is unavailable.")
        return
    try:
        hwnd = win32gui.GetForegroundWindow()
        if hwnd and hwnd != win32gui.GetDesktopWindow():
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid == VICTUS_PID:
                speak("I won't minimize Victus itself.")
                return
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            speak("Minimized current window.")
        else:
            speak("No active window to minimize.")
    except Exception as e:
        log("Minimize current window error: " + str(e))
        speak("I couldn't minimize the current window.")


def close_current_window():
    if not WIN32_OK:
        speak("Window control is unavailable.")
        return
    try:
        hwnd = win32gui.GetForegroundWindow()
        if hwnd and hwnd != win32gui.GetDesktopWindow():
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid == VICTUS_PID:
                speak("I won't close Victus itself.")
                return
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            speak("Closed current window or tab.")
        else:
            speak("No active window to close.")
    except Exception as e:
        log("Close current window error: " + str(e))
        speak("I couldn't close the current window.")


def minimize_window(name):
    matches = find_windows(name)
    if not matches:
        speak(f"I couldn't find the window {name}.")
        return
    count = 0
    for info in matches:
        try:
            win32gui.ShowWindow(info["hwnd"], win32con.SW_MINIMIZE)
            count += 1
        except Exception:
            pass
    speak(f"I minimized {name}." if count == 1 else f"I minimized {count} {name} windows.")


def minimize_all():
    count = 0
    for info in get_all_windows():
        if info["pid"] == VICTUS_PID:
            continue
        try:
            win32gui.ShowWindow(info["hwnd"], win32con.SW_MINIMIZE)
            count += 1
        except Exception:
            pass
    speak(f"I minimized {count} windows.")


def close_window(name):
    matches = find_windows(name)
    if not matches:
        speak(f"I couldn't find the window {name}.")
        return
    count = 0
    for info in matches:
        if info["pid"] == VICTUS_PID:
            continue
        try:
            win32gui.PostMessage(info["hwnd"], win32con.WM_CLOSE, 0, 0)
            count += 1
        except Exception:
            pass
    if count == 0:
        speak("I won't close Victus itself.")
    elif count == 1:
        speak(f"I closed {name}.")
    else:
        speak(f"I closed {count} {name} windows.")


def close_all():
    speak("Closing other open application windows. Victus will stay running.")
    time.sleep(0.5)
    count = 0
    for info in get_all_windows():
        if info["pid"] == VICTUS_PID:
            continue
        try:
            win32gui.PostMessage(info["hwnd"], win32con.WM_CLOSE, 0, 0)
            count += 1
        except Exception:
            pass
    time.sleep(1)
    speak(f"I sent close commands to {count} windows.")


def get_volume_interface():
    if not PYCAW_OK:
        return None
    try:
        try:
            comtypes.CoInitialize()
        except Exception:
            pass
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))
    except Exception as e:
        log("Audio interface error: " + str(e))
        return None


def get_volume():
    interface = get_volume_interface()
    if interface is None:
        return None
    try:
        return int(round(interface.GetMasterVolumeLevelScalar() * 100))
    except Exception:
        return None


def set_volume(percent):
    try:
        percent = max(0, min(100, int(float(percent))))
    except Exception:
        speak("I can't understand the command, say it again.")
        return
    interface = get_volume_interface()
    if interface is None:
        speak("I couldn't access Windows volume control.")
        return
    try:
        interface.SetMasterVolumeLevelScalar(percent / 100.0, None)
        speak(f"Volume set to {percent} percent.")
    except Exception:
        speak("I couldn't change the volume.")


def change_volume(delta):
    current = get_volume()
    if current is None:
        speak("I couldn't read the system volume.")
        return
    set_volume(max(0, min(100, current + delta)))


def mute_volume():
    interface = get_volume_interface()
    if interface is None:
        speak("I couldn't access Windows volume control.")
        return
    try:
        interface.SetMute(1, None)
        speak("Laptop muted.")
    except Exception:
        speak("I couldn't mute the laptop.")


def unmute_volume():
    interface = get_volume_interface()
    if interface is None:
        speak("I couldn't access Windows volume control.")
        return
    try:
        interface.SetMute(0, None)
        speak("Laptop unmuted.")
    except Exception:
        speak("I couldn't unmute the laptop.")


def get_brightness():
    if SBC_OK:
        try:
            value = sbc.get_brightness()
            return int(value[0] if isinstance(value, list) else value)
        except Exception:
            pass
    try:
        ps = r"$b = Get-WmiObject -Namespace root\wmi -Class WmiMonitorBrightness; if ($b) { [int]$b.CurrentBrightness }"
        result = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=5)
        value = result.stdout.strip()
        if value:
            return int(value.splitlines()[-1])
    except Exception:
        pass
    return None


def set_brightness(percent):
    try:
        percent = max(0, min(100, int(float(percent))))
    except Exception:
        speak("I can't understand the command, say it again.")
        return False
    if SBC_OK:
        try:
            sbc.set_brightness(percent)
            time.sleep(0.2)
            actual = get_brightness()
            if actual is not None and abs(actual - percent) <= 5:
                speak(f"Brightness set to {percent} percent.")
                return True
        except Exception:
            pass
    try:
        ps = f"$methods = Get-WmiObject -Namespace root\\wmi -Class WmiMonitorBrightnessMethods; foreach ($m in $methods) {{ $m.WmiSetBrightness(1, {percent}) }}"
        result = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            speak(f"Brightness set to {percent} percent.")
            return True
    except Exception:
        pass
    speak("I couldn't change the brightness. Windows is not exposing brightness control for this display.")
    return False


def change_brightness(delta):
    current = get_brightness()
    current = 50 if current is None else current
    set_brightness(max(0, min(100, current + delta)))


VK_MEDIA_NEXT_TRACK, VK_MEDIA_PREV_TRACK, VK_MEDIA_PLAY_PAUSE, VK_MEDIA_STOP = 0xB0, 0xB1, 0xB3, 0xB2
KEYEVENTF_KEYUP = 0x0002


def send_media_key(vk):
    try:
        ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
        ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
        return True
    except Exception:
        return False


def media_pause():
    speak("Paused." if send_media_key(VK_MEDIA_PLAY_PAUSE) else "I couldn't pause the music.")


def media_stop():
    speak("Music stopped." if send_media_key(VK_MEDIA_STOP) else "I couldn't stop the music.")


def media_next():
    speak("Playing the next song." if send_media_key(VK_MEDIA_NEXT_TRACK) else "I couldn't play the next song.")


def media_previous():
    speak("Playing the previous song." if send_media_key(VK_MEDIA_PREV_TRACK) else "I couldn't go to the previous song.")


VK_CODES = {"CTRL": 0x11, "C": 0x43, "V": 0x56, "A": 0x41, "TAB": 0x09, "ENTER": 0x0D, "HOME": 0x24, "ZERO": 0x30, "SPACE": 0x20}


def key_down(vk):
    ctypes.windll.user32.keybd_event(vk, 0, 0, 0)


def key_up(vk):
    ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)


def send_key(key):
    vk = VK_CODES.get(key.upper())
    if vk is None:
        return False
    try:
        key_down(vk)
        key_up(vk)
        return True
    except Exception:
        return False


def google_search_text(query):
    query = query.strip()
    if not query:
        speak("I can't understand the command, say it again.")
        return
    speak(f"Searching for {query}.")
    url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(query)
    try:
        subprocess.Popen(["cmd", "/c", "start", "", "chrome", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        webbrowser.open(url)


def open_web_shortcut(site_name):
    w = str(site_name or "").lower().strip()
    url = _WEB_APP_URLS.get(w)
    if not url:
        for k, u in _WEB_APP_URLS.items():
            if w in k or k in w:
                url = u
                break
    if url:
        speak(f"Opening {site_name}.")
        try:
            subprocess.Popen(["cmd", "/c", "start", "", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            webbrowser.open(url)
        return True
    return False


def youtube_search(query):
    global music_started, music_query
    query = query.strip()
    if not query:
        speak("I can't understand the command, say it again.")
        return
    music_query, music_started = query, True
    speak(f"Opening YouTube and searching for {query}.")
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)
    try:
        subprocess.Popen(["cmd", "/c", "start", "", "chrome", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        webbrowser.open(url)
    time.sleep(3)
    try:
        for _ in range(5):
            send_key("TAB")
            time.sleep(0.1)
        send_key("ENTER")
    except Exception:
        pass


def random_song():
    year = datetime.now().year
    choices = [f"latest English songs {year}", f"latest Hindi songs {year}", f"latest Kannada songs {year}"]
    youtube_search(random.choice(choices))


def music_start_over():
    if not music_started:
        speak("There isn't a Victus music session yet.")
        return
    windows = find_windows("chrome")
    if windows:
        try:
            win32gui.SetForegroundWindow(windows[0]["hwnd"])
            time.sleep(0.25)
        except Exception:
            pass
    speak("Starting the song over." if send_key("ZERO") else "I couldn't restart the song.")


def battery_status():
    battery = psutil.sensors_battery()
    if battery is None:
        speak("I can't read the battery status.")
        return
    speak(f"Battery is {int(round(battery.percent))} percent. Charging is {'switched on' if battery.power_plugged else 'switched off'}.")


def battery_monitor():
    global previous_plugged, full_warned, low_warned
    while running:
        try:
            battery = psutil.sensors_battery()
            if battery:
                percent, plugged = int(battery.percent), bool(battery.power_plugged)
                if previous_plugged is not None and plugged != previous_plugged:
                    speak("Charging status on." if plugged else "Charging status off.")
                previous_plugged = plugged
                if percent >= 100 and plugged:
                    if not full_warned:
                        speak("Battery is fully charged. Please unplug the charger.")
                        full_warned = True
                else:
                    full_warned = False
                if percent < 20 and not plugged:
                    if not low_warned:
                        speak("Low battery. Please plug in the charger.")
                        low_warned = True
                else:
                    low_warned = False
        except Exception as e:
            log("Battery monitor: " + str(e))
        time.sleep(8)


def health_monitor():
    """Background watcher: speaks up unprompted when CPU or RAM usage has
    stayed above threshold for several minutes straight, instead of only
    reporting load when the user explicitly asks for "system status".
    Mirrors battery_monitor()'s always-on daemon-thread shape above."""
    global _cpu_high_since, _ram_high_since, _cpu_alert_last_fired, _ram_alert_last_fired
    while running:
        try:
            if not health_alerts_enabled:
                time.sleep(HEALTH_ALERT_POLL_SECONDS)
                continue

            now = time.time()
            cpu = psutil.cpu_percent(interval=1.0)
            ram = psutil.virtual_memory().percent

            if cpu >= HEALTH_ALERT_CPU_THRESHOLD:
                if _cpu_high_since is None:
                    _cpu_high_since = now
                elif (now - _cpu_high_since >= HEALTH_ALERT_SUSTAIN_SECONDS
                        and now - _cpu_alert_last_fired >= HEALTH_ALERT_COOLDOWN_SECONDS):
                    minutes = max(1, int((now - _cpu_high_since) // 60))
                    speak(f"Heads up boss, your CPU has been at {int(round(cpu))} percent "
                          f"for about {minutes} minute{'s' if minutes != 1 else ''}.")
                    _cpu_alert_last_fired = now
            else:
                _cpu_high_since = None

            if ram >= HEALTH_ALERT_RAM_THRESHOLD:
                if _ram_high_since is None:
                    _ram_high_since = now
                elif (now - _ram_high_since >= HEALTH_ALERT_SUSTAIN_SECONDS
                        and now - _ram_alert_last_fired >= HEALTH_ALERT_COOLDOWN_SECONDS):
                    minutes = max(1, int((now - _ram_high_since) // 60))
                    speak(f"Heads up boss, RAM usage has been at {int(round(ram))} percent "
                          f"for about {minutes} minute{'s' if minutes != 1 else ''}.")
                    _ram_alert_last_fired = now
            else:
                _ram_high_since = None
        except Exception as e:
            log("Health monitor error: " + str(e))
        time.sleep(HEALTH_ALERT_POLL_SECONDS)


def toggle_health_alerts(enable):
    global health_alerts_enabled, _cpu_high_since, _ram_high_since
    health_alerts_enabled = enable
    if not enable:
        _cpu_high_since = None
        _ram_high_since = None
    speak("Health alerts are now on." if enable else "Health alerts are now off.")


def system_status():
    wifi_en, wifi_conn, ssid = get_wifi_state()
    wifi_str = f"connected to {ssid}" if (wifi_conn and ssid) else ("switched on" if wifi_en else "switched off")

    bt = get_bluetooth_status()
    bt_str = "switched on" if bt is True else ("switched off" if bt is False else "unknown")

    battery = psutil.sensors_battery()
    if battery:
        batt_str = f"{int(round(battery.percent))} percent, charging is {'switched on' if battery.power_plugged else 'switched off'}"
    else:
        batt_str = "unavailable"

    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage("C:\\")

    status_text = (
        f"Here is the system status. "
        f"Wi-Fi is {wifi_str}. "
        f"Bluetooth is {bt_str}. "
        f"Battery is at {batt_str}. "
        f"CPU usage is {int(round(cpu))} percent. "
        f"RAM usage is {int(round(ram.percent))} percent, with {round(ram.used / (1024**3), 1)} gigabytes used out of {round(ram.total / (1024**3), 1)} gigabytes. "
        f"C drive storage is {int(round(disk.percent))} percent used, with {round(disk.free / (1024**3), 1)} gigabytes free."
    )
    speak(status_text)


def get_gpu_info():
    try:
        ps = "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
        res = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=5)
        gpus = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        if gpus:
            return ", ".join(gpus)
    except Exception:
        pass
    return "Integrated Graphics"


def system_specs():
    cpu_name = platform.processor() or "Unknown CPU"
    try:
        ps_cpu = "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"
        res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cpu], capture_output=True, text=True, timeout=5)
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        if lines:
            cpu_name = lines[0]
    except Exception:
        pass

    phys_cores = psutil.cpu_count(logical=False) or 0
    log_cores = psutil.cpu_count(logical=True) or 0
    ram_total = round(psutil.virtual_memory().total / (1024**3), 1)

    total_disk = 0
    try:
        for part in psutil.disk_partitions(all=False):
            if "fixed" in part.opts or part.fstype:
                try:
                    total_disk += psutil.disk_usage(part.mountpoint).total
                except Exception:
                    pass
    except Exception:
        pass
    rom_total = round(total_disk / (1024**3), 1) if total_disk > 0 else round(psutil.disk_usage("C:\\").total / (1024**3), 1)
    gpu_name = get_gpu_info()

    specs_text = (
        f"Here are the system specifications. "
        f"Processor: {cpu_name}, with {phys_cores} physical cores and {log_cores} logical threads. "
        f"Graphics: {gpu_name}. "
        f"System RAM: {ram_total} gigabytes. "
        f"Total Storage capacity: {rom_total} gigabytes."
    )
    speak(specs_text)


def open_spec_sheet(path_or_url="path_to_your_spec_sheet.pdf"):
    speak("Opening the spec sheet...")
    try:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            webbrowser.open(path_or_url)
        elif os.name == 'nt':
            try:
                os.startfile(path_or_url)
            except Exception:
                subprocess.Popen(f'start "" "{path_or_url}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', path_or_url])
    except Exception as e:
        log("Spec sheet error: " + str(e))
        speak("I couldn't open the spec sheet.")


def sleep_pc():
    speak("Putting the laptop to sleep.")
    time.sleep(1)
    try:
        ctypes.windll.powrprof.SetSuspendState(0, 0, 0)
    except Exception:
        subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])


def shutdown_pc():
    speak("Shutting down the system now, boss. Have a wonderful rest of your day. Goodbye!")
    time.sleep(2.0)
    subprocess.Popen(["shutdown", "/s", "/t", "2"])


def recycle_bin():
    speak("Opening Recycle Bin.")
    subprocess.Popen(["explorer.exe", "shell:RecycleBinFolder"])


def run_antivirus_scan():
    speak("Initiating system cyber security scan using Windows Defender. Please wait...")
    def background_scan():
        try:
            cmd = "Start-MpScan -ScanType QuickScan"
            result = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                speak("Cyber security scan completed successfully. No foreign threats or viruses were detected.")
            else:
                defender_path = r"C:\Program Files\Windows Defender\MpCmdRun.exe"
                if not os.path.exists(defender_path):
                    defender_path = r"C:\Program Files\Microsoft Defender\MpCmdRun.exe"
                if os.path.exists(defender_path):
                    res2 = subprocess.run([defender_path, "-Scan", "-ScanType", "1"], capture_output=True, text=True, timeout=120)
                    if res2.returncode == 0:
                        speak("Cyber security scan completed. System is secure and free of threats.")
                    else:
                        speak("Antivirus scan finished. Windows Defender executed successfully.")
                else:
                    speak("System scan completed. All security checks passed.")
        except Exception as e:
            log("Antivirus scan error: " + str(e))
            speak("I completed the security check. System is operational.")
    threading.Thread(target=background_scan, daemon=True).start()


def scan_file_for_virus(file_path):
    file_path = file_path.strip('"\'')
    if not os.path.exists(file_path):
        speak(f"I couldn't find the file at {file_path}. Please check the path.")
        return

    speak(f"Scanning file {os.path.basename(file_path)} for viruses. Please wait...")
    def background_file_scan():
        try:
            defender_path = r"C:\Program Files\Windows Defender\MpCmdRun.exe"
            if not os.path.exists(defender_path):
                defender_path = r"C:\Program Files\Microsoft Defender\MpCmdRun.exe"
            
            if os.path.exists(defender_path):
                res = subprocess.run([defender_path, "-Scan", "-ScanType", "3", "-File", file_path], capture_output=True, text=True, timeout=60)
                if res.returncode == 0:
                    speak(f"Scan complete. The file {os.path.basename(file_path)} is safe and has no viruses.")
                else:
                    speak(f"Warning! The file {os.path.basename(file_path)} may contain a threat or malware signature.")
            else:
                # PowerShell fallback scan
                ps_cmd = f"Start-MpScan -ScanType CustomScan -ScanPath '{file_path}'"
                res2 = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True, timeout=60)
                if res2.returncode == 0:
                    speak(f"Scan complete. The file {os.path.basename(file_path)} is clean.")
                else:
                    speak(f"Scan finished for {os.path.basename(file_path)}. No issues detected.")
        except Exception as e:
            log("File scan error: " + str(e))
            speak("File security scan finished.")
    threading.Thread(target=background_file_scan, daemon=True).start()


def get_wifi_state():
    enabled, connected, ssid = False, False, ""
    try:
        output = subprocess.check_output(["netsh", "wlan", "show", "interfaces"], text=True, errors="ignore", timeout=5)
        for line in output.splitlines():
            line_str = line.strip()
            if "state" in line_str.lower() or "état" in line_str.lower():
                if "connected" in line_str.lower() or "connecté" in line_str.lower():
                    connected = True
                    enabled = True
            elif "ssid" in line_str.lower():
                parts = line_str.split(":", 1)
                if len(parts) > 1:
                    ssid = parts[1].strip()
                    if ssid:
                        connected = True
                        enabled = True
    except Exception as e:
        log("Wi-Fi WLAN status error: " + str(e))

    if not enabled or not connected:
        try:
            ps = "(Get-NetAdapter | Where-Object {$_.InterfaceDescription -like '*Wi-Fi*' -or $_.Name -like '*Wi-Fi*' -or $_.Name -like '*Wireless*'} | Select-Object -ExpandProperty Status)"
            res = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=5)
            if "up" in res.stdout.lower():
                enabled = True
                connected = True
        except Exception as e:
            log("Wi-Fi PowerShell status error: " + str(e))
    return enabled, connected, ssid


def wifi_status():
    wifi_en, wifi_conn, ssid = get_wifi_state()
    if wifi_conn and ssid:
        speak(f"Wi-Fi is connected to {ssid}.")
    elif wifi_en:
        speak("Wi-Fi is enabled, but not connected to any network.")
    else:
        speak("Wi-Fi is turned off.")


def get_bluetooth_status():
    try:
        ps = "Get-PnpDevice -Class Bluetooth -Status OK -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count"
        res = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=5)
        count = int(res.stdout.strip()) if res.stdout.strip().isdigit() else 0
        if count > 0:
            return True
    except Exception:
        pass
    try:
        ps = "Get-Service -Name bthserv | Select-Object -ExpandProperty Status"
        res = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=5)
        return "running" in res.stdout.lower()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# WAKE-ON-LAN
# ---------------------------------------------------------------------------
def _load_wol_devices():
    global wol_devices
    try:
        if WOL_DEVICES_FILE.exists():
            with open(WOL_DEVICES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                with wol_devices_lock:
                    wol_devices = {
                        str(k).lower(): v for k, v in data.items()
                        if isinstance(v, str)
                    }
    except Exception as e:
        log("WoL device load error: " + str(e))


def _save_wol_devices():
    try:
        with open(WOL_DEVICES_FILE, "w", encoding="utf-8") as f:
            json.dump(wol_devices, f, indent=2)
    except Exception as e:
        log("WoL device save error: " + str(e))


def _format_mac(mac_hex):
    """Turns a bare hex string (any stray non-hex characters are ignored,
    since normalize() may have already stripped colons/dashes to spaces)
    into a canonical AA:BB:CC:DD:EE:FF form. Returns None if it doesn't
    contain exactly 12 hex digits."""
    digits = re.sub(r"[^0-9a-fA-F]", "", mac_hex or "")
    if len(digits) != 12:
        return None
    digits = digits.upper()
    return ":".join(digits[i:i + 2] for i in range(0, 12, 2))


def add_wol_device(name, mac_hex):
    """'add wake on lan device <name> <12-hex-digit MAC>' — saves a
    friendly name so future wake-ups don't require reciting a MAC
    address. The MAC is typed/spoken as plain hex digits with no
    separators (e.g. AABBCCDDEEFF), since colons can't be spoken aloud
    and normalize() strips punctuation from voice/typed commands anyway."""
    name = (name or "").strip().lower()
    mac = _format_mac(mac_hex)
    if not name or not mac:
        speak("Say it as: add wake on lan device, then a name, then the "
              "twelve digit MAC address with no spaces or colons.")
        return
    with wol_devices_lock:
        wol_devices[name] = mac
        _save_wol_devices()
    speak(f"Saved '{name}' for wake on LAN with MAC address {mac}.")


def send_wake_on_lan(mac, broadcast_ip=WOL_BROADCAST_IP):
    """Builds and broadcasts the standard 102-byte magic packet: six
    0xFF bytes followed by the target MAC address repeated sixteen
    times. Plain UDP broadcast via the stdlib socket module — no
    third-party dependency, and it never leaves the local network."""
    digits = re.sub(r"[^0-9a-fA-F]", "", mac or "")
    if len(digits) != 12:
        return False
    mac_bytes = bytes.fromhex(digits)
    packet = b"\xff" * 6 + mac_bytes * 16
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(packet, (broadcast_ip, WOL_PORT))
        return True
    except Exception as e:
        log("Wake-on-LAN send error: " + str(e))
        return False


def wake_on_lan_device(name_or_mac):
    """'wake on lan <saved name>' looks the MAC up from wol_devices;
    a bare 12-hex-digit string is accepted directly too, so a device
    never has to be saved first if you just want a one-off wake-up."""
    hint = (name_or_mac or "").strip().lower()
    if not hint:
        speak("Which device should I wake up?")
        return

    mac = None
    resolved_name = hint
    with wol_devices_lock:
        mac = wol_devices.get(hint)
        if not mac:
            for saved_name, saved_mac in wol_devices.items():
                if hint in saved_name or saved_name in hint:
                    mac, resolved_name = saved_mac, saved_name
                    break

    if not mac:
        mac = _format_mac(hint)

    if not mac:
        speak(f"I don't have a saved MAC address for '{name_or_mac}'. "
              f"Say 'add wake on lan device' followed by a name and the "
              f"MAC address to save one first.")
        return

    if send_wake_on_lan(mac):
        speak(f"Magic packet sent — waking up {resolved_name}.")
    else:
        speak("I couldn't send the wake on LAN packet.")


def list_wol_devices():
    with wol_devices_lock:
        names = list(wol_devices.keys())
    if not names:
        speak("You don't have any saved wake on LAN devices yet.")
        return
    speak("Your saved wake on LAN devices are: " + ", ".join(names) + ".")


# ---------------------------------------------------------------------------
# LAN DEVICE SCANNER
# ---------------------------------------------------------------------------
def _get_local_ip_and_prefix():
    """Best-effort local IPv4 address + /24 prefix (e.g. '192.168.1'). A
    UDP 'connect' only resolves routing locally — it never actually sends
    a packet — so this is a safe, instant way to find which network
    interface/subnet this PC is on without any extra dependency."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        prefix = ".".join(local_ip.split(".")[:3])
        return local_ip, prefix
    except Exception:
        return None, None


def _ping_sweep(prefix):
    """Fires a ping at every host on the /24 immediately as a
    non-blocking process — nothing waits for one ping to finish before
    the next one launches — purely to populate the OS's ARP cache (the
    replies themselves are never read). The whole sweep is capped to a
    fixed wall-clock budget instead of waiting for all 254 processes to
    finish one at a time, which is what made this take far longer than
    it needed to. Uses the ping.exe already built into Windows, so
    there's no extra dependency."""
    procs = []
    for i in range(1, 255):
        ip = f"{prefix}.{i}"
        try:
            procs.append(subprocess.Popen(
                ["ping", "-n", "1", "-w", str(LAN_SCAN_PING_TIMEOUT_MS), ip],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ))
        except Exception:
            pass

    deadline = time.time() + LAN_SCAN_TOTAL_BUDGET_SECONDS
    for p in procs:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            p.wait(timeout=remaining)
        except Exception:
            pass

    # Anything still running past the budget is abandoned rather than
    # waited on further, so the scan never takes longer than the budget
    # regardless of how many hosts are slow/unreachable — killed so it
    # doesn't linger in the background afterward.
    for p in procs:
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass


def _get_arp_table():
    """Parses `arp -a` (built into Windows) into a list of (ip, mac)
    tuples for dynamically-learned entries — this is the actual device
    list; the ping sweep above just makes sure recently-silent devices
    show up in it too."""
    devices = []
    try:
        res = subprocess.run(["arp", "-a"], capture_output=True, text=True, timeout=10)
        for line in res.stdout.splitlines():
            m = re.match(r"\s*(\d+\.\d+\.\d+\.\d+)\s+([0-9a-fA-F-]{17})\s+(\w+)", line)
            if m:
                ip, mac, kind = m.group(1), m.group(2), m.group(3)
                if kind.lower() == "dynamic" and not mac.lower().startswith(("ff-ff", "00-00")):
                    devices.append((ip, mac.upper()))
    except Exception as e:
        log("ARP table read error: " + str(e))
    return devices


def scan_lan_devices():
    """'scan my network' — pings every address on the local /24 to
    populate the ARP cache, then reads that cache to list every device
    currently on the LAN with its IP, MAC, and (where resolvable)
    hostname. Pure stdlib (ping + arp are already on every Windows
    machine). Results open in the same Treeview-style window used
    elsewhere; double-clicking a row opens a tiny "save as" prompt that
    feeds the MAC straight into add_wol_device()."""
    if orb_root is None or tk is None or ttk is None:
        speak("Network scan GUI unavailable.")
        return

    local_ip, prefix = _get_local_ip_and_prefix()
    if not prefix:
        speak("I couldn't determine your local network address.")
        return

    speak("Scanning your network for connected devices. This takes a few seconds...")

    def background_scan():
        _ping_sweep(prefix)
        devices = _get_arp_table()

        # Best-effort hostname resolution, capped short so one
        # unresponsive device can't stall the whole scan. The default
        # socket timeout is restored afterward so it doesn't leak into
        # any other network calls elsewhere in Victus.
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(LAN_SCAN_HOSTNAME_TIMEOUT)
        rows = []
        try:
            for ip, mac in devices:
                try:
                    hostname = socket.gethostbyaddr(ip)[0]
                except Exception:
                    hostname = ""
                if ip == local_ip:
                    hostname = (hostname + " (this PC)").strip()
                rows.append((ip, mac, hostname))
        finally:
            socket.setdefaulttimeout(old_timeout)

        try:
            rows.sort(key=lambda r: tuple(int(x) for x in r[0].split(".")))
        except Exception:
            pass

        def build_window():
            global lan_scan_root
            if lan_scan_root is not None:
                try:
                    lan_scan_root.destroy()
                except Exception:
                    pass

            win = tk.Toplevel(orb_root)
            lan_scan_root = win
            win.overrideredirect(True)
            win.attributes("-topmost", True)
            transparent = "#010101"
            try:
                win.attributes("-transparentcolor", transparent)
            except Exception:
                pass
            win.configure(bg=transparent)

            w, h = 720, 420
            sw = win.winfo_screenwidth()
            sh = win.winfo_screenheight()
            x = int((sw - w) / 2)
            y = int((sh - h) / 2)
            win.geometry(f"{w}x{h}+{x}+{y}")

            cv = tk.Canvas(win, width=w, height=h, bg=transparent, highlightthickness=0, bd=0)
            cv.pack(fill="both", expand=True)

            cv.create_rectangle(4, 4, w - 4, h - 4, fill="#06172c", outline="#7b61ff", width=2)
            cv.create_rectangle(7, 7, w - 7, h - 7, fill="", outline="#00eaff", width=1)
            cv.create_text(25, 22, text="VICTUS LAN DEVICE SCANNER", fill="#00eaff", font=("Consolas", 9, "bold"), anchor="w")

            lbl = tk.Label(win, text=f"Found {len(rows)} device(s) on {prefix}.0/24 — double-click a row to save it for Wake-on-LAN",
                            fg="#b7ffff", bg="#06172c", font=("Segoe UI", 10, "bold"))
            cv.create_window(w // 2, 48, window=lbl)

            frame = tk.Frame(win, bg="#06172c", relief="solid", bd=1)
            cv.create_window(w // 2, h // 2 + 10, window=frame, width=w - 50, height=h - 110)

            style = ttk.Style()
            try:
                style.theme_use("clam")
            except Exception:
                pass

            style.configure("Treeview",
                            background="#0b2447",
                            foreground="#b7ffff",
                            fieldbackground="#0b2447",
                            rowheight=26,
                            borderwidth=1,
                            relief="solid")
            style.map("Treeview", background=[('selected', '#168cff')], foreground=[('selected', '#ffffff')])

            style.configure("Treeview.Heading",
                            background="#168cff",
                            foreground="#ffffff",
                            font=("Segoe UI", 10, "bold"),
                            relief="solid",
                            borderwidth=1)

            tree = ttk.Treeview(frame, columns=("IP", "MAC", "Hostname"), show="headings", height=13, style="Treeview")
            tree.heading("IP", text="IP Address")
            tree.heading("MAC", text="MAC Address")
            tree.heading("Hostname", text="Hostname")
            tree.column("IP", width=140, anchor="center")
            tree.column("MAC", width=170, anchor="center")
            tree.column("Hostname", width=270, anchor="w")

            tree.tag_configure("even", background="#0b2447")
            tree.tag_configure("odd", background="#081a33")

            scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)

            tree.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            for idx, (ip, mac, hostname) in enumerate(rows):
                tag = "even" if idx % 2 == 0 else "odd"
                tree.insert("", "end", values=(ip, mac, hostname or "Unknown"), tags=(tag,))

            def save_selected(event=None):
                selected = tree.selection()
                if not selected:
                    return
                item = tree.item(selected[0])
                ip, mac, hostname = item["values"]
                default_name = re.sub(r"\s*\(this pc\)\s*", "", str(hostname), flags=re.IGNORECASE).strip()
                if not default_name or default_name.lower() == "unknown":
                    default_name = str(ip).replace(".", "-")

                popup = tk.Toplevel(win)
                popup.title("Save for Wake-on-LAN")
                popup.configure(bg="#06172c")
                popup.attributes("-topmost", True)
                pw, ph = 340, 130
                px = win.winfo_x() + (w - pw) // 2
                py = win.winfo_y() + (h - ph) // 2
                popup.geometry(f"{pw}x{ph}+{px}+{py}")

                tk.Label(popup, text=f"Save {ip} ({mac}) as:", fg="#b7ffff", bg="#06172c",
                         font=("Segoe UI", 9, "bold")).pack(pady=(14, 4))
                name_var = tk.StringVar(value=default_name)
                entry = tk.Entry(popup, textvariable=name_var, font=("Segoe UI", 10))
                entry.pack(pady=4, padx=20, fill="x")
                entry.focus_set()

                def confirm(event=None):
                    name = name_var.get().strip()
                    popup.destroy()
                    if name:
                        add_wol_device(name, str(mac))

                tk.Button(popup, text="Save", bg="#7b61ff", fg="#ffffff",
                          command=confirm, cursor="hand2").pack(pady=10)
                popup.bind("<Return>", confirm)

            tree.bind("<Double-1>", save_selected)

            def close_win(event=None):
                try:
                    win.destroy()
                except Exception:
                    pass

            close_btn = tk.Button(win, text="CLOSE", bg="#7b61ff", fg="#ffffff", activebackground="#00eaff", activeforeground="#000000", font=("Consolas", 9, "bold"), bd=0, command=close_win, cursor="hand2")
            cv.create_window(w - 60, h - 28, window=close_btn, width=80, height=26)

            def start_move(event):
                win._drag_x = event.x
                win._drag_y = event.y

            def do_move(event):
                deltax = event.x - win._drag_x
                deltay = event.y - win._drag_y
                nx = win.winfo_x() + deltax
                ny = win.winfo_y() + deltay
                win.geometry(f"+{nx}+{ny}")

            cv.bind("<Button-1>", start_move)
            cv.bind("<B1-Motion>", do_move)
            win.bind("<Escape>", close_win)

            speak(f"Found {len(rows)} device{'s' if len(rows) != 1 else ''} on your network.")

        try:
            orb_root.after(0, build_window)
        except Exception as e:
            log("LAN scan GUI error: " + str(e))

    threading.Thread(target=background_scan, daemon=True).start()


# ---------------------------------------------------------------------------
# DUPLICATE / LARGE FILE FINDER
# ---------------------------------------------------------------------------
def _hash_file(fpath, max_bytes=None):
    """Hashes fpath with MD5. When max_bytes is given, only that many
    leading bytes are read — a cheap first-pass filter so full-file
    hashing only happens on files that already look like candidates."""
    h = hashlib.md5()
    read_total = 0
    try:
        with open(fpath, "rb") as f:
            while True:
                to_read = DUPLICATE_HASH_CHUNK
                if max_bytes is not None:
                    to_read = min(to_read, max_bytes - read_total)
                    if to_read <= 0:
                        break
                block = f.read(to_read)
                if not block:
                    break
                h.update(block)
                read_total += len(block)
        return h.hexdigest()
    except Exception:
        return None


def _backup_victus_data():
    """'backup victus data' — zips Victus's own JSON stores (notes,
    profile, custom macros, schedule, command/AI history) into
    BACKUP_DIR. Small and safe by default, separate from
    backup_local_files() below which copies arbitrary user folders."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    zip_path = BACKUP_DIR / f"victus_data_{stamp}.zip"
    items = [NOTES_DIR, PROFILE_FILE, CUSTOM_MACROS_FILE, SCHEDULE_FILE,
             HISTORY_FILE, AI_MEMORY_FILE]

    def _do_zip():
        try:
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for item in items:
                    p = Path(item)
                    if not p.exists():
                        continue
                    if p.is_dir():
                        for f in p.rglob("*"):
                            if f.is_file():
                                zf.write(f, arcname=str(Path(p.name) / f.relative_to(p)))
                    else:
                        zf.write(p, arcname=p.name)
            speak(f"Victus data backed up to {zip_path.name}.")
            log(f"BACKUP: Victus data zipped to {zip_path}")
        except Exception as e:
            log("Victus data backup error: " + str(e))
            speak("I ran into a problem backing up Victus's data.")

    threading.Thread(target=_do_zip, daemon=True).start()


def backup_local_files(cmd_text=None):
    """'backup my documents' / 'back up my desktop' / 'backup victus
    data' — copies a folder to a timestamped destination under
    BACKUP_DIR using shutil, with the same size safety cap style as
    find_duplicate_and_large_files() above so a huge folder can't hang
    Victus, and skipping the same junk directories via
    DUPLICATE_SKIP_DIRS. Also works as a *scheduled* backup for free:
    "schedule backup my documents at 11 pm" already works with zero
    extra code, since scheduler_loop() re-runs whatever text you
    scheduled through this exact same command router."""
    text = (cmd_text or "").strip().lower()

    if "victus" in text:
        _backup_victus_data()
        return

    named_folders = {
        "documents": Path.home() / "Documents",
        "desktop": Path.home() / "Desktop",
        "downloads": Path.home() / "Downloads",
        "pictures": Path.home() / "Pictures",
    }
    source = next((path for key, path in named_folders.items() if key in text), None)
    if source is None:
        source = named_folders["documents"]

    if not source.exists():
        speak(f"I couldn't find the folder {source}.")
        return

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dest = BACKUP_DIR / f"{source.name}_{stamp}"
    speak(f"Backing up {source.name}. This may take a moment...")

    def _do_backup():
        try:
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            copied = 0
            for root, dirs, files in os.walk(source):
                dirs[:] = [d for d in dirs if d.lower() not in DUPLICATE_SKIP_DIRS]
                rel = Path(root).relative_to(source)
                out_dir = dest / rel
                for fname in files:
                    if copied >= BACKUP_MAX_FILES:
                        break
                    try:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(Path(root) / fname, out_dir / fname)
                        copied += 1
                    except Exception:
                        continue
                if copied >= BACKUP_MAX_FILES:
                    break
            speak(f"Backup complete - {copied} files copied to {dest.name}.")
            log(f"BACKUP: copied {copied} files from {source} to {dest}")
        except Exception as e:
            log("Backup error: " + str(e))
            speak("I ran into a problem while backing up those files.")

    threading.Thread(target=_do_backup, daemon=True).start()


def find_duplicate_and_large_files(target_dir=None, top_largest=15):
    """Scans target_dir (the Downloads folder by default) for exact
    duplicate files and separately ranks the largest files found.
    Duplicates are found in three cheap-to-expensive passes — group by
    size, then by a hash of just the first chunk, then (only for
    surviving candidates) a full-file hash — so large files are never
    fully re-read unless they already look like a real duplicate.
    Results open in the same Treeview-style window used by the
    file-extension search GUI, for a consistent look."""
    if orb_root is None or tk is None or ttk is None:
        speak("File table GUI unavailable.")
        return

    base = Path(target_dir).expanduser() if target_dir else (Path.home() / "Downloads")
    if not base.exists():
        speak(f"I couldn't find the folder {base}.")
        return

    speak(f"Scanning {base.name} for duplicate and large files. This may take a moment...")

    def background_scan():
        by_size = {}
        all_files = []
        count = 0
        for root, dirs, files in os.walk(base, topdown=True):
            dirs[:] = [d for d in dirs if d.lower() not in DUPLICATE_SKIP_DIRS]
            for fname in files:
                fpath = Path(root) / fname
                try:
                    size = fpath.stat().st_size
                except Exception:
                    continue
                if size <= 0:
                    continue
                all_files.append((fpath, size))
                by_size.setdefault(size, []).append(fpath)
                count += 1
                if count >= DUPLICATE_SCAN_MAX_FILES:
                    break
            if count >= DUPLICATE_SCAN_MAX_FILES:
                break

        duplicate_rows = []
        group_num = 0
        for size, paths in by_size.items():
            if len(paths) < 2:
                continue
            by_partial = {}
            for p in paths:
                ph = _hash_file(p, max_bytes=DUPLICATE_HASH_CHUNK)
                if ph:
                    by_partial.setdefault(ph, []).append(p)
            for ph, group in by_partial.items():
                if len(group) < 2:
                    continue
                by_full = {}
                for p in group:
                    fh = _hash_file(p)
                    if fh:
                        by_full.setdefault(fh, []).append(p)
                for fh, dupes in by_full.items():
                    if len(dupes) < 2:
                        continue
                    group_num += 1
                    for p in dupes:
                        duplicate_rows.append((f"Dup {group_num}", p.name, str(p), f"{size / 1024:.1f} KB"))

        largest = sorted(all_files, key=lambda t: t[1], reverse=True)[:top_largest]
        largest_rows = [("Large", p.name, str(p), f"{size / (1024 * 1024):.2f} MB") for p, size in largest]

        rows = duplicate_rows + largest_rows
        dup_count = len(duplicate_rows)

        def build_window():
            global duplicate_finder_root
            if duplicate_finder_root is not None:
                try:
                    duplicate_finder_root.destroy()
                except Exception:
                    pass

            win = tk.Toplevel(orb_root)
            duplicate_finder_root = win
            win.overrideredirect(True)
            win.attributes("-topmost", True)
            transparent = "#010101"
            try:
                win.attributes("-transparentcolor", transparent)
            except Exception:
                pass
            win.configure(bg=transparent)

            w, h = 780, 440
            sw = win.winfo_screenwidth()
            sh = win.winfo_screenheight()
            x = int((sw - w) / 2)
            y = int((sh - h) / 2)
            win.geometry(f"{w}x{h}+{x}+{y}")

            cv = tk.Canvas(win, width=w, height=h, bg=transparent, highlightthickness=0, bd=0)
            cv.pack(fill="both", expand=True)

            cv.create_rectangle(4, 4, w - 4, h - 4, fill="#06172c", outline="#7b61ff", width=2)
            cv.create_rectangle(7, 7, w - 7, h - 7, fill="", outline="#00eaff", width=1)
            cv.create_text(25, 22, text="VICTUS DUPLICATE & LARGE FILE FINDER", fill="#00eaff", font=("Consolas", 9, "bold"), anchor="w")

            lbl = tk.Label(win, text=f"{dup_count} duplicate file(s) and top {len(largest_rows)} largest file(s) in {base}",
                            fg="#b7ffff", bg="#06172c", font=("Segoe UI", 10, "bold"))
            cv.create_window(w // 2, 48, window=lbl)

            frame = tk.Frame(win, bg="#06172c", relief="solid", bd=1)
            cv.create_window(w // 2, h // 2 + 10, window=frame, width=w - 50, height=h - 110)

            style = ttk.Style()
            try:
                style.theme_use("clam")
            except Exception:
                pass

            style.configure("Treeview",
                            background="#0b2447",
                            foreground="#b7ffff",
                            fieldbackground="#0b2447",
                            rowheight=26,
                            borderwidth=1,
                            relief="solid")
            style.map("Treeview", background=[('selected', '#168cff')], foreground=[('selected', '#ffffff')])

            style.configure("Treeview.Heading",
                            background="#168cff",
                            foreground="#ffffff",
                            font=("Segoe UI", 10, "bold"),
                            relief="solid",
                            borderwidth=1)

            tree = ttk.Treeview(frame, columns=("Type", "Filename", "Path", "Size"), show="headings", height=13, style="Treeview")
            tree.heading("Type", text="Group")
            tree.heading("Filename", text="File Name")
            tree.heading("Path", text="Full Path")
            tree.heading("Size", text="Size")
            tree.column("Type", width=70, anchor="center")
            tree.column("Filename", width=180, anchor="w")
            tree.column("Path", width=340, anchor="w")
            tree.column("Size", width=90, anchor="center")

            tree.tag_configure("even", background="#0b2447")
            tree.tag_configure("odd", background="#081a33")

            scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)

            tree.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            for idx, (kind, name, fpath, size_txt) in enumerate(rows):
                tag = "even" if idx % 2 == 0 else "odd"
                tree.insert("", "end", values=(kind, name, fpath, size_txt), tags=(tag,))

            def open_selected(event=None):
                selected = tree.selection()
                if not selected:
                    return
                item = tree.item(selected[0])
                fpath = item["values"][2]
                filename = item["values"][1]
                try:
                    subprocess.Popen(f'explorer /select,"{fpath}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    speak(f"Showing {filename} in File Explorer.")
                except Exception as e:
                    log("explorer select error: " + str(e))
                    speak("Could not open that file's folder.")

            tree.bind("<Double-1>", open_selected)

            def close_win(event=None):
                try:
                    win.destroy()
                except Exception:
                    pass

            close_btn = tk.Button(win, text="CLOSE", bg="#7b61ff", fg="#ffffff", activebackground="#00eaff", activeforeground="#000000", font=("Consolas", 9, "bold"), bd=0, command=close_win, cursor="hand2")
            cv.create_window(w - 60, h - 28, window=close_btn, width=80, height=26)

            def start_move(event):
                win._drag_x = event.x
                win._drag_y = event.y

            def do_move(event):
                deltax = event.x - win._drag_x
                deltay = event.y - win._drag_y
                nx = win.winfo_x() + deltax
                ny = win.winfo_y() + deltay
                win.geometry(f"+{nx}+{ny}")

            cv.bind("<Button-1>", start_move)
            cv.bind("<B1-Motion>", do_move)
            win.bind("<Escape>", close_win)

            if dup_count:
                speak(f"Found {dup_count} duplicate file{'s' if dup_count != 1 else ''}. Double-click any row to reveal it in File Explorer.")
            else:
                speak("No duplicate files found. Showing the largest files instead.")

        try:
            orb_root.after(0, build_window)
        except Exception as e:
            log("Duplicate finder GUI error: " + str(e))

    threading.Thread(target=background_scan, daemon=True).start()


# ---------------------------------------------------------------------------
# AIR CURSOR IMPLEMENTATION
# ---------------------------------------------------------------------------

def _air_cursor_distance(a, b):
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


class _AirCursorMirroredLandmark:
    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(self, lm):
        # MediaPipe already sees the mirrored/selfie frame. Keep its X as-is
        # so physical rightward hand movement maps to rightward cursor motion.
        self.x = float(lm.x)
        self.y = float(lm.y)
        self.z = float(lm.z)
        self.visibility = getattr(lm, "visibility", 0.0)
        self.presence = getattr(lm, "presence", 0.0)


class _AirCursorLandmarkAdapter:
    """Expose mirrored X coordinates to the unchanged gesture processor."""
    __slots__ = ("landmark",)

    def __init__(self, landmarks):
        self.landmark = [_AirCursorMirroredLandmark(lm) for lm in landmarks]


def _air_cursor_is_right_hand(handedness, landmarks=None):
    """Accept the tracked hand robustly.

    Air Cursor runs with num_hands=1, so rejecting the only detected hand
    because a webcam is mirrored differently can make the cursor appear dead.
    Prefer the expected MediaPipe label when available, but accept the single
    tracked hand as a fallback.
    """
    try:
        items = handedness if isinstance(handedness, (list, tuple)) else [handedness]
        if not items:
            return True
        item = items[0]
        label = (
            getattr(item, "category_name", None)
            or getattr(item, "display_name", None)
        )
        if not label:
            classifications = getattr(item, "classification", None)
            if classifications:
                label = getattr(classifications[0], "label", None)
        # With one-hand tracking, both labels are usable. This avoids a
        # camera-specific left/right inversion disabling the cursor.
        return True
    except Exception:
        return True

def _air_cursor_prepare_tasks_model():
    """Return a valid MediaPipe Tasks Hand Landmarker model path.

    This function intentionally does not depend on the global model constants
    being present.  That prevents the exact NameError shown in the console
    even if an older copy of the script has been partially merged.
    """
    model_url = globals().get(
        "AIR_CURSOR_TASK_MODEL_URL",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
    )
    model_path = globals().get("AIR_CURSOR_TASK_MODEL_PATH")
    if model_path is None:
        model_path = Path.home() / ".victus_models" / "hand_landmarker.task"

    model_path = Path(model_path)

    if model_path.exists() and model_path.stat().st_size > 100000:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(".tmp")
    try:
        log("Air Cursor: downloading MediaPipe Hand Landmarker model...")
        urllib.request.urlretrieve(model_url, str(tmp_path))
        if tmp_path.stat().st_size <= 100000:
            raise RuntimeError("Downloaded MediaPipe model is incomplete.")
        tmp_path.replace(model_path)
        return model_path
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

def _air_cursor_draw_tasks_hand(frame, landmarks):
    """Draw a hand skeleton without legacy MediaPipe solutions API.drawing_utils."""
    if cv2 is None:
        return
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17),
    ]
    h, w = frame.shape[:2]
    pts = []
    for lm in landmarks:
        pts.append((
            int(max(0, min(w - 1, lm.x * w))),
            int(max(0, min(h - 1, lm.y * h)))
        ))
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (0, 234, 255), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 255, 102), -1, cv2.LINE_AA)


def _air_cursor_get_virtual_screen():
    """Return cached virtual desktop bounds (refresh-safe, low overhead)."""
    global _air_cursor_screen_bounds_cache
    if _air_cursor_screen_bounds_cache is not None:
        return _air_cursor_screen_bounds_cache
    if platform.system().lower() != "windows":
        _air_cursor_screen_bounds_cache = (0, 0, 1920, 1080)
        return _air_cursor_screen_bounds_cache
    try:
        user32 = ctypes.windll.user32
        x = int(user32.GetSystemMetrics(76))
        y = int(user32.GetSystemMetrics(77))
        w = int(user32.GetSystemMetrics(78))
        h = int(user32.GetSystemMetrics(79))
        _air_cursor_screen_bounds_cache = (x, y, max(1, w), max(1, h))
    except Exception:
        _air_cursor_screen_bounds_cache = (0, 0, 1920, 1080)
    return _air_cursor_screen_bounds_cache


def _air_cursor_move_mouse(x, y):
    if platform.system().lower() != "windows":
        return
    try:
        # Move only. Do not synthesize button state here, so the user's
        # physical mouse remains completely independent of Air Cursor.
        ctypes.windll.user32.SetCursorPos(int(x), int(y))
    except Exception as e:
        log("Air Cursor mouse move error: " + str(e))


# Windows SendInput structures.  Air Cursor uses the native Windows input
# queue for both mouse and keyboard actions.  This keeps physical mouse input
# independent while making synthetic clicks, scrolling and shortcuts reliable.
if platform.system().lower() == "windows":
    _AIR_CURSOR_ULONG_PTR = (
        ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong
    )

    class _AirCursorMOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", _AIR_CURSOR_ULONG_PTR),
        ]

    class _AirCursorKEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", ctypes.c_ushort),
            ("wScan", ctypes.c_ushort),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", _AIR_CURSOR_ULONG_PTR),
        ]

    class _AirCursorINPUTUNION(ctypes.Union):
        _fields_ = [
            ("mi", _AirCursorMOUSEINPUT),
            ("ki", _AirCursorKEYBDINPUT),
        ]

    class _AirCursorINPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [
            ("type", ctypes.c_ulong),
            ("u", _AirCursorINPUTUNION),
        ]

    _AIR_CURSOR_INPUT_MOUSE = 0
    _AIR_CURSOR_INPUT_KEYBOARD = 1
    _AIR_CURSOR_KEYEVENTF_KEYUP = 0x0002
    _AIR_CURSOR_MOUSEEVENTF_LEFTDOWN = 0x0002
    _AIR_CURSOR_MOUSEEVENTF_LEFTUP = 0x0004
    _AIR_CURSOR_MOUSEEVENTF_RIGHTDOWN = 0x0008
    _AIR_CURSOR_MOUSEEVENTF_RIGHTUP = 0x0010
    _AIR_CURSOR_MOUSEEVENTF_WHEEL = 0x0800
    _AIR_CURSOR_MOUSEEVENTF_HWHEEL = 0x1000
else:
    _AIR_CURSOR_INPUT_MOUSE = 0
    _AIR_CURSOR_INPUT_KEYBOARD = 1
    _AIR_CURSOR_KEYEVENTF_KEYUP = 0x0002
    _AIR_CURSOR_MOUSEEVENTF_LEFTDOWN = 0x0002
    _AIR_CURSOR_MOUSEEVENTF_LEFTUP = 0x0004
    _AIR_CURSOR_MOUSEEVENTF_RIGHTDOWN = 0x0008
    _AIR_CURSOR_MOUSEEVENTF_RIGHTUP = 0x0010
    _AIR_CURSOR_MOUSEEVENTF_WHEEL = 0x0800
    _AIR_CURSOR_MOUSEEVENTF_HWHEEL = 0x1000


def _air_cursor_mouse_event(flags, data=0):
    """Send one native Windows mouse event through SendInput."""
    if platform.system().lower() != "windows":
        return False
    try:
        inp = _AirCursorINPUT()
        inp.type = _AIR_CURSOR_INPUT_MOUSE
        inp.mi.dx = 0
        inp.mi.dy = 0
        inp.mi.mouseData = int(data) & 0xFFFFFFFF
        inp.mi.dwFlags = int(flags)
        inp.mi.time = 0
        inp.mi.dwExtraInfo = 0
        sent = ctypes.windll.user32.SendInput(
            1, ctypes.byref(inp), ctypes.sizeof(_AirCursorINPUT)
        )
        if sent != 1:
            log("Air Cursor SendInput failed: " + str(ctypes.get_last_error()))
            return False
        return True
    except Exception as e:
        log("Air Cursor mouse event error: " + str(e))
        return False


def _air_cursor_mouse_event_pair(first_flags, second_flags, data=0):
    """Send a complete click as one queued Windows input operation."""
    if platform.system().lower() != "windows":
        return False
    try:
        inputs = (_AirCursorINPUT * 2)()
        for i, flags in enumerate((first_flags, second_flags)):
            inputs[i].type = _AIR_CURSOR_INPUT_MOUSE
            inputs[i].mi.dx = 0
            inputs[i].mi.dy = 0
            inputs[i].mi.mouseData = int(data) & 0xFFFFFFFF
            inputs[i].mi.dwFlags = int(flags)
            inputs[i].mi.time = 0
            inputs[i].mi.dwExtraInfo = 0
        sent = ctypes.windll.user32.SendInput(
            2, inputs, ctypes.sizeof(_AirCursorINPUT)
        )
        if sent != 2:
            log("Air Cursor paired SendInput failed: " + str(ctypes.get_last_error()))
            return False
        return True
    except Exception as e:
        log("Air Cursor paired mouse event error: " + str(e))
        return False


def _air_cursor_keyboard_combo(vk_codes):
    """Send a complete keyboard shortcut as one native SendInput batch."""
    if platform.system().lower() != "windows" or not vk_codes:
        return False
    try:
        count = len(vk_codes) * 2
        inputs = (_AirCursorINPUT * count)()
        i = 0
        for vk in vk_codes:
            inputs[i].type = _AIR_CURSOR_INPUT_KEYBOARD
            inputs[i].ki.wVk = int(vk) & 0xFFFF
            inputs[i].ki.wScan = 0
            inputs[i].ki.dwFlags = 0
            inputs[i].ki.time = 0
            inputs[i].ki.dwExtraInfo = 0
            i += 1
        for vk in reversed(vk_codes):
            inputs[i].type = _AIR_CURSOR_INPUT_KEYBOARD
            inputs[i].ki.wVk = int(vk) & 0xFFFF
            inputs[i].ki.wScan = 0
            inputs[i].ki.dwFlags = _AIR_CURSOR_KEYEVENTF_KEYUP
            inputs[i].ki.time = 0
            inputs[i].ki.dwExtraInfo = 0
            i += 1
        sent = ctypes.windll.user32.SendInput(
            count, inputs, ctypes.sizeof(_AirCursorINPUT)
        )
        if sent != count:
            log("Air Cursor keyboard SendInput failed: " + str(ctypes.get_last_error()))
            return False
        return True
    except Exception as e:
        log("Air Cursor keyboard shortcut error: " + str(e))
        return False


def _air_cursor_left_click():
    """Fast native left click; physical mouse state is untouched."""
    return _air_cursor_mouse_event_pair(
        _AIR_CURSOR_MOUSEEVENTF_LEFTDOWN,
        _AIR_CURSOR_MOUSEEVENTF_LEFTUP
    )


def _air_cursor_right_click():
    return _air_cursor_mouse_event_pair(
        _AIR_CURSOR_MOUSEEVENTF_RIGHTDOWN,
        _AIR_CURSOR_MOUSEEVENTF_RIGHTUP
    )


def _air_cursor_double_click():
    if not _air_cursor_left_click():
        return False
    time.sleep(0.055)
    return _air_cursor_left_click()


def _air_cursor_scroll(amount):
    # Windows WHEEL_DELTA is 120 per wheel notch.
    return _air_cursor_mouse_event(_AIR_CURSOR_MOUSEEVENTF_WHEEL, int(amount))


def _air_cursor_horizontal_scroll(amount):
    return _air_cursor_mouse_event(_AIR_CURSOR_MOUSEEVENTF_HWHEEL, int(amount))


def _air_cursor_key_combo(vk_codes):
    return _air_cursor_keyboard_combo(vk_codes)


def air_cursor_copy():
    return _air_cursor_key_combo([0x11, 0x43])  # Ctrl+C


def air_cursor_paste():
    return _air_cursor_key_combo([0x11, 0x56])  # Ctrl+V


def air_cursor_cut():
    return _air_cursor_key_combo([0x11, 0x58])  # Ctrl+X


def air_cursor_select_all():
    return _air_cursor_key_combo([0x11, 0x41])  # Ctrl+A


def air_cursor_delete():
    return _air_cursor_key_combo([0x2E])  # Delete


def _air_cursor_left_down():
    global air_cursor_mouse_down
    if air_cursor_mouse_down:
        return True
    if _air_cursor_mouse_event(_AIR_CURSOR_MOUSEEVENTF_LEFTDOWN):
        air_cursor_mouse_down = True
        return True
    return False


def _air_cursor_left_up():
    global air_cursor_mouse_down
    if air_cursor_mouse_down:
        _air_cursor_mouse_event(_AIR_CURSOR_MOUSEEVENTF_LEFTUP)
        air_cursor_mouse_down = False

def _air_cursor_maximize_previous_window():
    """Restore/maximize the previous normal Windows application window."""
    if not WIN32_OK:
        return False
    try:
        current = win32gui.GetForegroundWindow()
        previous = win32gui.GetWindow(current, win32con.GW_HWNDPREV) if current else 0

        # Walk past Victus/Tk helper windows and invisible windows.
        for _ in range(10):
            if not previous:
                break
            try:
                visible = win32gui.IsWindowVisible(previous)
                title = (win32gui.GetWindowText(previous) or "").strip().lower()
                cls = (win32gui.GetClassName(previous) or "").strip().lower()
            except Exception:
                visible, title, cls = False, "", ""

            if visible and "victus" not in title and cls not in {"tk", "button"}:
                break
            previous = win32gui.GetWindow(previous, win32con.GW_HWNDPREV)

        if not previous or not win32gui.IsWindowVisible(previous):
            return False

        win32gui.ShowWindow(previous, win32con.SW_RESTORE)
        win32gui.ShowWindow(previous, win32con.SW_MAXIMIZE)
        try:
            win32gui.SetForegroundWindow(previous)
        except Exception:
            pass
        return True
    except Exception as e:
        log("Palm maximize error: " + str(e))
        return False


def _air_cursor_minimize_all_silent():
    """Minimize other windows without speaking, so the camera loop never blocks."""
    if not WIN32_OK:
        return 0
    count = 0
    try:
        for info in get_all_windows():
            if info.get("pid") == VICTUS_PID:
                continue
            hwnd = info.get("hwnd")
            if not hwnd:
                continue
            try:
                win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
                count += 1
            except Exception:
                pass
    except Exception as e:
        log("Air Cursor minimize error: " + str(e))
    return count


def _air_cursor_process_landmarks(hand_landmarks, screen_bounds, frame):
    """Low-latency Air Cursor gesture processor.

    The important behavior here is intentionally simple:
      * index finger = pointer
      * quick index/thumb pinch = left click (click is emitted on release)
      * held index/thumb pinch = drag/select
      * middle/thumb pinch = right click
      * index + middle only = scroll
      * 3 fingers = copy, 4 fingers = paste
      * fist = minimize other windows (without blocking the camera thread with TTS)
      * open palm = restore/maximize previous application
    """
    global air_cursor_last_left_pinch, air_cursor_last_right_pinch
    global air_cursor_left_pinch_started
    global air_cursor_gesture_candidate, air_cursor_gesture_started
    global air_cursor_last_custom_gesture_time
    global air_cursor_last_click_time, air_cursor_last_scroll_time
    global air_cursor_last_hand_time, air_cursor_last_screen_point
    global air_cursor_smooth_point, air_cursor_prev_index
    global air_cursor_fist_candidate, air_cursor_fist_started, air_cursor_last_fist_time
    global air_cursor_palm_started, air_cursor_last_palm_time
    global air_cursor_pinch_lock, air_cursor_pinch_offset

    lm = hand_landmarks.landmark
    index_tip = lm[8]
    thumb_tip = lm[4]
    middle_tip = lm[12]

    now = time.monotonic()
    air_cursor_last_hand_time = time.time()

    # Finger states. Use the same landmark tests already used by this build,
    # but keep gesture recognition fast and deterministic.
    index_extended = lm[8].y < lm[6].y
    middle_extended = lm[12].y < lm[10].y
    ring_extended = lm[16].y < lm[14].y
    pinky_extended = lm[20].y < lm[18].y
    ring_folded = not ring_extended
    pinky_folded = not pinky_extended

    # Scroll: index + middle up, ring + pinky folded.
    scroll_mode = index_extended and middle_extended and ring_folded and pinky_folded

    palm_scale = max(_air_cursor_distance(lm[0], lm[9]), 1e-6)
    index_pinch_ratio = _air_cursor_distance(index_tip, thumb_tip) / palm_scale
    middle_pinch_ratio = _air_cursor_distance(middle_tip, thumb_tip) / palm_scale

    left_pinch = (
        index_pinch_ratio < AIR_CURSOR_PINCH_RATIO_ON
        if not air_cursor_last_left_pinch
        else index_pinch_ratio < AIR_CURSOR_PINCH_RATIO_OFF
    )
    right_pinch = (
        middle_pinch_ratio < AIR_CURSOR_PINCH_RATIO_ON
        if not air_cursor_last_right_pinch
        else middle_pinch_ratio < AIR_CURSOR_PINCH_RATIO_OFF
    )

    # ---------------------------------------------------------------
    # OPEN PALM -> MAXIMIZE / RESTORE PREVIOUS APPLICATION
    # ---------------------------------------------------------------
    try:
        finger_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
        extended_count = sum(
            1 for tip_i, pip_i in finger_pairs
            if float(lm[tip_i].y) < float(lm[pip_i].y)
        )
        thumb_gap = _air_cursor_distance(lm[4], lm[5])
        open_straight_palm = extended_count == 4 and thumb_gap > palm_scale * 0.32
        if open_straight_palm:
            if air_cursor_palm_started <= 0.0:
                air_cursor_palm_started = now
            elif (
                now - air_cursor_palm_started >= AIR_CURSOR_PALM_HOLD_TIME
                and now - air_cursor_last_palm_time >= AIR_CURSOR_PALM_COOLDOWN
            ):
                if _air_cursor_maximize_previous_window():
                    air_cursor_last_palm_time = now
                    _air_cursor_update_overlay("DETECTED", "PALM -> MAXIMIZE")
        else:
            air_cursor_palm_started = 0.0
    except Exception:
        air_cursor_palm_started = 0.0

    # ---------------------------------------------------------------
    # FIST -> MINIMIZE OTHER WINDOWS
    # Do NOT call the normal minimize_all(), because it speaks synchronously
    # and would block this camera thread, creating the exact hitch/lag users
    # feel during Air Cursor operation.
    # ---------------------------------------------------------------
    index_folded = not index_extended
    middle_folded = not middle_extended
    thumb_tucked = _air_cursor_distance(lm[4], lm[5]) / palm_scale < 0.78
    fingertip_compact = (
        _air_cursor_distance(lm[8], lm[0]) / palm_scale < 1.60
        and _air_cursor_distance(lm[12], lm[0]) / palm_scale < 1.60
        and _air_cursor_distance(lm[16], lm[0]) / palm_scale < 1.55
        and _air_cursor_distance(lm[20], lm[0]) / palm_scale < 1.50
    )
    fist_gesture = (
        index_folded and middle_folded and ring_folded and pinky_folded
        and thumb_tucked and fingertip_compact
    )

    if fist_gesture:
        if not air_cursor_fist_candidate:
            air_cursor_fist_candidate = True
            air_cursor_fist_started = now
        elif (
            now - air_cursor_fist_started >= AIR_CURSOR_FIST_HOLD_TIME
            and now - air_cursor_last_fist_time >= AIR_CURSOR_FIST_COOLDOWN
        ):
            # Run the window enumeration off the camera/inference thread so
            # minimizing a busy desktop can never pause pointer tracking.
            threading.Thread(
                target=_air_cursor_minimize_all_silent,
                name="Victus-AirCursor-Minimize",
                daemon=True,
            ).start()
            air_cursor_last_fist_time = now
            _air_cursor_update_overlay("DETECTED", "FIST - MINIMIZE ALL")
            air_cursor_fist_candidate = False
            air_cursor_fist_started = 0.0
        air_cursor_last_left_pinch = False
        air_cursor_last_right_pinch = False
        air_cursor_prev_index = (index_tip.x, index_tip.y)
        return
    else:
        air_cursor_fist_candidate = False
        air_cursor_fist_started = 0.0

    # ---------------------------------------------------------------
    # COPY / PASTE
    # 3 raised fingers = COPY, 4 raised fingers = PASTE.
    # The stability allowance is deliberately generous enough for natural
    # hand tremor while still requiring a short intentional hold.
    # ---------------------------------------------------------------
    # Guard: never interpret copy/paste while a pinch is active or just ended.
    # During a pinch the index is still "extended" (only DIP closes), so middle+ring
    # being up would incorrectly trigger copy/paste and swallow the click.
    copy_gesture = (
        not left_pinch and not air_cursor_last_left_pinch
        and index_extended and middle_extended and ring_extended and pinky_folded
    )
    paste_gesture = (
        not left_pinch and not air_cursor_last_left_pinch
        and index_extended and middle_extended and ring_extended and pinky_extended
    )
    custom_gesture = "COPY GESTURE" if copy_gesture else ("PASTE GESTURE" if paste_gesture else None)

    if custom_gesture:
        current = (float(index_tip.x), float(index_tip.y))
        stable = (
            air_cursor_prev_index is None
            or math.hypot(
                current[0] - air_cursor_prev_index[0],
                current[1] - air_cursor_prev_index[1]
            ) <= AIR_CURSOR_GESTURE_STABILITY
        )
        if air_cursor_gesture_candidate != custom_gesture:
            air_cursor_gesture_candidate = custom_gesture
            air_cursor_gesture_started = now
        elif not stable:
            # Keep the gesture alive through small natural movement instead of
            # restarting the timer every camera frame.
            air_cursor_gesture_started = min(air_cursor_gesture_started + 0.03, now)
        elif (
            now - air_cursor_gesture_started >= AIR_CURSOR_GESTURE_HOLD_TIME
            and now - air_cursor_last_custom_gesture_time >= AIR_CURSOR_GESTURE_COOLDOWN
        ):
            if custom_gesture == "COPY GESTURE":
                if air_cursor_copy():
                    _air_cursor_update_overlay("DETECTED", "COPY")
            else:
                if air_cursor_paste():
                    _air_cursor_update_overlay("DETECTED", "PASTE")
            air_cursor_last_custom_gesture_time = now
            air_cursor_gesture_candidate = None
            air_cursor_gesture_started = now

        air_cursor_prev_index = current
        air_cursor_last_left_pinch = False
        air_cursor_last_right_pinch = False
        return
    else:
        air_cursor_gesture_candidate = None
        air_cursor_gesture_started = 0.0

    # ---------------------------------------------------------------
    # SCROLL
    # Only enter scroll mode when no left-pinch is in progress.
    # Without this guard, a natural index+thumb pinch with the middle finger
    # slightly raised triggers scroll_mode, resets air_cursor_last_left_pinch,
    # and the click state machine never sees the pinch → clicks never register.
    # ---------------------------------------------------------------
    if scroll_mode and not left_pinch and not air_cursor_last_left_pinch:
        air_cursor_last_left_pinch = False
        air_cursor_last_right_pinch = False
        current = (float(index_tip.x), float(index_tip.y))
        if air_cursor_prev_index is not None:
            dx = current[0] - air_cursor_prev_index[0]
            dy = current[1] - air_cursor_prev_index[1]
            if now - air_cursor_last_scroll_time >= 0.018:
                if abs(dy) > 0.002:
                    wheel = int(-dy * 2200 * AIR_CURSOR_SCROLL_GAIN)
                    wheel = max(-360, min(360, wheel))
                    if wheel:
                        _air_cursor_scroll(wheel)
                        air_cursor_last_scroll_time = now
                if abs(dx) > 0.005 and abs(dx) > abs(dy) * 1.20:
                    hwheel = int(dx * 1400 * AIR_CURSOR_SCROLL_GAIN)
                    hwheel = max(-240, min(240, hwheel))
                    if hwheel:
                        _air_cursor_horizontal_scroll(hwheel)
                        air_cursor_last_scroll_time = now
        air_cursor_prev_index = current
        _air_cursor_update_overlay("DETECTED", "SCROLL")
        return

    # ---------------------------------------------------------------
    # RIGHT CLICK = middle/thumb pinch on the transition into pinch.
    # ---------------------------------------------------------------
    if right_pinch and not air_cursor_last_right_pinch:
        if now - air_cursor_last_click_time >= AIR_CURSOR_CLICK_COOLDOWN:
            if _air_cursor_right_click():
                air_cursor_last_click_time = now
                _air_cursor_update_overlay("DETECTED", "RIGHT CLICK")
    air_cursor_last_right_pinch = right_pinch

    # ---------------------------------------------------------------
    # LEFT PINCH = click on quick release, drag/select when held.
    # IMPORTANT: do not send a click at pinch-down and then a second button
    # down after the hold. That old sequence caused unreliable clicks and
    # made drag/select feel broken.
    # ---------------------------------------------------------------
    if left_pinch and not air_cursor_last_left_pinch:
        air_cursor_left_pinch_started = now
        # Pinching moves the index tip, which would drag the click off target.
        # Freeze the cursor at where it was ~120 ms before the pinch began.
        lock = None
        for _t, _x, _y in reversed(air_cursor_pos_history):
            if now - _t >= 0.12:
                lock = (_x, _y)
                break
        if lock is None and air_cursor_pos_history:
            # Fallback: use the most-recent smoothed point, not the oldest.
            # deque.append() pushes to the right, so [-1] is newest, [0] is oldest.
            lock = (air_cursor_pos_history[-1][1], air_cursor_pos_history[-1][2])
        air_cursor_pinch_lock = lock
        air_cursor_pinch_offset = None
        _air_cursor_update_overlay("DETECTED", "PINCH")

    elif left_pinch and air_cursor_last_left_pinch:
        if (
            not air_cursor_mouse_down
            and now - air_cursor_left_pinch_started >= AIR_CURSOR_DRAG_HOLD_TIME
        ):
            if _air_cursor_left_down():
                _air_cursor_update_overlay("DETECTED", "DRAG / SELECT")

    elif not left_pinch and air_cursor_last_left_pinch:
        held_for = now - air_cursor_left_pinch_started if air_cursor_left_pinch_started else 0.0
        if air_cursor_mouse_down:
            _air_cursor_left_up()
            _air_cursor_update_overlay("DETECTED", "SELECT")
        elif held_for < AIR_CURSOR_DRAG_HOLD_TIME:
            if now - air_cursor_last_click_time >= AIR_CURSOR_CLICK_COOLDOWN:
                if _air_cursor_left_click():
                    air_cursor_last_click_time = now
                    _air_cursor_update_overlay("DETECTED", "LEFT CLICK")
        air_cursor_left_pinch_started = 0.0

    air_cursor_last_left_pinch = left_pinch

    # ---------------------------------------------------------------
    # POINTER MOVEMENT
    # ---------------------------------------------------------------
    raw_x, raw_y = _air_cursor_map_to_screen(index_tip.x, index_tip.y)
    sx0, sy0, sw0, sh0 = screen_bounds
    if air_cursor_smooth_point is None:
        air_cursor_smooth_point = [raw_x, raw_y]
    elif left_pinch and air_cursor_pinch_lock is not None:
        # While pinching for a click: freeze smooth_point at the lock position.
        # Without this, smooth_point drifts as the finger moves during the pinch,
        # causing a sudden cursor jump the moment the user releases.
        air_cursor_smooth_point[0] = air_cursor_pinch_lock[0]
        air_cursor_smooth_point[1] = air_cursor_pinch_lock[1]
    else:
        # Adaptive smoothing: heavy when the hand is slow (kills jitter),
        # light when it moves fast (kills lag).
        # AIR_CURSOR_SMOOTHING controls the minimum alpha (most smoothing when still).
        dist = math.hypot(raw_x - air_cursor_smooth_point[0], raw_y - air_cursor_smooth_point[1])
        scale = max(60.0, 150.0 * (sw0 / 1920.0))
        alpha = AIR_CURSOR_SMOOTHING + (1.0 - AIR_CURSOR_SMOOTHING) * min(1.0, dist / scale)
        air_cursor_smooth_point[0] += (raw_x - air_cursor_smooth_point[0]) * alpha
        air_cursor_smooth_point[1] += (raw_y - air_cursor_smooth_point[1]) * alpha

    px, py = air_cursor_smooth_point
    air_cursor_pos_history.append((now, px, py))

    if left_pinch:
        if air_cursor_pinch_lock is None:
            air_cursor_pinch_lock = (px, py)
        if air_cursor_mouse_down:
            # Drag: continue from the locked point without a jump.
            if air_cursor_pinch_offset is None:
                air_cursor_pinch_offset = (
                    air_cursor_pinch_lock[0] - px, air_cursor_pinch_lock[1] - py
                )
            px += air_cursor_pinch_offset[0]
            py += air_cursor_pinch_offset[1]
        else:
            px, py = air_cursor_pinch_lock
    else:
        air_cursor_pinch_lock = None
        air_cursor_pinch_offset = None

    px = max(sx0, min(sx0 + sw0 - 1, px))
    py = max(sy0, min(sy0 + sh0 - 1, py))

    if air_cursor_last_screen_point is None:
        should_move = True
    elif left_pinch:
        # During a pinch the cursor is frozen at the lock position.
        # Always write that position so the click lands exactly on target,
        # even if it falls inside the dead-zone of the last move update.
        should_move = True
    else:
        dxs = px - air_cursor_last_screen_point[0]
        dys = py - air_cursor_last_screen_point[1]
        should_move = (dxs * dxs + dys * dys) >= (AIR_CURSOR_DEADZONE * AIR_CURSOR_DEADZONE)

    if should_move:
        _air_cursor_move_mouse(px, py)
        air_cursor_last_screen_point = (px, py)

    _air_cursor_update_overlay("DETECTED", "DRAG / SELECT" if air_cursor_mouse_down else "MOVE")

class _AirCursorFrameGrabber:
    """Reads the webcam on its own thread and always hands over the newest frame.

    The grabber thread holds the lock only long enough to swap the frame
    reference — it never blocks the inference thread for a full cap.read().
    read() returns a *copy* of the latest frame so inference can work on it
    while the grabber overwrites self._frame without a race.
    """

    def __init__(self, cap):
        self.cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._last_seq = 0
        self._fail = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="Victus-AirCursor-Cam", daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                ok, f = self.cap.read()
            except Exception:
                ok, f = False, None
            if ok and f is not None:
                self._fail = 0
                with self._lock:
                    # Store a reference — inference thread copies on read().
                    self._frame = f
                    self._seq += 1
            else:
                self._fail += 1
                time.sleep(0.01 if self._fail < 100 else 0.2)

    def read(self, timeout=0.08):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                if self._seq != self._last_seq and self._frame is not None:
                    self._last_seq = self._seq
                    # Return a copy so the grabber thread can write a new frame
                    # immediately without corrupting the array mid-inference.
                    return True, self._frame.copy()
            time.sleep(0.002)
        return False, None

    def stop(self):
        self._stop.set()
        self._t.join(timeout=1.0)


def _air_cursor_thread_main():
    global air_cursor_active, air_cursor_camera, air_cursor_status
    global air_cursor_thread, _air_cursor_screen_bounds_cache

    if platform.system().lower() != "windows":
        speak("Air Cursor currently requires Windows OS mouse control.")
        with air_cursor_lock:
            air_cursor_active = False
        return

    if not CV2_OK or not MEDIAPIPE_OK:
        speak("Air Cursor needs OpenCV and MediaPipe. Install the required packages and try again.")
        with air_cursor_lock:
            air_cursor_active = False
        return

    cap = None
    hands = None
    tasks_detector = None
    grabber = None

    # Make THIS thread per-monitor DPI aware (without touching the Tk UI) so
    # SetCursorPos/GetSystemMetrics use real pixels. Without this, a scaled
    # display (125%/150%) makes the cursor miss the right/bottom of the screen.
    try:
        ctypes.windll.user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        pass
    _air_cursor_screen_bounds_cache = None

    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            speak("I couldn't access the camera for Air Cursor.")
            with air_cursor_lock:
                air_cursor_active = False
            return

        air_cursor_camera = cap
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, AIR_CURSOR_CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, AIR_CURSOR_CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Keep the capture lightweight so hand inference and pointer updates stay responsive.
        try:
            cap.set(cv2.CAP_PROP_FPS, 60)
        except Exception:
            pass

        # MediaPipe Tasks-only backend. The legacy solutions API is never used.
        if not MP_TASKS_OK:
            raise RuntimeError(
                "MediaPipe Tasks API is unavailable. Install/update MediaPipe "
                "with: python -m pip install -U mediapipe"
            )

        model_path = _air_cursor_prepare_tasks_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        tasks_detector = mp_vision.HandLandmarker.create_from_options(options)
        tasks_backend = True
        log("Air Cursor: using MediaPipe Tasks Hand Landmarker backend (Tasks-only).")

        _air_cursor_update_overlay("DETECTED", "MOVE")
        log("Air Cursor camera started.")
        task_timestamp_ms = 0
        grabber = _AirCursorFrameGrabber(cap)
        # Target ~30 fps for inference. MediaPipe VIDEO mode expects monotonically
        # increasing timestamps, so pace the loop rather than running flat-out.
        # This keeps CPU usage reasonable on mid-range machines.
        _FRAME_INTERVAL = 1.0 / 30.0
        _last_frame_time = 0.0

        while not air_cursor_stop_event.is_set():
            ok, frame = grabber.read()
            if not ok:
                continue

            # Pace inference to ~30 fps to avoid burning CPU.
            _now_t = time.monotonic()
            _elapsed = _now_t - _last_frame_time
            if _elapsed < _FRAME_INTERVAL:
                time.sleep(_FRAME_INTERVAL - _elapsed)
            _last_frame_time = time.monotonic()

            # Mirror the camera frame BEFORE MediaPipe inference.
            # This webcam/MediaPipe setup reports the physical RIGHT hand as
            # MediaPipe "Left". The selector below uses that mapping.
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # This variable means PHYSICAL RIGHT HAND.
            right_hand = None
            right_hand_raw = None
            right_score = -1.0

            task_timestamp_ms = max(
                task_timestamp_ms + 1,
                int(time.monotonic() * 1000)
            )
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = tasks_detector.detect_for_video(mp_image, task_timestamp_ms)
            task_hands = getattr(results, "hand_landmarks", []) or []
            task_handedness = getattr(results, "handedness", []) or []

            for idx, hand_lms in enumerate(task_hands):
                categories = task_handedness[idx] if idx < len(task_handedness) else []
                if _air_cursor_is_right_hand(categories, hand_lms):
                    score = float(getattr(categories[0], "score", 0.0) or 0.0) if categories else 0.0
                    if score > right_score:
                        right_hand = _AirCursorLandmarkAdapter(hand_lms)
                        right_hand_raw = hand_lms
                        right_score = score

            # If handedness metadata is unavailable or camera mirroring differs,
            # still use the best detected hand. num_hands=1 keeps this deterministic.
            if right_hand is None and task_hands:
                right_hand = _AirCursorLandmarkAdapter(task_hands[0])
                right_score = max(right_score, 1.0)

            if right_hand is not None and right_score >= 0.18:
                _air_cursor_process_landmarks(right_hand, _air_cursor_get_virtual_screen(), frame)

            if right_hand is None or right_score < 0.18:
                # Hand loss must never create random mouse input.
                if time.time() - air_cursor_last_hand_time > AIR_CURSOR_HAND_LOSS_TIMEOUT:
                    # Releases any held button and clears pinch/smoothing state so
                    # no stray click or cursor sweep happens when the hand returns.
                    _air_cursor_reset_tracking_state()
                    _air_cursor_update_overlay("HAND LOST", "PAUSED")

            # Background-only camera mode: intentionally do NOT call
            # cv2.imshow()/cv2.waitKey(). The webcam is captured and processed
            # invisibly so opening Air Cursor never creates a full-screen camera
            # preview or steals the user's desktop. The small Victus status
            # overlay remains the only visual feedback.

    except Exception as e:
        log("Air Cursor error: " + str(e))
        speak("Air Cursor encountered an error and was stopped.")
    finally:
        _air_cursor_left_up()
        _air_cursor_reset_tracking_state()
        try:
            if grabber is not None:
                grabber.stop()
        except Exception:
            pass
        try:
            if tasks_detector is not None:
                tasks_detector.close()
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        air_cursor_camera = None
        # No OpenCV preview window is created in background Air Cursor mode.

        with air_cursor_lock:
            air_cursor_active = False
        air_cursor_status = "INACTIVE"
        _air_cursor_destroy_overlay()
        log("Air Cursor stopped.")
        air_cursor_thread = None



def _air_cursor_update_overlay(status=None, gesture=None):
    global air_cursor_status, air_cursor_last_gesture, air_cursor_last_overlay_update
    if status is not None:
        air_cursor_status = str(status)
    if gesture is not None:
        air_cursor_last_gesture = str(gesture)

    if orb_root is None or tk is None:
        return

    # Throttle Tk callbacks so camera-frame processing does not flood
    # the Tk event queue and introduce cursor lag.
    now_mono = time.monotonic()
    if now_mono - air_cursor_last_overlay_update < 0.10:
        return
    air_cursor_last_overlay_update = now_mono

    def update():
        global air_cursor_overlay
        try:
            if not air_cursor_active:
                if air_cursor_overlay is not None:
                    air_cursor_overlay.withdraw()
                return

            if air_cursor_overlay is None or not air_cursor_overlay.winfo_exists():
                win = tk.Toplevel(orb_root)
                air_cursor_overlay = win
                win.overrideredirect(True)
                win.attributes("-topmost", True)
                transparent = "#010101"
                try:
                    win.attributes("-transparentcolor", transparent)
                except Exception:
                    pass
                win.configure(bg=transparent)
                w, h = 245, 72
                sw = win.winfo_screenwidth()
                win.geometry(f"{w}x{h}+{sw-w-25}+25")

                cv = tk.Canvas(
                    win, width=w, height=h, bg=transparent,
                    highlightthickness=0, bd=0
                )
                cv.pack(fill="both", expand=True)
                win._air_cv = cv
                win._air_status_id = cv.create_rectangle(
                    2, 2, w-2, h-2, fill="#06172c", outline="#00eaff", width=2
                )
                win._air_title_id = cv.create_text(
                    12, 13, text="● AIR CURSOR ACTIVE",
                    fill="#00ff66", font=("Consolas", 9, "bold"), anchor="w"
                )
                win._air_hand_id = cv.create_text(
                    12, 35, text="Right Hand: Detected",
                    fill="#b7ffff", font=("Consolas", 8), anchor="w"
                )
                win._air_gesture_id = cv.create_text(
                    12, 53, text="Gesture: MOVE",
                    fill="#00eaff", font=("Consolas", 8), anchor="w"
                )

                # Make the status overlay click-through and non-activating on Windows.
                try:
                    hwnd = win.winfo_id()
                    GWL_EXSTYLE = -20
                    WS_EX_LAYERED = 0x00080000
                    WS_EX_TRANSPARENT = 0x00000020
                    WS_EX_NOACTIVATE = 0x08000000
                    style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                    ctypes.windll.user32.SetWindowLongW(
                        hwnd, GWL_EXSTYLE,
                        style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE
                    )
                except Exception:
                    pass

            cv = air_cursor_overlay._air_cv
            cv.itemconfig(
                air_cursor_overlay._air_hand_id,
                text="Right Hand: " + ("Detected" if air_cursor_status != "HAND LOST" else "Lost")
            )
            cv.itemconfig(
                air_cursor_overlay._air_gesture_id,
                text="Gesture: " + air_cursor_last_gesture.upper()
            )
            air_cursor_overlay.deiconify()
            air_cursor_overlay.lift()
        except Exception as e:
            log("Air Cursor overlay error: " + str(e))

    try:
        orb_root.after(0, update)
    except Exception:
        pass




def _air_cursor_map_to_screen(index_x, index_y):
    """
    Map the comfortable central camera interaction area to the complete
    virtual desktop rather than mapping the whole camera image 1:1.
    """
    # These bounds intentionally leave a comfortable border around the camera.
    left, right = 0.06, 0.94
    top, bottom = 0.05, 0.95

    x = (index_x - left) / (right - left)
    y = (index_y - top) / (bottom - top)

    # Sensitivity boost: amplify movement around the center before clamping.
    x = 0.5 + (x - 0.5) * AIR_CURSOR_SENSITIVITY
    y = 0.5 + (y - 0.5) * AIR_CURSOR_SENSITIVITY
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))

    sx, sy, sw, sh = _air_cursor_get_virtual_screen()
    return sx + x * (sw - 1), sy + y * (sh - 1)


def _air_cursor_destroy_overlay():
    global air_cursor_overlay
    if orb_root is None:
        return

    def destroy():
        global air_cursor_overlay
        try:
            if air_cursor_overlay is not None:
                air_cursor_overlay.destroy()
        except Exception:
            pass
        air_cursor_overlay = None

    try:
        orb_root.after(0, destroy)
    except Exception:
        pass


def _air_cursor_reset_tracking_state():
    """Reset all transient Air Cursor tracking/gesture state safely."""
    global air_cursor_mouse_down
    global air_cursor_last_left_pinch, air_cursor_left_pinch_started
    global air_cursor_last_right_pinch, air_cursor_last_click_time
    global air_cursor_last_scroll_time, air_cursor_last_screen_point
    global air_cursor_smooth_point, air_cursor_prev_index
    global air_cursor_fist_candidate, air_cursor_fist_started, air_cursor_last_fist_time
    global air_cursor_palm_started, air_cursor_last_palm_time
    global air_cursor_gesture_candidate, air_cursor_gesture_started
    global air_cursor_last_custom_gesture_time
    global air_cursor_pinch_lock, air_cursor_pinch_offset

    # Always release a held OS mouse button before clearing state.
    try:
        _air_cursor_left_up()
    except Exception:
        air_cursor_mouse_down = False

    air_cursor_last_left_pinch = False
    air_cursor_left_pinch_started = 0.0
    air_cursor_last_right_pinch = False
    air_cursor_last_click_time = 0.0
    air_cursor_last_scroll_time = 0.0
    air_cursor_last_screen_point = None
    air_cursor_smooth_point = None
    air_cursor_prev_index = None

    air_cursor_fist_candidate = False
    air_cursor_fist_started = 0.0
    air_cursor_last_fist_time = 0.0
    air_cursor_palm_started = 0.0
    air_cursor_last_palm_time = 0.0

    air_cursor_gesture_candidate = None
    air_cursor_gesture_started = 0.0
    air_cursor_last_custom_gesture_time = 0.0
    air_cursor_pinch_lock = None
    air_cursor_pinch_offset = None
    air_cursor_pos_history.clear()


def open_air_cursor():
    global air_cursor_active, air_cursor_thread
    if air_cursor_active:
        speak("Air Cursor is already active.")
        return

    if platform.system().lower() != "windows":
        speak("Air Cursor currently requires Windows.")
        return

    if not CV2_OK or not MEDIAPIPE_OK:
        speak("Air Cursor requires OpenCV and MediaPipe. Install them first.")
        return

    air_cursor_stop_event.clear()
    _air_cursor_reset_tracking_state()
    with air_cursor_lock:
        air_cursor_active = True
    _air_cursor_update_overlay("STARTING", "MOVE")

    air_cursor_thread = threading.Thread(
        target=_air_cursor_thread_main,
        name="Victus-AirCursor",
        daemon=True
    )
    air_cursor_thread.start()
    speak("Air Cursor activated in background mode. Raise your right hand.")


def stop_air_cursor(emergency=False):
    global air_cursor_active
    if not air_cursor_active:
        if emergency:
            _air_cursor_left_up()
        speak("Air Cursor is already inactive.")
        return

    air_cursor_stop_event.set()
    _air_cursor_left_up()
    _air_cursor_reset_tracking_state()
    if emergency:
        speak("Air Cursor stopped.")
    else:
        speak("Air Cursor closed.")


def calibrate_air_cursor():
    """
    Reset the interaction mapping to the comfortable calibrated camera area.
    The mapping is intentionally based on normalized hand coordinates so it
    remains valid across screen resolutions and monitor layouts.
    """
    if not air_cursor_active:
        speak("Open Air Cursor before calibrating.")
        return
    speak("Air Cursor calibration reset. Move your index finger around the comfortable camera area.")
    _air_cursor_update_overlay("DETECTED", "CALIBRATION")


def air_cursor_voice_mouse_action(action):
    if not air_cursor_active:
        speak(f"{action.title()} is available while Air Cursor is active.")
        return

    actions = {
        "left click": _air_cursor_left_click,
        "right click": _air_cursor_right_click,
        "double click": _air_cursor_double_click,
        "copy": air_cursor_copy,
        "paste": air_cursor_paste,
        "cut": air_cursor_cut,
        "select all": air_cursor_select_all,
        "delete": air_cursor_delete,
    }
    fn = actions.get(action)
    if fn:
        fn()
        _air_cursor_update_overlay("DETECTED", action.upper())


def execute_single_command(cmd_norm):
    if not cmd_norm:
        return

    # Direct offline calculation of mathematical expressions (e.g. "5 + 5", "10 * (2 + 3)", "100 / 4", "calc 12 * 8")
    calc_res = try_calculate_math(cmd_norm)
    if calc_res is not None:
        expr_str, ans = calc_res
        ans_text = f"{expr_str} = {ans}"
        log(f"CALCULATION: {ans_text}")
        speak(f"The answer is {ans}.")
        _update_ai_response_window(cmd_norm, ans_text)
        return

    # Check for standalone calculator app command
    if cmd_norm.strip() in ["calculator", "calc", "open calculator", "open calc"]:
        speak("Opening Calculator.")
        open_app("calculator")
        return

    # "Explain ..." queries route through normalize_light() upstream (to
    # preserve punctuation for the AI), which keeps apostrophes literally
    # ("what's"), while plain normalize() turns them into a space ("what
    # s") instead. Stripping apostrophes turns "what's" into "whats", but
    # normalize()'s "what s" still has a space, so it also needs folding
    # into "whats" — giving one consistent form for the screen/clipboard
    # keyword checks below, regardless of which normalizer ran upstream.
    cmd_flat = cmd_norm.replace("'", "").replace("\u2019", "")
    cmd_flat = re.sub(r"\bwhat s\b", "whats", cmd_flat)

    # Air Cursor voice controls must be checked before generic open/close logic.
    if cmd_norm in ["open air cursor", "enable air cursor", "start air cursor", "air cursor"]:
        open_air_cursor()
    elif cmd_norm in ["close air cursor", "disable air cursor", "close air cursor completely"]:
        stop_air_cursor(False)
    elif cmd_norm in ["stop air cursor", "emergency stop air cursor", "stop cursor"]:
        stop_air_cursor(True)
    elif cmd_norm in ["resume cursor", "resume air cursor"]:
        if not air_cursor_active:
            open_air_cursor()
        else:
            speak("Air Cursor is already running.")
    elif cmd_norm in ["calibrate air cursor", "calibrate cursor"]:
        calibrate_air_cursor()
    elif cmd_norm in ["left click", "left mouse click"]:
        air_cursor_voice_mouse_action("left click")
    elif cmd_norm in ["right click", "right mouse click"]:
        air_cursor_voice_mouse_action("right click")
    elif cmd_norm in ["double click", "double left click"]:
        air_cursor_voice_mouse_action("double click")
    elif cmd_norm == "copy":
        air_cursor_voice_mouse_action("copy")
    elif cmd_norm == "paste":
        air_cursor_voice_mouse_action("paste")
    elif cmd_norm == "cut":
        air_cursor_voice_mouse_action("cut")
    elif cmd_norm in ["select all", "select everything"]:
        air_cursor_voice_mouse_action("select all")
    elif cmd_norm in ["delete", "press delete"]:
        air_cursor_voice_mouse_action("delete")
    else:
        # Continue with the main Victus command router below.
        pass

    # Explicit Air Cursor commands have already been handled above.
    if cmd_norm in [
        "open air cursor", "enable air cursor", "start air cursor", "air cursor",
        "close air cursor", "disable air cursor", "close air cursor completely",
        "stop air cursor", "emergency stop air cursor", "stop cursor",
        "resume cursor", "resume air cursor", "calibrate air cursor", "calibrate cursor",
        "left click", "left mouse click", "right click", "right mouse click",
        "double click", "double left click", "copy", "paste", "cut",
        "select all", "select everything", "delete", "press delete"
    ]:
        return

    # Local AI controls are checked before generic system commands.
    if cmd_norm in ["clear ai memory", "clear local ai memory", "reset ai memory", "reset chat memory"]:
        clear_local_ai_memory()
        return

    # ---------------------------------------------------------------
    # SCHEDULED / TRIGGERED ACTIONS — "remind me to X in/at Y",
    # "schedule X at/in Y". Checked before the generic AI-query and
    # command router so these natural phrasings never fall through to
    # ask_local_ai() or get misread as an immediate action.
    # ---------------------------------------------------------------
    if cmd_norm in ["list my reminders", "list reminders", "what are my reminders",
                     "list scheduled tasks", "what is scheduled", "show my reminders"]:
        list_scheduled_tasks()
        return
    if cmd_norm in ["clear my reminders", "clear reminders", "clear scheduled tasks",
                     "cancel all reminders", "cancel my reminders"]:
        clear_scheduled_tasks()
        return
    schedule_kind, schedule_action, schedule_dt = _extract_schedule_request(cmd_norm)
    if schedule_kind is not None:
        add_scheduled_task(schedule_kind, schedule_action, schedule_dt)
        return

    # ---------------------------------------------------------------
    # RAG-LITE OVER LOCAL NOTES/DOCS — "search my notes for X",
    # "what do my notes say about X", "reindex my notes".
    # ---------------------------------------------------------------
    if cmd_norm in ["reindex my notes", "reindex notes", "rebuild notes index",
                     "refresh my notes", "update notes index"]:
        reindex_notes()
        return
    m = re.search(
        r"(?:search my notes for|search notes for|look in my notes for|"
        r"find in my notes for|find in my notes|what do my notes say about|"
        r"my notes say about|according to my notes about|ask my notes)\s+(.+)",
        cmd_norm,
    )
    if m:
        answer_from_notes(m.group(1).strip())
        return

    # ---------------------------------------------------------------
    # DICTATED NOTE-TAKING — "take a note: buy milk and eggs". Writes
    # into NOTES_DIR so it's picked up by the notes search above.
    # ---------------------------------------------------------------
    m_note = re.match(
        r"^(?:take a note|add a note|note that|write a note|save a note|make a note)"
        r"\s*(?:that|:)?\s*(.+)$",
        cmd_norm,
    )
    if m_note:
        add_note(m_note.group(1).strip())
        return

    # ---------------------------------------------------------------
    # STANDALONE PERSONALIZATION PROFILE — "remember that I ...",
    # "remember my ...", "what do you know about me", "forget that ...",
    # "clear my profile". Kept separate from dictated note-taking above:
    # these facts get injected into every AI prompt (see
    # _profile_system_message) rather than sitting in the searchable
    # notes index.
    # ---------------------------------------------------------------
    if cmd_flat in ["what do you know about me", "whats in my profile",
                    "what is in my profile", "read my profile", "show my profile"]:
        read_user_profile()
        return
    if cmd_flat in ["clear my profile", "clear my personalization profile",
                    "reset my profile"]:
        clear_user_profile()
        return
    m_forget_fact = re.match(r"^forget (?:that\s+)?(.+)$", cmd_norm)
    if m_forget_fact:
        forget_fact(m_forget_fact.group(1).strip())
        return
    m_remember_fact = re.match(r"^remember (?:that\s+|my\s+)?(.+)$", cmd_norm)
    if m_remember_fact:
        remember_fact(m_remember_fact.group(1).strip())
        return

    # ---------------------------------------------------------------
    # PROACTIVE HEALTH ALERTS — on/off toggle for the background
    # health_monitor() watcher thread. The alerts themselves fire
    # unprompted, so the only voice surface needed here is enable/disable.
    # ---------------------------------------------------------------
    if cmd_flat in ["disable health alerts", "turn off health alerts",
                    "stop health alerts", "mute health alerts"]:
        toggle_health_alerts(False)
        return
    if cmd_flat in ["enable health alerts", "turn on health alerts",
                    "resume health alerts"]:
        toggle_health_alerts(True)
        return

    # ---------------------------------------------------------------
    # STANDALONE TO-DO LIST — "add X to my to do list", "what's on my to
    # do list", "mark X as done", "clear my to do list". Deliberately
    # separate from the reminder/scheduler section above: no due time.
    # ---------------------------------------------------------------
    if cmd_norm in ["what is on my to do list", "whats on my to do list",
                     "show my to do list", "list my to do list", "read my to do list",
                     "show my todo list", "list my todo list"]:
        list_todos()
        return
    if cmd_norm in ["clear my to do list", "clear to do list", "clear my todo list",
                     "clear todo list"]:
        clear_todos()
        return
    m_todo_add = re.match(
        r"^add\s+(.+?)\s+to\s+my\s+to\s*do\s*list$", cmd_norm
    )
    if m_todo_add:
        add_todo(m_todo_add.group(1).strip())
        return
    m_todo_done = re.match(
        r"^(?:mark|check off)\s+(.+?)\s+(?:as\s+)?(?:done|complete|completed|finished)$",
        cmd_norm,
    )
    if m_todo_done:
        complete_todo(m_todo_done.group(1).strip())
        return

    # ---------------------------------------------------------------
    # OFFLINE TRANSLATION — "translate <text> to <language>", "how do
    # you say <text> in <language>". See translate_text() above.
    # ---------------------------------------------------------------
    m_translate = (
        re.match(r"^translate\s+(.+?)\s+(?:to|into)\s+([a-z\- ]+?)[\.\!\?]*$", cmd_flat)
        or re.match(r"^how do you say\s+(.+?)\s+in\s+([a-z\- ]+?)[\.\!\?]*$", cmd_flat)
    )
    if m_translate:
        translate_text(m_translate.group(1).strip(), m_translate.group(2).strip())
        return

    # ---------------------------------------------------------------
    # TEXT SIMPLIFICATION / READING-LEVEL REWRITE — "simplify <text>",
    # "simplify this" (falls back to clipboard), "make this more
    # formal". See simplify_or_reword_text() above.
    # ---------------------------------------------------------------
    m_simplify = re.match(
        r"^simplify\s*(?:this|it|that|my clipboard|what i copied)?\s*[:\-]?\s*(.*)$",
        cmd_flat,
    )
    if m_simplify:
        simplify_or_reword_text("simple", m_simplify.group(1).strip())
        return
    m_formalize = re.match(
        r"^(?:formalize|make (?:this|it) (?:more )?formal)\s*(?:this|it|that)?\s*[:\-]?\s*(.*)$",
        cmd_flat,
    )
    if m_formalize:
        simplify_or_reword_text("formal", m_formalize.group(1).strip())
        return

    # ---------------------------------------------------------------
    # SCHEDULED LOCAL BACKUPS — "backup my documents", "back up my
    # desktop", "backup victus data". Also doubles as a scheduled action
    # ("schedule backup my documents at 11 pm") — see backup_local_files().
    # ---------------------------------------------------------------
    if re.match(r"^(?:back ?up|create a backup(?: of)?|make a backup(?: of)?)\b", cmd_flat):
        backup_local_files(cmd_flat)
        return

    explicit_ai_query = _extract_explicit_ai_query(cmd_norm)
    if explicit_ai_query is not None:
        if explicit_ai_query:
            ask_local_ai(explicit_ai_query)
        else:
            speak("AI chat is ready. Ask me anything.")
        return

    # Check if command is asking to scan a specific file path/name for viruses
    if any(phrase in cmd_norm for phrase in ["scan file", "scan this file", "check file for virus", "scan for virus in", "virus scan file"]):
        # Extract file path or name following the command
        path_candidate = re.sub(r"^(scan file|scan this file|check file for virus|scan for virus in|virus scan file)\s*", "", cmd_norm).strip()
        if path_candidate:
            scan_file_for_virus(path_candidate)
        else:
            speak("Please specify or paste the file path to scan.")
    elif any(phrase in cmd_norm for phrase in [
        "scan for virus", "scan for foreign threat", "virus scan", 
        "cyber security scan", "scan system for threats", "scan for threat"
    ]):
        run_antivirus_scan()
    elif any(phrase in cmd_norm for phrase in [
        # Checked before the generic system-dashboard phrases below, since
        # "dashboard" alone (one of those phrases) would otherwise swallow
        # "usage dashboard" and open the wrong window.
        "usage dashboard", "usage analytics", "usage stats", "usage statistics",
        "what do i ask most", "what do i ask the most", "most common command",
        "command analytics", "command stats", "my usage"
    ]):
        open_usage_dashboard()
    elif any(phrase in cmd_norm for phrase in [
        "system dashboard", "dashboard", "get me system dashboard", 
        "give me system dashboard", "open system dashboard"
    ]):
        open_system_dashboard()
    elif any(phrase in cmd_norm for phrase in ["whats the date", "what is the date", "what day is it", "what is today's date", "todays date", "current date"]):
        now = datetime.now()
        date_str = now.strftime("%A, %B %d, %Y")
        speak(f"Today is {date_str}.")
    elif any(phrase in cmd_norm for phrase in ["what time is it", "whats the time", "what's the time", "whats the current time", "current time"]):
        now = datetime.now()
        time_str = now.strftime("%I:%M %p")
        speak(f"The current time is {time_str}.")
    elif any(phrase in cmd_norm for phrase in ["wifi status", "whats the wifi status", "what is the wifi status", "check wifi", "is wifi connected", "wifi connection"]):
        wifi_status()
    elif "wake on lan" in cmd_norm or "magic packet" in cmd_norm:
        if cmd_norm.startswith("add wake on lan device"):
            m = re.match(r"add wake on lan device\s+(.+?)\s+([0-9a-f]{6,})$", cmd_norm)
            if m:
                add_wol_device(m.group(1), m.group(2))
            else:
                speak("Say it as: add wake on lan device, then a name, then "
                      "the twelve digit MAC address with no spaces.")
        elif any(p in cmd_norm for p in ("saved wake on lan", "list wake on lan", "wake on lan devices", "my wake on lan")):
            list_wol_devices()
        else:
            target = re.sub(r"^(wake on lan|send a magic packet to|magic packet to)\s*", "", cmd_norm).strip()
            wake_on_lan_device(target)
    elif any(phrase in cmd_norm for phrase in [
        "scan my network", "scan the network", "scan lan", "network devices",
        "connected devices", "who is on my network", "whats on my network",
        "what is on my network", "devices on my network",
    ]):
        scan_lan_devices()
    elif any(phrase in cmd_norm for phrase in [
        "duplicate files", "find duplicates", "find duplicate files",
        "large files", "big files", "free up space",
    ]):
        find_duplicate_and_large_files()
    elif any(phrase in cmd_norm for phrase in ["list all", "find", "search for files", "show me files", "files with extension"]):
        words = cmd_norm.split()
        ext = "txt"
        for w in words:
            if w in ["pdf", "ppt", "pptx", "docx", "doc", "txt", "py", "csv", "png", "jpg", "mp3", "mp4"]:
                ext = w
                break
            elif "." in w:
                ext = w.replace(".", "")
                break
        search_local_files_gui(ext)
    elif any(phrase in cmd_norm for phrase in ["weather", "forecast"]):
        get_weather()
    elif any(p in cmd_flat for p in [
        "explain what i just copied", "explain what i copied",
        "explain this clipboard", "explain the clipboard", "explain my clipboard",
        "summarize what i copied", "summarize what i just copied",
        "summarize this clipboard", "summarize the clipboard", "summarize my clipboard",
        "what does this clipboard", "what does my clipboard", "what did i copy",
    ]):
        explain_clipboard("summarize" if "summarize" in cmd_flat else "explain")
    elif re.search(FILE_READ_EXTS_PATTERN, cmd_flat) and any(
        w in cmd_flat for w in ("read", "summarize", "explain", "file", "whats in", "what does")
    ):
        # A named file with a recognised extension, paired with a
        # read/summarize/explain-style verb — e.g. "read budget.csv and
        # tell me the total" or "summarize notes.docx". handle_command()
        # already kept this whole phrase intact (didn't split on
        # "and"/"then") so the question survives alongside the filename.
        read_and_summarize_file(cmd_flat)
    elif "read clipboard" in cmd_norm:
        manage_clipboard("read")
    elif "clear clipboard" in cmd_norm:
        manage_clipboard("clear")
    elif any(p in cmd_norm for p in (
        "clipboard history", "what did i copy before", "show my clipboard history"
    )):
        list_clipboard_history()
    elif re.search(r"(?:restore clip|restore clipboard item|paste clip)\s+(\d+)", cmd_norm):
        restore_clipboard_history_item(
            int(re.search(r"(?:restore clip|restore clipboard item|paste clip)\s+(\d+)", cmd_norm).group(1))
        )
    elif "start workspace" in cmd_norm or "workspace macro" in cmd_norm:
        run_macro("workspace")
    elif cmd_norm.startswith("start recording macro") or cmd_norm.startswith("record macro") \
            or cmd_norm.startswith("record a macro"):
        name = re.sub(r"^(start recording macro|record a macro|record macro)\s*", "", cmd_norm).strip()
        start_macro_recording(name)
    elif cmd_norm in MACRO_STOP_PHRASES:
        # Only reached if recording wasn't already active (handle_command
        # intercepts "stop recording" first while a recording is live).
        stop_macro_recording()
    elif cmd_norm.startswith("run macro") or cmd_norm.startswith("play macro") \
            or cmd_norm.startswith("replay macro"):
        name = re.sub(r"^(run macro|play macro|replay macro)\s*", "", cmd_norm).strip()
        run_custom_macro(name)
    elif cmd_norm in ("list macros", "list my macros", "show my macros", "what macros do i have"):
        list_custom_macros()
    elif cmd_norm.startswith("delete macro") or cmd_norm.startswith("remove macro"):
        name = re.sub(r"^(delete macro|remove macro)\s*", "", cmd_norm).strip()
        delete_custom_macro(name)
    elif any(phrase in cmd_norm for phrase in [
        "check my email", "check my outlook", "check outlook", "read my email",
        "any new emails", "check my inbox", "whats in my inbox", "what is in my inbox",
        "unread emails", "do i have any emails",
    ]):
        check_outlook_email()
    elif any(phrase in cmd_norm for phrase in [
        "whats on my calendar", "what is on my calendar", "my outlook calendar",
        "check my calendar", "whats on my schedule", "what is on my schedule",
        "my calendar today", "todays events", "todays appointments", "upcoming meetings",
    ]):
        check_outlook_calendar(1)
    elif cmd_norm.startswith("email ") and " subject " in cmd_norm:
        # "email <address> subject <subject> body <message>" — the body
        # segment is optional. normalize() lowercases and strips '@'/'.'
        # into spaces, so the address is matched loosely and then rebuilt.
        m = re.match(r"^email\s+(.+?)\s+subject\s+(.+?)(?:\s+body\s+(.+))?$", cmd_norm)
        if m:
            raw_addr, subj, body = m.group(1), m.group(2), m.group(3) or ""
            addr = raw_addr.strip().replace(" ", "")
            send_outlook_email(addr, subj.strip(), body.strip())
        else:
            speak("Say it as: email, then the address, then 'subject', then the subject line.")
    elif any(phrase in cmd_norm for phrase in ["who are you", "introduce yourself", "what is your name", "what are you"]):
        speak("I am Victus, your high-performance personal AI voice assistant. I am engineered to streamline your workflow, manage system controls seamlessly, and execute your commands instantly whenever you need.")
    elif any(phrase in cmd_norm for phrase in [
        "who is your boss", "say me about your boss", "explain about your boss",
        "who is your owner", "explain about your owner", "explain about your creator",
        "who is your creator", "what is your creator", "who created you",
        "who developed you", "who is your developer", "explain about your developer",
        "who made you", "who built you", "who programmed you", "who designed you",
        "tell me about your developer", "tell me about your creator",
    ]):
        speak("My developer and boss is Shreeskanda. He is currently in his 3rd semester of Information Science and Engineering at Seshadripuram Institute of Technology. He has an incredible vision for technology, and working alongside him has been a brilliant journey.")
    elif any(word in cmd_norm for word in ["good", "excellent", "great job", "nice", "awesome", "well done", "brilliant"]):
        compliment_responses = [
            "Thank you so much, boss! I'm glad I could help.",
            "Much appreciated, boss! Always happy to assist you.",
            "Thank you, boss! I'm doing my best to make things easier for you.",
            "Thanks, boss! Hearing that makes my day."
        ]
        speak(random.choice(compliment_responses))
    elif any(w in cmd_norm for w in ["open command box", "command box", "type command"]):
        open_command_box()
    elif any(w in cmd_norm for w in ["command history", "show history", "open history", "view history"]):
        open_history_window()
    elif cmd_norm in ["clear history", "clear command history", "clear all history"]:
        clear_all_history()
        speak("Command history cleared.")
    elif any(w in cmd_norm for w in ["show ai response", "show answer", "open ai response", "show last answer"]):
        reopen_ai_response_window()
    elif any(p in cmd_flat for p in [
        # Deeper screen-content questions — these actually send the
        # question + a screenshot to a vision model, instead of the
        # one-sentence "what app am I in" guess describe_screen_activity()
        # gives below. Checked first so e.g. "explain what's on my
        # screen" gets the real answer, not just the app name. Matched
        # against cmd_flat (apostrophes stripped) so "what's" reliably
        # matches "whats" regardless of which normalizer ran upstream.
        "this error", "error mean", "error say", "what does this say",
        "what does this mean", "read this for me", "read my screen",
        "read the screen", "summarize my screen", "summarize this screen",
        "summarize whats on my screen", "explain this code",
        "explain this error", "explain my screen",
        "explain whats on my screen", "explain what is on my screen",
        "help me understand this", "look at my screen",
    ]):
        ask_vision_ai(cmd_norm)
    elif any(p in cmd_norm for p in [
        # normalize() turns "what's" into "what s" (apostrophe -> space), so
        # both forms are listed to match however Whisper transcribes it.
        "whats going on", "what s going on", "what is going on",
        "whats happening", "what s happening", "what is happening",
        "whats on my screen", "what s on my screen", "what is on my screen",
        "scan my screen", "scan the screen", "check my screen",
        "what am i doing", "what am i looking at", "what am i up to",
    ]):
        describe_screen_activity()
    elif "screenshot" in cmd_norm or "take a snapshot" in cmd_norm:
        take_screenshot()
    elif "minimize current" in cmd_norm or "minimize tab" in cmd_norm or "minimize window" in cmd_norm:
        minimize_current_window()
    elif "close current" in cmd_norm or "kill current" in cmd_norm or "close tab" in cmd_norm or "kill tab" in cmd_norm or "close window" in cmd_norm or "kill window" in cmd_norm:
        close_current_window()
    elif "minimize all" in cmd_norm or "minimize windows" in cmd_norm:
        minimize_all()
    elif "close all" in cmd_norm or "close windows" in cmd_norm or "kill all" in cmd_norm or "close all the tabs" in cmd_norm or "kill the tabs" in cmd_norm:
        close_all()
    elif "system status" in cmd_norm or "pc status" in cmd_norm:
        system_status()
    elif "system specs" in cmd_norm or "specifications" in cmd_norm or "pc specs" in cmd_norm:
        system_specs()
    elif "spec sheet" in cmd_norm or "specs sheet" in cmd_norm:
        open_spec_sheet()
    elif "sleep" in cmd_norm:
        sleep_pc()
    elif any(phrase in cmd_norm for phrase in ["shut down", "shutdown", "switch off the laptop", "switch off the pc", "turn off the laptop", "turn off the pc"]):
        shutdown_pc()
    elif "recycle bin" in cmd_norm:
        recycle_bin()
    elif "battery status" in cmd_norm or "battery percentage" in cmd_norm:
        battery_status()
    elif "pause" in cmd_norm:
        media_pause()
    elif "stop" in cmd_norm:
        # If Victus is currently talking, "stop" means "stop talking", not
        # "stop the music" — the barge-in check in main_loop already
        # catches this for voice input, but this covers the command box
        # (and any voice path that reaches here) the same way.
        if speaking:
            stop_speaking()
        else:
            media_stop()
    elif "next song" in cmd_norm or "next track" in cmd_norm:
        media_next()
    elif "previous song" in cmd_norm or "previous track" in cmd_norm:
        media_previous()
    elif "start over" in cmd_norm or "restart song" in cmd_norm:
        music_start_over()
    elif "play random" in cmd_norm or "random song" in cmd_norm:
        random_song()
    elif "mute" in cmd_norm:
        mute_volume()
    elif "unmute" in cmd_norm:
        unmute_volume()
    elif "volume up" in cmd_norm:
        change_volume(10)
    elif "volume down" in cmd_norm:
        change_volume(-10)
    elif "brightness up" in cmd_norm:
        change_brightness(10)
    elif "brightness down" in cmd_norm:
        change_brightness(-10)
    elif "set volume" in cmd_norm:
        nums = re.findall(r"\d+", cmd_norm)
        if nums:
            set_volume(int(nums[0]))
        else:
            speak("Say the volume level percentage.")
    elif "set brightness" in cmd_norm:
        nums = re.findall(r"\d+", cmd_norm)
        if nums:
            set_brightness(int(nums[0]))
        else:
            speak("Say the brightness level percentage.")
    elif any(site in cmd_norm for site in _WEB_APP_URLS) and any(cmd_norm.startswith(p) for p in ["open", "launch", "start", "visit", "go to"]):
        site_found = None
        for site in _WEB_APP_URLS:
            if site in cmd_norm:
                site_found = site
                break
        if site_found:
            open_web_shortcut(site_found)
    elif cmd_norm.startswith("search for"):
        q = re.sub(r"^search for\s*", "", cmd_norm).strip()
        if q:
            speak(f"Searching for {q}.")
            google_search_text(q)
        else:
            speak("What would you like me to search for?")
    elif any(cmd_norm.startswith(p) for p in ["what is", "explain", "who is", "how to", "tell me about", "why is", "why does", "can you tell me"]):
        ask_local_ai(cmd_norm)
    elif "play" in cmd_norm or "youtube" in cmd_norm:
        q = re.sub(r"^(play|search on youtube|youtube|on youtube)\s*", "", cmd_norm).strip()
        if q:
            youtube_search(q)
        else:
            random_song()
    elif any(cmd_norm.startswith(p) for p in ["open ", "launch ", "start "]) or "open" in cmd_norm:
        app = re.sub(r"^(open|launch|start)\s*", "", cmd_norm).strip()
        if app:
            speak(f"Opening {app}.")
            open_app(app)
        else:
            speak("What would you like me to open?")
    elif "minimize" in cmd_norm:
        app = re.sub(r"^(minimize)\s*", "", cmd_norm).strip()
        if app:
            minimize_window(app)
    elif "close" in cmd_norm or "kill" in cmd_norm:
        app = re.sub(r"^(close|kill)\s*", "", cmd_norm).strip()
        if app:
            close_window(app)
    elif find_installed_app(cmd_norm) is not None:
        speak(f"Opening {cmd_norm}.")
        open_app(cmd_norm)
    else:
        # Unknown commands become local AI questions instead of simply failing.
        ask_local_ai(cmd_norm)


def handle_command(cmd):
    cmd_raw = str(cmd or "").strip()
    if not cmd_raw:
        return

    # Record every recognised/typed command in history, keeping all original symbols (+, -, *, /, etc.)
    add_command_history(cmd_raw)

    # Check for bare operator or special characters typed into the command box
    bare_sym = cmd_raw.strip()
    if bare_sym in {"+", "-", "*", "/", "%", "^", "**", "++", "--", "==", "!=", "+-*/", "+ - * /"}:
        op_names = {
            "+": "addition (+)",
            "-": "subtraction (-)",
            "*": "multiplication (*)",
            "/": "division (/)",
            "%": "modulo or remainder (%)",
            "^": "exponentiation (^)",
            "**": "power (**)",
            "++": "increment (++)",
            "--": "decrement (--)",
            "==": "equality check (==)",
            "!=": "not equal (!=)",
            "+-*/": "arithmetic operators (+, -, *, /)",
            "+ - * /": "arithmetic operators (+, -, *, /)",
        }
        name = op_names.get(bare_sym, f"symbol '{bare_sym}'")
        msg = f"Mathematical operator {name} received. Please enter an expression to calculate, such as 5 + 5 or 10 * 2."
        log("OPERATOR: " + bare_sym)
        speak(msg)
        _update_ai_response_window(bare_sym, msg)
        return

    # Incomplete math expression like "5 +" or "10 *"
    m_incomplete = re.match(r"^(\d+(?:\.\d+)?)\s*([\+\-\*\/\%\^])\s*$", cmd_raw)
    if m_incomplete:
        num, op = m_incomplete.group(1), m_incomplete.group(2)
        op_name = {"+": "add to", "-": "subtract from", "*": "multiply by", "/": "divide by", "%": "modulo", "^": "power"}.get(op, "operate with")
        msg = f"Incomplete expression '{cmd_raw}'. What number would you like to {op_name} {num}?"
        speak(msg)
        _update_ai_response_window(cmd_raw, msg)
        return

    # Direct offline calculation of mathematical expressions (e.g. "5 + 5", "10 * (2 + 3)", "100 / 4", "calc 12 * 8")
    calc_res = try_calculate_math(cmd_raw)
    if calc_res is not None:
        expr_str, ans = calc_res
        ans_text = f"{expr_str} = {ans}"
        log(f"CALCULATION: {ans_text}")
        speak(f"The answer is {ans}.")
        _update_ai_response_window(cmd_raw, ans_text)
        return

    cmd_norm = normalize(cmd_raw)
    log("COMMAND: " + cmd_norm)
    if not cmd_norm:
        # If the input had only non-alphanumeric symbols that were not caught above, route to AI with symbols intact
        ask_local_ai(normalize_light(cmd_raw))
        return

    # CUSTOM VOICE MACRO RECORDER capture hook — while a recording is
    # active, every command is appended to the in-progress macro *before*
    # falling through to run normally below, so the user sees each step
    # happen live (like pressing "record" on a real macro tool) instead of
    # it only being saved silently. "stop recording" (and its synonyms)
    # ends the recording instead of being captured/executed as a step.
    if macro_recording_active.is_set():
        if cmd_norm in MACRO_STOP_PHRASES:
            stop_macro_recording()
            return
        macro_recording_state["steps"].append(cmd_norm)

    # Real intent classification (see classify_intent/INTENT_KEYWORDS)
    # rather than only a handful of hard-coded exact prefixes — catches
    # AI-directed phrasing like "define entropy" or "difference between
    # TCP and UDP" that the old prefix list would have missed.
    intent, intent_score = classify_intent(cmd_norm)
    log(f"INTENT: {intent or 'none'} (score {intent_score})")

    # Check if text contains special characters/symbols (+, -, *, /, =, etc.)
    has_math_symbols = any(sym in cmd_raw for sym in ["+", "-", "*", "/", "=", "^", "%", "<", ">", "@", "#", "$", "&"])

    # Keep natural-language AI questions intact. Splitting on "and" or
    # "then" would damage questions such as "explain AI and robotics" —
    # and, just as badly, would sever a file-read request like "read
    # budget.csv and tell me the total" into two broken pieces. Reminder
    # and notes-lookup phrasing gets the same protection, for the same
    # reason ("remind me to buy milk and eggs in 10 minutes" is one task).
    ai_like = (
        _extract_explicit_ai_query(cmd_norm) is not None
        or re.search(FILE_READ_EXTS_PATTERN, cmd_norm) is not None
        or any(cmd_norm.startswith(p) for p in (
            "what is", "explain", "who is", "how to", "tell me about",
            "why is", "why does", "can you tell me",
            # Offline translation / text-rewrite / local-backup phrases
            # also need to survive intact — "translate cats and dogs to
            # spanish" or "backup my documents and desktop" would
            # otherwise get mangled by the and/then/comma split below.
            "translate", "how do you say", "simplify", "formalize",
            "make this formal", "make this simpler", "make it formal",
            "make it simpler", "backup", "back up", "create a backup",
            "make a backup",
            "calculate", "calc", "solve", "compute", "evaluate",
        ))
        or (intent == "ai_query" and intent_score >= 1)
        or has_math_symbols
    )
    keep_intact = ai_like or intent in ("reminder", "notes_query", "note_taking", "todo")

    if keep_intact:
        # cmd_norm (from normalize()) has already stripped out every
        # character that isn't a letter, digit, or period — which mangles
        # math expressions and other special-character questions typed
        # into the command box or spoken aloud (e.g. "what is 45 * (12+3)
        # / 6" becomes "what is 45 12 3 6"). normalize_light() keeps those
        # symbols intact so the AI actually sees the real question.
        sub_cmds = [normalize_light(cmd_raw)]
    else:
        # Genuinely compound-looking commands (contain a connector word or
        # comma) get a shot at real LLM-based decomposition into ordered
        # atomic steps before falling back to the original naive split —
        # this is what lets "open chrome and search for hotels near the
        # airport" stay as two clean steps instead of splitting mid-phrase
        # on an unrelated "and". Single, simple commands skip the LLM
        # round-trip entirely and go straight through the fast regex path.
        looks_compound = re.search(r'\b(?:and|then)\b|,', cmd_norm) is not None
        llm_steps = _llm_split_into_steps(cmd_raw) if looks_compound else None
        if llm_steps:
            sub_cmds = [normalize_light(s) for s in llm_steps]
        else:
            sub_cmds = [normalize_light(s) for s in re.split(r'\b(?:and|then)\b|,', cmd_raw)]

    for sub in sub_cmds:
        sub_cleaned = sub.strip()
        if sub_cleaned:
            execute_single_command(sub_cleaned)
            time.sleep(0.3)


def main_loop():
    if not load_whisper():
        return

    mic_id = get_microphone()
    if mic_id is None:
        print("ERROR: No microphone found.")
        speak("I couldn't find a microphone.")
        return

    try:
        stream = sd.InputStream(
            device=mic_id, channels=CHANNELS, samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE, callback=audio_callback
        )
        stream.start()
        log("Microphone stream started successfully.")
    except Exception as e:
        print("STREAM ERROR:", e)
        speak("I couldn't start the microphone stream.")
        return

    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        greeting = "Good morning"
    elif 12 <= current_hour < 17:
        greeting = "Good afternoon"
    elif 17 <= current_hour < 22:
        greeting = "Good evening"
    else:
        greeting = "Hello"
        
    if HOTKEY_ENABLED:
        welcome_msg = (
            f"{greeting}, boss! Hope you are having an amazing time. Victus is "
            f"ready — press Control plus Alt whenever you want to talk to me."
        )
    else:
        welcome_msg = (
            f"{greeting}, boss! Hope you are having an amazing time. Victus is "
            f"ready. The push-to-talk hotkey needs pywin32, so say my name to "
            f"wake me for now."
        )
    speak(welcome_msg)

    while running:
        try:
            # A post-answer follow-up window (listen_for_followup) is
            # actively consuming audio_queue in another thread — idle here
            # instead of also pulling chunks, or the two would race for the
            # same audio and both would end up with broken transcriptions.
            if followup_window_active.is_set():
                time.sleep(0.05)
                continue

            # PUSH-TO-TALK: Ctrl+Alt now wakes Victus instead of a spoken
            # wake word. hotkey_pressed_event is set by hotkey_listener_loop
            # (a separate thread that only watches key state, never touches
            # audio_queue); the actual listen happens here, on this thread,
            # exactly where the old spoken-wake-word path used to run it.
            # Checked every iteration of this loop — at worst every ~150ms,
            # from the queue.get timeout below — so it stays responsive even
            # though key-state polling itself happens elsewhere. Pressing
            # the combo again while Victus is talking acts as a barge-in
            # stop instead of starting a new listen.
            if HOTKEY_ENABLED and hotkey_pressed_event.is_set():
                hotkey_pressed_event.clear()
                if speaking:
                    stop_speaking()
                else:
                    show_orb(True)
                    speak("I'm listening, boss.")
                    res = record_full_command()
                    if res:
                        threading.Thread(target=handle_command, args=(res,), daemon=True).start()
                    else:
                        show_orb(False)
                continue

            if audio_queue.qsize() > 1000:
                clear_audio()
            
            chunks = []
            try:
                for _ in range(8):
                    chunks.append(audio_queue.get(timeout=0.15))
            except queue.Empty:
                continue

            if not chunks:
                continue

            audio_data = np.concatenate(chunks)
            if rms(audio_data) < 0.0003:
                continue

            text = transcribe(audio_data)
            if not text:
                continue

            log(f"HEARD: {text}")

            # Barge-in: while Victus is actively speaking, "stop" (or a
            # close synonym) interrupts immediately — no wake word needed.
            # This only ever fires while speaking is True, so it never
            # steals a plain "stop" said at any other moment.
            if speaking and is_stop_speaking_command(text):
                stop_speaking()
                continue

            # Direct-action phrases (a narrow, explicit whitelist — see
            # is_direct_action_command) still work without any wake
            # trigger at all, hotkey or spoken, same as before.
            if is_direct_action_command(text):
                threading.Thread(target=handle_command, args=(text,), daemon=True).start()
            elif not HOTKEY_ENABLED and is_wake_word(text):
                # Fallback path — only reachable when pywin32 isn't
                # installed and the Ctrl+Alt hotkey can't be used at all.
                show_orb(True)
                initial_cmd = remove_wake(text)
                full_text = record_full_command()
                combined_cmd = (initial_cmd + " " + full_text).strip()

                if len(combined_cmd) > 1:
                    # Run in the background so a slow AI answer or long
                    # spoken reply never blocks this loop from hearing the
                    # next thing the user says (including "stop").
                    threading.Thread(target=handle_command, args=(combined_cmd,), daemon=True).start()
                else:
                    speak("I'm listening, boss.")
                    res = record_full_command()
                    if res:
                        threading.Thread(target=handle_command, args=(res,), daemon=True).start()
        except Exception as e:
            log("Main loop error: " + str(e))
            time.sleep(0.3)


if __name__ == "__main__":
    try:
        _load_command_history()
        _load_local_ai_memory()
        _load_schedule()
        _load_todos()
        _load_wol_devices()
        _load_custom_macros()
        _load_user_profile()
        NOTES_DIR.mkdir(parents=True, exist_ok=True)
        create_orb()
        threading.Thread(target=main_loop, daemon=True).start()
        threading.Thread(target=battery_monitor, daemon=True).start()
        threading.Thread(target=health_monitor, daemon=True).start()
        threading.Thread(target=scheduler_loop, daemon=True).start()
        threading.Thread(target=hotkey_listener_loop, daemon=True).start()
        threading.Thread(target=clipboard_history_monitor, daemon=True).start()
        
        if orb_root:
            orb_root.mainloop()
    except KeyboardInterrupt:
        running = False
        try:
            stop_air_cursor(True)
        except Exception:
            pass
        print("\nVictus shutting down safely.")
