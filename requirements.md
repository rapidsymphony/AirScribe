System Context and Role:
You are an Expert AI Systems Architect and Senior Python Engineer. Your objective is to write a complete, production-ready Python application that acts as a fully local, privacy-first alternative to commercial dictation software like "Wispr Flow" or "Superwhisper". This project will be published as an open-source repository on GitHub, so all code, documentation, and files must be production-grade, highly modular, and clear.

Application Objective:
Create a background desktop application that allows the user to hold down a global hotkey, speak into their microphone, and upon releasing the hotkey, have their speech accurately transcribed, semantically cleaned by a local LLM, and automatically pasted into their currently active window. The system must operate entirely offline.

Technical Stack Requirements:
- Language: Python 3.11+
- Speech-to-Text: faster-whisper (for low-latency, quantized local transcription).
- LLM Cleanup Pass: ollama (via local API requests to http://localhost:11434/api/generate).
- Audio Capture: sounddevice and numpy.
- Hotkey Management: pynput (for cross-platform global key hooking).
- OS Automation: pyautogui and pyperclip (for clipboard management and simulated pasting).
- Window Context: Cross-platform approach (handling Windows/macOS/Linux) to detect the active window.

Architecture & Multithreading Constraints:
This application must be highly responsive. The global hotkey listener lifecycle must be managed cleanly on the main thread. Audio processing, ASR inference, and LLM network calls MUST run in a separate worker thread via a thread-safe queue.Queue. The application must not freeze while the AI models are executing.

Deliverable Requirements:
Please provide the complete code and documentation, separated into well-structured, modular files. Provide a repository-ready codebase containing requirements.txt, .gitignore, a professional README.md, and a main.py that orchestrates the following discrete modules:

Module 1: Configuration (config.py)
Define variables for:
- HOTKEY: The key combination (e.g., 'alt' or a pynput Key object) used for push-to-talk.
- WHISPER_MODEL: Default to "base.en" or "small.en" with compute_type="int8".
- OLLAMA_MODEL: Default to "llama3" or "phi3".
- OLLAMA_URL: http://localhost:11434/api/generate.
- SAMPLE_RATE: 16000.

Module 2: Audio Capture (audio_capture.py)
Implement a class that handles real-time audio recording via sounddevice.
- When the push-to-talk key is held, begin capturing a 16kHz mono float32 audio stream into a dynamic buffer.
- When the key is released, stop the stream, convert the buffer to a flat NumPy array, and place it into the thread-safe queue.
- Ensure the stream is opened and closed efficiently to prevent OS resource leaks.

Module 3: Active Window Context (window_context.py)
Implement a lightweight, cross-platform function to retrieve the title of the currently focused window.
- Handle OS differences conditionally (e.g., pygetwindow for Windows, a brief osascript subprocess call for macOS, xdotool or fallback for Linux).
- Use graceful try/except fallbacks returning an empty string if the OS restricts access or tools are missing.
- This string will be passed down the pipeline to inform the LLM of the user's current environmental context.

Module 4: The AI Pipeline (ai_pipeline.py)
This is the core worker thread. It pulls audio arrays from the queue and executes a two-stage process:
- Stage 1: Transcription (faster-whisper)
  Initialize the WhisperModel once on startup to persist it in VRAM/RAM. Run the audio array through model.transcribe(). Extract the raw text string.
- Stage 2: Semantic Cleanup (Ollama)
  If the raw text is extremely short (under 15 characters), bypass the LLM and return the raw text to conserve processing time. If valid, construct a JSON payload for the Ollama API.
  Use this exact System Prompt:
  "You are a specialized dictation post-processor. Your sole function is to receive raw speech-to-text output and return clean, formatted text ready to be pasted into an application. Rules: 1. Remove all filler words (e.g., um, uh, like). 2. Correct grammatical errors and punctuation. 3. Resolve spoken self-corrections (e.g., if the input is 'meet at two actually three', output 'meet at three'). 4. Output ONLY the cleaned text. Do not include any conversational preamble, XML tags, or commentary. If the active window is provided, adapt formatting appropriately (e.g., code syntax for IDEs, formal paragraphs for email)."
- Append the Active Window Title and the Raw Transcript to the user prompt.
- Parse the Ollama response to extract the final cleaned string.

Module 5: Clipboard and Injection Manager (injector.py)
Implement a highly defensive text pasting mechanism utilizing clipboard injection to prevent typing simulation errors.
- Acquire a threading lock.
- Save the current contents of the user's clipboard via pyperclip.paste().
- Clear the clipboard to prevent race conditions.
- Write the LLM-cleaned text to the clipboard via pyperclip.copy().
- Use time.sleep(0.05) to allow the OS to register the clipboard change asynchronously.
- Trigger a paste command using the correct cross-platform shortcut: pyautogui.hotkey('command', 'v') for macOS, or pyautogui.hotkey('ctrl', 'v') for Windows/Linux.
- Use time.sleep(0.05) to ensure the active application completes the paste event.
- Restore the original clipboard content via pyperclip.copy().
- Release the thread lock.

Module 6: Orchestration (main.py)
Wire all modules together into a cohesive application.
- Initialize the AI pipeline in a background daemon thread.
- Initialize the pynput keyboard listener, ensuring key press/release events manage the audio capture state seamlessly without dropping frames or freezing.
- Include graceful shutdown handling for KeyboardInterrupt to clean up threads and audio streams.

Module 7: Repository Documentation & Meta Files
- .gitignore: Provide a comprehensive Python .gitignore, explicitly ensuring python caches, virtual environments (.venv), log files, and temporary audio exports are ignored.
- README.md: Generate a polished, beautifully formatted GitHub repository README. It must include:
  1. Project Title and a compelling description highlighting privacy-first, 100% local operation.
  2. Features list (Cross-platform window awareness, multi-threaded pipeline, clipboard defense).
  3. Prerequisites (Ollama installation, system dependencies like portaudio/xdotool if needed).
  4. Step-by-step Installation Guide (virtual environment setup, pip install).
  5. Configuration breakdown.
  6. Architecture block diagram (using simple text/ASCII art to map out the Main Thread vs. Worker Thread pipeline).

Code Constraints:
- Include extensive logging using Python's logging module so the user can observe inference times and latency bottlenecks in the console.
- Handle all edge cases gracefully (e.g., Ollama server is offline, microphone permissions denied, clipboard is locked by another process).
- Do not use placeholder comments for core logic; write the actual implementation for the entire data flow.
- Provide the code files sequentially with clear layout boundaries.