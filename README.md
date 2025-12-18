🎥 AI Meeting Summarizer (Audio + Video + Visual Understanding)

An AI-powered web application that automatically transcribes, summarizes, and analyzes meetings from audio and video files.
For video uploads, the system goes beyond audio by understanding visual content such as presentation slides (PPT), on-screen text, and objects, and generates a combined intelligent summary.

🚀 Features
🔊 Audio-Based Intelligence

Upload audio files (.mp3, .wav, .m4a)

Accurate speech-to-text transcription using Whisper

Concise meeting summary generation using Transformer-based NLP models

🎥 Video-Based Multimodal Intelligence

When a video file (.mp4) is uploaded, the system performs multimodal analysis:

Audio Extraction

Extracts audio track from video using FFmpeg

Converts speech to text using Whisper ASR

Visual Understanding

Extracts key video frames at fixed intervals

Detects:

📊 Presentation slides (PPT)

🧾 On-screen text (via OCR)

🖼 Objects & scenes

Text & Object Detection

Uses OCR to read visible slide text

Uses image captioning models to describe visual content

Combined AI Summary

Generates:

Speech summary (what was said)

Visual summary (what was shown)

Final combined summary (audio + visuals)

🧠 How It Works (Pipeline)
📌 Audio Upload
Audio File
   ↓
Whisper ASR
   ↓
Transcript
   ↓
Text Summarization (BART)

📌 Video Upload
Video File
   ↓
Audio Extraction (FFmpeg) ─────────┐
   ↓                              │
Whisper ASR                        │
   ↓                              │
Speech Summary                     │
                                   ↓
Frame Extraction (OpenCV)
   ↓
OCR + Image Captioning
   ↓
Visual Summary
   ↓
Combined AI Summary

🧰 Tech Stack
🔙 Backend

Python

Flask – Web framework

🧠 AI / ML

Whisper (faster-whisper) – Speech-to-Text

BART (facebook/bart-large-cnn) – Text summarization

BLIP – Image captioning (visual understanding)

EasyOCR – Slide & on-screen text detection

🎥 Video & Image Processing

FFmpeg – Audio extraction from video

OpenCV – Frame extraction

Pillow – Image handling

🎨 Frontend

HTML5

Tailwind CSS

JavaScript

📁 Supported File Formats
Type	Formats
Audio	.mp3, .wav, .m4a, .aac
Video	.mp4, .webm, .mov
⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/ai-meeting-summarizer.git
cd ai-meeting-summarizer

2️⃣ Create Virtual Environment
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install flask werkzeug faster-whisper transformers torch torchvision
pip install opencv-python pillow easyocr

4️⃣ Install FFmpeg

Windows: https://ffmpeg.org/download.html

Linux:

sudo apt install ffmpeg

5️⃣ Run the App
python app.py


Open browser:

http://localhost:5000

🖥 Application Pages

🏠 Home – Upload audio/video and view summary

💬 Chat – Ask questions about the meeting using AI

ℹ️ About – Project overview and technical details

📊 Output Example (Video Upload)

Transcript

Full text of spoken conversation from the video

Speech Summary

Key discussion points extracted from audio

Visual Summary

Detected slides, charts, titles, and on-screen content

Combined Summary

Unified understanding of what was said and what was shown

🔮 Future Enhancements

Speaker diarization (who spoke when)

Sentiment analysis of meetings

Auto slide-to-summary mapping

Cloud deployment & multi-user support

Meeting action-item extraction

👨‍💻 Author

Aditya Kaper
B.E. Computer Science
AI & Full-Stack Development Enthusiast

⭐ Why This Project Matters

This project demonstrates real-world multimodal AI by combining:

Speech Recognition

Computer Vision

Natural Language Processing

It solves a practical productivity problem by converting long meetings into actionable insights
