import os
import tempfile
import cv2
from PIL import Image
import subprocess
import torch
from faster_whisper import WhisperModel
from transformers import pipeline
import easyocr

# Lazy loading models
asr_model = None
summarizer = None
qa_model = None
ocr_reader = None

def get_asr_model():
    global asr_model
    if asr_model is None:
        asr_model = WhisperModel("base", device="cpu")
    return asr_model

def get_summarizer():
    global summarizer
    if summarizer is None:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return summarizer

def get_qa_model():
    global qa_model
    if qa_model is None:
        qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    return qa_model

def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        ocr_reader = easyocr.Reader(['en'])
    return ocr_reader

def extract_audio_from_video(video_path):
    """Extracts audio track from video as .wav"""
    audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    command = [
        "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


def transcribe_and_summarize(file_path):
    """Transcribes and summarizes an audio or video file."""
    if file_path.lower().endswith(('.mp4', '.mov', '.avi', '.webm')):

        return process_video(file_path)
    else:

        asr = get_asr_model()
        segments, _ = asr.transcribe(file_path)
        full_text = " ".join([segment.text for segment in segments])

        if len(full_text.strip()) == 0:
            return "No speech detected.", "No summary could be generated."

        summ = get_summarizer()
        summary = summ(full_text, max_length=200, min_length=80, do_sample=False)[0]['summary_text']
        return full_text, summary



def extract_keyframes(video_path, frame_interval=5):
    """Extracts representative frames from a video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    count = 0
    success, image = cap.read()

    while success:
        if int(count % (fps * frame_interval)) == 0:
            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(temp_file.name, image)
            frames.append(temp_file.name)
        success, image = cap.read()
        count += 1

    cap.release()
    return frames


def analyze_frame(image_path):
    """Analyzes a frame for text (e.g., from PPT slides)."""
    # Extract visible text (like from slides)
    ocr = get_ocr_reader()
    ocr_text = " ".join(ocr.readtext(image_path, detail=0))
    return ocr_text



def process_video(video_path):
    """
    Processes a video to:
      - Transcribe and summarize spoken content
      - Detect text in visuals (e.g., PPT slides)
      - Merge all into a unified summary
    """
    # Step 1: Transcribe & summarize audio
    audio_path = extract_audio_from_video(video_path)
    asr = get_asr_model()
    segments, _ = asr.transcribe(audio_path)
    transcript = " ".join([segment.text for segment in segments])

    if len(transcript.strip()) == 0:
        return "No speech detected.", "No summary could be generated."

    summ = get_summarizer()
    speech_summary = summ(transcript, max_length=200, min_length=80, do_sample=False)[0]['summary_text']

    # Step 2: Extract keyframes
    frames = extract_keyframes(video_path, frame_interval=5)
    ocr_texts = []

    # Step 3: Analyze visuals (OCR for text detection)
    for f in frames[:10]:  # process limited frames for speed
        ocr_txt = analyze_frame(f)
        ocr_texts.append(ocr_txt)
        os.remove(f)

    # Step 4: Summarize visual content
    combined_visual_text = " ".join(ocr_texts)

    if len(combined_visual_text.strip()) > 20:
        visual_summary = summ(
            combined_visual_text,
            max_length=150,
            min_length=50,
            do_sample=False
        )[0]['summary_text']
    else:
        visual_summary = "No significant text detected in visuals."

    # Step 5: Combine both
    combined_summary = summ(
        f"Speech Summary: {speech_summary}. Visual Summary: {visual_summary}.",
        max_length=250,
        min_length=80,
        do_sample=False
    )[0]['summary_text']

    return transcript, combined_summary



def chat_with_ai(user_message, transcript, summary):
    """Generates a response to user query based on transcript and summary."""

    # Check if we have meeting data
    if not transcript.strip() and not summary.strip():
        return "I don't have any meeting data to work with yet. Please upload an audio or video file first to get started!"

    user_lower = user_message.lower().strip()

    # Define keyword groups for better matching
    summary_keywords = ["summary", "summarize", "what happened", "overview", "about the meeting", "meeting summary", "summarize the meeting"]
    transcript_keywords = ["transcript", "full text", "what was said", "details", "everything", "full transcript", "meeting transcript"]
    keypoints_keywords = ["key points", "main points", "highlights", "important", "key takeaways", "main ideas"]
    duration_keywords = ["how long", "duration", "length", "time", "meeting length", "how long was"]
    agenda_keywords = ["agenda", "topics", "discussed", "talked about", "topic of the meeting", "meeting topics", "what was discussed", "meeting agenda", "agenda of the meeting"]
    context_keywords = ["context", "background", "context of the meeting", "meeting context", "what was the context", "background of the meeting"]

    # Check for keyword matches (more flexible matching)
    def contains_any_keyword(message, keywords):
        return any(keyword in message for keyword in keywords)

    # Handle summary requests
    if contains_any_keyword(user_lower, summary_keywords):
        if summary.strip():
            return f"Here's a summary of the meeting:\n\n{summary}"
        else:
            return "I have the transcript but couldn't generate a summary. Here's what was said:\n\n" + transcript[:500] + "..." if len(transcript) > 500 else transcript

    # Handle transcript requests
    elif contains_any_keyword(user_lower, transcript_keywords):
        if transcript.strip():
            return f"Here's the full transcript:\n\n{transcript[:2000]}{'...' if len(transcript) > 2000 else ''}"
        else:
            return "I don't have a transcript available. The audio processing might have failed."

    # Handle key points or main ideas
    elif contains_any_keyword(user_lower, keypoints_keywords):
        if summary.strip():
            # Extract key sentences from summary
            sentences = summary.split('. ')
            key_points = '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else summary
            return f"Key points from the meeting:\n\n{key_points}"
        else:
            return "I don't have a summary to extract key points from. Please ask for the transcript instead."

    # Handle length or duration if available (but we don't have it)
    elif contains_any_keyword(user_lower, duration_keywords):
        return "I'm sorry, I don't have information about the meeting duration from the audio file."

    # Handle agenda or topics
    elif contains_any_keyword(user_lower, agenda_keywords):
        if summary.strip():
            return f"Based on the meeting content, here are the main topics discussed:\n\n{summary}"
        else:
            return "I don't have enough information to determine the agenda. Please check the transcript for details."

    # Handle context or background
    elif contains_any_keyword(user_lower, context_keywords):
        if summary.strip():
            return f"Here's the context and background from the meeting:\n\n{summary}"
        else:
            return "I don't have enough information about the meeting context. Please check the transcript for details."

    # For other questions, use QA model on transcript and summary
    else:
        if transcript.strip() or summary.strip():
            context = f"Transcript: {transcript}\n\nSummary: {summary}"
            try:
                qa = get_qa_model()
                answer = qa(question=user_message, context=context)['answer']
                # Check if the answer is meaningful (not just a fragment or generic greeting)
                answer_clean = answer.lower().strip()
                generic_responses = ['[cls]', '[sep]', '', 'hello everyone', 'hello everyone.', 'hello', 'hi', 'hey', 'yes', 'no', 'okay', 'ok', 'sure', 'alright', 'fine', 'good', 'thanks', 'thank you']
                if len(answer.strip()) < 10 or answer_clean in generic_responses or answer_clean.startswith('hello'):
                    return "I couldn't find a specific answer to your question in the meeting content. Try asking about the summary, key points, or transcript instead."
                return answer
            except Exception as e:
                return "I had trouble analyzing your question. Please try asking about the summary, transcript, or key points from the meeting."
        else:
            return "I don't have any meeting data to answer questions about. Please upload an audio or video file first!"
