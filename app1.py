import streamlit as st
import requests
import tempfile
import os
from pathlib import Path
import json
import re
from typing import Dict, Tuple, Optional
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
import io
import wave

# Configure Streamlit page
st.set_page_config(
    page_title="Accent Detection Tool",
    page_icon="🎤",
    layout="wide"
)

class AccentDetectorWindows:
    def __init__(self):
        self.recognizer = sr.Recognizer()

        self.accent_patterns = {
            'American': {
                'keywords': ['really', 'water', 'better', 'letter', 'butter', 'car', 'hard', 'start'],
                'indicators': ['rhotic r', 'flat a', 'dropped t'],
                'common_words': ['dance', "can't", 'ask', 'answer']
            },
            'British': {
                'keywords': ['bath', 'dance', 'ask', 'class', 'half', 'rather', 'after'],
                'indicators': ['non-rhotic', 'broad a', 'received pronunciation'],
                'common_words': ['water', 'better', 'matter', 'butter']
            },
            'Australian': {
                'keywords': ['day', 'mate', 'today', 'place', 'face', 'way', 'say'],
                'indicators': ['rising intonation', 'diphthong variation'],
                'common_words': ['no', 'go', 'so', 'know']
            },
            'Canadian': {
                'keywords': ['about', 'house', 'out', 'now', 'how', 'south'],
                'indicators': ['canadian raising', 'eh marker'],
                'common_words': ['sorry', 'process', 'been']
            }
        }

    def save_uploaded_file(self, uploaded_file) -> str:
        try:
            suffix = Path(uploaded_file.name).suffix
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, f"uploaded{suffix}")
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.read())
            return file_path
        except Exception as e:
            raise Exception(f"File saving error: {str(e)}")

    def download_video(self, url: str) -> str:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, stream=True, timeout=60, headers=headers)
            response.raise_for_status()
            temp_dir = tempfile.mkdtemp()
            ext = '.mp4' if '.mp4' in url else '.webm' if '.webm' in url else '.avi'
            path = os.path.join(temp_dir, f"video{ext}")
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return path
        except Exception as e:
            raise Exception(f"Download error: {str(e)}")

    def extract_audio(self, video_path: str) -> str:
        try:
            base = os.path.splitext(video_path)[0]
            audio_path = base + '.wav'
            audio = AudioSegment.from_file(video_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(audio_path, format="wav")
            return audio_path
        except Exception as e:
            raise Exception(f"Audio extraction error: {str(e)}")

    def transcribe_audio(self, audio_path: str) -> str:
        try:
            with sr.AudioFile(audio_path) as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio_data = self.recognizer.record(source)
            try:
                return self.recognizer.recognize_google(audio_data).lower()
            except sr.RequestError:
                return self.recognizer.recognize_sphinx(audio_data).lower()
        except sr.UnknownValueError:
            raise Exception("Speech not understood")
        except Exception as e:
            raise Exception(f"Transcription error: {str(e)}")

    def analyze_text_patterns(self, text: str) -> Dict:
        result = {'word_count': len(text.split()), 'accent_indicators': {}}
        for accent, data in self.accent_patterns.items():
            score = 0
            features = []
            for k in data['keywords']:
                if k in text:
                    score += 15
                    features.append(f"keyword: {k}")
            for w in data['common_words']:
                if w in text:
                    score += 10
                    features.append(f"variant: {w}")
            if result['word_count'] > 20:
                score += min(result['word_count'] // 5, 20)
            result['accent_indicators'][accent] = {'score': score, 'features': features}
        return result

    def detect_accent(self, text: str) -> Tuple[str, float, str]:
        if not text.strip():
            return "Uncertain", 0, "No valid speech"
        analysis = self.analyze_text_patterns(text)
        scores = analysis['accent_indicators']
        best = max(scores, key=lambda x: scores[x]['score'])
        conf = min(scores[best]['score'], 100)
        if conf < 25:
            return "Uncertain", conf, "Low confidence detection"
        explain = ', '.join(scores[best]['features'][:3])
        return best, conf, f"Features: {explain}"

    def process_video(self, video_path: str) -> Dict:
        try:
            st.info("Extracting audio...")
            audio = self.extract_audio(video_path)
            st.info("Transcribing audio...")
            text = self.transcribe_audio(audio)
            st.info("Detecting accent...")
            accent, confidence, explanation = self.detect_accent(text)
            return {
                'success': True,
                'accent': accent,
                'confidence': confidence,
                'explanation': explanation,
                'transcription': text,
                'word_count': len(text.split())
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

def main():
    st.title("🎤 Accent Detection Tool")
    st.markdown("**Windows Version with Upload & URL Support**")
    st.markdown("---")

    detector = AccentDetectorWindows()

    option = st.radio("Select input method:", ("Upload video file", "Use video URL"))

    video_path = None
    if option == "Upload video file":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "webm", "avi", "mov"])
        if uploaded_file:
            video_path = detector.save_uploaded_file(uploaded_file)

    elif option == "Use video URL":
        video_url = st.text_input("Enter video URL:", placeholder="https://example.com/video.mp4")
        if video_url and video_url.startswith(('http://', 'https://')):
            if st.button("Download and Process"):
                with st.spinner("Downloading and Processing..."):
                    try:
                        video_path = detector.download_video(video_url)
                    except Exception as e:
                        st.error(str(e))
                        return
        else:
            st.warning("Please enter a valid video URL")

    if video_path and st.button("Analyze Accent"):
        with st.spinner("Processing..."):
            result = detector.process_video(video_path)

        if result['success']:
            st.success("Analysis complete")
            st.subheader("Detected Accent")
            st.markdown(f"**{result['accent']}** with confidence **{result['confidence']}%**")
            st.info(result['explanation'])
            st.text_area("Transcription", result['transcription'], height=150)
            st.metric("Words", result['word_count'])
        else:
            st.error(result['error'])

if __name__ == "__main__":
    main()
