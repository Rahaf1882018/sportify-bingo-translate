import gradio as gr
import torch
from transformers import MarianMTModel, MarianTokenizer
import whisper
from gtts import gTTS
import os
from pydub import AudioSegment
import tempfile

# تحميل نموذج Whisper (الصغير لسرعة أكبر)
whisper_model = whisper.load_model("base")

# تحميل نموذج الترجمة المخصص من Hugging Face
model_name = "Rahaf2018/Sportify-Bingo-Translator-Finetuned"
translator = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# الدالة: تحويل الصوت إلى نص باستخدام Whisper
def transcribe_audio(audio_path):
    audio = whisper_model.transcribe(audio_path, fp16=False)
    return audio["text"]

# الدالة: ترجمة النص باستخدام النموذج المخصص
def translate_text(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = translator.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# الدالة: تحويل النص العربي إلى صوت
def text_to_speech(text):
    tts = gTTS(text, lang="ar")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# اقتصاص الصوت إلى مقاطع صغيرة <= 30 ثانية
def split_audio_chunks(file_path, max_chunk_length=30 * 1000):
    audio = AudioSegment.from_file(file_path)
    chunks = []
    for i in range(0, len(audio), max_chunk_length):
        chunk = audio[i:i + max_chunk_length]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        chunk.export(temp_file.name, format="wav")
        chunks.append(temp_file.name)
    return chunks

# الدالة الكاملة: تحويل صوت إلى ترجمة صوتية عربية
def full_pipeline(audio_path):
    chunks = split_audio_chunks(audio_path)
    full_transcription = ""
    for chunk_path in chunks:
        full_transcription += transcribe_audio(chunk_path) + " "
    translation = translate_text(full_transcription.strip())
    audio_output = text_to_speech(translation)
    return full_transcription, translation, audio_output

# واجهة Gradio
with gr.Blocks(title="Sportify Bingo") as demo:
    gr.Markdown("## 🎙️ Sportify Bingo - Sports Commentary Translator")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="🎧 ارفع تعليقك الصوتي")
            submit = gr.Button("ابدأ الترجمة")

        with gr.Column():
            transcription_output = gr.Textbox(label="🎤 النص الإنجليزي")
            translation_output = gr.Textbox(label="📘 الترجمة العربية")
            tts_output = gr.Audio(label="🔊 الصوت المترجم")

    submit.click(full_pipeline, inputs=audio_input, outputs=[transcription_output, translation_output, tts_output])

demo.launch()
