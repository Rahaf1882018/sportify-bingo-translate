import gradio as gr
import whisper
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
import tempfile
import os

# تحميل نموذج Whisper
whisper_model = whisper.load_model("small")

# تحميل نموذج الترجمة من الإنجليزية إلى العربية
model_name = "Helsinki-NLP/opus-mt-en-ar"
translation_model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# تحويل الصوت إلى نص
def transcribe(audio):
    result = whisper_model.transcribe(audio)
    return result['text']

# ترجمة النص
def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = translation_model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# تحويل النص العربي إلى صوت باستخدام gTTS
def text_to_speech_arabic(text):
    tts = gTTS(text, lang='ar')
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)
    return tmp_file.name

# دالة شاملة: صوت ➡️ نص ➡️ ترجمة ➡️ صوت عربي
def full_process(audio):
    if audio is None:
        return "🚫 الرجاء رفع ملف صوتي", "", None

    # 1. تحويل الصوت إلى نص
    text = transcribe(audio)

    # 2. ترجمة النص
    translated = translate(text)

    # 3. تحويل الترجمة إلى صوت
    arabic_audio_path = text_to_speech_arabic(translated)

    return text, translated, arabic_audio_path

# واجهة Gradio
demo = gr.Interface(
    fn=full_process,
    inputs=gr.Audio(type="filepath", label="🎧 ارفع المقطع الصوتي"),
    outputs=[
        gr.Textbox(label="📜 النص الأصلي (بالإنجليزية)"),
        gr.Textbox(label="🈶 الترجمة (بالعربية)"),
        gr.Audio(label="🔊 الترجمة المنطوقة", type="filepath")
    ],
    title="🎙️ Sportify Bingo - الترجمة الصوتية",
    description="ارفع مقطع صوتي للتعليق الرياضي وسنقوم بتحويله إلى نص، ترجمته للعربية، ثم نطقه بالصوت.",
    theme="soft"
)

# تشغيل التطبيق
demo.launch(share=True)
