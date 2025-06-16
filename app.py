import streamlit as st
import whisper
import openai

st.set_page_config(page_title="Arabic Audio Transcriber & Summarizer", layout="centered")
st.title("🎙️ Arabic Audio Transcriber & Summarizer")

# Step 1: Ask for API Key securely
api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()
openai.api_key = api_key

# Step 2: Upload audio file
audio_file = st.file_uploader("📤 Upload an audio file", type=["mp3", "wav"])
if audio_file:
    with st.spinner("Saving audio file..."):
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_file.read())
    
    # Step 3: Transcribe
    st.info("🗣️ Transcribing audio with Whisper...")
    model = whisper.load_model("base")  # You can use "medium" or "large" if needed
    result = model.transcribe("temp_audio.wav", language="ar")
    
    st.subheader("📄 Transcript")
    st.write(result["text"])
    
    # Step 4: Summarize
    st.info("🧠 Summarizing with GPT...")
    prompt = f"لخص النص التالي باللغة العربية في فقرة واحدة:\n\n{result['text']}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo" if needed
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=300
        )
        summary = response.choices[0].message.content
        st.subheader("📝 Summary")
        st.write(summary)
    except Exception as e:
        st.error(f"❌ An error occurred during summarization: {str(e)}")
