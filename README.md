# 🎤 Indic Parler TTS - Interactive UI

High-quality Text-to-Speech with **69 speakers**, **21 languages**, and **12 emotions**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/indic-parler-tts-ui/blob/main/demo.ipynb)

## ✨ Features

- 🎚️ **Full Audio Quality Control**: Pitch, Speed, Expressivity, Noise, Reverb
- 👥 **69 Named Speakers** across Hindi, Tamil, Telugu, Bengali, and more
- 🎭 **12 Emotions**: Happy, Sad, Anger, Narration, News, etc.
- 🚀 **GPU Optimized**: SDPA attention + bfloat16
- 🌐 **Gradio UI**: Works in Colab with public URL

## 🚀 Quick Start (Google Colab)

1. Click the **Open in Colab** badge above
2. Run all cells
3. Click **Load Model** in the UI
4. Start generating speech!

## 💻 Local Installation

```bash
# Clone the repository
git clone https://github.com/beginner4a3/ui.git
cd ui

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

## 📁 Project Structure

```
ui/
├── app.py                    # Gradio frontend
├── demo.ipynb               # Colab notebook
├── requirements.txt         # Dependencies
├── README.md               # This file
├── AUDIO_QUALITY_GUIDE.md  # Quality settings reference
└── frontend/               # HTML version (local only)
```

## 🎛️ Audio Quality Settings

| Setting | Options |
|---------|---------|
| **Pitch** | Low → Moderate → High |
| **Speed** | Slow → Moderate → Fast |
| **Expressivity** | Monotone → Slightly Expressive → Expressive |
| **Quality** | Good → High → Very High |
| **Noise** | Noisy → Slightly Noisy → Very Clear |
| **Reverb** | Distant → Slightly Distant → Close |

## 🎭 Supported Emotions

Command, Anger, Narration, Conversation, Disgust, Fear, Happy, Neutral, News, Sad, Surprise

## 🌍 Supported Languages

Assamese, Bengali, Bodo, Dogri, English, Gujarati, Hindi, Kannada, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu

## 📝 License

Based on [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts)

---

**Note**: Replace `beginner4a3` with your GitHub username in the Colab badge URL.
