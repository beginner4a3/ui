"""
Indic Parler TTS - Interactive Audio Quality Control
Gradio-based frontend for Google Colab

Usage:
1. First run setup cell to load model
2. Then run: python app.py (or import and call launch_app())
"""

import torch
import gradio as gr
import numpy as np
import os

# ==========================================
# Configuration - Official Speaker List
# From: https://huggingface.co/ai4bharat/indic-parler-tts
# ==========================================

# Speakers organized by language with recommended voices first
SPEAKERS_BY_LANGUAGE = {
    "Hindi": {
        "recommended": ["Divya", "Karan"],
        "all": ["Divya", "Karan", "Sita", "Bikram", "Maya", "Rohit"]
    },
    "Tamil": {
        "recommended": ["Jaya", "Thamizh"],
        "all": ["Jaya", "Thamizh", "Aditi", "Sunita", "Tapan"]
    },
    "Telugu": {
        "recommended": ["Anjali", "Amrita"],
        "all": ["Anjali", "Amrita"]
    },
    "Bengali": {
        "recommended": ["Arjun", "Aditi"],
        "all": ["Arjun", "Aditi", "Tapan", "Rashmi", "Arnav", "Riya"]
    },
    "Kannada": {
        "recommended": ["Kavya", "Priya"],
        "all": ["Kavya", "Priya"]
    },
    "Malayalam": {
        "recommended": ["Meera", "Lakshmi"],
        "all": ["Meera", "Lakshmi"]
    },
    "Marathi": {
        "recommended": ["Neha", "Pooja"],
        "all": ["Neha", "Pooja"]
    },
    "Gujarati": {
        "recommended": ["Aisha"],
        "all": ["Aisha"]
    },
    "Odia": {
        "recommended": ["Leela"],
        "all": ["Leela"]
    },
    "Punjabi": {
        "recommended": ["Indira"],
        "all": ["Indira"]
    },
    "Assamese": {
        "recommended": ["Kiran"],
        "all": ["Kiran"]
    },
    "English (Indian)": {
        "recommended": ["Aarav", "Naina"],
        "all": ["Aarav", "Naina"]
    }
}

# Build dropdown choices with recommended voices on top
def build_speaker_choices():
    choices = ["-- Random Voice --"]
    
    # First add all recommended speakers
    choices.append("--- RECOMMENDED VOICES ---")
    for lang, data in SPEAKERS_BY_LANGUAGE.items():
        for speaker in data["recommended"]:
            choices.append(f"⭐ {speaker} ({lang})")
    
    # Then add all speakers by language
    choices.append("--- ALL VOICES BY LANGUAGE ---")
    for lang, data in SPEAKERS_BY_LANGUAGE.items():
        for speaker in data["all"]:
            if f"⭐ {speaker} ({lang})" not in choices:  # Avoid duplicates
                choices.append(f"{speaker} ({lang})")
    
    return choices

SPEAKER_CHOICES = build_speaker_choices()

EMOTIONS = [
    "None", "Neutral", "Happy", "Sad", "Anger", "Fear", 
    "Surprise", "Disgust", "Narration", "News", "Conversation", "Command"
]

PITCH_OPTIONS = {
    1: ("Low", "low-pitched"),
    2: ("Slightly Low", "slightly low-pitched"),
    3: ("Moderate", "moderate pitch"),
    4: ("Slightly High", "slightly high-pitched"),
    5: ("High", "high-pitched")
}

SPEED_OPTIONS = {
    1: ("Slow", "slow pace"),
    2: ("Slightly Slow", "slightly slow pace"),
    3: ("Moderate", "moderate pace"),
    4: ("Slightly Fast", "slightly fast pace"),
    5: ("Fast", "fast pace")
}

EXPRESSIVITY_OPTIONS = {
    1: ("Monotone", "monotone"),
    2: ("Slightly Expressive", "slightly expressive"),
    3: ("Expressive & Animated", "expressive and animated")
}

# ==========================================
# Global Model Variables
# ==========================================

model = None
tokenizer = None
description_tokenizer = None
device = None

# ==========================================
# Model Loading (Called from Colab cell)
# ==========================================

def setup_model(hf_token: str):
    """
    Load the model. Call this BEFORE launching the app.
    
    Usage in Colab:
        from app import setup_model
        setup_model("hf_your_token_here")
    """
    global model, tokenizer, description_tokenizer, device
    
    print("=" * 50)
    print("🔐 Setting up Indic Parler TTS")
    print("=" * 50)
    
    # Login to HuggingFace
    from huggingface_hub import login
    try:
        login(token=hf_token.strip())
        print("✅ Logged into HuggingFace")
    except Exception as e:
        print(f"❌ HF Login failed: {e}")
        return False
    
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Use float32 instead of bfloat16 to avoid unsupported ScalarType error
    torch_dtype = torch.float32
    
    print(f"🔧 Loading model on {device}...")
    print("   (This may take a few minutes on first run)")
    
    try:
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            "ai4bharat/indic-parler-tts",
            torch_dtype=torch_dtype,
            attn_implementation="eager",  # T5Encoder doesn't support sdpa
            token=hf_token.strip()
        ).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(
            "ai4bharat/indic-parler-tts",
            token=hf_token.strip()
        )
        description_tokenizer = AutoTokenizer.from_pretrained(
            model.config.text_encoder._name_or_path
        )
        
        print("=" * 50)
        print(f"✅ Model loaded successfully on {device}!")
        print("   You can now launch the UI")
        print("=" * 50)
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def is_model_loaded():
    """Check if model is loaded."""
    return model is not None

# ==========================================
# Description Generator
# ==========================================

def generate_description(
    speaker: str,
    gender: str,
    accent: str,
    emotion: str,
    pitch: int,
    speed: int,
    expressivity: int,
    quality: str,
    noise: str,
    reverb: str
) -> str:
    """Generate the description prompt based on settings."""
    
    pitch_desc = PITCH_OPTIONS[pitch][1]
    speed_desc = SPEED_OPTIONS[speed][1]
    expr_desc = EXPRESSIVITY_OPTIONS[expressivity][1]
    
    # Build description
    if speaker and speaker not in ["-- Random Voice --", "--- RECOMMENDED VOICES ---", "--- ALL VOICES BY LANGUAGE ---"]:
        # Extract speaker name (remove ⭐ prefix and language suffix)
        speaker_name = speaker.replace("⭐ ", "").split(" (")[0]
        desc = f"{speaker_name}'s voice is {expr_desc} with a {pitch_desc} tone"
    else:
        if accent and accent != "None":
            desc = f"A {gender} {accent} speaker with a {pitch_desc} voice delivers {expr_desc} speech"
        else:
            desc = f"A {gender} speaker with a {pitch_desc} voice delivers {expr_desc} speech"
    
    desc += f" at a {speed_desc}"
    
    if emotion and emotion != "None":
        desc += f" with a {emotion} tone"
    
    desc += "."
    desc += f" The recording is of {quality}"
    desc += f", with {noise} audio"
    desc += f" and a {reverb} environment."
    
    return desc

# ==========================================
# TTS Generation
# ==========================================

def generate_speech(
    text: str,
    speaker: str,
    gender: str,
    accent: str,
    emotion: str,
    pitch: int,
    speed: int,
    expressivity: int,
    quality: str,
    noise: str,
    reverb: str
):
    """Generate speech from text using the specified settings."""
    
    global model, tokenizer, description_tokenizer, device
    
    if model is None:
        return None, "❌ Model not loaded! Run the setup cell first."
    
    if not text.strip():
        return None, "❌ Please enter some text to speak."
    
    # Skip section headers
    if speaker in ["--- RECOMMENDED VOICES ---", "--- ALL VOICES BY LANGUAGE ---"]:
        speaker = "-- Random Voice --"
    
    description = generate_description(
        speaker, gender, accent, emotion,
        pitch, speed, expressivity,
        quality, noise, reverb
    )
    
    try:
        desc_inputs = description_tokenizer(description, return_tensors="pt").to(device)
        text_inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generation = model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=text_inputs.input_ids,
                prompt_attention_mask=text_inputs.attention_mask
            )
        
        audio_arr = generation.cpu().numpy().squeeze()
        sample_rate = model.config.sampling_rate
        
        return (sample_rate, audio_arr), f"✅ Generated!\n\n📝 Description:\n{description}"
    
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def preview_description(
    speaker: str,
    gender: str,
    accent: str,
    emotion: str,
    pitch: int,
    speed: int,
    expressivity: int,
    quality: str,
    noise: str,
    reverb: str
) -> str:
    """Preview the description without generating audio."""
    return generate_description(
        speaker, gender, accent, emotion,
        pitch, speed, expressivity,
        quality, noise, reverb
    )

# ==========================================
# Gradio Interface (No Load Button)
# ==========================================

def create_interface():
    """Create the Gradio interface."""
    
    # Check if model is loaded
    model_status = "✅ Model loaded and ready!" if is_model_loaded() else "⚠️ Model not loaded - run setup cell first"
    
    with gr.Blocks(
        title="Indic Parler TTS",
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="pink",
        ),
        css="""
        .gradio-container { max-width: 1200px !important; }
        .main-title { text-align: center; margin-bottom: 1rem; }
        """
    ) as app:
        
        gr.Markdown(
            """
            # 🎤 Indic Parler TTS - Audio Quality Control
            **Generate high-quality speech with 69 speakers, 21 languages, and 12 emotions**
            """,
            elem_classes="main-title"
        )
        
        # Status bar (no load button)
        gr.Markdown(f"**Status:** {model_status}")
        
        gr.Markdown("---")
        
        with gr.Row():
            # Left Column: Settings
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Text Input")
                text_input = gr.Textbox(
                    label="Text to Speak",
                    placeholder="Enter the text you want to convert to speech...",
                    value="Hello, welcome to Indic Parler TTS. This is a demonstration of high-quality text to speech synthesis.",
                    lines=4
                )
                
                gr.Markdown("### 👤 Voice Settings")
                gr.Markdown("*⭐ = Recommended voice for best quality*")
                
                speaker = gr.Dropdown(
                    choices=SPEAKER_CHOICES,
                    value="-- Random Voice --",
                    label="Speaker"
                )
                
                with gr.Row():
                    gender = gr.Radio(
                        choices=["female", "male"],
                        value="female",
                        label="Gender"
                    )
                    accent = gr.Dropdown(
                        choices=["None", "Indian", "British", "American"],
                        value="Indian",
                        label="Accent"
                    )
                
                emotion = gr.Dropdown(
                    choices=EMOTIONS,
                    value="None",
                    label="Emotion"
                )
                
                gr.Markdown("### 🎚️ Voice Controls")
                
                pitch = gr.Slider(
                    minimum=1, maximum=5, step=1, value=3,
                    label="Pitch (Low → High)"
                )
                speed = gr.Slider(
                    minimum=1, maximum=5, step=1, value=3,
                    label="Speaking Rate (Slow → Fast)"
                )
                expressivity = gr.Slider(
                    minimum=1, maximum=3, step=1, value=2,
                    label="Expressivity (Monotone → Expressive)"
                )
                
                gr.Markdown("### 🎧 Audio Quality")
                
                quality = gr.Radio(
                    choices=["very high quality", "high quality", "good quality"],
                    value="very high quality",
                    label="Recording Quality"
                )
                noise = gr.Radio(
                    choices=["very clear", "slightly noisy", "noisy"],
                    value="very clear",
                    label="Background Noise"
                )
                reverb = gr.Radio(
                    choices=["close-sounding", "slightly distant", "distant-sounding"],
                    value="close-sounding",
                    label="Reverberation"
                )
            
            # Right Column: Output
            with gr.Column(scale=1):
                gr.Markdown("### 🔊 Generated Audio")
                
                generate_btn = gr.Button("🎙️ Generate Speech", variant="primary", size="lg")
                
                audio_output = gr.Audio(
                    label="Audio Output",
                    type="numpy"
                )
                
                status_output = gr.Textbox(
                    label="Status & Description",
                    lines=6,
                    interactive=False
                )
                
                gr.Markdown("### 📋 Description Preview")
                
                preview_btn = gr.Button("👁️ Preview Description", variant="secondary")
                
                description_preview = gr.Textbox(
                    label="Generated Description",
                    lines=4,
                    interactive=False
                )
                
                gr.Markdown(
                    """
                    ### 💡 Tips
                    - Use **⭐ recommended speakers** for best quality
                    - Add **punctuation** for natural pauses
                    - Use **"very clear"** noise for best quality
                    - Higher **expressivity** = more dynamic speech
                    """
                )
        
        # Event handlers
        all_inputs = [
            text_input, speaker, gender, accent, emotion,
            pitch, speed, expressivity, quality, noise, reverb
        ]
        
        generate_btn.click(
            fn=generate_speech,
            inputs=all_inputs,
            outputs=[audio_output, status_output]
        )
        
        preview_btn.click(
            fn=preview_description,
            inputs=all_inputs[1:],
            outputs=description_preview
        )
        
        # Auto-preview on setting change
        for inp in all_inputs[1:]:
            inp.change(
                fn=preview_description,
                inputs=all_inputs[1:],
                outputs=description_preview
            )
        
        gr.Markdown(
            """
            ---
            **Model**: [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts) | 
            **Languages**: 21 | **Speakers**: 69 | **Emotions**: 12
            """
        )
    
    return app

def launch_app():
    """Launch the Gradio app."""
    app = create_interface()
    app.launch(share=True, debug=True, show_error=True)

# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    print("🎤 Indic Parler TTS UI")
    print("=" * 50)
    
    if not is_model_loaded():
        print("⚠️  Model not pre-loaded!")
        print("   In Colab, run the setup cell first:")
        print("")
        print("   from app import setup_model")
        print("   setup_model('hf_your_token')")
        print("")
        print("   Then run this cell again.")
        print("=" * 50)
    
    app = create_interface()
    app.launch(share=True, debug=True, show_error=True)
