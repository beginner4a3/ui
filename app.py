"""
Indic Parler TTS - Interactive Audio Quality Control
Gradio-based frontend for Google Colab

Run with: python app.py
"""

import torch
import gradio as gr
import numpy as np

# ==========================================
# Configuration
# ==========================================

SPEAKERS = {
    "-- Random Voice --": "",
    "Hindi": ["Divya", "Rohit", "Maya", "Karan", "Sita", "Bikram"],
    "Tamil": ["Aditi", "Sunita", "Tapan"],
    "Telugu": ["Anjali", "Amrita"],
    "Bengali": ["Leela"],
    "Kannada": ["Kavya", "Priya"],
    "Malayalam": ["Meera", "Lakshmi"],
    "Marathi": ["Neha", "Pooja"],
}

# Flatten speakers for dropdown
SPEAKER_CHOICES = ["-- Random Voice --"]
for lang, speakers in SPEAKERS.items():
    if isinstance(speakers, list):
        SPEAKER_CHOICES.extend([f"{s} ({lang})" for s in speakers])

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
# Model Loading
# ==========================================

model = None
tokenizer = None
description_tokenizer = None
device = None

def load_model():
    """Load the Indic Parler TTS model with GPU optimization."""
    global model, tokenizer, description_tokenizer, device
    
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print(f"🔧 Loading model on {device}...")
    
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "ai4bharat/indic-parler-tts",
        torch_dtype=torch_dtype,
        attn_implementation="sdpa"
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    description_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path
    )
    
    print("✅ Model loaded successfully!")
    return f"✅ Model loaded on {device}"

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
    if speaker and speaker != "-- Random Voice --":
        # Extract speaker name (remove language suffix)
        speaker_name = speaker.split(" (")[0]
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
    
    # Audio quality
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
        return None, "❌ Model not loaded! Click 'Load Model' first."
    
    if not text.strip():
        return None, "❌ Please enter some text to speak."
    
    # Generate description
    description = generate_description(
        speaker, gender, accent, emotion,
        pitch, speed, expressivity,
        quality, noise, reverb
    )
    
    try:
        # Tokenize
        desc_inputs = description_tokenizer(description, return_tensors="pt").to(device)
        text_inputs = tokenizer(text, return_tensors="pt").to(device)
        
        # Generate
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

# ==========================================
# Preview Description
# ==========================================

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
# Gradio Interface
# ==========================================

def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="Indic Parler TTS",
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="pink",
        ),
        css="""
        .gradio-container { max-width: 1200px !important; }
        .main-title { text-align: center; margin-bottom: 1rem; }
        .desc-preview { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem; border-radius: 10px; color: white;
        }
        """
    ) as app:
        
        gr.Markdown(
            """
            # 🎤 Indic Parler TTS - Audio Quality Control
            **Generate high-quality speech with 69 speakers, 21 languages, and 12 emotions**
            """,
            elem_classes="main-title"
        )
        
        with gr.Row():
            load_btn = gr.Button("🚀 Load Model", variant="primary", scale=1)
            load_status = gr.Textbox(
                label="Status", 
                value="⏳ Click 'Load Model' to start",
                interactive=False,
                scale=2
            )
        
        load_btn.click(fn=load_model, outputs=load_status)
        
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
                    - Use **named speakers** for consistent voice
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
            inputs=all_inputs[1:],  # Exclude text_input
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

# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    print("🎤 Starting Indic Parler TTS UI...")
    print("=" * 50)
    
    app = create_interface()
    
    # Launch with share=True for Colab
    app.launch(
        share=True,  # Creates public URL for Colab
        debug=True,
        show_error=True
    )
