/**
 * Indic Parler TTS - Audio Quality Control
 * JavaScript Application Logic
 */

// ==========================================
// Configuration Maps
// ==========================================

const PITCH_MAP = {
    1: 'low-pitched',
    2: 'slightly low-pitched',
    3: 'moderate pitch',
    4: 'slightly high-pitched',
    5: 'high-pitched'
};

const PITCH_DISPLAY = {
    1: 'Low',
    2: 'Slightly Low',
    3: 'Moderate',
    4: 'Slightly High',
    5: 'High'
};

const SPEED_MAP = {
    1: 'slow pace',
    2: 'slightly slow pace',
    3: 'moderate pace',
    4: 'slightly fast pace',
    5: 'fast pace'
};

const SPEED_DISPLAY = {
    1: 'Slow',
    2: 'Slightly Slow',
    3: 'Moderate',
    4: 'Slightly Fast',
    5: 'Fast'
};

const EXPRESSIVITY_MAP = {
    1: 'monotone',
    2: 'slightly expressive',
    3: 'expressive and animated'
};

const EXPRESSIVITY_DISPLAY = {
    1: 'Monotone',
    2: 'Slightly Expressive',
    3: 'Expressive & Animated'
};

// ==========================================
// DOM Elements
// ==========================================

const elements = {
    textInput: document.getElementById('textInput'),
    speaker: document.getElementById('speaker'),
    accent: document.getElementById('accent'),
    emotion: document.getElementById('emotion'),
    pitch: document.getElementById('pitch'),
    speed: document.getElementById('speed'),
    expressivity: document.getElementById('expressivity'),
    pitchValue: document.getElementById('pitchValue'),
    speedValue: document.getElementById('speedValue'),
    expressivityValue: document.getElementById('expressivityValue'),
    descriptionOutput: document.getElementById('descriptionOutput'),
    codeOutput: document.getElementById('codeOutput'),
    genderGroup: document.getElementById('genderGroup'),
    accentGroup: document.getElementById('accentGroup')
};

// ==========================================
// Helper Functions
// ==========================================

function getSelectedRadio(name) {
    const selected = document.querySelector(`input[name="${name}"]:checked`);
    return selected ? selected.value : null;
}

function getGender() {
    return getSelectedRadio('gender');
}

function getQuality() {
    return getSelectedRadio('quality');
}

function getNoise() {
    return getSelectedRadio('noise');
}

function getReverb() {
    return getSelectedRadio('reverb');
}

// ==========================================
// Description Generator
// ==========================================

function generateDescription() {
    const speaker = elements.speaker.value;
    const gender = getGender();
    const accent = elements.accent.value;
    const emotion = elements.emotion.value;
    const pitch = PITCH_MAP[elements.pitch.value];
    const speed = SPEED_MAP[elements.speed.value];
    const expressivity = EXPRESSIVITY_MAP[elements.expressivity.value];
    const quality = getQuality();
    const noise = getNoise();
    const reverb = getReverb();

    let description = '';

    // Build speaker/voice part
    if (speaker) {
        description = `${speaker}'s voice is ${expressivity} with a ${pitch} tone`;
    } else {
        let voicePart = `A ${gender} speaker`;
        if (accent) {
            voicePart = `A ${gender} ${accent} speaker`;
        }
        description = `${voicePart} with a ${pitch} voice delivers ${expressivity} speech`;
    }

    // Add speed
    description += ` at a ${speed}`;

    // Add emotion
    if (emotion) {
        description += ` with a ${emotion} tone`;
    }

    description += '.';

    // Add quality descriptors
    description += ` The recording is of ${quality}`;
    description += `, with ${noise} audio`;
    description += ` and a ${reverb} environment.`;

    return description;
}

// ==========================================
// Code Generator
// ==========================================

function generateCode() {
    const text = elements.textInput.value || 'Hello, welcome to Indic Parler TTS!';
    const description = generateDescription();

    const code = `import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Load model with GPU optimization
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "ai4bharat/indic-parler-tts",
    torch_dtype=torch_dtype,
    attn_implementation="sdpa"  # 1.4x speedup
).to(device)

# Load tokenizers
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)

# Your text and description
prompt = """${escapeString(text)}"""

description = """${escapeString(description)}"""

# Tokenize inputs
description_input_ids = description_tokenizer(
    description, return_tensors="pt"
).to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

# Generate audio
with torch.no_grad():
    generation = model.generate(
        input_ids=description_input_ids.input_ids,
        attention_mask=description_input_ids.attention_mask,
        prompt_input_ids=prompt_input_ids.input_ids,
        prompt_attention_mask=prompt_input_ids.attention_mask
    )

# Save audio
audio_arr = generation.cpu().numpy().squeeze()
sf.write("output.wav", audio_arr, model.config.sampling_rate)
print("✅ Audio saved to output.wav")`;

    return code;
}

function escapeString(str) {
    return str.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n');
}

// ==========================================
// UI Update Functions
// ==========================================

function updateSliderDisplays() {
    elements.pitchValue.textContent = PITCH_DISPLAY[elements.pitch.value];
    elements.speedValue.textContent = SPEED_DISPLAY[elements.speed.value];
    elements.expressivityValue.textContent = EXPRESSIVITY_DISPLAY[elements.expressivity.value];
}

function updateSpeakerDependentFields() {
    const hasSpeaker = elements.speaker.value !== '';
    elements.genderGroup.style.display = hasSpeaker ? 'none' : 'block';
    elements.accentGroup.style.display = hasSpeaker ? 'none' : 'block';
}

function updateOutput() {
    const description = generateDescription();
    const code = generateCode();

    elements.descriptionOutput.textContent = description;
    elements.codeOutput.textContent = code;
}

// ==========================================
// Copy to Clipboard
// ==========================================

async function copyCode() {
    const code = elements.codeOutput.textContent;
    const btn = document.querySelector('.copy-btn');
    const btnText = btn.querySelector('.copy-text');

    try {
        await navigator.clipboard.writeText(code);
        btnText.textContent = 'Copied!';
        btn.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';

        setTimeout(() => {
            btnText.textContent = 'Copy Code';
            btn.style.background = '';
        }, 2000);
    } catch (err) {
        console.error('Failed to copy:', err);
        btnText.textContent = 'Failed!';

        setTimeout(() => {
            btnText.textContent = 'Copy Code';
        }, 2000);
    }
}

// ==========================================
// Speaker Tag Click Handler
// ==========================================

function setupSpeakerTags() {
    document.querySelectorAll('.speaker-tag:not(.more)').forEach(tag => {
        tag.addEventListener('click', () => {
            const speakerName = tag.textContent;
            // Find and select the speaker in dropdown
            const option = Array.from(elements.speaker.options).find(
                opt => opt.value === speakerName
            );
            if (option) {
                elements.speaker.value = speakerName;
                updateSpeakerDependentFields();
                updateOutput();
            }
        });
    });
}

// ==========================================
// Event Listeners
// ==========================================

function setupEventListeners() {
    // Text input
    elements.textInput.addEventListener('input', updateOutput);

    // Dropdowns
    elements.speaker.addEventListener('change', () => {
        updateSpeakerDependentFields();
        updateOutput();
    });
    elements.accent.addEventListener('change', updateOutput);
    elements.emotion.addEventListener('change', updateOutput);

    // Sliders
    elements.pitch.addEventListener('input', () => {
        updateSliderDisplays();
        updateOutput();
    });
    elements.speed.addEventListener('input', () => {
        updateSliderDisplays();
        updateOutput();
    });
    elements.expressivity.addEventListener('input', () => {
        updateSliderDisplays();
        updateOutput();
    });

    // Radio buttons
    document.querySelectorAll('input[type="radio"]').forEach(radio => {
        radio.addEventListener('change', updateOutput);
    });
}

// ==========================================
// Initialization
// ==========================================

function init() {
    updateSliderDisplays();
    updateSpeakerDependentFields();
    updateOutput();
    setupEventListeners();
    setupSpeakerTags();

    console.log('🎤 Indic Parler TTS Control Studio initialized!');
}

// Run on DOM ready
document.addEventListener('DOMContentLoaded', init);
