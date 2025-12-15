# 🎤 Indic Parler TTS - Audio Quality Guide

> Complete reference for improving voice quality with all available settings

---

## 📊 Quick Reference: Audio Quality Settings

| Setting | Best Quality | Medium | Low |
|---------|-------------|--------|-----|
| **Background Noise** | `very clear audio` | `slightly noisy` | `very noisy audio` |
| **Reverberation** | `close-sounding` | `slightly distant` | `distant-sounding` |
| **Expressivity** | `expressive and animated` | `slightly expressive` | `monotone` |
| **Pitch** | Any (preference) | `moderate pitch` | Any |
| **Speaking Rate** | `moderate pace` | `slow pace` / `fast pace` | - |
| **Voice Quality** | `very high quality` | `good quality` | `basic quality` |

---

## 🎯 Settings Deep Dive

### 1. Background Noise
Controls the clarity of the recording environment.

```
"very clear audio"       → Studio-quality, no background noise
"slightly noisy"         → Minor ambient sounds
"very noisy audio"       → Noticeable background noise
```

**Best Practice**: Always include `"very clear audio"` for production use.

---

### 2. Reverberation / Distance
Controls how close the speaker sounds.

```
"close-sounding"         → Intimate, direct sound (like a podcast mic)
"slightly distant"       → Some room ambiance
"distant-sounding"       → Echoey, far from mic
```

**Best Practice**: Use `"close-sounding"` or `"very close up"` for clarity.

---

### 3. Expressivity
Controls the emotional range and dynamics.

```
"expressive and animated" → Lively, dynamic speech with emotion
"slightly expressive"     → Natural but controlled
"monotone"               → Flat, robotic delivery
```

**Best Practice**: Use `"expressive and animated"` for natural speech, `"slightly expressive"` for professional content.

---

### 4. Pitch
Controls voice frequency.

```
"high-pitched"           → Higher voice
"slightly high-pitched"  → Moderately high
"moderate pitch"         → Neutral
"slightly low-pitched"   → Moderately low
"low-pitched"           → Deeper voice
```

---

### 5. Speaking Rate
Controls the pace of speech.

```
"fast pace"              → Quick delivery
"slightly fast"          → Moderately quick
"moderate pace"          → Normal speed
"slow pace"              → Deliberate, clear
```

**Best Practice**: Use `"moderate pace"` for clarity, `"slow pace"` for narration.

---

### 6. Voice Quality
Overall production quality indicator.

```
"very high quality"      → Best output quality
"high quality"          → Very good quality
"good quality"          → Standard quality
"basic quality"         → Lower quality
```

---

## 🎭 Emotion Tags

### Available Emotions (12)
| Emotion | Use Case |
|---------|----------|
| **Command** | Instructions, directions |
| **Anger** | Frustrated speech |
| **Narration** | Storytelling, audiobooks |
| **Conversation** | Natural dialogue |
| **Disgust** | Negative reactions |
| **Fear** | Worried, scared speech |
| **Happy** | Joyful, positive content |
| **Neutral** | Informational content |
| **News** | Broadcast-style delivery |
| **Sad** | Melancholic content |
| **Surprise** | Amazed, unexpected reactions |
| **Proper Noun** | Names emphasis |

### Languages with Tested Emotion Support
- Assamese, Bengali, Bodo, Dogri, Kannada
- Malayalam, Marathi, Sanskrit, Nepali, Tamil

> **Note**: Emotions may work for other languages but are not officially tested.

---

## 👥 Named Speakers (69 Total)

Using a specific speaker name ensures **consistent voice** across generations.

### How to Use
```python
description = "Divya's voice is slightly expressive at a moderate pace..."
```

### Sample Speakers by Language
| Language | Speakers |
|----------|----------|
| Hindi | Divya, Rohit, Maya, Karan, Sita, Bikram |
| Tamil | Aditi, Sunita, Tapan |
| Telugu | Anjali, Amrita |
| Bengali | Leela |
| + More | 57 additional speakers |

---

## 📝 Description Templates

### Maximum Quality Female
```
A female speaker with a slightly high pitch delivers expressive and 
animated speech at a moderate pace. The recording is of very high 
quality, with very clear audio and the speaker sounds very close up.
```

### Maximum Quality Male
```
A male speaker with a moderate pitch delivers expressive speech at 
a moderate pace. The recording is of very high quality, with very 
clear audio and a close-sounding environment.
```

### Named Speaker Template
```
Divya's voice is slightly expressive and animated with a moderate 
pace, captured in a very clear, close-sounding recording of 
very high quality.
```

### News Anchor Style
```
A professional news anchor delivers clear, articulate speech with 
a neutral tone at a moderate pace. The recording is of very high 
quality with very clear audio.
```

### Audiobook Narrator
```
Karan speaks with a low-pitched, calm voice at a slow pace. The 
recording is of very high quality, with very clear audio and a 
close-sounding environment.
```

---

## ⚡ Performance Optimization

### 1. SDPA Attention (Default - 1.4x Faster)
```python
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "ai4bharat/indic-parler-tts",
    attn_implementation="sdpa"
)
```

### 2. torch.compile (Up to 4.5x Faster)
```python
model.generation_config.cache_implementation = "static"
model.forward = torch.compile(model.forward, mode="reduce-overhead")

# Requires 2 warmup runs
for _ in range(2):
    _ = model.generate(...)
```

### 3. Batch Generation
Process multiple texts simultaneously for efficiency.

### 4. GPU with bfloat16
```python
model = model.to("cuda", dtype=torch.bfloat16)
```

---

## ✅ Best Practices Checklist

- [ ] Always include `"very clear audio"` 
- [ ] Use `"close-sounding"` or `"very close up"`
- [ ] Set `"very high quality"` for production
- [ ] Use **named speakers** for voice consistency
- [ ] Add **punctuation** in text for natural pauses
- [ ] Enable **torch.compile** for 4x+ speedup
- [ ] Use **bfloat16** dtype on GPU
- [ ] Use **batch generation** for multiple texts

---

## 🌍 Supported Languages

### Official (21)
Assamese, Bengali, Bodo, Dogri, Kannada, Malayalam, Marathi, Sanskrit, 
Nepali, English, Telugu, Hindi, Gujarati, Konkani, Maithili, Manipuri, 
Odia, Santali, Sindhi, Tamil, Urdu

### Unofficial (3)
Chhattisgarhi, Kashmiri, Punjabi

---

## 📐 Quality Metrics (Native Speaker Scores)

| Language | NSS Score |
|----------|-----------|
| Sanskrit | 99.79 |
| Maithili | 95.36 |
| Bodo | 94.47 |
| Sindhi | 76.46 |
| Kashmiri | 55.30 |

---

*Generated from ai4bharat/indic-parler-tts documentation*
