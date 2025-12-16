
from transformers import AutoConfig
try:
    config = AutoConfig.from_pretrained("ai4bharat/indic-parler-tts")
    print(f"Max Position Embeddings: {getattr(config, 'max_position_embeddings', 'Not found')}")
    print(f"Max Length: {getattr(config, 'max_length', 'Not found')}")
    
    # Check text encoder config if available
    if hasattr(config, "text_encoder"):
        print(f"Text Encoder Max Pos: {getattr(config.text_encoder, 'max_position_embeddings', 'Not found')}")
        
    # Check decoder config if available
    if hasattr(config, "decoder"):
        print(f"Decoder Max Pos: {getattr(config.decoder, 'max_position_embeddings', 'Not found')}")
        
except Exception as e:
    print(f"Error: {e}")
