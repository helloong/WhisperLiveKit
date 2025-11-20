
import torch
from whisperlivekit.whisper.model import TextDecoder, ModelDimensions

def test_tensor_mismatch():
    # Mock dimensions
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=512,
        n_audio_head=8,
        n_audio_layer=4,
        n_vocab=51865,
        n_text_ctx=448, # Standard context length
        n_text_state=512,
        n_text_head=8,
        n_text_layer=4
    )

    decoder = TextDecoder(
        dims.n_vocab,
        dims.n_text_ctx,
        dims.n_text_state,
        dims.n_text_head,
        dims.n_text_layer
    )

    # Mock inputs
    batch_size = 1
    # Create a KV cache with a large offset
    # Let's say we are near the end of the context
    current_offset = 440
    kv_cache = {
        "k": torch.randn(batch_size, current_offset, dims.n_text_state),
        "v": torch.randn(batch_size, current_offset, dims.n_text_state)
    }
    
    # Input tokens that will push us over the limit
    # 440 + 10 = 450 > 448
    x = torch.randint(0, dims.n_vocab, (batch_size, 10))
    xa = torch.randn(batch_size, dims.n_audio_ctx, dims.n_audio_state)

    print(f"Testing with offset {current_offset} and input length {x.shape[-1]}")
    print(f"Total length {current_offset + x.shape[-1]} > Context {dims.n_text_ctx}")

    try:
        decoder(x, xa, kv_cache=kv_cache)
        print("Success! No crash.")
    except RuntimeError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

if __name__ == "__main__":
    test_tensor_mismatch()
