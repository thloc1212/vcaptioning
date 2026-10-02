"""Exercise the custom T5 generation path with tiny weights before full YC2 eval."""

import torch
from transformers import T5Config
from transformers.modeling_outputs import BaseModelOutput

from model.modeling_t5 import T5ForConditionalGeneration


def main():
    config = T5Config(
        vocab_size=32,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
        use_cache=False,
    )
    model = T5ForConditionalGeneration(config).eval()
    encoder_outputs = BaseModelOutput(last_hidden_state=torch.zeros(1, 3, config.d_model))
    with torch.no_grad():
        output = model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            max_new_tokens=2,
            num_beams=2,
            use_cache=False,
        )
    assert output.ndim == 2 and output.shape[0] == 1, output.shape
    print("Custom T5 generation smoke test passed:", tuple(output.shape))


if __name__ == "__main__":
    main()
