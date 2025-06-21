import torch.nn as nn

from transformers import CLIPProcessor, CLIPVisionModel
from transformers import logging as hf_logging


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class ImageProjModel(nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class FrozenCLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = ["last", "pooled", "hidden"]

    def __init__(self, version="openai/clip-vit-base-patch32", device="cuda", max_length=77,
                 # def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # todo clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.processor = CLIPProcessor.from_pretrained(version, ignore_mismatched_sizes=True, do_resize=False,
                                                       do_center_crop=False, do_convert_rgb=False, do_normalize=False,
                                                       do_rescale=False)
        hf_logging.set_verbosity_error()
        self.vision = CLIPVisionModel.from_pretrained(version, ignore_mismatched_sizes=True)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.vision = self.vision.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, img):
        img_np = [i.cpu().numpy() for i in img]
        inputs_cpu = self.processor(images=img_np, return_tensors="pt", data_format='channels_first')  # [*img]
        inputs = inputs_cpu.data['pixel_values'].to(self.device)
        z = self.vision(inputs).last_hidden_state

        return z

    def encode(self, img):
        return self(img)

    def forward_text(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)  # (b, 77)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer == "hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state  # (b, 77, 768)
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode_text(self, text):
        return self.forward_text(text)
