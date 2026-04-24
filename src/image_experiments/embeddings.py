from __future__ import annotations

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


class ImageEmbedder:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Force safetensors on Hyperion while shared Torch is still below 2.6.
        self.model = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()

    def _extract_feature_tensor(self, output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output

        for attr in ("image_embeds", "pooler_output"):
            value = getattr(output, attr, None)
            if isinstance(value, torch.Tensor):
                return value

        hidden = getattr(output, "last_hidden_state", None)
        if isinstance(hidden, torch.Tensor):
            if hidden.ndim == 3:
                return hidden[:, 0]
            return hidden

        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            first = output[0]
            if first.ndim == 3:
                return first[:, 0]
            return first

        raise TypeError(f"Unsupported image feature output type: {type(output)!r}")

    def generate_embeddings(self, dataloader) -> tuple[torch.Tensor, torch.Tensor]:
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Generating embeddings"):
                # Note: Images are PIL if transform is not applied in Dataset, 
                # but standard practice is to handle processing in the loader.
                # If images are already tensors from Dataset:
                inputs = images.to(self.device)
                
                if hasattr(self.model, "get_image_features"):
                    features = self._extract_feature_tensor(
                        self.model.get_image_features(pixel_values=inputs)
                    )
                else:
                    features = self._extract_feature_tensor(self.model(pixel_values=inputs))
                
                all_embeddings.append(features.detach().cpu())
                all_labels.append(labels.detach().cpu())

        return torch.cat(all_embeddings), torch.cat(all_labels)
