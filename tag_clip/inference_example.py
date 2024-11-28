import torch
import open_clip

from tag_clip.utils.models import create_custom_model
from huggingface_hub import hf_hub_download
from tag_clip.utils.dataset import ImageProcessor


if __name__ == "__main__":
    repo_id = "dudcjs2779/anime-style-tag-clip"
    filename = "model.safetensors"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    downloaded_path  = hf_hub_download(repo_id=repo_id, filename=filename)
    model = create_custom_model("EVA02-B-16", "amp", downloaded_path, device)
    model.eval()
                    
    processor = ImageProcessor()
    tokenizer = open_clip.get_tokenizer('EVA02-B-16')

    image = processor("sample/Blue_Archive_Memorial_Lobby_Yuuka_(Sportswear).jpg").unsqueeze(0).to(device)
    tags = ["dress", "black suit", "gym uniform"]
    text = tokenizer(tags).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        scores = (100.0 * image_features @ text_features.T)
        probs = scores.softmax(dim=-1)

    [print(f"{tags[i]}\t score:{scores[0][i]:.4f}\t prob:{probs[0][i]:.4f}") for i in range(len(tags))]

