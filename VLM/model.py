import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, BitsAndBytesConfig
from accelerate import load_checkpoint_and_dispatch
import os
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, BitsAndBytesConfig
import os

class AntifakePrompt:
    """Plain inference implementation for detecting fake images."""

    def __init__(
        self,
        pretrained_path: str = "Salesforce/instructblip-vicuna-7b",
        finetuned_path: str = None,
        quant_4bit: bool = True,
        device_map: str = "auto"
    ):
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.pseudo_word = "[*]"
        self.prompt = f"Is this photo real {self.pseudo_word}?"

        # Initialize the processor with custom normalization
        self.processor = InstructBlipProcessor.from_pretrained(
            pretrained_path,
            image_mean=[0.4730, 0.4499, 0.4129],
            image_std=[0.2780, 0.2713, 0.2872]
        )

        # Load the model directly using from_pretrained.
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            pretrained_path,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )

        # Tie the weights as required before further operations.
        self.model.tie_weights()

        # Optionally load fine-tuned weights if provided.
        if finetuned_path and os.path.exists(finetuned_path):
            print(f"Loading fine-tuned weights from {finetuned_path}")
            checkpoint = torch.load(finetuned_path, map_location="cpu")
            self.model.load_state_dict(checkpoint, strict=False)

        # Set the model to evaluation mode.
        self.model.eval()

        # Generation parameters for inference.
        self.gen_kwargs = {
            "max_new_tokens": 5,
            "num_beams": 5,
            "do_sample": False,
            "temperature": 1.0,
            "synced_gpus": False  # Not needed for plain inference.
        }

    def preprocess_image(self, image_input):
        """Preprocess the input image for inference."""
        if isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                image = Image.open(BytesIO(requests.get(image_input).content)).convert('RGB')
            else:
                image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise ValueError("Unsupported image input type")
            
        return self.processor(images=image, text=self.prompt, return_tensors="pt")

    def predict(self, image_input):
        """Run inference on a single image input."""
        inputs = self.preprocess_image(image_input)
        # Move inputs to the same device as the model.
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **self.gen_kwargs)
        text = self.processor.decode(outputs[0], skip_special_tokens=True)
        return "yes" in text.lower()
        # return text

if __name__ == "__main__":
    # Initialize the detector for plain inference.
    detector = AntifakePrompt(
        finetuned_path="./weights/ckpt/COCO_150k_SD3_SD2IP_lama.pth",
        quant_4bit=True,
        device_map="auto"
    )
    
    # Provide an image path (or URL) for inference.
    image_path = "./Low_res_image.png"
    result = detector.predict(image_path)
    # print(f"{image_path}: {'Real' if result else 'Fake'}")
    print(result)
