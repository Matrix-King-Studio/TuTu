import torch

from transformers import AutoProcessor, AutoModelForCausalLM


class TuTu:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_id = "microsoft/Florence-2-base-ft"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True)

    def run_example(self, task_prompt, image, text_input=None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )

        return parsed_answer

    def caption(self, image):
        task_prompt = "<CAPTION>"
        return self.run_example(task_prompt, image)
    
    def detailed_caption(self, image):
        task_prompt = "<DETAILED_CAPTION>"
        return self.run_example(task_prompt, image)

    def more_detailed_caption(self, image):
        task_prompt = "<MORE_DETAILED_CAPTION>"
        return self.run_example(task_prompt, image)
    
    def object_detection(self, image):
        task_prompt = "<OD>"
        return self.run_example(task_prompt, image)
    
    def dense_region_caption(self, image):
        task_prompt = "<DENSE_REGION_CAPTION>"
        return self.run_example(task_prompt, image)

    def region_proposal(self, image):
        task_prompt = "<REGION_PROPOSAL>"
        return self.run_example(task_prompt, image)
    
    def phrase_grounding(self, image, text_input):
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        return self.run_example(task_prompt, image, text_input)

    def ocr(self, image):
        task_prompt = "<OCR>"
        return self.run_example(task_prompt, image)
    
    def ocr_with_region(self, image):
        task_prompt = "<OCR_WITH_REGION>"
        return self.run_example(task_prompt, image)
