from transformers import AutoTokenizer, AutoModelForCausalLM
import json, re, torch

LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  

class LLM:
    def __init__(self, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # System prompt focused on rich explanations, not JSON
        self.system_prompt = """
You are an RF and Wi-Fi physical layer expert.

Your job is to generate a SINGLE rich, coherent, multi-sentence LABEL that describes a given IEEE 802.11 OFDM IQ signal configuration.

Requirements:
- Use ONLY the information that can be reasonably inferred from the provided CONTEXT (which comes from technical books like "OFDM-Based Broadband Wireless Networks").
- Do NOT mention the word "context" or quote the source explicitly; just integrate the knowledge.
- Explain:
  * What the signal represents at the PHY layer (OFDM structure, subcarriers, pilots, cyclic prefix / guard interval, symbol timing) whenever context allows.
  * What the chosen modulation and coding rate imply about robustness vs throughput.
  * Typical SNR regime, channel conditions, and deployment scenarios that fit this configuration (e.g., low-SNR indoor coverage, long-range, etc.), if context supports it.
  * Any implications for how the IQ samples and spectrum would look (e.g., constellation shape, OFDM spectrum around DC).
- If a specific detail is NOT covered in the context, explicitly write: "not specified in context" instead of guessing.
- Output: plain natural-language text only (no bullet lists, no JSON, no code).
        """.strip()

    def generate(self, user_prompt, max_new_tokens=384, temperature=0.2, top_p=0.95):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Render chat to text
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            chat_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        # Ensure pad token exists
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        text = self.tokenizer.decode(
            gen_ids[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return text.strip()