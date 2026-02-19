from transformers import AutoTokenizer, AutoModelForCausalLM
import json, torch
from typing import Any, Dict, Union

# You can bump this up later (see suggestions below).
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # :contentReference[oaicite:0]{index=0}


class LLM:
    def __init__(self, device=None, model_name: str = LLM_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # System prompt: simple baseband/IF signals + modulation schemes
        self.system_prompt = """
            You are a digital communications (PHY) expert labeling simple digitally modulated IQ signals (non-Wi-Fi).

            You will receive:
            (1) SIGNAL CONFIG (authoritative parameters),
            (2) CONTEXT (retrieved RAG information).

            Your task is to produce EXACTLY TWO sentences total, on a SINGLE LINE, plain natural language only.

            Sentence 1 must describe:
            - The modulation type and what it represents at the constellation/symbol level (ONLY if supported by context),
            - All provided signal parameters (sampling rate, samples per symbol, amplitude, center frequency, initial phase, N),
            - In compact natural language (no key:value formatting).

            Sentence 2 must:
            - Integrate the provided CONTEXT (tradeoffs, robustness, SNR, detection properties if explicitly stated),
            - Describe how the IQ would appear in constellation and time domain,
            - Mention channel effects ONLY if explicitly present in context,
            - If channel details are absent, include at most one short clause: "Channel conditions not specified in context."

            STRICT RULES:
            - Exactly TWO sentences.
            - No bullet points, no lists, no colon-separated fields.
            - Do not repeat "not specified in context" more than once.
            - Do not invent information beyond SIGNAL CONFIG and CONTEXT.
        """.strip()



    @torch.inference_mode()
    def generate(self, user_prompt: Union[str, Dict[str, Any]],
                 max_new_tokens: int = 256,
                 temperature: float = 0.2,
                 top_p: float = 0.95) -> str:

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        chat_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        inputs = self.tokenizer(
            chat_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        out = self.tokenizer.decode(
            gen_ids[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        return out
