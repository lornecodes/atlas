"""Translator agent — stub that wraps text with language markers."""

from atlas.runtime.base import AgentBase


class TranslatorAgent(AgentBase):
    async def execute(self, input_data: dict) -> dict:
        text = input_data["text"]
        target_lang = input_data["target_lang"]
        return {
            "translated_text": f"[{target_lang}] {text}",
            "source_lang": "en",
            "target_lang": target_lang,
        }
