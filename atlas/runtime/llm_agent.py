"""LLMAgent — base class for agents backed by an LLM provider."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from atlas.llm.provider import LLMProvider, LLMResponse
from atlas.runtime.base import AgentBase

if TYPE_CHECKING:
    from atlas.contract.types import AgentContract
    from atlas.runtime.context import AgentContext


class LLMAgent(AgentBase):
    """Base class for agents that call an LLM.

    Subclasses implement build_prompt() and parse_response().
    The LLM provider is created once in on_startup() and reused
    across all execute() calls (warm slot model).

    Example::

        class MySummarizer(LLMAgent):
            def build_prompt(self, input_data):
                return f"Summarize: {input_data['text']}"

            def parse_response(self, response, input_data):
                return {"summary": response.text}
    """

    def __init__(self, contract: AgentContract, context: AgentContext) -> None:
        super().__init__(contract, context)
        self._provider: LLMProvider | None = None

    async def on_startup(self) -> None:
        """Create the LLM provider.

        If ``context.providers["llm_provider"]`` is set, uses that
        (dependency injection for testing / composition). Otherwise
        falls back to ``_create_provider()``.
        """
        injected = self.context.providers.get("llm_provider")
        if injected is not None:
            self._provider = injected
        else:
            self._provider = self._create_provider()

    def _create_provider(self) -> LLMProvider:
        """Create an LLM provider instance.

        Default: AnthropicProvider using the contract's model preference.
        Override this to use a different provider or custom configuration.
        """
        from atlas.llm.anthropic import AnthropicProvider, model_for_preference

        preference = self.contract.model.preference
        return AnthropicProvider(model=model_for_preference(preference))

    @abstractmethod
    def build_prompt(self, input_data: dict[str, Any]) -> str:
        """Build the prompt string from input data.

        Called before every LLM invocation. The input has already
        been validated against the contract's input schema.
        """
        ...

    @abstractmethod
    def parse_response(
        self, response: LLMResponse, input_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Parse the LLM response into the output dict.

        Must return a dict that conforms to the contract's output schema.
        """
        ...

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Build prompt → call LLM → parse response."""
        if not self._provider:
            raise RuntimeError("LLMAgent not started — on_startup() not called")
        prompt = self.build_prompt(input_data)
        response = await self._provider.complete(prompt)
        # Capture token/model info for tracing
        self.context.execution_metadata.update({
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "model": response.model,
        })
        return self.parse_response(response, input_data)
