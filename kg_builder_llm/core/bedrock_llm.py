"""Minimal Bedrock LLM wrapper matching neo4j-graphrag's OpenAILLM interface."""

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class _LLMResponse:
    content: str


class BedrockLLM:
    """Calls AWS Bedrock Converse API; interface matches neo4j-graphrag OpenAILLM."""

    def __init__(self, model_id: str, region: str, temperature: float = 0.1):
        import boto3
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.temperature = temperature

    def invoke(self, prompt: str) -> _LLMResponse:
        response = self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": self.temperature},
            additionalModelRequestFields={"thinking": {"type": "enabled", "budget_tokens": 4096}},
        )
        # Response may contain a thinking block followed by the text block; take the text.
        content_blocks = response["output"]["message"]["content"]
        text = next(b["text"] for b in content_blocks if "text" in b)
        return _LLMResponse(content=text)

    async def ainvoke(self, prompt: str) -> _LLMResponse:
        return await asyncio.get_event_loop().run_in_executor(None, self.invoke, prompt)
