"""
title: Anthropic Prompt Cache
author: daniel
version: 0.2.0
license: MIT
description: Adds cache_control breakpoints to Anthropic/Claude requests for cost savings. Caches tools, system prompt, and last user message. Supports 5m and 1h TTL.
"""

from pydantic import BaseModel, Field
from typing import Optional, Callable, Awaitable
import copy
import logging

log = logging.getLogger(__name__)


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Filter processing priority (lower = earlier)"
        )
        enabled: bool = Field(
            default=True,
            description="Enable/disable prompt caching"
        )
        cache_ttl: str = Field(
            default="5m",
            description="Cache TTL: '5m' (default, 1.25x write cost) or '1h' (2x write cost)",
            json_schema_extra={"enum": ["5m", "1h"]}
        )
        cache_tools: bool = Field(
            default=True,
            description="Add cache_control to tools (Function Calling)"
        )
        cache_system: bool = Field(
            default=True,
            description="Add cache_control to system message"
        )
        cache_last_user: bool = Field(
            default=True,
            description="Add cache_control to last user message (incremental caching)"
        )
        show_cache_status: bool = Field(
            default=True,
            description="Show cache status in UI (inlet: breakpoints, outlet: hit/miss)"
        )
        debug: bool = Field(
            default=False,
            description="Log modified messages to console"
        )

    def __init__(self):
        self.type = "filter"  # Required for Open WebUI to call inlet/outlet
        self.valves = self.Valves()

    def _is_anthropic_model(self, model: str) -> bool:
        """Check if model is Anthropic/Claude"""
        model_lower = model.lower()
        # Match: claude, anthropic, sonnet, opus, haiku (Anthropic model names)
        anthropic_keywords = ["claude", "anthropic", "sonnet", "opus", "haiku"]
        return any(kw in model_lower for kw in anthropic_keywords)

    def _get_cache_control(self) -> dict:
        """Get cache_control dict based on TTL setting"""
        if self.valves.cache_ttl == "1h":
            return {"type": "ephemeral", "ttl": "1h"}
        return {"type": "ephemeral"}

    def _add_cache_to_content(self, content) -> list:
        """Convert content to list format and add cache_control to last block"""
        cache_control = self._get_cache_control()

        # String content -> convert to list
        if isinstance(content, str):
            return [{"type": "text", "text": content, "cache_control": cache_control}]

        # List content -> add to last block
        if isinstance(content, list) and len(content) > 0:
            content = copy.deepcopy(content)
            content[-1] = {**content[-1], "cache_control": cache_control}
            return content

        return content

    def _remove_existing_cache_control(self, messages: list) -> list:
        """Remove any existing cache_control to avoid duplicates/conflicts"""
        cleaned = []
        for msg in messages:
            msg = copy.deepcopy(msg)
            content = msg.get("content")

            if isinstance(content, list):
                new_content = []
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        block = {k: v for k, v in block.items() if k != "cache_control"}
                    new_content.append(block)
                msg["content"] = new_content

            cleaned.append(msg)
        return cleaned

    def _remove_cache_control_from_tools(self, tools: list) -> list:
        """Remove existing cache_control from tools"""
        cleaned = []
        for tool in tools:
            tool = copy.deepcopy(tool)
            if "cache_control" in tool:
                del tool["cache_control"]
            cleaned.append(tool)
        return cleaned

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> dict:
        # Skip if disabled
        if not self.valves.enabled:
            return body

        # Check if Anthropic model
        model = body.get("model", "")
        if not self._is_anthropic_model(model):
            return body

        cache_points_added = []
        cache_control = self._get_cache_control()

        # Clean existing cache_control from messages
        if "messages" in body:
            body["messages"] = self._remove_existing_cache_control(body["messages"])

        # Clean existing cache_control from tools
        if "tools" in body and body["tools"]:
            body["tools"] = self._remove_cache_control_from_tools(body["tools"])

        # Cache Hierarchy (Anthropic order): tools -> system -> messages
        # Max 4 breakpoints total

        # 1. Cache tools (last tool gets cache_control)
        if self.valves.cache_tools and "tools" in body and body["tools"]:
            body["tools"][-1]["cache_control"] = cache_control
            cache_points_added.append("Tools")

        # 2. Cache system message
        if self.valves.cache_system and "messages" in body:
            for i, msg in enumerate(body["messages"]):
                if msg.get("role") == "system":
                    body["messages"][i]["content"] = self._add_cache_to_content(msg["content"])
                    cache_points_added.append("System")
                    break

        # 3. Cache last user message (for incremental conversation caching)
        if self.valves.cache_last_user and "messages" in body:
            for i in range(len(body["messages"]) - 1, -1, -1):
                if body["messages"][i].get("role") == "user":
                    body["messages"][i]["content"] = self._add_cache_to_content(
                        body["messages"][i]["content"]
                    )
                    cache_points_added.append("User")
                    break

        # Emit status event (inlet: show what we're caching)
        if self.valves.show_cache_status and __event_emitter__ and cache_points_added:
            points_str = " + ".join(cache_points_added)
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"Cache: {points_str} | TTL: {self.valves.cache_ttl}",
                    "done": False
                }
            })

        # Debug logging
        if self.valves.debug:
            import json
            log.info(f"[Anthropic Cache] Model: {model}")
            log.info(f"[Anthropic Cache] Breakpoints: {cache_points_added}")
            if "tools" in body:
                log.info(f"[Anthropic Cache] Tools: {json.dumps(body.get('tools', []), indent=2)}")
            log.info(f"[Anthropic Cache] Messages: {json.dumps(body.get('messages', []), indent=2)}")

        return body

    async def stream(
        self,
        event: dict,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> dict:
        """
        Intercept streaming chunks to catch usage data in final chunk.
        Attempts to emit cache stats via __event_emitter__ (experimental).
        """
        if not self.valves.enabled or not self.valves.show_cache_status:
            return event

        # Check for final chunk with usage data
        choices = event.get("choices", [])
        if choices and choices[0].get("finish_reason") == "stop":
            usage = event.get("usage")
            if usage and __event_emitter__:
                cache_read = usage.get("cache_read_input_tokens", 0)
                cache_write = usage.get("cache_creation_input_tokens", 0)

                if cache_read > 0:
                    savings = int(cache_read * 0.9)
                    status_msg = f"Cache HIT: {cache_read:,} tokens (saved ~{savings:,})"
                elif cache_write > 0:
                    status_msg = f"Cache WRITE: {cache_write:,} tokens"
                else:
                    return event

                try:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": status_msg, "done": True}
                    })
                except Exception:
                    pass  # Silently fail if event_emitter not available

        return event

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> dict:
        """
        Log cache hit/miss statistics for debugging.
        Note: UI status updates are handled by stream() function.
        """
        if not self.valves.enabled or not self.valves.debug:
            return body

        # Try to extract usage stats from response
        usage = body.get("usage")
        if not usage and "response" in body and isinstance(body["response"], dict):
            usage = body["response"].get("usage")

        if usage:
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_write = usage.get("cache_creation_input_tokens", 0)
            input_tokens = usage.get("input_tokens", 0)

            if cache_read > 0:
                savings = int(cache_read * 0.9)
                log.info(f"[Anthropic Cache] HIT: {cache_read:,} tokens (saved ~{savings:,})")
            elif cache_write > 0:
                log.info(f"[Anthropic Cache] WRITE: {cache_write:,} tokens cached")
            else:
                log.info(f"[Anthropic Cache] No cache (input: {input_tokens:,} tokens)")

        return body
