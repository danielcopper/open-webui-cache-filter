"""
title: Anthropic Prompt Cache
author: daniel
version: 0.8.2
license: MIT
description: Displays Anthropic prompt cache statistics (hit/miss, tokens saved, cost savings) in the chat UI.

Requirements:
- Open WebUI: ENABLE_FORWARD_USER_INFO_HEADERS=true (sends x-openwebui-chat-id header)
- LiteLLM config:
    extra_spend_tag_headers:
      - "x-openwebui-chat-id"
    cache_control_injection_points:
      - location: message
        role: system
      - location: message
        index: -1

The filter queries LiteLLM spend logs to get cache statistics and displays them in the UI.
Note: Actual cache_control injection is done by LiteLLM, not this filter.
"""

from pydantic import BaseModel, Field
from typing import Optional, Callable, Awaitable
import asyncio
import copy
import logging
from datetime import datetime, timedelta

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
        litellm_url: str = Field(
            default="http://litellm:4000",
            description="LiteLLM proxy URL for querying spend logs"
        )
        litellm_api_key: str = Field(
            default="",
            description="LiteLLM master API key for spend logs access"
        )

    def __init__(self):
        self.type = "filter"  # Required for Open WebUI to call inlet/outlet
        self.citation = False  # Prevent Open WebUI from overriding our custom citations
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

        # Debug logging
        if self.valves.debug:
            log.info(f"[Anthropic Cache] Model: {model}, Breakpoints: {cache_points_added}")

        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> dict:
        """
        Query LiteLLM spend logs for cache statistics and display in UI.
        """
        if not self.valves.enabled:
            return body

        # Need API key to query spend logs
        if not self.valves.litellm_api_key:
            if self.valves.debug:
                log.info("[Anthropic Cache] No LiteLLM API key configured, skipping cache stats")
            return body

        # Get user ID and chat_id for filtering
        user_id = __user__.get("id") if __user__ else None
        chat_id = body.get("chat_id")

        if self.valves.debug:
            log.info(f"[Anthropic Cache] OUTLET chat_id: {chat_id}")

        if not user_id:
            return body

        try:
            import aiohttp

            # Wait for spend logs to be written (async delay in LiteLLM)
            await asyncio.sleep(2)

            # Query spend logs for today
            today = datetime.utcnow().strftime("%Y-%m-%d")
            tomorrow = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")

            url = f"{self.valves.litellm_url}/spend/logs"
            params = {
                "start_date": today,
                "end_date": tomorrow,
                "summarize": "false"
            }
            headers = {
                "Authorization": f"Bearer {self.valves.litellm_api_key}"
            }

            # Retry logic - spend logs may not be written immediately
            latest = None
            chat_id_tag = f"x-openwebui-chat-id: {chat_id}" if chat_id else None

            for attempt in range(3):
                async with aiohttp.ClientSession() as http_session:
                    async with http_session.get(url, params=params, headers=headers, timeout=5) as response:
                        if response.status != 200:
                            if self.valves.debug:
                                log.warning(f"[Anthropic Cache] LiteLLM API error: {response.status}")
                            return body
                        logs = await response.json()

                # Filter by chat_id tag (exact match) or fall back to user_id
                matching_logs = []
                for entry in logs:
                    if entry.get("user") != user_id:
                        continue
                    # Prefer exact chat_id match via request_tags
                    if chat_id_tag:
                        tags = entry.get("request_tags", [])
                        if chat_id_tag not in tags:
                            continue
                    matching_logs.append(entry)

                if matching_logs:
                    # ─────────────────────────────────────────────────────────────
                    # Heuristic to distinguish Main Response from Title Generation
                    # ─────────────────────────────────────────────────────────────
                    # Open WebUI makes parallel requests for each user message:
                    # 1. Main completion (answer to user) - more tokens, has cache hits
                    # 2. Title generation (chat title) - few tokens (~20-100), no cache hits
                    #
                    # Both have the same chat_id, so we need to identify the main response:
                    # - If cache_read > 0: definitely main response (title gen uses different prompt)
                    # - If cache_read = 0: take highest total_tokens (main > title gen)
                    #
                    # This works because:
                    # - Title gen has short output (~20 tokens for a title)
                    # - Main responses have more tokens (even short answers > title)
                    # - After cache expires, we still get the right entry via token count
                    # ─────────────────────────────────────────────────────────────

                    # Sort by endTime descending, take recent entries for this chat
                    matching_logs.sort(key=lambda x: x.get("endTime", ""), reverse=True)
                    recent_logs = matching_logs[:10]

                    # Priority 1: Entry with cache_read > 0 (definitely main response)
                    for entry in recent_logs:
                        usage = entry.get("metadata", {}).get("usage_object", {})
                        if (usage.get("cache_read_input_tokens", 0) or 0) > 0:
                            latest = entry
                            break

                    # Priority 2: Entry with highest total_tokens (main response > title gen)
                    if not latest:
                        latest = max(recent_logs, key=lambda x: x.get("total_tokens", 0))

                    break

                # Wait before retry
                if attempt < 2:
                    await asyncio.sleep(1)

            if not latest:
                if self.valves.debug:
                    log.info(f"[Anthropic Cache] No logs found for chat_id {chat_id}")
                return body

            # Extract cache stats
            usage = latest.get("metadata", {}).get("usage_object", {})
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_write = usage.get("cache_creation_input_tokens", 0)

            if self.valves.debug:
                log.info(f"[Anthropic Cache] Matched: tokens={latest.get('total_tokens')}, cache_read={cache_read}, cache_write={cache_write}")

            # Emit cache status
            if self.valves.show_cache_status and __event_emitter__:
                if cache_read > 0:
                    # Cache HIT - calculate money saved using actual model pricing
                    model_info = latest.get("metadata", {}).get("model_map_information", {}).get("model_map_value", {})
                    input_cost = model_info.get("input_cost_per_token", 0) or 0
                    cache_read_cost = model_info.get("cache_read_input_token_cost", 0) or 0

                    if input_cost > 0 and cache_read_cost > 0:
                        saved_dollars = cache_read * (input_cost - cache_read_cost)
                        if saved_dollars >= 0.01:
                            status_msg = f"Cache hit: {cache_read:,} (-${saved_dollars:.2f})"
                        else:
                            status_msg = f"Cache hit: {cache_read:,}"
                    else:
                        status_msg = f"Cache hit: {cache_read:,}"

                elif cache_write > 0:
                    # Cache WRITE - show tokens cached with TTL
                    status_msg = f"Cached: {cache_write:,} tokens (5m)"

                else:
                    # No cache activity
                    status_msg = "Cache expired"

                # Status at top (replaces "Cache active" from inlet)
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": status_msg,
                        "done": True
                    }
                })

        except ImportError:
            if self.valves.debug:
                log.warning("[Anthropic Cache] aiohttp not available")
        except Exception as e:
            if self.valves.debug:
                log.warning(f"[Anthropic Cache] Error querying spend logs: {e}")

        return body
