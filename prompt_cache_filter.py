"""
title: Prompt Cache Stats
author: daniel
version: 1.0.1
license: MIT
description: Displays prompt cache statistics (hit/miss, tokens saved, cost savings) for Anthropic and OpenAI.

Supported:
- Anthropic: 90% discount, manual cache_control via LiteLLM
- OpenAI: 50% discount, automatic caching (no config needed)

Requirements:
- Open WebUI: ENABLE_FORWARD_USER_INFO_HEADERS=true
- LiteLLM config:
    extra_spend_tag_headers:
      - "x-openwebui-chat-id"
    # For Anthropic only:
    cache_control_injection_points:
      - location: message
        role: system
      - location: message
        index: -1
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
            description="[Anthropic only] Cache TTL: '5m' (1.25x write) or '1h' (2x write)",
            json_schema_extra={"enum": ["5m", "1h"]}
        )
        cache_tools: bool = Field(
            default=True,
            description="[Anthropic only] Add cache_control to tools"
        )
        cache_system: bool = Field(
            default=True,
            description="[Anthropic only] Add cache_control to system message"
        )
        cache_last_user: bool = Field(
            default=True,
            description="[Anthropic only] Add cache_control to last user message"
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

    def _get_provider(self, model: str) -> str | None:
        """Get provider from model name"""
        model_lower = model.lower()
        # Anthropic
        if model_lower.startswith("anthropic/"):
            return "anthropic"
        if any(kw in model_lower for kw in ["claude", "sonnet", "opus", "haiku"]):
            return "anthropic"
        # OpenAI
        if model_lower.startswith("openai/"):
            return "openai"
        if any(kw in model_lower for kw in ["gpt-", "gpt_"]):
            return "openai"
        return None

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

        # Get provider from model name
        model = body.get("model", "")
        provider = self._get_provider(model)
        if not provider:
            return body

        cache_points_added = []

        # Anthropic: Manual cache_control injection
        if provider == "anthropic":
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

        # OpenAI: Caching is automatic, no injection needed

        # Debug logging
        if self.valves.debug:
            log.info(f"[Cache] {provider}: Model={model}, Breakpoints={cache_points_added}")

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

        # Get provider from model name
        model = body.get("model", "")
        provider = self._get_provider(model)
        if not provider:
            return body

        # Need API key to query spend logs
        if not self.valves.litellm_api_key:
            if self.valves.debug:
                log.info("[Cache] No LiteLLM API key configured, skipping cache stats")
            return body

        # Get user ID and chat_id for filtering
        user_id = __user__.get("id") if __user__ else None
        chat_id = body.get("chat_id")

        if self.valves.debug:
            log.info(f"[Cache] OUTLET provider={provider}, chat_id={chat_id}")

        if not user_id:
            return body

        try:
            import aiohttp

            # Show loading indicator while waiting for spend logs
            if self.valves.show_cache_status and __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "Checking cache...",
                        "done": False
                    }
                })

            # Record start time to filter out old log entries
            outlet_start_time = datetime.utcnow()

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

                    # Filter to entries from CURRENT request only (not old messages)
                    # Use outlet_start_time minus small buffer for clock skew
                    cutoff_time = (outlet_start_time - timedelta(seconds=5)).isoformat()
                    current_entries = [
                        entry for entry in matching_logs
                        if entry.get("endTime", "") >= cutoff_time
                    ]

                    # If no fresh entries, retry (logs might not be written yet)
                    if not current_entries:
                        if attempt < 2:
                            await asyncio.sleep(1)
                            continue
                        # Give up, use most recent entry as fallback
                        current_entries = matching_logs[:1]

                    # Sort by endTime descending (most recent first)
                    current_entries.sort(key=lambda x: x.get("endTime", ""), reverse=True)

                    # Among current entries, prefer one with cache activity (main response)
                    for entry in current_entries:
                        usage = entry.get("metadata", {}).get("usage_object", {})
                        # Anthropic: cache_read_input_tokens
                        # OpenAI: prompt_tokens_details.cached_tokens
                        anthropic_cache = (usage.get("cache_read_input_tokens", 0) or 0) > 0
                        prompt_details = usage.get("prompt_tokens_details", {}) or {}
                        openai_cache = (prompt_details.get("cached_tokens", 0) or 0) > 0
                        if anthropic_cache or openai_cache:
                            latest = entry
                            break

                    # Fallback: highest token count among current entries
                    if not latest and current_entries:
                        latest = max(current_entries, key=lambda x: x.get("total_tokens", 0))

                    break

                # Wait before retry
                if attempt < 2:
                    await asyncio.sleep(1)

            if not latest:
                if self.valves.debug:
                    log.info(f"[Cache] No logs found for chat_id {chat_id}")
                return body

            # Extract cache stats based on provider
            usage = latest.get("metadata", {}).get("usage_object", {})

            if provider == "openai":
                # OpenAI: prompt_tokens_details.cached_tokens
                prompt_details = usage.get("prompt_tokens_details", {}) or {}
                cache_read = prompt_details.get("cached_tokens", 0) or 0
                cache_write = 0  # OpenAI doesn't report cache writes
            else:
                # Anthropic: cache_read_input_tokens, cache_creation_input_tokens
                cache_read = usage.get("cache_read_input_tokens", 0) or 0
                cache_write = usage.get("cache_creation_input_tokens", 0) or 0

            if self.valves.debug:
                log.info(f"[Cache] {provider}: tokens={latest.get('total_tokens')}, cache_read={cache_read}, cache_write={cache_write}")

            # Emit cache status
            if self.valves.show_cache_status and __event_emitter__:
                if cache_read > 0:
                    # Cache HIT - calculate money saved
                    model_info = latest.get("metadata", {}).get("model_map_information", {}).get("model_map_value", {})
                    input_cost = model_info.get("input_cost_per_token", 0) or 0

                    if provider == "openai":
                        # OpenAI: 50% discount on cached tokens
                        if input_cost > 0:
                            saved_dollars = cache_read * input_cost * 0.5
                            if saved_dollars >= 0.01:
                                status_msg = f"Cache hit: {cache_read:,} (-${saved_dollars:.2f})"
                            else:
                                status_msg = f"Cache hit: {cache_read:,}"
                        else:
                            status_msg = f"Cache hit: {cache_read:,}"
                    else:
                        # Anthropic: use actual cache_read_cost from model info
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
                    # Cache WRITE (Anthropic only)
                    status_msg = f"Cached: {cache_write:,} tokens (5m)"

                else:
                    # No cache activity
                    status_msg = "Cache expired"

                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": status_msg,
                        "done": True
                    }
                })

        except ImportError:
            if self.valves.debug:
                log.warning("[Cache] aiohttp not available")
        except Exception as e:
            if self.valves.debug:
                log.warning(f"[Cache] Error querying spend logs: {e}")

        return body
