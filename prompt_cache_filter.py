"""
title: Prompt Cache Stats
author: daniel
version: 3.0.0
license: MIT
description: Displays prompt cache statistics (hit/miss, tokens saved, cost savings) for Anthropic and OpenAI.

=== HOW IT WORKS ===

This filter queries LiteLLM spend logs AFTER the LLM response completes.
If pricing info is missing from spend logs, it fetches from /model/info.

=== CACHING OVERVIEW ===

Anthropic Prompt Caching:
- Minimum tokens: 4096 (Opus 4.5), 1024 (Sonnet), 2048 (Haiku)
- TTL: 5 minutes (default) or 1 hour
- Read discount: 90% off input price

OpenAI Automatic Caching:
- Automatic for prompts > 1024 tokens
- 50% discount on cached tokens

=== LITELLM INTEGRATION ===

Works with LiteLLM's cache_control_injection_points.
When using LiteLLM proxy, disable filter-side injection:
- cache_system: false
- cache_last_user: false

Requirements:
- Open WebUI: ENABLE_FORWARD_USER_INFO_HEADERS=true
- LiteLLM: extra_spend_tag_headers: ["x-openwebui-chat-id"]
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
        priority: int = Field(default=0, description="Filter processing priority")
        enabled: bool = Field(default=True, description="Enable/disable prompt caching")
        cache_ttl: str = Field(
            default="5m",
            description="[Anthropic only] Cache TTL: '5m' or '1h'",
            json_schema_extra={"enum": ["5m", "1h"]},
        )
        cache_tools: bool = Field(
            default=True,
            description="[Anthropic only] Add cache_control to tools",
        )
        cache_system: bool = Field(
            default=False,
            description="[Anthropic only] Add cache_control to system",
        )
        cache_last_user: bool = Field(
            default=False,
            description="[Anthropic only] Add cache_control to last user",
        )
        show_cache_status: bool = Field(
            default=True, description="Show cache status in UI"
        )
        debug: bool = Field(default=False, description="Log debug information")
        litellm_url: str = Field(
            default="http://litellm:4000", description="LiteLLM proxy URL"
        )
        litellm_api_key: str = Field(default="", description="LiteLLM master API key")

    def __init__(self):
        self.type = "filter"
        self.citation = False
        self.valves = self.Valves()
        self._model_prices: dict = {}
        self._prices_fetched: bool = False

    # ── Provider detection ──────────────────────────────────────────────

    def _get_provider(self, model: str) -> str | None:
        model_lower = model.lower()
        if model_lower.startswith("anthropic/") or any(
            kw in model_lower for kw in ["claude", "sonnet", "opus", "haiku"]
        ):
            return "anthropic"
        if model_lower.startswith("openai/") or any(
            kw in model_lower for kw in ["gpt-", "gpt_", "o3", "o4"]
        ):
            return "openai"
        return None

    # ── Cache control helpers (inlet) ───────────────────────────────────

    def _get_cache_control(self) -> dict:
        if self.valves.cache_ttl == "1h":
            return {"type": "ephemeral", "ttl": "1h"}
        return {"type": "ephemeral"}

    def _add_cache_to_content(self, content) -> list:
        cache_control = self._get_cache_control()
        if isinstance(content, str):
            return [{"type": "text", "text": content, "cache_control": cache_control}]
        if isinstance(content, list) and len(content) > 0:
            content = copy.deepcopy(content)
            content[-1] = {**content[-1], "cache_control": cache_control}
            return content
        return content

    def _remove_existing_cache_control(self, messages: list) -> list:
        cleaned = []
        for msg in messages:
            msg = copy.deepcopy(msg)
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        block = {
                            k: v for k, v in block.items() if k != "cache_control"
                        }
                    new_content.append(block)
                msg["content"] = new_content
            cleaned.append(msg)
        return cleaned

    def _remove_cache_control_from_tools(self, tools: list) -> list:
        cleaned = []
        for tool in tools:
            tool = copy.deepcopy(tool)
            if "cache_control" in tool:
                del tool["cache_control"]
            cleaned.append(tool)
        return cleaned

    # ── Spend log helpers (outlet) ──────────────────────────────────────

    def _safe_get_usage(self, entry: dict) -> dict:
        return (entry.get("metadata") or {}).get("usage_object") or {}

    def _extract_cache_tokens(self, usage: dict, provider: str) -> tuple[int, int]:
        if provider == "openai":
            prompt_details = usage.get("prompt_tokens_details") or {}
            return (prompt_details.get("cached_tokens") or 0, 0)
        return (
            usage.get("cache_read_input_tokens") or 0,
            usage.get("cache_creation_input_tokens") or 0,
        )

    def _find_best_entry(
        self,
        logs: list,
        user_id: str,
        chat_id_tag: str | None,
        cutoff_time: str,
        provider: str,
    ) -> dict | None:
        """Find the best matching spend log entry for the current request."""
        # Filter by user, chat_id tag, and time window
        candidates = []
        skipped_user = 0
        skipped_tag = 0
        skipped_time = 0
        for entry in logs:
            if entry.get("user") != user_id:
                skipped_user += 1
                continue
            if chat_id_tag:
                tags = entry.get("request_tags") or []
                if chat_id_tag not in tags:
                    skipped_tag += 1
                    if self.valves.debug:
                        log.info(
                            f"[Cache]   skip tag: want={chat_id_tag}, "
                            f"got={tags}, model={entry.get('model')}"
                        )
                    continue
            if entry.get("endTime", "") < cutoff_time:
                skipped_time += 1
                continue
            candidates.append(entry)

        if self.valves.debug:
            log.info(
                f"[Cache] Filter: {len(logs)} logs -> {len(candidates)} candidates "
                f"(skip: user={skipped_user}, tag={skipped_tag}, time={skipped_time})"
            )

        if not candidates:
            return None

        # Sort by endTime descending (most recent first)
        candidates.sort(key=lambda x: x.get("endTime", ""), reverse=True)

        # Prefer entry with cache activity (= main response, not title generation)
        for entry in candidates:
            usage = self._safe_get_usage(entry)
            cache_read, cache_write = self._extract_cache_tokens(usage, provider)
            if cache_read > 0 or cache_write > 0:
                return entry

        # Fallback: entry with most output tokens (main response > title gen)
        return max(candidates, key=lambda x: x.get("total_tokens") or 0)

    def _get_spend_log_pricing(self, entry: dict) -> tuple[float, float]:
        """Extract per-token pricing from spend log metadata. Returns (input_cost, cache_read_cost)."""
        metadata = entry.get("metadata") or {}
        model_map = (metadata.get("model_map_information") or {}).get(
            "model_map_value"
        ) or {}
        input_cost = model_map.get("input_cost_per_token") or 0
        cache_read_cost = model_map.get("cache_read_input_token_cost") or 0
        return (input_cost, cache_read_cost)

    def _calculate_savings(
        self, cache_read: int, provider: str, input_cost: float, cache_read_cost: float
    ) -> float:
        if cache_read <= 0 or input_cost <= 0:
            return 0.0
        if provider == "openai":
            return cache_read * input_cost * 0.5
        if cache_read_cost > 0:
            return cache_read * (input_cost - cache_read_cost)
        return cache_read * input_cost * 0.9

    def _format_cache_status(
        self, cache_read: int, cache_write: int, saved_dollars: float
    ) -> str:
        if cache_read > 0:
            if saved_dollars >= 0.01:
                return f"Cache hit: {cache_read:,} (-${saved_dollars:.2f})"
            if saved_dollars > 0:
                return f"Cache hit: {cache_read:,} (-${saved_dollars:.4f})"
            return f"Cache hit: {cache_read:,}"
        if cache_write > 0:
            ttl = "1h" if self.valves.cache_ttl == "1h" else "5m"
            return f"Cached: {cache_write:,} tokens ({ttl})"
        return "Cache miss"

    # ── Model pricing from /model/info (fallback) ──────────────────────

    async def _fetch_model_prices(self, session) -> None:
        if self._prices_fetched:
            return
        try:
            url = f"{self.valves.litellm_url}/model/info"
            headers = {"Authorization": f"Bearer {self.valves.litellm_api_key}"}
            async with session.get(url, headers=headers, timeout=5) as response:
                if response.status != 200:
                    return
                data = await response.json()

            for model in data.get("data") or []:
                model_name = model.get("model_name", "")
                info = model.get("model_info") or {}
                input_cost = info.get("input_cost_per_token") or 0
                if not input_cost:
                    continue

                cache_read_cost = info.get("cache_read_input_token_cost") or (
                    input_cost * 0.1
                    if any(
                        kw in model_name.lower()
                        for kw in ["anthropic", "claude", "sonnet", "opus", "haiku"]
                    )
                    else input_cost * 0.5
                )

                self._model_prices[model_name] = {
                    "input_cost_per_token": input_cost,
                    "cache_read_cost_per_token": cache_read_cost,
                }

            self._prices_fetched = True
            if self.valves.debug:
                log.info(
                    f"[Cache] Loaded {len(self._model_prices)} model prices from /model/info"
                )
        except Exception as e:
            if self.valves.debug:
                log.warning(f"[Cache] Failed to fetch model prices: {e}")

    def _get_model_price(self, model: str, litellm_model: str = "") -> dict:
        # Exact match
        if model in self._model_prices:
            return self._model_prices[model]
        # With provider prefix
        for prefix in ["openai/", "anthropic/", ""]:
            key = f"{prefix}{model}"
            if key in self._model_prices:
                return self._model_prices[key]
        # LiteLLM model name
        if litellm_model:
            if litellm_model in self._model_prices:
                return self._model_prices[litellm_model]
            if "/" in litellm_model:
                base = litellm_model.split("/")[-1]
                for key in self._model_prices:
                    if base in key:
                        return self._model_prices[key]
        # Partial match
        model_lower = model.lower()
        for key, value in self._model_prices.items():
            if model_lower in key.lower() or key.lower() in model_lower:
                return value
        return {}

    # ── Inlet ───────────────────────────────────────────────────────────

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> dict:
        if not self.valves.enabled:
            return body

        model = body.get("model", "")
        provider = self._get_provider(model)
        if not provider or provider != "anthropic":
            return body

        cache_control = self._get_cache_control()

        if "messages" in body:
            body["messages"] = self._remove_existing_cache_control(body["messages"])
        if "tools" in body and body["tools"]:
            body["tools"] = self._remove_cache_control_from_tools(body["tools"])

        if self.valves.cache_tools and "tools" in body and body["tools"]:
            body["tools"][-1]["cache_control"] = cache_control

        if self.valves.cache_system and "messages" in body:
            for i, msg in enumerate(body["messages"]):
                if msg.get("role") == "system":
                    body["messages"][i]["content"] = self._add_cache_to_content(
                        msg["content"]
                    )
                    break

        if self.valves.cache_last_user and "messages" in body:
            for i in range(len(body["messages"]) - 1, -1, -1):
                if body["messages"][i].get("role") == "user":
                    body["messages"][i]["content"] = self._add_cache_to_content(
                        body["messages"][i]["content"]
                    )
                    break

        return body

    # ── Outlet ──────────────────────────────────────────────────────────

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __chat_id__: Optional[str] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> dict:
        if not self.valves.enabled or not self.valves.litellm_api_key:
            return body

        model = body.get("model", "")
        provider = self._get_provider(model)
        if not provider:
            return body

        user_id = (__user__ or {}).get("id")
        if not user_id:
            return body

        chat_id = __chat_id__ or body.get("chat_id")

        async def emit_status(description: str, done: bool = False):
            if self.valves.show_cache_status and __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": description, "done": done},
                    }
                )

        try:
            import aiohttp

            await emit_status("Checking cache...")

            outlet_start_time = datetime.utcnow()
            cutoff_time = (outlet_start_time - timedelta(seconds=5)).isoformat()
            chat_id_tag = f"x-openwebui-chat-id: {chat_id}" if chat_id else None

            today = datetime.utcnow().strftime("%Y-%m-%d")
            tomorrow = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
            url = f"{self.valves.litellm_url}/spend/logs"
            params = {
                "start_date": today,
                "end_date": tomorrow,
                "user_id": user_id,
                "summarize": "false",
            }
            headers = {"Authorization": f"Bearer {self.valves.litellm_api_key}"}

            latest = None

            async with aiohttp.ClientSession() as session:
                if not self._prices_fetched:
                    await self._fetch_model_prices(session)

                # Wait for LiteLLM to write the spend log (~3-6s after response)
                await asyncio.sleep(2.0)

                for attempt in range(1, 6):
                    if attempt > 1:
                        await emit_status(f"Checking cache ({attempt}/5)...")
                        await asyncio.sleep(2.0)

                    try:
                        async with session.get(
                            url, params=params, headers=headers, timeout=15
                        ) as response:
                            if response.status != 200:
                                log.warning(f"[Cache] API error: {response.status}")
                                await emit_status("Cache: API error", done=True)
                                return body
                            logs = await response.json()
                    except Exception as e:
                        log.warning(f"[Cache] Request error ({type(e).__name__}): {e}")
                        continue

                    latest = self._find_best_entry(
                        logs, user_id, chat_id_tag, cutoff_time, provider
                    )

                    if self.valves.debug:
                        log.info(
                            f"[Cache] Attempt {attempt}: {'found' if latest else 'no match'}, "
                            f"{len(logs)} total logs"
                        )

                    if latest:
                        break

            if not latest:
                if self.valves.debug:
                    log.info("[Cache] No log found after 5 attempts")
                await emit_status("Cache: No data", done=True)
                return body

            # Extract cache tokens
            usage = self._safe_get_usage(latest)
            cache_read, cache_write = self._extract_cache_tokens(usage, provider)

            # Get pricing: try spend log first, then /model/info fallback
            input_cost, cache_read_cost = self._get_spend_log_pricing(latest)

            if not input_cost:
                litellm_model = latest.get("model") or ""
                price_info = self._get_model_price(model, litellm_model)
                input_cost = price_info.get("input_cost_per_token") or 0
                cache_read_cost = price_info.get("cache_read_cost_per_token") or 0
                if self.valves.debug and input_cost:
                    log.info(f"[Cache] Using /model/info price for {model}")

            saved_dollars = self._calculate_savings(
                cache_read, provider, input_cost, cache_read_cost
            )
            status_msg = self._format_cache_status(cache_read, cache_write, saved_dollars)
            await emit_status(status_msg, done=True)

        except ImportError:
            log.warning("[Cache] aiohttp not available")
        except Exception as e:
            log.warning(f"[Cache] Error: {type(e).__name__}: {e}")
            if self.valves.debug:
                log.exception("[Cache] Full traceback:")
            await emit_status("Cache: Error", done=True)

        return body
