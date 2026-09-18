"""Offline checks of the production methods; no AstrBot service or network needed."""

import ast
import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch


ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
constants = [node for node in TREE.body if isinstance(node, ast.Assign)]
plugin_class = next(
    node
    for node in TREE.body
    if isinstance(node, ast.ClassDef) and node.name == "IsItTrue"
)
plugin_class.decorator_list = []
for method in plugin_class.body:
    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
        method.decorator_list = [
            d
            for d in method.decorator_list
            if isinstance(d, ast.Name) and d.id in ("staticmethod", "classmethod")
        ]
module = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *constants,
        plugin_class,
    ],
    type_ignores=[],
)
namespace = {
    "re": re,
    "asyncio": asyncio,
    "datetime": datetime,
    "json": json,
    "logger": Mock(),
    "Star": object,
}
exec(
    compile(ast.fix_missing_locations(module), str(ROOT / "main.py"), "exec"), namespace
)
Plugin = namespace["IsItTrue"]


class TemporalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.plugin = Plugin.__new__(Plugin)
        self.calls = []

        async def chat(**kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                completion_text="CLAIM: 某产品今日发布\nSEARCH: 某产品 发布\nNEED_SEARCH: yes\nNOTE: 核验日期"
            )

        self.provider = SimpleNamespace(text_chat=chat)
        self.plugin._resolve_providers = lambda: [self.provider]
        self.plugin._provider_label = lambda p: "offline"
        self.plugin.enable_vision = True
        self.plugin.plan_prompt = "CUSTOM PLANNER"
        self.plugin.system_prompt = "CUSTOM FINAL"
        self.plugin.max_content_chars = 4000
        self.plugin.max_search_chars = 4000
        self.plugin.max_search_queries = 2
        self.plugin.true_label = "TRUE"
        self.plugin.false_label = "FALSE"
        self.plugin.unknown_label = "UNKNOWN"

    async def test_planner_and_final_payload_get_fresh_local_time(self):
        times = [
            datetime(2031, 5, 1, 23, 59, tzinfo=timezone(timedelta(hours=-4))),
            datetime(2031, 5, 2, 0, 1, tzinfo=timezone(timedelta(hours=-4))),
        ]
        # Mock local timezone conversion, not the machine's actual timezone.
        clock = Mock()
        clock.now.side_effect = [
            SimpleNamespace(astimezone=lambda t=t: t) for t in times
        ]
        with patch.dict(namespace, datetime=clock):
            plan = await self.plugin._plan_verification(
                text="long claim" * 15,
                images=["offline-image"],
                supplement="",
                source="message",
            )
            self.assertTrue(plan["need_search"])
            await self.plugin._chat_with_fallback(
                prompt_with_images="evidence",
                prompt_without_images="evidence",
                image_urls=[],
                system_prompt=self.plugin.system_prompt,
                stage="终判",
            )
        self.assertEqual(len(self.calls), 2)
        for call, original, stamp in zip(
            self.calls, ("CUSTOM PLANNER", "CUSTOM FINAL"), times
        ):
            self.assertTrue(call["system_prompt"].startswith(original))
            self.assertIn(stamp.isoformat(timespec="seconds"), call["system_prompt"])
            self.assertIn("unknown", call["system_prompt"])
            self.assertIn("知识截止", call["system_prompt"])
        self.assertEqual(self.plugin.system_prompt, "CUSTOM FINAL")

    async def test_short_text_keeps_search_and_final_time_without_planner_call(self):
        plan = await self.plugin._plan_verification(
            text="某产品今天发布了吗", images=[], supplement="", source="message"
        )
        self.assertTrue(plan["need_search"])
        self.assertEqual(self.calls, [])
        await self.plugin._chat_with_fallback(
            prompt_with_images="short",
            prompt_without_images="short",
            image_urls=[],
            system_prompt=self.plugin.system_prompt,
            stage="终判",
        )
        self.assertEqual(len(self.calls), 1)
        self.assertIn("本次核查时间", self.calls[0]["system_prompt"])

    async def test_search_failure_remains_explicitly_unverified(self):
        self.plugin.search_provider = "tavily"
        self.plugin._search_with = AsyncMock(
            side_effect=RuntimeError("offline failure")
        )
        evidence = await self.plugin._web_search("recent release")
        self.assertEqual(evidence, "")
        prompt = self.plugin._build_user_prompt(
            source="message",
            text="某产品今日发布",
            images=[],
            supplement="",
            search_block=evidence,
            notes=[],
            claim="某产品今日发布",
        )
        self.assertIn("无可用联网证据", prompt)
        self.assertIn("unknown", prompt)
        await self.plugin._chat_with_fallback(
            prompt_with_images=prompt,
            prompt_without_images=prompt,
            image_urls=[],
            system_prompt=self.plugin.system_prompt,
            stage="终判",
        )
        self.assertEqual(self.calls[0]["prompt"], prompt)

    async def test_image_retry_also_gets_temporal_guard(self):
        calls = []

        async def chat(**kwargs):
            calls.append(kwargs)
            if kwargs["image_urls"]:
                raise RuntimeError("offline vision failure")
            return SimpleNamespace(completion_text="unknown")

        self.provider.text_chat = chat
        await self.plugin._chat_with_fallback(
            prompt_with_images="image",
            prompt_without_images="no image",
            image_urls=["offline"],
            system_prompt="custom",
            stage="终判",
        )
        self.assertEqual(len(calls), 2)
        self.assertTrue(all("本次核查时间" in c["system_prompt"] for c in calls))

    def test_schema_defaults_match_and_uncertain_text_is_not_false(self):
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        self.assertEqual(
            schema["system_prompt"]["default"], namespace["DEFAULT_SYSTEM_PROMPT"]
        )
        self.assertEqual(
            schema["plan_prompt"]["default"], namespace["DEFAULT_PLAN_PROMPT"]
        )
        planner_prompt = self.plugin._dated_system_prompt(namespace["DEFAULT_PLAN_PROMPT"])
        self.assertIn("严格只输出四行", planner_prompt)
        self.assertIn("规划阶段保持原有检索协议，不作真假结论", planner_prompt)
        self.assertIn("最终判定时，时效主张没有可靠核验资料应为 unknown", planner_prompt)
        self.assertEqual(self.plugin._format_verdict("无法确认是否谣言"), "UNKNOWN")
        self.assertEqual(
            self.plugin._format_verdict("false\n可靠证据否定主张"),
            "FALSE\n可靠证据否定主张",
        )


if __name__ == "__main__":
    unittest.main()
