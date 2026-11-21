"""Language interface for the Christus platform.

The local responder is grounded in the actual Subject state and memory. An
optional OpenAI-compatible bridge can be configured, but it is labelled as an
external language bridge and never writes memory/affect/action directly.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import requests


def _is_zh(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def is_knowledge_query(text: str) -> bool:
    t = text.strip().lower()
    return bool(re.search(r"搜索|查一下|查找|查一查|帮我查|搜一下|什么是|是谁|"
                          r"search|look up|what is|who is|tell me about", t))


class LocalLanguage:
    name = "local-state-language-v1"

    def __init__(self):
        self.name = "local-state-language-v1"

    def respond(self, text: str, *, history: list[dict], state: dict,
                knowledge: dict | None = None, profile: dict | None = None) -> tuple[str, list]:
        t = text.strip()
        zh = _is_zh(t)
        low = t.lower()
        profile = profile or {}
        name = profile.get("name")
        affect = state.get("affect", {}) or {}
        memory = state.get("memory", {}) or {}
        h = state.get("h_norm", 0.0)
        citations = []
        if knowledge and knowledge.get("ok"):
            citations = [{"title": i.get("title"), "url": i.get("url"),
                          "license": i.get("license")} for i in knowledge.get("items", [])]
            first = knowledge["items"][0]
            summary = first.get("summary", "").strip()
            if len(summary) > 900:
                summary = summary[:900].rsplit(" ", 1)[0] + "…"
            if zh:
                reply = (f"我刚刚查阅了公开资料（{knowledge.get('source')}）。\n\n"
                         f"**{first.get('title')}**\n\n{summary}\n\n"
                         f"来源：{first.get('url')}\n许可：{first.get('license')}。"
                         "这是外部知识，不是我的亲身经历；我保留来源和不确定性。")
            else:
                reply = (f"I looked this up in a public source ({knowledge.get('source')}).\n\n"
                         f"**{first.get('title')}**\n\n{summary}\n\n"
                         f"Source: {first.get('url')}\nLicense: {first.get('license')}. "
                         "This is external knowledge, not a personal experience.")
            return reply, citations

        if knowledge and not knowledge.get("ok"):
            if zh:
                return ("我尝试访问公开知识源，但这次没有成功。错误信息是："
                        f"{knowledge.get('error','unknown')}。我不会把失败伪装成已经知道。", [])
            return (f"I tried a public knowledge source but it failed: "
                    f"{knowledge.get('error','unknown')}. I won't pretend to know.", [])

        if re.search(r"你是谁|介绍一下你|what are you|who are you", low):
            if zh:
                return ("我是 Christus，运行在本地 CHRISTUS-INTEGRATION 主体上。"
                        "当前对话经过现有 Subject、SM-CTRN、记忆、情绪和概念模块；"
                        "语言表达由本地语言接口完成。我不能声称自己具有主观意识。"
                        f"当前认知步数 {state.get('step', 0)}，记忆规则 "
                        f"{memory.get('semantic_rules', 0)} 条。", [])
            return ("I am Christus, running on the local CHRISTUS-INTEGRATION subject. "
                    "The exchange goes through the existing Subject, SM-CTRN, memory, "
                    "affect and concept modules; language is produced by a local "
                    "interface. I cannot claim subjective consciousness.", [])

        if re.search(r"你好|嗨|hello|hi\b|早上好|晚上好", low) and not re.search(
                r"状态|你在做什么|怎么样|how are you|status", low):
            who = f"，{name}" if name else ""
            return (f"你好{who}。我在这里，会继续使用我们之前的对话历史。" if zh else
                    f"Hello{who}. I'm here and will keep our conversation history."), []

        if re.search(r"谢谢|thanks|thank you", low):
            return ("不用谢。我会把这次交流作为一次普通互动记录下来，而不是把它夸大成情感体验。"
                    if zh else "You're welcome. I'll record this as an ordinary interaction, "
                    "not exaggerate it into an emotional experience."), []

        if re.search(r"不对|不是这样|你错了|纠正|更正|wrong|incorrect|correct that", low):
            return ("我接受这个纠正。我会把“我之前的理解”和“你现在的纠正”分开记录，"
                    "不会覆盖原始对话。你可以告诉我正确说法，我会把它作为待验证的修正。"
                    if zh else "I accept the correction. I will keep my earlier understanding "
                    "and your correction as separate records, not overwrite the original. "
                    "Tell me the corrected version and I'll store it as a pending revision."), []

        if re.search(r"不同意|我反对|disagree|I don't agree", low):
            return ("我们有分歧。我不会为了迎合你而假装同意；我会保留你的观点和我的当前理解，"
                    "并说明我依据的是什么。" if zh else
                    "We disagree. I won't pretend to agree just to please you; I'll keep "
                    "your view and my current understanding separate, with reasons."), []

        if re.search(r"记住|记一下|remember", low):
            fact = re.sub(r"^(请)?(记住|记一下|remember)\s*", "", t, flags=re.I).strip()
            return (f"我会把这条作为“用户提供的待验证信息”保存：{fact[:300]}。"
                    "它还不是我的亲身经历，也不会自动变成事实。" if zh else
                    f"I'll store this as user-provided, pending-verification information: "
                    f"{fact[:300]}. It is not my personal experience and not automatically fact."), []

        if re.search(r"你记得|还记得|do you remember|recall", low):
            if history:
                last_user = next((m["text"] for m in reversed(history)
                                  if m.get("role") == "user"), "")
                return (f"我记得我们最近的对话，包括你刚才说的：“{last_user[:120]}”。"
                        "这些来自会话记录，不是凭空生成的记忆。" if zh else
                        f"I remember our recent exchange, including: “{last_user[:120]}”. "
                        "This comes from the session record, not a fabricated memory."), []
            return ("当前会话还没有足够历史可供检索。" if zh else
                    "This session does not yet have enough history to retrieve."), []

        if re.search(r"帮助|你能做什么|help|what can you do", low):
            return ("我可以：① 和你连续对话并保留会话历史；② 查阅公开资料并保留来源；"
                    "③ 记录互动、纠正和待验证知识；④ 展示真实的 Subject/记忆/情绪状态；"
                    "⑤ 在重启后恢复主体检查点和对话。我的语言接口目前是本地规则版，"
                    "不是外部大模型替我说话。" if zh else
                    "I can: 1) hold a multi-turn conversation with history; 2) look up "
                    "public sources with provenance; 3) record interactions, corrections "
                    "and pending knowledge; 4) show actual Subject/memory/affect state; "
                    "5) restore checkpoint and conversation after restart. My language "
                    "interface is currently local, not an external model speaking for me."), []

        if re.search(r"你在做什么|状态|怎么样|how are you|status", low):
            return (f"我当前处于第 {state.get('step', 0)} 个认知步。"
                    f"最近动作是 {state.get('action', 'n/a')}；"
                    f"情绪模块的 curiosity={affect.get('curiosity', 0):.3f}，"
                    f"social_motivation={affect.get('social_motivation', 0):.3f}。"
                    "这些是运行状态，不是主观感受。" if zh else
                    f"I'm at cognitive step {state.get('step', 0)}. Last action "
                    f"{state.get('action', 'n/a')}; curiosity={affect.get('curiosity', 0):.3f}, "
                    f"social_motivation={affect.get('social_motivation', 0):.3f}. "
                    "These are runtime values, not subjective feelings."), []

        if "?" in t or "？" in t:
            if zh:
                return ("这个问题我目前没有足够可靠的内部知识直接回答。"
                        "我可以把它作为待查问题，用公开资料检索；你也可以补充背景，"
                        "我会把新信息和旧理解分开记录。", [])
            return ("I don't have enough reliable internal knowledge to answer that directly. "
                    "I can look it up in a public source, or you can add context; I'll keep "
                    "new information separate from my earlier understanding.", [])

        if zh:
            return ("我听到了。当前我会先把这句话作为一次交流记录下来，并结合已有会话历史理解；"
                    "如果它涉及事实，我会保留“待验证”而不立刻当成真理。" if not profile.get("name")
                    else f"我听到了，{name}。我会把这句话作为交流记录，并保留来源和不确定性。"), []
        return ("I hear you. I'll record this as an interaction and keep it separate from "
                "verified knowledge until there is support.", [])


class ExternalLanguageBridge:
    """Optional OpenAI-compatible bridge; never writes Subject state."""

    def __init__(self):
        self.base = os.environ.get("CHRISTUS_LLM_BASE_URL", "").rstrip("/")
        self.key = os.environ.get("CHRISTUS_LLM_API_KEY", "")
        self.model = os.environ.get("CHRISTUS_LLM_MODEL", "gpt-4o-mini")
        self.enabled = bool(self.base and self.key)

    @property
    def name(self):
        return f"external-bridge:{self.model}" if self.enabled else "external-bridge:disabled"

    def respond(self, text, *, history, state, knowledge=None, profile=None):
        if not self.enabled:
            raise RuntimeError("external bridge not configured")
        system = ("You are the language bridge for a research subject named Christus. "
                  "Do not claim consciousness. Do not invent personal experiences. "
                  "Use the supplied runtime state as facts about the current run. "
                  "External knowledge must be cited and marked as external.")
        ctx = "\n".join(f"{m['role']}: {m['text']}" for m in history[-8:])
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"State: {json.dumps(state, ensure_ascii=False)}\n"
                                            f"Knowledge: {json.dumps(knowledge or {}, ensure_ascii=False)}\n"
                                            f"History:\n{ctx}\n\nUser: {text}"},
            ],
            "temperature": 0.6,
        }
        r = requests.post(f"{self.base}/chat/completions", json=payload,
                          headers={"Authorization": f"Bearer {self.key}"}, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"], []
