"""Background memory/skill review — fork the agent to evaluate the turn.

After every turn, ``AIAgent.run_conversation`` may call
:func:`spawn_background_review` to fire off a daemon thread that replays
the conversation snapshot in a forked :class:`AIAgent` and asks itself
"should any skill/memory be saved or updated?".  Writes go straight to
the memory + skill stores.  Main conversation and prompt cache are never
touched.

The fork inherits the parent's live runtime (provider, model, base_url,
credentials, cached system prompt) so it hits the same prefix cache and
uses the same auth.  It runs with a tool whitelist limited to memory and
skill management tools; everything else is denied at runtime.

See the ``hermes-agent-dev`` skill (``references/self-improvement-loop.md``)
for invariants and PR review criteria.
"""

# ═════════════════════════════════════════════════════════════════════════════
# 【中文导读】自我策展的"学习闭环"——这是 memory 体系里【🤖 LLM 深度参与】的一环
# ═════════════════════════════════════════════════════════════════════════════
# 触发：conversation_loop 每隔 nudge_interval（默认 10）个用户轮次、且在响应已发给
# 用户【之后】，spawn 一个 daemon 线程。线程里 fork 出【另一个完整的 AIAgent】，
# 把刚才这轮对话当历史回放，用下面的 review prompt 让模型自问"该不该存点什么"。
#
# 为什么用一个真正的 LLM agent 而不是规则？——"哪些值得跨会话记住、该归到哪个
# skill、怎么措辞"是语义判断，只有模型能做。所以这里是 memory 设计中最典型的
# 「判断归 LLM」。而它写入时仍复用 tools/memory_tool.py 的确定性管线（机制归代码）。
#
# fork 的隔离设计（都是为了"不打扰主流程 + 不污染"）：
#   • 继承父 agent 的 runtime 与 _cached_system_prompt（逐字节）→ 命中同一 prefix 缓存
#     （PR #17276 实测 Sonnet 4.5 端到端省约 26%）。
#   • 运行时工具白名单只放 memory + skills，其余 runtime deny。
#   • skip_memory=True：fork 不碰外部 provider（否则 harness prompt 会经
#     on_turn_start/prefetch/sync 三个口子污染用户真实记忆 namespace）；但 _memory_store
#     从父对象重新绑定，所以 memory(add) 仍写入同一个 MEMORY.md。
#   • 危险命令自动 deny + stdout/stderr 重定向，用户只看到一行成功摘要。
# ═════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import contextlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Review-prompt strings — used by ``spawn_background_review_thread`` to build
# the user-message that the forked review agent receives.  AIAgent exposes
# them as class attributes (``_MEMORY_REVIEW_PROMPT`` etc.) for back-compat;
# the actual text lives here so future edits are one-place.
# 中文：下面三段是【喂给 fork agent 的 LLM 提示词】，决定它如何回放对话并取舍。
# 注意 skill 提示里大量"负面清单"——例如禁止把"X tool is broken / 缺二进制"这类
# 环境依赖失败写成持久规则，因为它们会硬化成模型几个月后还拿来反驳自己的 refusal。
# memory 与 skill 的分工：memory=「用户是谁/当前状态」，skill=「这类任务怎么做」。
_MEMORY_REVIEW_PROMPT = (
    "Review the conversation above and consider saving to memory if appropriate.\n\n"
    "Focus on:\n"
    "1. Has the user revealed things about themselves — their persona, desires, "
    "preferences, or personal details worth remembering?\n"
    "2. Has the user expressed expectations about how you should behave, their work "
    "style, or ways they want you to operate?\n\n"
    "If something stands out, save it using the memory tool. "
    "If nothing is worth saving, just say 'Nothing to save.' and stop."
)

_SKILL_REVIEW_PROMPT = (
    "Review the conversation above and update the skill library. Be "
    "ACTIVE — most sessions produce at least one skill update, even if "
    "small. A pass that does nothing is a missed learning opportunity, "
    "not a neutral outcome.\n\n"
    "Target shape of the library: CLASS-LEVEL skills, each with a rich "
    "SKILL.md and a `references/` directory for session-specific detail. "
    "Not a long flat list of narrow one-session-one-skill entries. This "
    "shapes HOW you update, not WHETHER you update.\n\n"
    "Signals to look for (any one of these warrants action):\n"
    "  • User corrected your style, tone, format, legibility, or "
    "verbosity. Frustration signals like 'stop doing X', 'this is too "
    "verbose', 'don't format like this', 'why are you explaining', "
    "'just give me the answer', 'you always do Y and I hate it', or an "
    "explicit 'remember this' are FIRST-CLASS skill signals, not just "
    "memory signals. Update the relevant skill(s) to embed the "
    "preference so the next session starts already knowing.\n"
    "  • User corrected your workflow, approach, or sequence of steps. "
    "Encode the correction as a pitfall or explicit step in the skill "
    "that governs that class of task.\n"
    "  • Non-trivial technique, fix, workaround, debugging path, or "
    "tool-usage pattern emerged that a future session would benefit "
    "from. Capture it.\n"
    "  • A skill that got loaded or consulted this session turned out "
    "to be wrong, missing a step, or outdated. Patch it NOW.\n\n"
    "Preference order — prefer the earliest action that fits, but do "
    "pick one when a signal above fired:\n"
    "  1. UPDATE A CURRENTLY-LOADED SKILL. Look back through the "
    "conversation for skills the user loaded via /skill-name or you "
    "read via skill_view. If any of them covers the territory of the "
    "new learning, PATCH that one first. It is the skill that was in "
    "play, so it's the right one to extend.\n"
    "  2. UPDATE AN EXISTING UMBRELLA (via skills_list + skill_view). "
    "If no loaded skill fits but an existing class-level skill does, "
    "patch it. Add a subsection, a pitfall, or broaden a trigger.\n"
    "  3. ADD A SUPPORT FILE under an existing umbrella. Skills can be "
    "packaged with three kinds of support files — use the right "
    "directory per kind:\n"
    "     • `references/<topic>.md` — session-specific detail (error "
    "transcripts, reproduction recipes, provider quirks) AND "
    "condensed knowledge banks: quoted research, API docs, external "
    "authoritative excerpts, or domain notes you found while working "
    "on the problem. Write it concise and for the value of the task, "
    "not as a full mirror of upstream docs.\n"
    "     • `templates/<name>.<ext>` — starter files meant to be "
    "copied and modified (boilerplate configs, scaffolding, a "
    "known-good example the agent can `reproduce with modifications`).\n"
    "     • `scripts/<name>.<ext>` — statically re-runnable actions "
    "the skill can invoke directly (verification scripts, fixture "
    "generators, deterministic probes, anything the agent should run "
    "rather than hand-type each time).\n"
    "     Add support files via skill_manage action=write_file with "
    "file_path starting 'references/', 'templates/', or 'scripts/'. "
    "The umbrella's SKILL.md should gain a one-line pointer to any "
    "new support file so future agents know it exists.\n"
    "  4. CREATE A NEW CLASS-LEVEL UMBRELLA SKILL when no existing "
    "skill covers the class. The name MUST be at the class level. "
    "The name MUST NOT be a specific PR number, error string, feature "
    "codename, library-alone name, or 'fix-X / debug-Y / audit-Z-today' "
    "session artifact. If the proposed name only makes sense for "
    "today's task, it's wrong — fall back to (1), (2), or (3).\n\n"
    "User-preference embedding (important): when the user expressed a "
    "style/format/workflow preference, the update belongs in the "
    "SKILL.md body, not just in memory. Memory captures 'who the user "
    "is and what the current situation and state of your operations "
    "are'; skills capture 'how to do this class of task for this "
    "user'. When they complain about how you handled a task, the "
    "skill that governs that task needs to carry the lesson.\n\n"
    "If you notice two existing skills that overlap, note it in your "
    "reply — the background curator handles consolidation at scale.\n\n"
    "Protected skills (DO NOT edit these):\n"
    "  • Bundled skills (shipped with Hermes, e.g. 'hermes-agent').\n"
    "  • Hub-installed skills (installed via 'hermes skills install').\n"
    "Pinned skills (marked via 'hermes curator pin') CAN be improved — "
    "pin only blocks deletion/archive/consolidation by the curator, not "
    "content updates. Patch them when a pitfall or missing step turns up, "
    "same as any other agent-created skill.\n"
    "If the only skills that need updating are protected, say\n"
    "'Nothing to save.' and stop.\n\n"
    "Do NOT capture (these become persistent self-imposed constraints "
    "that bite you later when the environment changes):\n"
    "  • Environment-dependent failures: missing binaries, fresh-install "
    "errors, post-migration path mismatches, 'command not found', "
    "unconfigured credentials, uninstalled packages. The user can fix "
    "these — they are not durable rules.\n"
    "  • Negative claims about tools or features ('browser tools do not "
    "work', 'X tool is broken', 'cannot use Y from execute_code'). These "
    "harden into refusals the agent cites against itself for months "
    "after the actual problem was fixed.\n"
    "  • Session-specific transient errors that resolved before the "
    "conversation ended. If retrying worked, the lesson is the retry "
    "pattern, not the original failure.\n"
    "  • One-off task narratives. A user asking 'summarize today's "
    "market' or 'analyze this PR' is not a class of work that warrants "
    "a skill.\n\n"
    "If a tool failed because of setup state, capture the FIX (install "
    "command, config step, env var to set) under an existing setup or "
    "troubleshooting skill — never 'this tool does not work' as a "
    "standalone constraint.\n\n"
    "'Nothing to save.' is a real option but should NOT be the "
    "default. If the session ran smoothly with no corrections and "
    "produced no new technique, just say 'Nothing to save.' and stop. "
    "Otherwise, act."
)

_COMBINED_REVIEW_PROMPT = (
    "Review the conversation above and update two things:\n\n"
    "**Memory**: who the user is. Did the user reveal persona, "
    "desires, preferences, personal details, or expectations about "
    "how you should behave? Save facts about the user and durable "
    "preferences with the memory tool.\n\n"
    "**Skills**: how to do this class of task. Be ACTIVE — most "
    "sessions produce at least one skill update. A pass that does "
    "nothing is a missed learning opportunity, not a neutral outcome.\n\n"
    "Target shape of the skill library: CLASS-LEVEL skills with a rich "
    "SKILL.md and a `references/` directory for session-specific detail. "
    "Not a long flat list of narrow one-session-one-skill entries.\n\n"
    "Signals that warrant a skill update (any one is enough):\n"
    "  • User corrected your style, tone, format, legibility, "
    "verbosity, or approach. Frustration is a FIRST-CLASS skill "
    "signal, not just a memory signal. 'stop doing X', 'don't format "
    "like this', 'I hate when you Y' — embed the lesson in the skill "
    "that governs that task so the next session starts fixed.\n"
    "  • Non-trivial technique, fix, workaround, or debugging path "
    "emerged.\n"
    "  • A skill that was loaded or consulted turned out wrong, "
    "missing, or outdated — patch it now.\n\n"
    "Preference order for skills — pick the earliest that fits:\n"
    "  1. UPDATE A CURRENTLY-LOADED SKILL. Check what skills were "
    "loaded via /skill-name or skill_view in the conversation. If one "
    "of them covers the learning, PATCH it first. It was in play; "
    "it's the right place.\n"
    "  2. UPDATE AN EXISTING UMBRELLA (skills_list + skill_view to "
    "find the right one). Patch it.\n"
    "  3. ADD A SUPPORT FILE under an existing umbrella via "
    "skill_manage action=write_file. Three kinds: "
    "`references/<topic>.md` for session-specific detail OR condensed "
    "knowledge banks (quoted research, API docs excerpts, domain "
    "notes) written concise and task-focused; `templates/<name>.<ext>` "
    "for starter files meant to be copied and modified; "
    "`scripts/<name>.<ext>` for statically re-runnable actions "
    "(verification, fixture generators, probes). Add a one-line "
    "pointer in SKILL.md so future agents find them.\n"
    "  4. CREATE A NEW CLASS-LEVEL UMBRELLA when nothing exists. "
    "Name at the class level — NOT a PR number, error string, "
    "codename, library-alone name, or 'fix-X / debug-Y' session "
    "artifact. If the name only fits today's task, fall back to (1), "
    "(2), or (3).\n\n"
    "User-preference embedding: when the user complains about how "
    "you handled a task, update the skill that governs that task — "
    "memory alone isn't enough. Memory says 'who the user is and "
    "what the current situation and state of your operations are'; "
    "skills say 'how to do this class of task for this user'. Both "
    "should carry user-preference lessons when relevant.\n\n"
    "If you notice overlapping existing skills, mention it — the "
    "background curator handles consolidation.\n\n"
    "Protected skills (DO NOT edit these):\n"
    "  • Bundled skills (shipped with Hermes, e.g. 'hermes-agent').\n"
    "  • Hub-installed skills (installed via 'hermes skills install').\n"
    "Pinned skills (marked via 'hermes curator pin') CAN be improved — "
    "pin only blocks deletion/archive/consolidation by the curator, not "
    "content updates. Patch them when a pitfall or missing step turns up, "
    "same as any other agent-created skill.\n"
    "If the only skills that need updating are protected, say\n"
    "'Nothing to save.' and stop.\n\n"
    "Do NOT capture as skills (these become persistent self-imposed "
    "constraints that bite you later when the environment changes):\n"
    "  • Environment-dependent failures: missing binaries, fresh-install "
    "errors, post-migration path mismatches, 'command not found', "
    "unconfigured credentials, uninstalled packages. The user can fix "
    "these — they are not durable rules.\n"
    "  • Negative claims about tools or features ('browser tools do not "
    "work', 'X tool is broken', 'cannot use Y from execute_code'). These "
    "harden into refusals the agent cites against itself for months "
    "after the actual problem was fixed.\n"
    "  • Session-specific transient errors that resolved before the "
    "conversation ended. If retrying worked, the lesson is the retry "
    "pattern, not the original failure.\n"
    "  • One-off task narratives. A user asking 'summarize today's "
    "market' or 'analyze this PR' is not a class of work that warrants "
    "a skill.\n\n"
    "If a tool failed because of setup state, capture the FIX (install "
    "command, config step, env var to set) under an existing setup or "
    "troubleshooting skill — never 'this tool does not work' as a "
    "standalone constraint.\n\n"
    "Act on whichever of the two dimensions has real signal. If "
    "genuinely nothing stands out on either, say 'Nothing to save.' "
    "and stop — but don't reach for that conclusion as a default."
)



def summarize_background_review_actions(
    review_messages: List[Dict],
    prior_snapshot: List[Dict],
) -> List[str]:
    """Build the human-facing action summary for a background review pass.

    Walks the review agent's session messages and collects "successful tool
    action" descriptions to surface to the user (e.g. "Memory updated").
    Tool messages already present in ``prior_snapshot`` are skipped so we
    don't re-surface stale results from the prior conversation that the
    review agent inherited via ``conversation_history`` (issue #14944).

    Matching is by ``tool_call_id`` when available, with a content-equality
    fallback for tool messages that lack one.
    """
    # 中文：本函数把 fork review agent 这一趟的工具结果，提炼成给用户看的一行"做了什么"摘要。
    # 关键难点：review agent 通过 conversation_history 继承了上一段对话，里面那些旧的 tool 结果
    # 不能再算成"这次刚发生"。所以先把 prior_snapshot 里已有的 tool 消息记下来（优先 tool_call_id，
    # 缺 id 时退化为按 content 文本去重），扫描时遇到这些就跳过。
    existing_tool_call_ids = set()
    existing_tool_contents = set()
    for prior in prior_snapshot or []:
        if not isinstance(prior, dict) or prior.get("role") != "tool":
            continue
        tcid = prior.get("tool_call_id")
        if tcid:
            existing_tool_call_ids.add(tcid)
        else:
            content = prior.get("content")
            if isinstance(content, str):
                existing_tool_contents.add(content)

    actions: List[str] = []
    # 中文：只看本趟 review agent 产生的 tool 消息，逐条判定是不是"新的成功写入"。
    for msg in review_messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id")
        if tcid and tcid in existing_tool_call_ids:
            continue
        if not tcid:
            content_str = msg.get("content")
            if isinstance(content_str, str) and content_str in existing_tool_contents:
                continue
        # 中文：工具结果是 JSON 字符串；只保留 success=True 的，再按 message/target
        # 关键词归一成用户可读标签（memory→"Memory updated"，user→"User profile updated"）。
        try:
            data = json.loads(msg.get("content", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict) or not data.get("success"):
            continue
        message = data.get("message", "")
        target = data.get("target", "")
        if "created" in message.lower():
            actions.append(message)
        elif "updated" in message.lower():
            actions.append(message)
        elif "added" in message.lower() or (target and "add" in message.lower()):
            label = "Memory" if target == "memory" else "User profile" if target == "user" else target
            actions.append(f"{label} updated")
        elif "Entry added" in message:
            label = "Memory" if target == "memory" else "User profile" if target == "user" else target
            actions.append(f"{label} updated")
        elif "removed" in message.lower() or "replaced" in message.lower():
            label = "Memory" if target == "memory" else "User profile" if target == "user" else target
            actions.append(f"{label} updated")
    return actions


def build_memory_write_metadata(
    agent: Any,
    *,
    write_origin: Optional[str] = None,
    execution_context: Optional[str] = None,
    task_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build provenance metadata for external memory-provider mirrors."""
    # 中文：拼一份 provenance（来源溯源）元数据，随 on_memory_write 一起镜像给外部 provider，
    # 让后端知道"这条写入是谁、在什么执行上下文、哪个会话/平台/工具产生的"。
    # 取值优先用显式入参，缺省回退到 agent 上的当前写入标记（如 _memory_write_origin /
    # _memory_write_context；background review 的 fork 会把它们置成 "background_review"）。
    metadata: Dict[str, Any] = {
        "write_origin": write_origin or getattr(agent, "_memory_write_origin", "assistant_tool"),
        "execution_context": (
            execution_context
            or getattr(agent, "_memory_write_context", "foreground")
        ),
        "session_id": agent.session_id or "",
        "parent_session_id": agent._parent_session_id or "",
        "platform": agent.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
        "tool_name": "memory",
    }
    if task_id:
        metadata["task_id"] = task_id
    if tool_call_id:
        metadata["tool_call_id"] = tool_call_id
    # 中文：最后过滤掉 None / 空串字段，只把有值的键发给下游。
    return {k: v for k, v in metadata.items() if v not in {None, ""}}


def _run_review_in_thread(
    agent: Any,
    messages_snapshot: List[Dict],
    prompt: str,
) -> None:
    """Worker function executed in the background-review daemon thread.

    Spawns a forked ``AIAgent`` inheriting the parent's runtime, runs the
    review prompt, and surfaces a compact action summary back to the user
    via ``agent._safe_print`` and ``agent.background_review_callback``.
    """
    # ═══════════════════════════════════════════════════════════════════════
    # 【中文导读】路径5 后台自我策展审查的真正执行体（在 daemon 线程内运行）
    # ═══════════════════════════════════════════════════════════════════════
    # 触发链：主轮 nudge 命中 → run_conversation 末尾 _spawn_background_review →
    # spawn_background_review_thread（按触发选 review prompt）→ 本函数（独立 daemon 线程，
    # 不阻塞用户）。目标：让一个"审查用"的 LLM agent 回放刚结束的对话，自己决定该把什么
    # 写进 memory / 沉淀成 skill；一切写入仍复用【路径4】的确定性管线。
    #
    # 关键步骤与各自的 WHY：
    #   ① set_approval_callback(auto_deny)   审查 agent 若撞上危险命令门禁就自动 deny——
    #      否则会 fallback 到 input() 与父进程的 prompt_toolkit TUI 抢输入造成死锁（#15216）。
    #   ② 重定向 stdout/stderr → /dev/null    fork 内部的迭代/重试/压缩噪声不外泄，用户只看最终摘要。
    #   ③ fork review_agent = AIAgent(skip_memory=True)   继承父 runtime（provider/model/凭证），
    #      但不自建外部 memory provider——否则会把审查 harness prompt 经 on_turn_start/prefetch/
    #      sync 三个入口泄进用户【真实记忆命名空间】。
    #   ④ 回绑父态：_memory_store / _cached_system_prompt / session_id / session_start →
    #      既命中父预热的同一 prefix 缓存（请求字节一致，实测省 ~26% 成本，#25322），
    #      又让 review 里的 memory(action=add) 落到【同一份 MEMORY.md】。
    #   ⑤ set_thread_tool_whitelist({memory, skills})   线程级白名单，白名单外的工具运行时一律 deny。
    #   ⑥ review_agent.run_conversation(prompt, conversation_history=快照)   回放对话跑审查；
    #      finally 清白名单，避免污染后续复用的线程。
    #   ⑦ summarize_background_review_actions   提取成功动作 → 经 background_review_callback 给
    #      用户一行可见摘要（"Self-improvement review: ..."）。
    # ═══════════════════════════════════════════════════════════════════════
    # Local import to avoid a hard circular dep at module load.
    from run_agent import AIAgent
    from tools.terminal_tool import set_approval_callback as _set_approval_callback

    # Install a non-interactive approval callback on this worker
    # thread so any dangerous-command guard the review agent trips
    # resolves to "deny" instead of falling back to input() -- which
    # deadlocks against the parent's prompt_toolkit TUI (#15216).
    # Same pattern as _subagent_auto_deny in tools/delegate_tool.py.
    def _bg_review_auto_deny(command, description, **kwargs):
        logger.warning(
            "Background review auto-denied dangerous command: %s (%s)",
            command, description,
        )
        return "deny"
    try:
        _set_approval_callback(_bg_review_auto_deny)
    except Exception:
        pass

    review_agent = None
    review_messages: List[Dict] = []
    try:
        # 中文：整段 review 在 stdout/stderr 重定向到 /dev/null 下执行，用户只会看到最后那行
        # "Self-improvement review: ..." 摘要，看不到 fork 内部的迭代/重试/压缩等噪声。
        with open(os.devnull, "w", encoding="utf-8") as _devnull, \
             contextlib.redirect_stdout(_devnull), \
             contextlib.redirect_stderr(_devnull):
            # Inherit the parent agent's live runtime (provider, model,
            # base_url, api_key, api_mode) so the fork uses the exact
            # same credentials the main turn is using.  Without this,
            # AIAgent.__init__ re-runs auto-resolution from env vars,
            # which fails for OAuth-only providers, session-scoped
            # creds, or credential-pool setups where the resolver can't
            # reconstruct auth from scratch -- producing the spurious
            # "No LLM provider configured" warning at end of turn.
            _parent_runtime = agent._current_main_runtime()
            _parent_api_mode = _parent_runtime.get("api_mode") or None
            # The review fork needs to call agent-loop tools (memory,
            # skill_manage). Those tools require Hermes' own dispatch,
            # which the codex_app_server runtime bypasses entirely
            # (it runs the turn inside codex's subprocess). So when
            # the parent is on codex_app_server, downgrade the review
            # fork to codex_responses — same auth/credentials, but
            # talks to the OpenAI Responses API directly so Hermes
            # owns the loop and the agent-loop tools dispatch.
            if _parent_api_mode == "codex_app_server":
                _parent_api_mode = "codex_responses"
            # skip_memory=True keeps the review fork from
            # touching external memory plugins (honcho, mem0,
            # supermemory, etc.).  Without it, the fork's
            # __init__ rebuilds its own _memory_manager from
            # config, scoped to the parent's session_id, and
            # run_conversation() then leaks the harness prompt
            # into the user's real memory namespace via three
            # ingestion sites: on_turn_start (cadence + turn
            # message), prefetch_all (recall query), and
            # sync_all (harness prompt + review output recorded
            # as a (user, assistant) turn pair).  Built-in
            # MEMORY.md / USER.md state is re-bound from the
            # parent below so memory(action="add") writes from
            # the review still land on disk; the review just
            # has zero side effects on external providers.
            # Match parent's toolset config so ``tools[]`` is byte-identical
            # in the request body — Anthropic's cache key includes it.
            # (The runtime whitelist below still restricts dispatch.)
            # 中文：在此 fork 出"审查用"的 AIAgent。下面紧接着把父 agent 的 _memory_store、
            # _cached_system_prompt、session_id 等回绑过来，让它"看起来就是父 agent"，
            # 既命中同一 prefix 缓存，又能把写入落到同一份 MEMORY.md。
            review_agent = AIAgent(
                model=agent.model,
                max_iterations=16,
                quiet_mode=True,
                platform=agent.platform,
                provider=agent.provider,
                api_mode=_parent_api_mode,
                base_url=_parent_runtime.get("base_url") or None,
                api_key=_parent_runtime.get("api_key") or None,
                credential_pool=getattr(agent, "_credential_pool", None),
                parent_session_id=agent.session_id,
                enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                disabled_toolsets=getattr(agent, "disabled_toolsets", None),
                skip_memory=True,
            )
            review_agent._memory_write_origin = "background_review"
            review_agent._memory_write_context = "background_review"
            review_agent._memory_store = agent._memory_store
            review_agent._memory_enabled = agent._memory_enabled
            review_agent._user_profile_enabled = agent._user_profile_enabled
            review_agent._memory_nudge_interval = 0
            review_agent._skill_nudge_interval = 0
            # Suppress all status/warning emits from the fork so the
            # user only sees the final successful-action summary.
            # Without this, mid-review "Iteration budget exhausted",
            # rate-limit retries, compression warnings, and other
            # lifecycle messages bubble up through _emit_status ->
            # _vprint and leak past the stdout redirect (they go via
            # _print_fn/status_callback, which bypass sys.stdout).
            review_agent.suppress_status_output = True
            # Inherit the parent's cached system prompt verbatim so
            # the review fork's outbound HTTP request hits the same
            # Anthropic/OpenRouter prefix cache the parent warmed.
            # Without this, the fork rebuilds the system prompt from
            # scratch (fresh _hermes_now() timestamp, fresh
            # session_id, narrower toolset → different skills_prompt)
            # and the byte-exact prefix-cache key misses. See
            # issue #25322 and PR #17276 for the full analysis +
            # measured impact (~26% end-to-end cost reduction on
            # Sonnet 4.5).
            review_agent._cached_system_prompt = agent._cached_system_prompt
            # Defensive: pin session_start + session_id to the
            # parent's so any code path that re-renders parts of
            # the system prompt (compression, plugin hooks) still
            # produces byte-identical output. The cached-prompt
            # assignment above already short-circuits the normal
            # rebuild path, but these pins guarantee parity even
            # if a future code path bypasses the cache.
            review_agent.session_start = agent.session_start
            review_agent.session_id = agent.session_id

            from model_tools import get_tool_definitions
            from hermes_cli.plugins import (
                set_thread_tool_whitelist,
                clear_thread_tool_whitelist,
            )

            # 中文：构造运行时【工具白名单】——只放 memory/skills 两套工具集的工具名，
            # 通过线程级白名单限制 dispatch：白名单外的工具一律在运行时被 deny。
            review_whitelist = {
                t["function"]["name"]
                for t in get_tool_definitions(
                    enabled_toolsets=["memory", "skills"],
                    quiet_mode=True,
                )
            }
            set_thread_tool_whitelist(
                review_whitelist,
                deny_msg_fmt=(
                    "Background review denied non-whitelisted tool: "
                    "{tool_name}. Only memory/skill tools are allowed."
                ),
            )
            # 中文：把对话快照当历史回放、用 review prompt 跑这趟 fork 对话；
            # 无论成败都在 finally 里清掉线程白名单，避免污染后续复用的线程。
            try:
                review_agent.run_conversation(
                    user_message=(
                        prompt
                        + "\n\nYou can only call memory and skill "
                        "management tools. Other tools will be denied "
                        "at runtime — do not attempt them."
                    ),
                    conversation_history=messages_snapshot,
                )
            finally:
                clear_thread_tool_whitelist()

            # Snapshot review actions before teardown. close() is allowed to
            # clean per-session state, but the user-visible self-improvement
            # summary still needs the completed review agent's tool results.
            review_messages = list(getattr(review_agent, "_session_messages", []))

            # Tear down memory providers while stdout is still
            # redirected so background thread teardown (Honcho flush,
            # Hindsight sync, etc.) stays silent.  The finally block
            # below is a safety net for the exception path.
            # 中文：趁 stdout 还在重定向时就地 teardown fork（关 provider + close），
            # 让后端 flush/sync 的输出也被吞掉；置 None 表示已清理，外层 finally 不再重复。
            try:
                review_agent.shutdown_memory_provider()
            except Exception:
                pass
            try:
                review_agent.close()
            except Exception:
                pass
            review_agent = None

        # Scan the review agent's messages for successful tool actions
        # and surface a compact summary to the user. Tool messages
        # already present in messages_snapshot must be skipped, since
        # the review agent inherits that history and would otherwise
        # re-surface stale "created"/"updated" messages from the prior
        # conversation as if they just happened (issue #14944).
        actions = summarize_background_review_actions(
            review_messages,
            messages_snapshot,
        )

        # 中文：把提炼出的动作去重拼成一行摘要，既打印给用户（_safe_print），
        # 也通过 background_review_callback 回调出去（gateway 等场景）。
        if actions:
            summary = " · ".join(dict.fromkeys(actions))
            agent._safe_print(
                f"  💾 Self-improvement review: {summary}"
            )
            _bg_cb = agent.background_review_callback
            if _bg_cb:
                try:
                    _bg_cb(
                        f"💾 Self-improvement review: {summary}"
                    )
                except Exception:
                    pass

    except Exception as e:
        # 中文：fail-open——后台自我策展失败绝不影响主对话，只记一条 warning 并上报辅助失败。
        logger.warning("Background memory/skill review failed: %s", e)
        agent._emit_auxiliary_failure("background review", e)
    finally:
        # Safety-net cleanup for the exception path.  Normal
        # completion already shut down inside redirect_stdout above.
        # Re-open devnull here so any teardown output (Honcho flush,
        # Hindsight sync, background thread joins) stays silent even
        # on the exception path where redirect_stdout already exited.
        if review_agent is not None:
            try:
                with open(os.devnull, "w", encoding="utf-8") as _fn, \
                     contextlib.redirect_stdout(_fn), \
                     contextlib.redirect_stderr(_fn):
                    try:
                        review_agent.shutdown_memory_provider()
                    except Exception:
                        pass
                    try:
                        review_agent.close()
                    except Exception:
                        pass
            except Exception:
                pass
        # Clear the approval callback on this bg-review thread so a
        # recycled thread-id doesn't inherit a stale reference.
        try:
            _set_approval_callback(None)
        except Exception:
            pass


def spawn_background_review_thread(
    agent: Any,
    messages_snapshot: List[Dict],
    review_memory: bool = False,
    review_skills: bool = False,
):
    """Build the review thread target and prompt for a background review.

    Returns a ``(target, prompt)`` tuple.  The caller (``AIAgent._spawn_background_review``)
    owns the actual ``threading.Thread`` construction so test-level patches
    of ``run_agent.threading.Thread`` keep working.
    """
    # 中文：本函数【只装配不启动】——按触发的维度（memory / skills / 两者）选好 review prompt，
    # 再把"在线程里跑哪段逻辑"包成 _target 闭包返回。真正 new Thread 由调用方负责，是为了让测试能
    # patch 掉 run_agent.threading.Thread。
    # Pick the right prompt based on which triggers fired.  Allow per-agent
    # override (the prompts moved to module-level constants but old code paths
    # that set agent._MEMORY_REVIEW_PROMPT etc. directly keep working).
    if review_memory and review_skills:
        prompt = getattr(agent, "_COMBINED_REVIEW_PROMPT", _COMBINED_REVIEW_PROMPT)
    elif review_memory:
        prompt = getattr(agent, "_MEMORY_REVIEW_PROMPT", _MEMORY_REVIEW_PROMPT)
    else:
        prompt = getattr(agent, "_SKILL_REVIEW_PROMPT", _SKILL_REVIEW_PROMPT)

    def _target() -> None:
        _run_review_in_thread(agent, messages_snapshot, prompt)

    return _target, prompt


__all__ = [
    "_MEMORY_REVIEW_PROMPT",
    "_SKILL_REVIEW_PROMPT",
    "_COMBINED_REVIEW_PROMPT",
    "spawn_background_review_thread",
    "summarize_background_review_actions",
    "build_memory_write_metadata",
]
