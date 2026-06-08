"""Abstract base class for pluggable memory providers.

Memory providers give the agent persistent recall across sessions.
The MemoryManager enforces a one-external-provider limit to prevent
tool schema bloat and conflicting memory backends.

External providers (Honcho, Hindsight, Mem0, etc.) are registered
and managed via MemoryManager. Only one external provider runs at a
time.

Registration:
  Plugins ship in plugins/memory/<name>/ and are activated via
  the memory.provider config key.

Lifecycle (called by MemoryManager, wired in run_agent.py):
  initialize()          — connect, create resources, warm up
  system_prompt_block()  — static text for the system prompt
  prefetch(query)        — background recall before each turn
  sync_turn(user, asst)  — async write after each turn
  get_tool_schemas()     — tool schemas to expose to the model
  handle_tool_call()     — dispatch a tool call
  shutdown()             — clean exit

Optional hooks (override to opt in):
  on_turn_start(turn, message, **kwargs) — per-turn tick with runtime context
  on_session_end(messages)               — end-of-session extraction
  on_session_switch(new_session_id, **kwargs) — mid-process session_id rotation
  on_pre_compress(messages) -> str       — extract before context compression
  on_memory_write(action, target, content, metadata=None) — mirror built-in memory writes
  on_delegation(task, result, **kwargs)  — parent-side observation of subagent work
"""

# ═════════════════════════════════════════════════════════════════════════════
# 【中文导读】外部记忆 Provider 的抽象基类（接口契约）
# ═════════════════════════════════════════════════════════════════════════════
# 想接一个新的记忆后端（向量库 / 知识图谱 / 第三方 SaaS），就继承本类、实现下面的
# 生命周期方法，放到 plugins/memory/<name>/，再用 config 里的 memory.provider 选中。
# MemoryManager 会按这套接口在合适的时机回调你（见 memory_manager.py）。
#
# 必须实现：name / is_available / initialize / get_tool_schemas（+ 通常还有
#   handle_tool_call / prefetch / sync_turn / system_prompt_block / shutdown）。
# 可选 hook（不实现就是 no-op）：on_turn_start / on_session_end / on_session_switch /
#   on_pre_compress / on_memory_write / on_delegation。
#
# 性能约定：prefetch() 必须快——真正的召回放后台线程，prefetch() 只返缓存结果；
# sync_turn() 也应非阻塞（有延迟就排队后台处理）。否则会拖慢每一轮主对话。
# ═════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryProvider(ABC):
    """Abstract base class for memory providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        # 中文：provider 的短标识符（用于日志、config 选择、单 provider 去重）。
        """Short identifier for this provider (e.g. 'builtin', 'honcho', 'hindsight')."""

    # -- Core lifecycle (implement these) ------------------------------------

    @abstractmethod
    def is_available(self) -> bool:
        # 中文：判断本 provider 是否已配置且依赖就绪（只查 config/依赖，不发网络请求）。
        """Return True if this provider is configured, has credentials, and is ready.

        Called during agent init to decide whether to activate the provider.
        Should not make network calls — just check config and installed deps.
        """

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        # 中文·会话启动时调用一次：建连接 / 建资源（表、bank）/ 起后台线程 / 预热。
        # kwargs 里的 hermes_home（profile 隔离存储路径）、user_id、agent_identity 是
        # 【记忆隔离的关键】——按用户、按 profile 分桶，避免不同用户或不同人格的记忆串味；
        # agent_context 非 "primary"（cron/subagent/flush）时应跳过写入，否则会污染用户画像。
        """Initialize for a session.

        Called once at agent startup. May create resources (banks, tables),
        establish connections, start background threads, etc.

        kwargs always include:
          - hermes_home (str): The active HERMES_HOME directory path. Use this
            for profile-scoped storage instead of hardcoding ``~/.hermes``.
          - platform (str): "cli", "telegram", "discord", "cron", etc.

        kwargs may also include:
          - agent_context (str): "primary", "subagent", "cron", or "flush".
            Providers should skip writes for non-primary contexts (cron system
            prompts would corrupt user representations).
          - agent_identity (str): Profile name (e.g. "coder"). Use for
            per-profile provider identity scoping.
          - agent_workspace (str): Shared workspace name (e.g. "hermes").
          - parent_session_id (str): For subagents, the parent's session_id.
          - user_id (str): Platform user identifier (gateway sessions).
          - user_id_alt (str): Optional alternate stable platform user identifier.
        """

    def system_prompt_block(self) -> str:
        # 中文：返回要拼进 system prompt 的【静态】文本（说明/状态）；召回内容走 prefetch。
        """Return text to include in the system prompt.

        Called during system prompt assembly. Return empty string to skip.
        This is for STATIC provider info (instructions, status). Prefetched
        recall context is injected separately via prefetch().
        """
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        # 中文·召回（读路径）：每轮 API 调用前被调，返回文本会被 manager 加 <memory-context>
        # 围栏后注入到当前 user 消息（ephemeral，不落库）。【契约：必须快】——真正的检索放
        # 后台线程（见 queue_prefetch），这里只返回上一轮预热好的缓存。于是典型实现是
        # 「召回延迟一轮」：本轮注入的其实是上一轮 query 搜出来的内容，以此保证永不阻塞当前 turn。
        """Recall relevant context for the upcoming turn.

        Called before each API call. Return formatted text to inject as
        context, or empty string if nothing relevant. Implementations
        should be fast — use background threads for the actual recall
        and return cached results here.

        session_id is provided for providers serving concurrent sessions
        (gateway group chats, cached agents). Providers that don't need
        per-session scoping can ignore it.
        """
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # 中文·预热（与 prefetch 配对）：当前轮结束后被调，在后台线程发起对【下一轮】的召回，
        # 结果缓存起来等下一轮 prefetch() 取走。把检索延迟搬到后台、与主对话解耦的关键一环。
        # 默认 no-op——只有做后台预取的 provider（如 Mem0）才需要 override。
        """Queue a background recall for the NEXT turn.

        Called after each turn completes. The result will be consumed
        by prefetch() on the next turn. Default is no-op — providers
        that do background prefetching should override this.
        """

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        # 中文·写入（写路径）：每轮结束后把这轮 (user, assistant) 交流写进后端。
        # 【契约：非阻塞】——后端有网络延迟就丢后台线程，绝不能卡住主对话。
        # messages 给的是本轮完整 OpenAI 格式消息（含工具调用与结果），需要原始上下文的
        # 后端（如做服务端事实抽取的 Mem0）才用，否则可忽略只看 user/assistant 文本。
        """Persist a completed turn to the backend.

        Called after each turn. Should be non-blocking — queue for
        background processing if the backend has latency.

        ``messages`` is the OpenAI-style conversation message list as of the
        completed turn, including any assistant tool calls and tool results.
        Providers that do not need raw turn context can ignore it.
        """

    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # 中文：返回本 provider 要暴露给模型的工具 schema（无工具就返空列表）。
        """Return tool schemas this provider exposes.

        Each schema follows the OpenAI function calling format:
        {"name": "...", "description": "...", "parameters": {...}}

        Return empty list if this provider has no tools (context-only).
        """

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        # 中文：分发并执行本 provider 自己工具的一次调用，返回 JSON 字符串结果。
        """Handle a tool call for one of this provider's tools.

        Must return a JSON string (the tool result).
        Only called for tool names returned by get_tool_schemas().
        """
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    def shutdown(self) -> None:
        # 中文：干净退出——刷空队列、关闭连接。
        """Clean shutdown — flush queues, close connections."""

    # -- Optional hooks (override to opt in) ---------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        # 中文：每轮开始时的 tick（带运行时上下文），用于计轮次、调 scope、周期维护。
        """Called at the start of each turn with the user message.

        Use for turn-counting, scope management, periodic maintenance.

        kwargs may include: remaining_tokens, model, platform, tool_count.
        Providers use what they need; extras are ignored.
        """

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        # 中文：会话真正结束时（退出/超时，非每轮）触发——做收尾抽取、汇总。
        """Called when a session ends (explicit exit or timeout).

        Use for end-of-session fact extraction, summarization, etc.
        messages is the full conversation history.

        NOT called after every turn — only at actual session boundaries
        (CLI exit, /reset, gateway session expiry).
        """

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        # 中文·进程内 session_id 轮换时调用（/resume /branch /reset /new 与上下文压缩都会触发，
        # 但 provider 不被销毁）：若在 initialize 里缓存了会话级状态（_session_id/_document_id/累积
        # 缓冲/计数器），必须在这里刷新，否则后续写入会落到【错误会话】的记录上。
        # reset=True 表示全新会话（应清空累积缓冲）；rewound=True 表示 session_id 没变但 transcript
        # 被截断（缓存 per-turn 文档状态的要失效）。
        """Called when the agent switches session_id mid-process.

        Fires on ``/resume``, ``/branch``, ``/reset``, ``/new`` (CLI), the
        gateway equivalents, and context compression — any path that
        reassigns ``AIAgent.session_id`` without tearing the provider down.

        Providers that cache per-session state in ``initialize()``
        (``_session_id``, ``_document_id``, accumulated turn buffers,
        counters) should update or reset that state here so subsequent
        writes land in the correct session's record.

        Parameters
        ----------
        new_session_id:
            The session_id the agent just switched to.
        parent_session_id:
            The previous session_id, if meaningful — set for ``/branch``
            (fork lineage), context compression (continuation lineage),
            and ``/resume`` (the session we're leaving). Empty string
            when no lineage applies.
        reset:
            ``True`` when this is a genuinely new conversation, not a
            resumption of an existing one. Fired by ``/reset`` / ``/new``.
            Providers should flush accumulated per-session buffers
            (``_session_turns``, ``_turn_counter``, etc.) when this is
            set. ``False`` for ``/resume`` / ``/branch`` / compression
            where the logical conversation continues under the new id.
        rewound:
            ``True`` if session_id is unchanged but the transcript was
            truncated; providers caching per-turn document state should
            invalidate.

        Default is no-op for backward compatibility.
        """

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        # 中文·上下文压缩丢弃旧消息【之前】调用：messages 是即将被摘要/丢弃的旧消息。
        # provider 可在此从中抽取洞见并【返回文本】，这段文本会被并进压缩摘要的 prompt，
        # 从而把本要丢失的信息保留进摘要里。返回空串则不贡献（向后兼容默认）。
        """Called before context compression discards old messages.

        Use to extract insights from messages about to be compressed.
        messages is the list that will be summarized/discarded.

        Return text to include in the compression summary prompt so the
        compressor preserves provider-extracted insights. Return empty
        string for no contribution (backwards-compatible default).
        """
        return ""

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        # 中文：子 agent 完成时在【父 agent】侧触发——把"委派任务+返回结果"作为一条观察记下（子 agent 自身 skip_memory）。
        """Called on the PARENT agent when a subagent completes.

        The parent's memory provider gets the task+result pair as an
        observation of what was delegated and what came back. The subagent
        itself has no provider session (skip_memory=True).

        task: the delegation prompt
        result: the subagent's final response
        child_session_id: the subagent's session_id
        """

    def get_config_schema(self) -> List[Dict[str, Any]]:
        # 中文：声明 setup 时需收集的配置字段（供 'hermes memory setup' 引导填写）。
        """Return config fields this provider needs for setup.

        Used by 'hermes memory setup' to walk the user through configuration.
        Each field is a dict with:
          key:         config key name (e.g. 'api_key', 'mode')
          description: human-readable description
          secret:      True if this should go to .env (default: False)
          required:    True if required (default: False)
          default:     default value (optional)
          choices:     list of valid values (optional)
          url:         URL where user can get this credential (optional)
          env_var:     explicit env var name for secrets (default: auto-generated)

        Return empty list if no config needed (e.g. local-only providers).
        """
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        # 中文：把非机密配置写到 provider 自己的原生位置（机密走 .env）；纯 env 的 provider 留 no-op。
        """Write non-secret config to the provider's native location.

        Called by 'hermes memory setup' after collecting user inputs.
        ``values`` contains only non-secret fields (secrets go to .env).
        ``hermes_home`` is the active HERMES_HOME directory path.

        Providers with native config files (JSON, YAML) should override
        this to write to their expected location. Providers that use only
        env vars can leave the default (no-op).

        All new memory provider plugins MUST implement either:
        - save_config() for native config file formats, OR
        - use only env vars (in which case get_config_schema() fields
          should all have ``env_var`` set and this method stays no-op).
        """

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        # 中文·镜像钩子：内置 memory 工具写 MEMORY.md/USER.md 时（add/replace），manager 把同一条
        # 写入转发到这里，让外部后端与内置记忆保持同步——内置是"少量常驻关键事实"，外部是"无界
        # 语义层"，两边同源才不会各记各的。metadata 是 provenance 溯源（write_origin / execution_context
        # / session_id / platform 等），后端可据此做归因、去重、按来源过滤（如忽略 background_review 的写）。
        """Called when the built-in memory tool writes an entry.

        action: 'add', 'replace', or 'remove'
        target: 'memory' or 'user'
        content: the entry content
        metadata: structured provenance for the write, when available. Common
          keys include ``write_origin``, ``execution_context``, ``session_id``,
          ``parent_session_id``, ``platform``, and ``tool_name``.

        Use to mirror built-in memory writes to your backend.
        """
