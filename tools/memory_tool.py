#!/usr/bin/env python3
"""
Memory Tool Module - Persistent Curated Memory

Provides bounded, file-backed memory that persists across sessions. Two stores:
  - MEMORY.md: agent's personal notes and observations (environment facts, project
    conventions, tool quirks, things learned)
  - USER.md: what the agent knows about the user (preferences, communication style,
    expectations, workflow habits)

Both are injected into the system prompt as a frozen snapshot at session start.
Mid-session writes update files on disk immediately (durable) but do NOT change
the system prompt -- this preserves the prefix cache for the entire session.
The snapshot refreshes on the next session start.

Entry delimiter: § (section sign). Entries can be multiline.
Character limits (not tokens) because char counts are model-independent.

Design:
- Single `memory` tool with action parameter: add, replace, remove, read
- replace/remove use short unique substring matching (not full text or IDs)
- Behavioral guidance lives in the tool schema description
- Frozen snapshot pattern: system prompt is stable, tool responses show live state
"""

# ═════════════════════════════════════════════════════════════════════════════
# 【中文导读】内置策展记忆的「存储引擎」——纯确定性代码，本文件不含任何 LLM 调用
# ═════════════════════════════════════════════════════════════════════════════
# 本文件实现 MEMORY.md（agent 自己的笔记）和 USER.md（用户画像）两个文件的读写。
# 它只负责【机制】：读写 / 去重 / 容量限额 / 并发加锁 / 原子落盘 / 漂移检测 /
# 注入前威胁扫描。一切都是确定性的，可单测、与模型无关。
#
# 谁决定"记什么内容"？——是 LLM。模型在对话中调用 memory 工具，产出
# action/target/content；本文件拿到后只做校验与落盘，绝不二次加工语义。
# 这条「判断归 LLM、机制归代码」的边界是刻意的：换模型 / 模型乱来都不会
# 破坏持久化与安全保证。
#
# 三个最关键的设计点（详见对应函数注释）：
#   1) 冻结快照 _system_prompt_snapshot：load 时把磁盘渲染成一份快照注入
#      system prompt，整个 session 不变 —— 保住 LLM 的 prefix 缓存不失效。
#      会话中的写入即时落盘、但不动快照，下次 session 启动才刷新（见 load_from_disk）。
#   2) 外部漂移检测 _detect_external_drift：发现别的写入者（patch 工具 / shell /
#      手改 / 并发 session）往文件里塞了非法内容，就先备份再"拒写"——宁可失败
#      也不静默覆盖用户数据（issue #26045）。这是 fail-closed。
#   3) 注入前威胁扫描 _scan_memory_content + 加载期 _sanitize_entries_for_snapshot：
#      记忆会进 system prompt 且跨会话存活，是高价值注入靶子，按最严 scope 扫两道。
# ═════════════════════════════════════════════════════════════════════════════

import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, Any, List, Optional

from utils import atomic_replace

# fcntl is Unix-only; on Windows use msvcrt for file locking
msvcrt = None
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

logger = logging.getLogger(__name__)

# Where memory files live — resolved dynamically so profile overrides
# (HERMES_HOME env var changes) are always respected.  The old module-level
# constant was cached at import time and could go stale if a profile switch
# happened after the first import.
def get_memory_dir() -> Path:
    """Return the profile-scoped memories directory."""
    return get_hermes_home() / "memories"

ENTRY_DELIMITER = "\n§\n"


# ---------------------------------------------------------------------------
# Memory content scanning — lightweight check for injection/exfiltration
# in content that gets injected into the system prompt.
#
# Patterns live in ``tools/threat_patterns.py`` — the single source of truth
# shared with the context-file scanner and the tool-result delimiter system.
# Memory uses the "strict" scope (broadest pattern set) because:
#  - memory entries are user-curated; the user can rewrite a flagged entry
#  - memory enters the system prompt as a FROZEN snapshot, so a poisoned
#    entry persists for the entire session and across sessions until
#    explicitly removed.
# ---------------------------------------------------------------------------

from tools.threat_patterns import first_threat_message as _first_threat_message


def _scan_memory_content(content: str) -> Optional[str]:
    """Scan memory content for injection/exfil patterns. Returns error string if blocked."""
    # 中文：记忆写入前的【威胁扫描】总入口。复用 threat_patterns 共享库，按最严的
    # "strict" scope（最宽的模式集）扫一遍待写入内容；命中返回错误串、调用方据此拒写，
    # 没命中返回 None。记忆会进 system prompt 且跨会话存活，是高价值注入靶子，故用最严尺度。
    return _first_threat_message(content, scope="strict")


def _drift_error(path: "Path", bak_path: str) -> Dict[str, Any]:
    """Build the error dict returned when external drift is detected.

    The on-disk memory file contains content that wouldn't round-trip
    through the tool's parser/serializer — flushing would discard the
    appended/edited content from a patch tool, shell append, manual edit,
    or sister-session write. We refuse the mutation, point the operator at
    the .bak.<ts> snapshot we took, and tell them what to do next.
    """
    # 中文：漂移检测命中时统一构造的「拒写」错误包。把 .bak.<ts> 备份路径与处理建议
    # 一并回给 LLM，让它别再硬写、而是先消化外部写入者（patch 工具 / shell 追加 / 手改 /
    # 并发 session）塞进文件的内容。这是 fail-closed：宁可这次写失败，也不静默原子覆盖。
    return {
        "success": False,
        "error": (
            f"Refusing to write {path.name}: file on disk has content that "
            f"wouldn't round-trip through the memory tool (likely added by "
            f"the patch tool, a shell append, a manual edit, or a "
            f"concurrent session). A snapshot was saved to {bak_path}. "
            f"Resolve the drift first — either rewrite the file as a clean "
            f"§-delimited list of entries, or move the extra content out — "
            f"then retry. This guard exists to prevent silent data loss "
            f"(issue #26045)."
        ),
        "drift_backup": bak_path,
        "remediation": (
            "Open the .bak file, integrate the missing entries into the "
            "memory tool one at a time via memory(action=add, content=...), "
            "then remove or rewrite the original file to a clean state."
        ),
    }


class MemoryStore:
    """
    Bounded curated memory with file persistence. One instance per AIAgent.

    Maintains two parallel states:
      - _system_prompt_snapshot: frozen at load time, used for system prompt injection.
        Never mutated mid-session. Keeps prefix cache stable.
      - memory_entries / user_entries: live state, mutated by tool calls, persisted to disk.
        Tool responses always reflect this live state.
    """

    # 中文：记忆存储引擎，每个 AIAgent 持有一个实例。核心是【两份平行状态】：
    #   · _system_prompt_snapshot —— load 时冻结的快照，专供注入 system prompt，
    #     会话内永不改动，保住 prefix 缓存；
    #   · memory_entries / user_entries —— 活状态，add/replace/remove 即时改并落盘，
    #     工具响应永远反映这份活状态。
    # 二者刻意分离：磁盘随时变，但 system prompt 字节稳定，下次 session 启动才重新冻结。
    def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375):
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        # Frozen snapshot for system prompt -- set once at load_from_disk()
        # 冻结快照：注入 system prompt 用的就是这份，整个 session 不变（保 prefix 缓存）。
        # 它与上面的 memory_entries/user_entries（活状态，工具写入即时改）是两份平行状态。
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}

    def load_from_disk(self):
        """Load entries from MEMORY.md and USER.md, capture system prompt snapshot.

        The frozen snapshot is what enters the system prompt. We scan each
        entry for injection/promptware patterns at snapshot-build time —
        ANY hit replaces the entry text in the snapshot with a placeholder
        like ``[BLOCKED: …]``, so a poisoned-on-disk memory file (supply
        chain, compromised tool, sister-session write) cannot inject into
        the system prompt.

        The live ``memory_entries`` / ``user_entries`` lists keep the
        original text so the user can still SEE poisoned entries via
        ``memory(action=read)`` and remove them — silently dropping them
        would hide the attack from the user.

        Scanning is deterministic from disk bytes, so the snapshot remains
        stable for the entire session (prefix-cache invariant holds).
        """
        # 中文：读盘 → 按 § 切条 → 去重 → 逐条威胁扫描（命中替换为 [BLOCKED] 占位符，
        # 只换快照、不换活状态）→ 渲染带容量条的块 → 冻结进 _system_prompt_snapshot。
        # 整个过程纯代码、无 LLM；结果对相同磁盘字节是确定的，故快照在 session 内稳定。
        mem_dir = get_memory_dir()
        mem_dir.mkdir(parents=True, exist_ok=True)

        self.memory_entries = self._read_file(mem_dir / "MEMORY.md")
        self.user_entries = self._read_file(mem_dir / "USER.md")

        # Deduplicate entries (preserves order, keeps first occurrence)
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))

        # Sanitize entries for the system-prompt snapshot only.  Live state
        # (memory_entries / user_entries) keeps the raw text so the user
        # can see + remove poisoned entries via the memory tool.
        sanitized_memory = self._sanitize_entries_for_snapshot(self.memory_entries, "MEMORY.md")
        sanitized_user = self._sanitize_entries_for_snapshot(self.user_entries, "USER.md")

        # Capture frozen snapshot for system prompt injection
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", sanitized_memory),
            "user": self._render_block("user", sanitized_user),
        }

    @staticmethod
    def _sanitize_entries_for_snapshot(entries: List[str], filename: str) -> List[str]:
        """Return ``entries`` with any threat-matching entry replaced by a placeholder.

        Each entry is scanned with the shared threat-pattern library at the
        ``"strict"`` scope (same as memory writes).  On match, the entry is
        replaced in the returned list with ``"[BLOCKED: <filename> entry
        contained threat pattern: <ids>. Removed from system prompt.]"`` —
        the placeholder enters the snapshot, the original entry stays in
        live state for the user to inspect and delete.

        Empty or already-block-marker entries pass through unchanged.
        """
        # 中文：加载期的【脱毒】环节，是注入前威胁扫描的第二道（第一道是写入期 add/replace）。
        # 逐条按 strict scope 扫描磁盘上的条目，命中者在【返回列表】里换成 [BLOCKED:...] 占位符，
        # 占位符进冻结快照、原文仍留在活状态供用户 read/remove。专防「磁盘上已被投毒」的
        # 条目（供应链、被攻陷工具、并发 session 写入）借冻结快照注入 system prompt 并跨会话存活。
        from tools.threat_patterns import scan_for_threats

        sanitized: List[str] = []
        for entry in entries:
            if not entry or entry.startswith("[BLOCKED:"):
                sanitized.append(entry)
                continue
            findings = scan_for_threats(entry, scope="strict")
            if findings:
                logger.warning(
                    "Memory entry from %s blocked at load time: %s",
                    filename, ", ".join(findings),
                )
                sanitized.append(
                    f"[BLOCKED: {filename} entry contained threat pattern(s): "
                    f"{', '.join(findings)}. Removed from system prompt; "
                    f"use memory(action=read) to inspect and memory(action=remove) "
                    f"to delete the original.]"
                )
            else:
                sanitized.append(entry)
        return sanitized

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """Acquire an exclusive file lock for read-modify-write safety.

        Uses a separate .lock file so the memory file itself can still be
        atomically replaced via os.replace().
        """
        # 中文：所有 add/replace/remove 都在这把锁内"重读磁盘→改→落盘"，从而能拿到
        # 并发 session 的写入。锁加在独立 .lock 文件上，主文件仍可被 rename 原子替换。
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        if fcntl is None and msvcrt is None:
            yield
            return

        fd = open(lock_path, "a+", encoding="utf-8")
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except (OSError, IOError):
                    pass
            elif msvcrt:
                try:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            fd.close()

    @staticmethod
    def _path_for(target: str) -> Path:
        # 中文：把 target（"user"/"memory"）映射到对应磁盘文件路径，小工具。
        mem_dir = get_memory_dir()
        if target == "user":
            return mem_dir / "USER.md"
        return mem_dir / "MEMORY.md"

    def _reload_target(self, target: str) -> Optional[str]:
        """Re-read entries from disk into in-memory state.

        Called under file lock to get the latest state before mutating.
        Returns the backup path if external drift was detected (the on-disk
        file contains content that wouldn't round-trip through our
        parser/serializer, OR an entry larger than the store's char limit).
        When drift is detected the caller must abort the mutation —
        flushing would discard the un-roundtrippable content.
        Returns None on clean reload.
        """
        # 中文：每次 add/replace/remove 在锁内做的第一步——先跑漂移检测，再从磁盘重读
        # 最新条目灌进活状态。这一步保证我们改的是「别的 session 也写过之后」的最新版本，
        # 而非内存里的陈旧值。返回非 None（备份路径）即代表检测到漂移，调用方必须立刻放弃本次写。
        path = self._path_for(target)
        bak = self._detect_external_drift(target)
        fresh = self._read_file(path)
        fresh = list(dict.fromkeys(fresh))  # deduplicate
        self._set_entries(target, fresh)
        return bak

    def save_to_disk(self, target: str):
        """Persist entries to the appropriate file. Called after every mutation."""
        # 中文：把目标的活状态条目【原子落盘】（委托 _write_file 走临时文件+rename）。
        # 每次成功写入后都会调用，保证磁盘与活状态一致；写入即时持久，即使会话崩溃也不丢。
        get_memory_dir().mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))

    def _entries_for(self, target: str) -> List[str]:
        # 中文：按 target 取对应活状态条目列表的引用（注意是引用，非拷贝），小工具。
        if target == "user":
            return self.user_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        # 中文：按 target 把活状态条目列表整体替换为新列表，小工具（与 _entries_for 对称）。
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        # 中文：算目标当前占用的字符数——按 § 分隔符 join 后取长度，与限额口径一致。小工具。
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        # 中文：按 target 返回该库的字符上限（容量限额按字符而非 token，与模型无关）。小工具。
        if target == "user":
            return self.user_char_limit
        return self.memory_char_limit

    def add(self, target: str, content: str) -> Dict[str, Any]:
        """Append a new entry. Returns error if it would exceed the char limit."""
        # 中文·写入校验顺序（任一不过即拒，全代码）：
        #   ① 非空 → ② 威胁扫描 → ③ 加锁重读磁盘 → ④ 漂移检测(命中则备份+拒写)
        #   → ⑤ 去重(已存在则成功但不重复加) → ⑥ 限额(超 char_limit 则报错并回传现有条目)
        #   → ⑦ 追加 + 原子落盘。"该不该加这条"由 LLM 在调用前决定，这里只兜底。
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        # Scan for injection/exfiltration before accepting
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            # Re-read from disk under lock to pick up writes from other sessions.
            # If external drift was detected, the file was backed up to .bak.<ts>
            # — refuse the mutation so we don't clobber the un-roundtrippable
            # content the patch tool / shell append / sister session wrote.
            bak = self._reload_target(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

            entries = self._entries_for(target)
            limit = self._char_limit(target)

            # Reject exact duplicates
            if content in entries:
                return self._success_response(target, "Entry already exists (no duplicate added).")

            # Calculate what the new total would be
            new_entries = entries + [content]
            new_total = len(ENTRY_DELIMITER.join(new_entries))

            if new_total > limit:
                current = self._char_count(target)
                return {
                    "success": False,
                    "error": (
                        f"Memory at {current:,}/{limit:,} chars. "
                        f"Adding this entry ({len(content)} chars) would exceed the limit. "
                        f"Replace or remove existing entries first."
                    ),
                    "current_entries": entries,
                    "usage": f"{current:,}/{limit:,}",
                }

            entries.append(content)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry added.")

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """Find entry containing old_text substring, replace it with new_content."""
        # 中文：以 old_text【短唯一子串】定位一条已存在的条目，整条替换为 new_content。
        # 关键设计：用子串匹配而非全文/ID，LLM 只需给出一小段独特文字即可定位。
        # 流程：非空校验 → 对 new_content 威胁扫描 → 加锁重读+漂移检测 → 匹配（多条且
        # 不完全相同则拒绝、要求更具体；全相同则操作第一条）→ 限额校验 → 落盘。
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}

        # Scan replacement content for injection/exfiltration
        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), operate on the first one
                unique_texts = {e for _, e in matches}
                if len(unique_texts) > 1:
                    previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to replace just the first

            idx = matches[0][0]
            limit = self._char_limit(target)

            # Check that replacement doesn't blow the budget
            test_entries = entries.copy()
            test_entries[idx] = new_content
            new_total = len(ENTRY_DELIMITER.join(test_entries))

            if new_total > limit:
                return {
                    "success": False,
                    "error": (
                        f"Replacement would put memory at {new_total:,}/{limit:,} chars. "
                        f"Shorten the new content or remove other entries first."
                    ),
                }

            entries[idx] = new_content
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Remove the entry containing old_text substring."""
        # 中文：以 old_text 短唯一子串定位并删除整条条目，是 add 的逆操作。
        # 同样在锁内先漂移检测+重读；多条匹配且文本不完全相同则拒绝并要求更具体，
        # 全相同则删第一条。删除无需威胁扫描（不向 system prompt 引入新内容）。
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), remove the first one
                unique_texts = {e for _, e in matches}
                if len(unique_texts) > 1:
                    previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to remove just the first

            idx = matches[0][0]
            entries.pop(idx)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry removed.")

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """
        Return the frozen snapshot for system prompt injection.

        This returns the state captured at load_from_disk() time, NOT the live
        state. Mid-session writes do not affect this. This keeps the system
        prompt stable across all turns, preserving the prefix cache.

        Returns None if the snapshot is empty (no entries at load time).
        """
        # 中文：注意返回的是 load 时的【冻结快照】，不是活状态。会话中的写入不影响它，
        # 因此 system prompt 在整个 session 内字节稳定（prefix 缓存命中）。
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    # -- Internal helpers --

    def _success_response(self, target: str, message: str = None) -> Dict[str, Any]:
        # 中文：统一构造「成功」响应体。关键点：回传的是【活状态】的完整条目与实时占用率，
        # 而非冻结快照——这样 LLM 每次写入后都能看到记忆库的真实最新状态（与 system prompt
        # 里的冻结快照刻意区分）。各 add/replace/remove 成功分支共用此函数。
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        resp = {
            "success": True,
            "target": target,
            "entries": entries,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        return resp

    def _render_block(self, target: str, entries: List[str]) -> str:
        """Render a system prompt block with header and usage indicator."""
        # 中文：把一组条目渲染成注入 system prompt 用的【文本块】——带标题（USER PROFILE /
        # MEMORY）、占用率指示和分隔线，条目间用 § 连接。仅在 load_from_disk 冻结快照时被调用，
        # 输入是已脱毒(sanitize)的条目，故被 [BLOCKED] 替换过的内容不会进入 system prompt。
        if not entries:
            return ""

        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        if target == "user":
            header = f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"

        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """Read a memory file and split into entries.

        No file locking needed: _write_file uses atomic rename, so readers
        always see either the previous complete file or the new complete file.
        """
        # 中文：读盘并按 § 分隔符切成条目列表（去空白、丢空条）。关键点：用完整的
        # ENTRY_DELIMITER 切分而非裸 "§"，否则条目正文里含 § 会被误切。因 _write_file 走
        # 原子 rename，读侧无需加锁——永远看到「旧文件全貌」或「新文件全貌」，不会读到半截。
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return []

        if not raw.strip():
            return []

        # Use ENTRY_DELIMITER for consistency with _write_file. Splitting by "§"
        # alone would incorrectly split entries that contain "§" in their content.
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    def _detect_external_drift(self, target: str) -> Optional[str]:
        """Return a backup-path string if on-disk content shows external drift.

        The memory file is supposed to be a list of small entries the tool
        wrote, joined by §. Detect drift via two signals:

        1. Round-trip mismatch — re-parsing and re-serializing the file
           doesn't produce identical bytes (rare; would catch oddly-encoded
           delimiters).
        2. Entry-size overflow — any single parsed entry exceeds the
           store's whole-file char limit. The tool budgets the ENTIRE store
           against that limit; no single tool-written entry can exceed it.
           When we see one entry larger than the limit, an external writer
           (patch tool, shell append, manual edit, sister session) appended
           free-form content into what the tool will treat as one entry.
           Flushing would then truncate that entry to the model's new
           content, discarding the appended bytes — issue #26045.

        Returns the absolute path of the .bak file when drift was found and
        backed up; returns None when the file looks tool-shaped.

        Note: this is an INSTANCE method (not static) because we need the
        per-target char_limit for signal #2.
        """
        # 中文：两条漂移信号——①重新序列化后字节不一致；②任何单条超过整库 char_limit
        #（说明外部写入者塞了自由文本）。命中则备份成 .bak.<ts> 并返回路径，调用方据此拒写，
        # 避免把别人追加的内容连同模型的新值一起原子覆盖掉（#26045，fail-closed）。
        path = self._path_for(target)
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return None
        if not raw.strip():
            return None

        parsed = [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]
        roundtrip = ENTRY_DELIMITER.join(parsed)

        char_limit = self._char_limit(target)
        max_entry_len = max((len(e) for e in parsed), default=0)

        drift_detected = (raw.strip() != roundtrip) or (max_entry_len > char_limit)
        if not drift_detected:
            return None

        # Drift confirmed — snapshot the file so the operator can recover
        # whatever the external writer added, then return the .bak path so
        # the caller can refuse the mutation.
        ts = int(time.time())
        bak_path = path.with_suffix(path.suffix + f".bak.{ts}")
        try:
            bak_path.write_text(raw, encoding="utf-8")
        except (OSError, IOError):
            return str(bak_path) + " (BACKUP FAILED — file unchanged on disk)"
        return str(bak_path)

    @staticmethod
    def _write_file(path: Path, entries: List[str]):
        """Write entries to a memory file using atomic temp-file + rename.

        Previous implementation used open("w") + flock, but "w" truncates the
        file *before* the lock is acquired, creating a race window where
        concurrent readers see an empty file. Atomic rename avoids this:
        readers always see either the old complete file or the new one.
        """
        # 中文：所有落盘的唯一出口，实现【原子写】——先写同目录临时文件、fsync 落地，
        # 再 atomic_replace(rename) 顶替目标。同目录保证同一文件系统、rename 才原子。
        # 这样读侧永不见空文件或半截文件（这正是 _read_file 敢不加锁的前提）；
        # 任何失败都清理临时文件并抛错，不留垃圾。
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        try:
            # Write to temp file in same directory (same filesystem for atomic rename)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".mem_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                atomic_replace(tmp_path, path)
            except BaseException:
                # Clean up temp file on any failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write memory file {path}: {e}")


def memory_tool(
    action: str,
    target: str = "memory",
    content: str = None,
    old_text: str = None,
    store: Optional[MemoryStore] = None,
) -> str:
    """
    Single entry point for the memory tool. Dispatches to MemoryStore methods.

    Returns JSON string with results.
    """
    # 中文：memory 工具的【唯一入口与 dispatch】，把 LLM 的调用路由到 MemoryStore 对应方法。
    # 自身只做参数把关（store 必须存在、target 仅 memory/user、各 action 的必填项），
    # 真正的写入/校验逻辑全在 store.add/replace/remove 里。最终把结果 dict 序列化成 JSON 串返回。
    # fail-closed：store 缺失（配置禁用或环境不支持）直接报错，不静默跳过。
    if store is None:
        return tool_error("Memory is not available. It may be disabled in config or this environment.", success=False)

    if target not in {"memory", "user"}:
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False)

    if action == "add":
        if not content:
            return tool_error("Content is required for 'add' action.", success=False)
        result = store.add(target, content)

    elif action == "replace":
        if not old_text:
            return tool_error("old_text is required for 'replace' action.", success=False)
        if not content:
            return tool_error("content is required for 'replace' action.", success=False)
        result = store.replace(target, old_text, content)

    elif action == "remove":
        if not old_text:
            return tool_error("old_text is required for 'remove' action.", success=False)
        result = store.remove(target, old_text)

    else:
        return tool_error(f"Unknown action '{action}'. Use: add, replace, remove", success=False)

    return json.dumps(result, ensure_ascii=False)


def check_memory_requirements() -> bool:
    """Memory tool has no external requirements -- always available."""
    # 中文：注册表用的可用性探针。memory 是纯本地文件存储、无任何外部依赖（无网络/无密钥），
    # 故恒返回 True，工具始终可注册可用。小工具。
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

# 中文：OpenAI function-calling 工具 schema —— 这段是【给 LLM 看的行为引导】，不是给代码的。
# description 里详尽写明「该记什么/不该记什么、两个 target 的分工、三种 action 的用法」，
# 本文件其余部分只管机制，而「判断该不该记、记成什么」的语义引导全靠这段文字喂给模型。
# 即本仓库刻意的「判断归 LLM、机制归代码」边界中，属于 LLM 那一侧的提示来源。
MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable information to persistent memory that survives across sessions. "
        "Memory is injected into future turns, so keep it compact and focused on facts "
        "that will still matter later.\n\n"
        "WHEN TO SAVE (do this proactively, don't wait to be asked):\n"
        "- User corrects you or says 'remember this' / 'don't do that again'\n"
        "- User shares a preference, habit, or personal detail (name, role, timezone, coding style)\n"
        "- You discover something about the environment (OS, installed tools, project structure)\n"
        "- You learn a convention, API quirk, or workflow specific to this user's setup\n"
        "- You identify a stable fact that will be useful again in future sessions\n\n"
        "PRIORITY: User preferences and corrections > environment facts > procedural knowledge. "
        "The most valuable memory prevents the user from having to repeat themselves.\n\n"
        "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO "
        "state to memory; use session_search to recall those from past transcripts.\n"
        "If you've discovered a new way to do something, solved a problem that could be "
        "necessary later, save it as a skill with the skill tool.\n\n"
        "TWO TARGETS:\n"
        "- 'user': who the user is -- name, role, preferences, communication style, pet peeves\n"
        "- 'memory': your notes -- environment facts, project conventions, tool quirks, lessons learned\n\n"
        "ACTIONS: add (new entry), replace (update existing -- old_text identifies it), "
        "remove (delete -- old_text identifies it).\n\n"
        "SKIP: trivial/obvious info, things easily re-discovered, raw data dumps, and temporary task state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
                "description": "The action to perform."
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace'."
            },
            "old_text": {
                "type": "string",
                "description": "Short unique substring identifying the entry to replace or remove."
            },
        },
        "required": ["action", "target"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="memory",
    toolset="memory",
    schema=MEMORY_SCHEMA,
    handler=lambda args, **kw: memory_tool(
        action=args.get("action", ""),
        target=args.get("target", "memory"),
        content=args.get("content"),
        old_text=args.get("old_text"),
        store=kw.get("store")),
    check_fn=check_memory_requirements,
    emoji="🧠",
)




