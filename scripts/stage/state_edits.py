"""Evidence-bounded edits to a persistent claim list; no benchmark gold inputs."""
from __future__ import annotations

from copy import deepcopy


EDIT_INSTRUCTIONS = """
Maintain the previous_state as a persistent list of claims. Claims you do not
explicitly remove or replace will be carried forward verbatim. Edit only items
whose validity or importance changes. Do not infer character knowledge from an
unwitnessed event. Preserve independent goals when relationships change.
Return a JSON object with exactly these keys:
{"remove": [{"claim": "exact previous claim", "evidence_scene_orders": [1]}],
 "replace": [{"previous_claim": "exact previous claim", "claim": "updated claim",
              "evidence_scene_orders": [1]}],
 "add": [{"claim": "new current claim", "evidence_scene_orders": [1]}],
 "developments_since_previous_checkpoint": [{"claim": "effective interval change",
                                             "evidence_scene_orders": [1]}]}
Use empty lists when no operation is needed. Every edit must cite at least one
scene in the supplied interval memory. Never cite future scenes. Select at most
8 current claims after applying the edits, and at most 4 developments. If new
claims require space, explicitly remove a less salient prior claim, citing the
new evidence that makes its replacement more salient. Do not repeat a previous
claim in add. Do not remove and replace the same claim. No analysis or extra keys.
"""

EDIT_INSTRUCTIONS_ZH = """
将 previous_state 作为持久的断言列表维护。未显式删除或替换的断言将逐字保留。
仅修改有效性或重要性发生变化的条目。角色未目睹的事件不代表角色知道该事件。
关系改变时，保留仍有效的独立目标。
只返回包含以下字段的 JSON：
{"remove": [{"claim": "原有断言的完整原文", "evidence_scene_orders": [1]}],
 "replace": [{"previous_claim": "原有断言的完整原文", "claim": "更新后的断言",
              "evidence_scene_orders": [1]}],
 "add": [{"claim": "新的当前状态", "evidence_scene_orders": [1]}],
 "developments_since_previous_checkpoint": [{"claim": "本区间生效的变化",
                                             "evidence_scene_orders": [1]}]}
无需操作时使用空列表。每次修改必须引用给定区间记忆中的至少一个场景，禁止引用未来场景。
应用修改后，当前状态最多包含 8 条断言，变化最多包含 4 条。
如需为新状态腾出空间，必须显式删除较不重要的旧断言，并引用使新状态更重要的区间证据。
不要在 add 中重复旧断言，也不要同时删除和替换同一断言。不要输出分析或额外字段。
"""


def apply_edits(previous: list[str], patch: dict, allowed_scenes: set[int]) -> dict:
    """Apply validated edits while retaining unmentioned claims exactly."""
    if set(patch) != {"remove", "replace", "add", "developments_since_previous_checkpoint"}:
        raise ValueError("Unexpected edit fields")
    if any(not isinstance(patch[k], list) for k in patch):
        raise ValueError("Edit fields must be lists")
    current = [{"claim": text, "evidence_scene_orders": []} for text in previous]
    original = set(previous)
    touched = set()
    for operation in ("remove", "replace", "add"):
        for row in patch[operation]:
            keys = {"claim", "evidence_scene_orders"}
            if operation == "replace":
                keys.add("previous_claim")
            if not isinstance(row, dict) or set(row) != keys:
                raise ValueError(f"Invalid {operation} fields")
            evidence = row["evidence_scene_orders"]
            if (not isinstance(evidence, list) or not evidence
                    or any(type(n) is not int or n not in allowed_scenes for n in evidence)):
                raise ValueError("Edits need interval evidence")
            if not isinstance(row["claim"], str) or not row["claim"].strip():
                raise ValueError("Empty edit claim")
            if operation != "add":
                old = row["previous_claim"] if operation == "replace" else row["claim"]
                if old not in original or old in touched:
                    raise ValueError("Unknown or multiply edited previous claim")
                touched.add(old)
                index = next(i for i, item in enumerate(current) if item["claim"] == old)
                if operation == "remove":
                    current.pop(index)
                else:
                    current[index] = {"claim": row["claim"], "evidence_scene_orders": evidence}
            else:
                if row["claim"] in original:
                    raise ValueError("Use replace for existing claims")
                current.append(deepcopy(row))
    if len(current) > 8:
        raise ValueError("Edited current state exceeds eight claims")
    normalized = [" ".join(row["claim"].casefold().split()) for row in current]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate resulting state claims")
    return {"current_state": current,
            "developments_since_previous_checkpoint": patch["developments_since_previous_checkpoint"]}
