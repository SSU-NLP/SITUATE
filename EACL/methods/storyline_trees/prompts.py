"""Storyline Trees 장면 추출 프롬프트 — 논문 Appendix Table 14 문구 그대로.

PDF 추출본의 하드랩만 풀었고 어휘·구조·번호는 바꾸지 않았다. 원문 대조용은
`prompt_paper_verbatim.txt` 에 추출 그대로 보관한다.

저자 레포(github.com/Leonard907/StoryTree)에 코드가 없으므로 이 프롬프트가
재구현의 유일한 사양이다.
"""

SCENE_EXTRACTION = """You are an expert literary editor and structural analyst. Your task is to identify and extract the *major narrative scenes* from the provided book chapter.

# 1. Definition of a Scene
A **scene** is a distinct unit of dramatic action. While scenes often follow the unity of time and place, **Narrative Unity** is the most important factor.

**Criteria for a Scene:**
* **The Container:** It typically occurs in a continuous block of time.
* **The Content:** It centers on a character pursuing a specific goal or dealing with a specific conflict.
* **The Change:** It results in a change of state (emotional, physical, or informational) by the end.

**Criteria for Scene Breaks (When to Cut):**
* **Significant Time Jump:** The narrative moves from "afternoon" to "evening," or skips days/years.
* **Major Location Shift:** A complete change of setting (e.g., moving from a house to a workplace).
* *Exception:* Do NOT create a new scene for minor transit (e.g., walking from the kitchen to the living room, or driving a car) unless a major plot twist occurs specifically during the transit. Merge these transitions into the main scene they support.
* **POV Switch:** The narrator changes (e.g., strictly switching from Character A's head to Character B's head).

**What is NOT a Scene:**
* Brief moments of introspection without action.
* Short transitions between locations.
* Summary passages ("The next three months were hard...") -- treat these as the introduction to the next scene or the conclusion of the previous one.

# 2. Strict Constraints (Crucial)
1. **Full Coverage:** The extracted scenes must cover the **entirety** of the chapter text. There must be no "gaps" or text left out between scenes. `ending_sentence` represents the end of the scene as the sentence number (e.g., `5` means sentence number 5, which corresponds to the sentence labeled `[5]` in the chapter text). The first scene will automatically start at the beginning of the chapter text, and every subsequent scene will start from the text immediately after the `ending_sentence` of the previous scene. The final scene MUST have `ending_sentence` equal to the last sentence number of the chapter text.
2. **Sentence Numbers:** Each sentence in the chapter text is labeled with a number in brackets (e.g., `[1]`, `[2]`, `[3]`). You must use **only the sentence number** (as an integer, without brackets) as the `ending_sentence` value. Do NOT use the full sentence text. For example, if a scene ends at the sentence labeled `[5]`, set `ending_sentence` to `5` (not the actual sentence text).

# 3. Output Instructions
Analyze the chapter text. Extract a list of scenes. For each scene, output a JSON object with the following fields:
* **`scene_title`**: A short, punchy, 3-6 word title capturing the essence of the scene.
* **`ending_sentence`**: The sentence number (as an integer) that marks the absolute end of this scene. This must be **only the number** from the label, not the full sentence text (e.g., if the sentence is labeled `[5]`, use `5`, not the actual sentence content).
* **`synopsis`**: A concise summary of the *external action* (Who did what? What events occurred?).
* **`significance`**: An analysis of the *internal importance*. Why does this scene exist? (e.g., reveals Character X's weakness, advances the mystery plot, establishes the theme of betrayal).

# 4. Final Output Format
Return ONLY a valid JSON array. Do not wrap the JSON in markdown code blocks or add conversational text.

Example output:
[
{
"scene_title": "The Argument in the Kitchen",
"ending_sentence": 5,
"synopsis": "John confronts Mary about the missing money. Mary denies it but reveals she lost her job.",
"significance": "The scene shifts the dynamic from partnership to suspicion. It introduces the theme of financial insecurity and establishes Mary's tendency to lie under pressure."
}
]

Now, analyze the following book chapter and extract the scenes.

<chapter>
{{{chapter_text}}}
</chapter>"""
