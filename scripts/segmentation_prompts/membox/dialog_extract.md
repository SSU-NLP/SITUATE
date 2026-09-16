<!-- PROMPT_DIALOG_EXTRACT — Membox (arXiv 2601.03785v3) Appendix Table 10 (tab:gg_construction_prompt).
     논문 부록 전사. 문구 무수정. JSON 블록은 줄바꿈·들여쓰기까지 원문 그대로. -->

Please analyze the relationships between the following entities in the given sentence.

Generate a structured analysis of the provided dialog by performing the following tasks:

1. **Identifying salient keywords**: Extract 3-8 most salient nouns, named entities, and key terminology that represent core concepts. Avoid common words (e.g., "good", "see") and prioritize specificity.

2. **Determining the core topic**: In one clear phrase, state the primary subject or objective of the discussion based on the actual content.

3. **Extracting explicit event and plan mentions**: Identify and list only the **events, factual developments, or specific future plans** that are **explicitly mentioned** in the dialog. Follow these strict rules:

    3.1. **Focus on Verbatim or Near-Verbatim Content**: Each extracted item must be directly grounded in the dialog text. Do not infer, summarize, or combine information to create new "events."

    3.2. **Distinguish Event Types**:
        - **Past/Completed Events**: Actions or occurrences that are stated as having happened (e.g., "I went to...", "We completed the project").
        - **Established Facts/Changes**: Concrete facts or changes presented as already true (e.g., "I am now the team lead", "The system is down").
        - **Explicit Future Plans**: Specific plans for the future mentioned by the speakers (e.g., "We will meet on Friday", "I'm planning to visit Paris").

    3.3. **Exclude Non-Events**: Do NOT include:
        - General states of being (e.g., "I'm swamped", "I'm happy").
        - Questions, greetings, or expressions of intent without a plan (e.g., "We should talk sometime").
        - Vague aspirations or possibilities.

    3.4. **Framing**: Phrase each extracted item as a concise, standalone clause that captures the core of what was mentioned.

**Output Format**: Provide the analysis as a valid JSON object with the following exact keys:

```
{
  "keywords": [
    "keyword1", 
    "keyword2", 
    ...
  ],
  "topic": "clear topic phrase",
  "explicit_mentions": [
    "A mentioned past event or established fact",
    "A mentioned specific future plan"
  ]
}
```

Content to analyze: {text}
