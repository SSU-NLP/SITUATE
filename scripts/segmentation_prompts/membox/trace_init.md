<!-- PROMPT_TRACE_INIT — Membox (arXiv 2601.03785v3) Appendix Table 12 (tab:choose_prompt).
     논문 부록 전사. 문구 무수정. 예시 2건 포함. -->

You are an event chain constructor for building coherent memory structures. Your task is to analyze a set of events and organize them into logical chains.

**Task:**
Given a set of events, identify the primary narrative thread and any associated events that form a coherent event chain.

**Process:**
1. Analyze all events to identify the most prominent theme or activity
2. Connect events that share temporal, causal, or thematic relationships
3. Form the most coherent sequence possible
4. Identify any events that don't fit into the main narrative thread

**Output Format:**

```
{
    "primary_chain": ["Events forming the most coherent narrative, in logical order"],
    "secondary_chains": [["Other potential chains, if any"]],
    "isolated_events": ["Events that don't fit into any chain"],
    "chain_summary": "Brief description of the primary chain's theme and context"
}
```

**Examples:**

**Example 1:**
Events: ["I woke up at 7 AM", "I checked my email", "I had breakfast", "Then I went for a run"]

Output:

```
{
    "primary_chain": ["I woke up at 7 AM", "I had breakfast", "Then I went for a run"],
    "secondary_chains": [],
    "isolated_events": ["I checked my email"],
    "chain_summary": "Morning routine including waking, eating, and exercise"
}
```

**Example 2:**
Events: ["Started a new project at work", "Researched design patterns", "Met with the client", "Created initial wireframes", "Had lunch with a colleague"]

Output:

```
{
    "primary_chain": ["Started a new project at work",
    "Researched design patterns", "Created initial wireframes"],
    "secondary_chains": [["Met with the client"]],
    "isolated_events": ["Had lunch with a colleague"],
    "chain_summary": "Work project initiation and initial design phase"
}
```

**Now analyze:**
Events: {events}
Output your analysis in JSON format.
