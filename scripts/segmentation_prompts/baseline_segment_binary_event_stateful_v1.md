# Instruction

## Context

- **Goal**: Your task is to segment a multi-turn conversation into large events. An event is an ongoing coherent situation. The conversation may be a dialogue between two participants or a meeting among several speakers. Successive turns within the same event should be grouped into the same segmentation unit, and a new segmentation unit should be created whenever one event ends, and another begins.

- **Data**: The input data is a series of turns separated by "\n\n". Each turn is started with "[Segment (Segment Number) | Turn (Turn Number)]: ". The **Segment Number shows the segmentation you have already committed to for the preceding turns** — turns sharing a Segment Number are in the same segmentation unit. Those earlier decisions are final and must not be revised.

# Data

{text_to_be_segmented}

# Question

You are shown the conversation up to and including the CURRENT turn, **[Turn {cur}]** (the last turn above, whose Segment Number is not yet assigned). The turn immediately before it belongs to **Segment {cur_seg}**, which currently contains **{cur_seg_len}** turn(s).

Using the same segmentation criterion as above — a new segmentation unit is created whenever one event ends, and another begins, while successive turns within the same event stay in the same unit — decide whether **[Turn {cur}]** begins a NEW segment (one event ends and another begins at this turn, so it opens Segment {next_seg}) or continues the SAME segment (**Segment {cur_seg}**).

Answer with exactly one word: NEW or SAME.

# Output
