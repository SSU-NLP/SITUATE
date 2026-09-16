# Instruction

## Context

- **Goal**: Your task is to segment a narrative into topically coherent units based on semantics. The narrative may be a written story or a spoken account told by a narrator. Successive sentences on the same topic should be grouped into the same segmentation unit, and a new segmentation unit should be created when a topic shift occurs.

- **Data**: The input data is a series of sentences separated by "\n\n". Each sentence is started with "[Segment (Segment Number) | Sentence (Sentence Number)]: ". The **Segment Number shows the segmentation you have already committed to for the preceding sentences** — sentences sharing a Segment Number are in the same segmentation unit. Those earlier decisions are final and must not be revised.

# Data

{text_to_be_segmented}

# Question

You are shown the narrative up to and including the CURRENT sentence, **[Sentence {cur}]** (the last sentence above, whose Segment Number is not yet assigned). The sentence immediately before it belongs to **Segment {cur_seg}**, which currently contains **{cur_seg_len}** sentence(s).

Using the same segmentation criterion as above — a new segmentation unit is created when a topic shift occurs, while successive sentences on the same topic stay in the same unit — decide whether **[Sentence {cur}]** begins a NEW segment (a topic shift occurs at this sentence, so it opens Segment {next_seg}) or continues the SAME segment (**Segment {cur_seg}**).

Answer with exactly one word: NEW or SAME.

# Output
