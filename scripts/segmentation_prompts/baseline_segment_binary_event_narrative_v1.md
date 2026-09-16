# Instruction

## Context

- **Goal**: Your task is to segment a narrative into large events. An event is an ongoing coherent situation. The narrative may be a written story or a spoken account told by a narrator. Successive sentences within the same event should be grouped into the same segmentation unit, and a new segmentation unit should be created whenever one event ends, and another begins.

- **Data**: The input data is a series of sentences separated by "\n\n". Each sentence is a single sentence of the narrative, started with "[Sentence (Sentence Number)]: ".

# Data

{text_to_be_segmented}

# Question

You are shown the narrative up to and including the CURRENT sentence, **[Sentence {cur}]** (the last sentence above). Using the same segmentation criterion as above — a new segmentation unit is created whenever one event ends, and another begins, while successive sentences within the same event stay in the same unit — decide whether **[Sentence {cur}]** begins a NEW segment (one event ends and another begins at this sentence) or continues the SAME segment as the preceding sentences.

Answer with exactly one word: NEW or SAME.

# Output
