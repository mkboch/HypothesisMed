# SC-3 Protocol

SC-3 was a prospective same-prompt self-consistency control added after the primary benchmark protocol was frozen. It did not reopen the model panel, dataset membership, parser, Direct prompt, Structured prompt, or primary benchmark generations.

## Prompt and samples

SC-3 uses the frozen CoT prompt with SHA256 `cd1ab2c4e0d4aff7dab21f64a7c4b237b641300bfef0df52ac203577ccae4956`. Each item receives three prospective CoT samples.

## Seeds

The frozen SC-3 sample seeds are 20260831, 20260832, and 20260833.

## Aggregation rule

If at least two recovered samples agree, SC-3 selects the majority answer. If all recovered valid answers disagree, SC-3 selects the first recovered valid answer using the primary frozen sample priority sample_1 > sample_2 > sample_3. All six sample-priority permutations were evaluated for sensitivity.

## Matched generation count

SC-3 is matched to heterogeneous Fusion by model-call count: three model calls per item. It is not matched on output-token compute.

## Freeze chronology

The SC-3 protocol, seeds, prompt hash, parser hash, generation settings, and sample-priority rule were frozen before SC-3 outcome analysis.
