# Primary Protocol

## Frozen design

The primary benchmark was frozen before full outcome analysis. Model selection did not use outcome accuracy. The full panel was evaluated under native-chat serialization.

## Dataset split names and counts

- MedQA: 1,273 items
- MedMCQA: 4,183 items
- PubMedQA: 500 items
- Total per model: 5,956 items

Question text, answer options, and gold answers are not redistributed in this repository.

## Model panel

The five frozen models and full revisions are listed in `configs/model_panel.json`.

## Prompts and parser

- Direct prompt SHA256: 4c55da300b12678fce4edd7350a3d4c4aa1a6ad974090f2d6f341f911ab246c4
- CoT prompt SHA256: cd1ab2c4e0d4aff7dab21f64a7c4b237b641300bfef0df52ac203577ccae4956
- Structured prompt SHA256: 0f49a734980a15198f163c001ab903fb7d425f353b3ed0fa6c5d08f0018f8068
- Parser SHA256: cbaa75e20ff467ed81a3272e79de4ee7e97e3e9e94379ee40ac3d0332c959223

## Fusion definition

Fusion is derived after generation. If at least two of Direct, CoT, and Structured agree, Fusion selects the majority answer. If all three disagree, the primary fallback order is Direct > CoT > Structured. All six fallback permutations were used for sensitivity analysis.

## Freeze chronology

The prompt files, parser, dataset membership hash, model panel, serving condition, and generation settings were frozen before primary outcome analysis.
