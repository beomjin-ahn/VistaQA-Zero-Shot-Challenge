# VistaQA Zero-Shot Grounding Challenge

System report: "A VLM and SAM 3 Pipeline for the VistaQA Zero-Shot Grounding Challenge",
NeurIPS 2026 Workshop VLM4RWD.

GROVE (x100) on the 1,157 samples of VistaQA v1.1:

| epsilon 0.01 | epsilon 0.05 | epsilon 0.1 | answer accuracy T | mask score M |
|---|---|---|---|---|
| 77.52 | 79.21 | 80.61 | 83.75 | 81.54 |

## Files

- `vistaqa.py`: the pipeline. Gemini 4 Argon (OpenRouter, Google Vertex) answers and boxes the evidence, SAM 3 segments the boxes, SAM 2.1 refines small targets.
- `results/predictions.jsonl`: predictions in the official format.
- `results/eps0.01`, `results/eps0.05`, `results/eps0.1`: official evaluation reports.

## Predict

```bash
pip install -r requirements.txt
OPENROUTER_API_KEY=<key> python vistaqa.py --inputs <VistaQA folder> --out predictions.jsonl --workers 8
```

The input folder holds the per-sample json files and images of `mozhina/VistaQA`. `facebook/sam3` is gated on Hugging Face.

## Score

Official code: https://github.com/VistaQA/VistaQA-Evaluation (commit `6ad724f`). Judge: `Qwen/Qwen2.5-14B-Instruct` served by vLLM.
