# hinglish-mlops

End-to-end MLOps pipeline for **Hinglish** (Hindi + English, Roman script) sentiment
classification. Built to be cloned and run in ~10 minutes on a free, CPU-only setup.

> Status: under active construction (built one phase at a time). Full documentation,
> architecture diagram, results table and limitations arrive in the final phase.

## Dataset

- **Train/val/test:** SemEval-2020 Task 9 "SentiMix" (Hinglish), 3-class sentiment.
- **Cross-dataset evaluation:** `shae2977/hinglish-youtube-sentiments-dataset` (CC-BY-4.0),
  used only for an honest generalization check — never for training.

See `reports/data_card.md` (added in Phase 2) for details and licensing.

## License

MIT — see [LICENSE](LICENSE).
