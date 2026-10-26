# Conference paper: what is still needed before submission

Draft: `main.tex` (+ `refs.bib`), compiled `main.pdf`. Every red [TODO] in the PDF is listed here.

## Experiments (in priority order)
1. **Latency + entity accuracy vs. LLM baselines** (the paper's main claim, Table III).
   Ours (templates) vs. English LLM -> NLLB/IndicTrans2 vs. LLM writing Bengali directly.
   Same GPU; per-event ms to text and to start of speech; % wrong/invented player names.
2. **Tracking on SoccerNet-Tracking test (49 clips)**: `python benchmarks/tracking_benchmark.py`
   (HOTA/IDF1/IDsw, each tracker with and without our ID layer, Table I).
3. **Spotting on SoccerNet-v2 test**: `python benchmarks/spotting_benchmark.py`; check the four
   official baselines reproduce published mAP before citing anything.
4. **Detector**: mAP of COCO YOLO26 vs. our fine-tuned football YOLO26 on SoccerNet-Tracking test.
5. **Own labels on 2-3 full broadcast matches** (Labels-v2 format) for fused events
   (scoreboard + cards need raw video) and for cut handling, which SoccerNet clips don't test.
6. **Shirt reading** accuracy on the SoccerNet jersey-number test set; fallback level usage.
7. **Human ratings**: >=3 native speakers, >=100 events, fluency / correctness / excitement.
8. End-to-end fps per stage on one GPU.

## Writing
- Verify every new bib entry (marked VERIFY in refs.bib) and the Bengali speaker count.
- Add E2E-Spot / T-DEED citations; confirm no prior Bengali video-commentary work.
- Supervisor email and author order; check the venue's page limit and template.
