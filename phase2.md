## Data-wise justification 

* SoccerNet provides full-length match videos plus timestamped event annotations, which is exactly what’s required to train and evaluate a temporal action model and to generate match timelines and analytics.
* Unlike RTÉ/GAA sources (restricted/no API/no reuse guarantees), SoccerNet is a research dataset with consistent structure, making Phase 2 pipeline reproducible, testable, and completable.



## Final Year Project: Soccer Match Video Analytics + LLM Match Review Dashboard

### Best-quality action understanding (temporal model, not single-frame)
* Replace current **middle-frame ResNet50** approach with a **temporal video model** that uses multiple frames so it truly “sees” the action.
* Model options (quality-first):
  * **SlowFast / X3D / I3D** (strong, proven for action recognition)
  * **Video Transformer** (e.g., **TimeSformer / VideoMAE fine-tuning**)
* Training/Evaluation:
  * Train and evaluate on **SoccerNet action labels**
  * Use a short temporal window during inference (e.g., **32–64 frames** per clip window)

---

### Full-match dense timeline (spotting, not just classifying extracted clips)
* Run the trained temporal model across the **entire half/match** to generate an **action probability timeline** over time (per class).
* Post-processing:
  * Use **peak picking / non-maximum suppression (NMS)** to select final event timestamps.
* Output artifact:
  * `match_events.json` containing:
    * `timestamp`, `half`, `label`, `confidence`, `clip_reference` (path or time range)

---

### Player-wise analytics (the biggest “dashboard value”)
* Add **player detection + multi-object tracking** across the match footage.
* For each tracked player (Track-1, Track-2, etc.):
  * **Heatmap / zone occupancy**
  * **Average speed (approx)** + **distance proxy**
  * **Time on screen**
  * **Event involvement**: which tracks are closest / most present during detected actions
* Optional “extra”:
  * **Jersey number OCR** when visible → probabilistic player labeling

---

### Ball + possession-style metrics (advanced analytics)
* Add **ball detection** (hard but high value).
* Even imperfect ball detection enables:
  * **Ball-in-play segments**
  * **Attacking phase proxy** (ball location + player density)
  * **Tempo/momentum graphs** (event bursts + ball movement proxies)

---

### LLM match review (grounded + explainable)
* LLM does **not** interpret raw video.
* LLM reads only your computed outputs:
  * `match_events.json` (timeline + counts)
  * player tracking summaries (player stats JSON)
  * key clips list (top confidence + review queue)
* Generates:
  * Match summary
  * Key moments list (with timestamps)
  * Discipline + set-piece breakdown
  * “Most involved players” based on tracking stats
  * “Review list” for uncertain events (low confidence + timestamps)

---

## What’s different from others (uniqueness)
* **Quality-first temporal modeling** (not single-frame classification)
* **Full-match timeline spotting** with structured outputs (`match_events.json`)
* **Player tracking analytics** (heatmaps, involvement, movement proxies)
* **Ball/possession-style metrics** for deeper match understanding
* **LLM-generated reports grounded in model outputs** (no hallucination from video)
* **Dashboard + review workflow** (product-like end-to-end system)  
Most papers stop at reporting mAP—this delivers an **analytics platform**.
---

**Title:** *"Two-Stage Action Spotting with LLM-Grounded Interactive Match Analytics Dashboard"*

### Contribution 1: Two-Stage Spotting Pipeline
- Stage 1: Lightweight proposal network (MobileNet/EfficientNet at 2–5 fps)
- Stage 2: Temporal classifier (TSM/SlowFast) only on candidates
- **Must show** speedup vs dense sliding window **without** significant mAP drop

### Contribution 2: LLM-Grounded Dashboard (not just a report)
- Structured CV outputs → LLM → interactive Streamlit/Gradio dashboard
- Frame it as: *"grounded multimodal analytics interface"* — the dashboard IS the contribution, not just a PDF summary
- **Must formally evaluate**: human study (coaches/analysts rate usefulness) + automated metrics (ROUGE/BERTScore of generated summaries vs real match reports)

### Contribution 3: End-to-End System Design
- Full pipeline: video → spotting → tracking → structured JSON → LLM → interactive dashboard
- This "systems" contribution is valid for IEEE applied/multimedia conferences

---

---


## 5-Week Plan

### Week 1 - Full Data + Temporal Model
- Download **full SoccerNet-v2** (~500 games), replace your 2-game setup
- Train **TSM-ResNet50 or SlowFast** on multi-frame clips, evaluate with **mAP@{1,2,5}s**

### Week 2 - Two-Stage Pipeline + Baselines
- Build Stage 1 (EfficientNet-Lite at 2–5 fps → candidate timestamps) + Stage 2 (temporal classifier on candidates only)
- Run **NetVLAD++, CALF baselines** + **ablation table** (Stage 1 only / Stage 2 only / combined / dense) + log **FPS per stage**

### Week 3 - Player Tracking + Dashboard
- Add **YOLOv8 + ByteTrack** → `player_stats.json` (heatmaps, event involvement)
- Build **Streamlit dashboard**: event timeline, player heatmaps, action filters, export (CSV/JSON/PDF)

### Week 4 -  LLM Integration + Evaluation
- Wire **LLM** into dashboard (reads `match_events.json` + `player_stats.json` → interactive match summary, key moments, turning points)
- Run **ROUGE/BERTScore** on LLM summaries + **user study** (5–10 people, Likert scale rating on dashboard usefulness)

### Week 5 - Paper Writing
- Write Project report  & IEEE paper: pipeline diagram, mAP table, ablation table, latency table, dashboard screenshots, user study results
- Clean repo for reproducibility + final README

---
