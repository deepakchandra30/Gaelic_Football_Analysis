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

## 2 stage inference (optional — speed focused, if you later want it)
### A) Stage 1: cheap proposal generation (fast)
* Sample video at low FPS (e.g., **2–5 fps**)
* Use a lightweight model (e.g., **MobileNetV3 / EfficientNet-Lite**) or precomputed features
* Output **candidate timestamps** where something likely happens

### B) Stage 2: accurate classification (only on candidates)
* For each candidate timestamp, run a stronger model on a short window (e.g., **2–6 seconds**):
  * **TSM (Temporal Shift Module)** (fast + strong)
  * or a **small Video Transformer** for higher accuracy  
* Result: far fewer clips processed than sliding window → much faster end-to-end

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
