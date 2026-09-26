# Project--R: Bridging the Semantic Gap in Bengali Sports Commentary

![Status](https://img.shields.io/badge/Status-Active-success)
![Course](https://img.shields.io/badge/Course-CSE498R-blue)
![Language](https://img.shields.io/badge/Output-Bengali-green)

**Project--R** is an automated multimodal pipeline for real-time sports commentary and post-match journalism generation using Large Language Models (LLMs). Developed as a CSE498R Directed Research initiative, this system bridges the "semantic gap" between raw mathematical telemetry—such as millisecond timestamps and $X/Y$ coordinates—and engaging human broadcast consumption.

By decoupling visual perception from narrative generation, the system executes locally on edge devices to synthesize high-energy, colloquially accurate Bengali sports commentary for grassroots and amateur matches without the need for expensive human media teams.

---

## 🛠️ System Architecture

The pipeline is engineered for low-latency edge inference and is structured into four sequential phases:

1. **Phase 1: Data Ingestion and Event Structuring**
   Parses high-velocity JSON telemetry streams and utilizes rule-based event triggers to instantly initiate translation protocols for critical events like shots, fouls, or goals.
2. **Phase 2: Semantic Translation Middleware**
   Geometrically enriches triggered event data by calculating exact shooting distances and angles via Euclidean coordinates, which are then injected into strict, constraint-enforced prompt templates.
3. **Phase 3: Bilingual Natural Language Generation**
   A quantized open-weights LLM (e.g., Llama-3 or Mistral) generates short live commentary batches in English, while a secondary streaming prompt concurrently translates this text into broadcast-ready Bengali.
4. **Phase 4: Streaming Audio Synthesis and Synchronization**
   The Bengali stream routes into a multilingual Text-to-Speech (TTS) API (e.g., ElevenLabs). A 1.5-second video buffer delay synchronizes the synthesized voice track with the live visual action on-screen.

### Pipeline Visualization

```mermaid
flowchart TD
    A["Phase 1: Event Structuring<br/>(Action filters & trigger rules)"] --> B["Phase 2: Semantic Translation<br/>(Euclidean distance, xG, prompt injection)"]
    B --> C["Phase 3: LLM Generation Engine<br/>(Llama-3 / Mistral constraint generation)"]
    
    C --> D["English Live Micro-Batch"]
    
    D --> E["Bengali Translation Stream"]
    D --> F["Text Output"]
    
    E --> G["Phase 4: Streaming TTS Engine<br/>(ElevenLabs Multilingual v2)"]
    
    G --> H["Synced Audio Broadcast"]
    F --> H
