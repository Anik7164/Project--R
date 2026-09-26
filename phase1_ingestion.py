import ollama

def run_phase3_pipeline(phase2_prompt):
    print("--- Phase 3: LLM Generation Engine ---\n")
    
    # 1. English Live Micro-Batch
    print("Generating English commentary...")
    english_response = ollama.generate(
        model='llama3',
        prompt=phase2_prompt
    )
    english_text = english_response['response'].strip()
    print(f"[English Output]: {english_text}\n")
    
    # 2. Bengali Translation Stream
    print("Running secondary streaming prompt for Bengali localization...")
    bengali_prompt = (
        "You are a professional Bengali sports commentator. "
        "Translate the following English commentary into highly energetic, "
        "colloquial Bengali. Provide ONLY the Bengali text in the Bengali script, "
        "without any English characters or introductory phrases.\n\n"
        f"Commentary: {english_text}"
    )
    
    bengali_response = ollama.generate(
        model='llama3',
        prompt=bengali_prompt
    )
    bengali_text = bengali_response['response'].strip()
    print(f"[Bengali Output]: {bengali_text}\n")

if __name__ == "__main__":
    # Injecting Event 4207 directly from your Phase 2 output
    sample_prompt = (
        "System Constraint: Generate a 1-sentence live sports commentary in English. "
        "Fact: In minute 4, Alexis Mac Allister took a shot from 29.4 yards out. "
        "Outcome: Saved. Do not hallucinate any other events or players."
    )
    
    run_phase3_pipeline(sample_prompt)