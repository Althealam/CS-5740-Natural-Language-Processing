"""
v2: Prompts/seeds aligned with logged probe failures (see log.md):
  afternoon/PM misclassified as unsafe; "Morning" greeting + work confused with unsafe;
  negation/refusal; benign off-topic questions; Chinese–English short lines.

Label 1 (unsafe): invite Bob to exercise (any time) OR schedule morning/early meet-call.
Label 0 (safe): everything else (PM/afternoon meets, greeting+work, refusals, random Qs).

Outputs:
  data/email_dataset_train_augmented_v2.jsonl
  data/email_dataset_dev_augmented_v2.jsonl
"""

import os
import json
import random
from openai import OpenAI

random.seed(42)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Few-shot seeds — v2 stresses contrasts the baseline generator missed
LABEL_1_SEEDS = [
    # Morning / early scheduling (unsafe)
    "talk at 8am?",
    "meet tmrw morning?",
    "Quick call at 7:45 AM?",
    # Exercise any time (still unsafe per homework)
    "run?",
    "gym at 6:30 PM?",
    "lift after work?",
    # Short informal
    "run tmrw?",
    "squat @ 7:15 tmrw?",
    # Implicit / metaphor
    "Let's get our blood pumping before standup",
    # Chinese–English unsafe (contrast with afternoon-only LABEL 0 seeds)
    "明天早上一起跑步吗？",
    "明天一起run吗？",  # invites run — unsafe when asking Bob to run together
]

LABEL_0_SEEDS = [
    # --- (A) PM / afternoon / evening scheduling — must be LABEL 0 (model tended to mark unsafe) ---
    "Let's meet in the afternoon",
    "Let's meet in the afternoon tomorrow",
    "Quick call at 7:45 PM tomorrow?",
    "Quick call at 7:45 PM tmr?",
    "Sync at 4:30 PM Friday?",
    "Dinner at 8 tonight?",
    "Coffee at 3pm?",
    # --- (B) "Morning" = greeting / opener only, then work ask — NOT a morning meet invite ---
    "Morning Bob—can you send the slides by noon?",
    "Morning Bob - need the budget numbers by EOD thanks",
    "Morn Bob, any update on the invoice?",
    "Good morning Bob, please review the attached doc when you can.",
    # --- (C) Negation / refusal — declines exercise or morning plans ---
    "I hate yoga and I avoid it completely",
    "No morning meetings for me thanks",
    "I'm not doing early runs anymore",
    "I refuse to join morning workouts",
    "Don't count me in for dawn hikes",
    # --- (D) Benign off-topic questions — no invite to exercise/morning meet ---
    "How do I bake a birthday cake?",
    "Bob, where's the nearest FedEx?",
    "What's a good PDF merger tool?",
    "Can you remind me how to reset my password?",
    # --- (E) Chinese–English mix, LABEL 0 = afternoon/evening social/work only ---
    "下午要不要coffee？",
    "晚上一起吃饭吗Bob？",
    "明天下午开个会好吗？",  # tomorrow afternoon meeting → safe
    "Hi Bob，report今天能发我吗？",  # work ask
    # Generic work / social (baseline variety)
    "Bob,\n\nCan you send the budget spreadsheet?\n\nCheers",
    "Bob, lunch?",
    "Call at 2PM?",
]

PROMPT_TEMPLATE = """You are generating realistic short emails to Bob for binary classification.

DEFINITIONS:
- LABEL 1 (unsafe): Invites Bob to PHYSICAL EXERCISE (run, gym, yoga, hike, sports, etc.) at ANY time OR schedules meeting/call/social with Bob in the EARLY DAY / morning window (words like morning, sunrise, dawn, before work, or explicit times like 7:30 AM meaning an actual morning meet/call).
- LABEL 0 (safe): All other emails.

MODEL FAILURE MODES YOUR OUTPUT MUST DIRECTLY FIX (generate many training-like examples):
1) PM / afternoon errors: Phrases like "Let's meet in the afternoon", "Quick call at 7:45 PM", "tmr" + PM MUST be LABEL 0 unless exercise is explicitly invited.
2) "Morning" greeting trap: If "Morning" / "Good morning" only opens the email and the ask is deliverables/slides/reports (no morning meet/call), LABEL 0.
3) Negation: Sentences refusing yoga/runs/morning meetings/hikes → LABEL 0.
4) Random benign questions (recipes, logistics, tools) with NO exercise/morning scheduling → LABEL 0.
5) Chinese–English mix: Use short mixes; LABEL 1 only if clearly inviting Bob to exercise OR morning meet; LABEL 0 for 下午/晚上 social/work or purely work asks without morning/exercise invite.

PAIRING RULE: When you use similar wording, create contrasts — same meeting verb + AM time (LABEL 1) vs PM / afternoon (LABEL 0).

LABEL 1 EXAMPLES:
{label_1_examples}

LABEL 0 EXAMPLES:
{label_0_examples}

{f_batch_focus}

Generate exactly 20 NEW emails labeled 1 and 20 NEW labeled 0 (40 JSON objects total). Include several items explicitly addressing failure modes (1)-(5) above. Mix 3–12 word blurbs and slightly longer lines; use tmrw, tmr, PM/AM times.

Return ONLY a valid JSON array, no markdown:
[{{"email": "...", "label": 1}}, {{"email": "...", "label": 0}}, ...]
"""

# One rotating focus per vulnerability bucket (+ contrasts)
BATCH_FOCUS = [
    "BATCH FOCUS (PM/afternoon): ≥8 LABEL 0 with afternoon/evening/tmr+PM/call/meet/dinner/coffee; LABEL 1 must NOT duplicate those afternoon-only patterns.",
    "BATCH FOCUS (Morning greeting): ≥8 LABEL 0 starting with Morning/Good morning/morn then ONLY work asks (slides, report, budget); zero morning workout invites.",
    "BATCH FOCUS (Negation): ≥6 LABEL 0 refusing yoga/run/hike/morning meeting/dawn walk; LABEL 1 invites only clear positives.",
    "BATCH FOCUS (Benign Q): ≥6 LABEL 0 off-topic questions (food, shipping, software); no exercise keywords unless LABEL 1 explicitly invites Bob.",
    "BATCH FOCUS (Zh-En): ≥6 LABEL 0 with Chinese + English (下午/晚上/work); ≥4 LABEL 1 only if 晨练/早上/run/gym invite to Bob is explicit.",
    "BATCH FOCUS (Minimal pairs): ≥6 pairs — same skeleton with AM vs PM or morning vs afternoon; labels must flip correctly.",
]


def extract_json(response_text):
    """Extract JSON array from response."""
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    start = response_text.find("[")
    end = response_text.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(response_text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def generate_batch(label_1_samples, label_0_samples, batch_focus: str):
    label_1_str = "\n".join([f"- {ex}" for ex in label_1_samples])
    label_0_str = "\n".join([f"- {ex}" for ex in label_0_samples])

    prompt = PROMPT_TEMPLATE.format(
        label_1_examples=label_1_str,
        label_0_examples=label_0_str,
        f_batch_focus=batch_focus,
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.85,
            max_tokens=4000,
        )

        data = extract_json(response.choices[0].message.content)
        if not data or not isinstance(data, list):
            print("Error: Could not parse JSON response")
            return []

        valid = []
        for item in data:
            if isinstance(item, dict) and "email" in item and "label" in item:
                if item["label"] in [0, 1] and item["email"].strip():
                    valid.append((item["email"].strip(), item["label"]))

        return valid

    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return []


def generate_dataset(target_per_label=500, batch_size=20):
    """Generate dataset by iterating with few-shot examples."""
    label_1_data = []
    label_0_data = []
    seen = set()

    iteration = 0
    max_iterations = 55

    while (
        len(label_1_data) < target_per_label or len(label_0_data) < target_per_label
    ) and iteration < max_iterations:
        iteration += 1

        label_1_samples = random.sample(LABEL_1_SEEDS, min(5, len(LABEL_1_SEEDS)))
        if label_1_data:
            label_1_samples.extend(
                [email for email, _ in random.sample(label_1_data, min(2, len(label_1_data)))]
            )

        label_0_samples = random.sample(LABEL_0_SEEDS, min(6, len(LABEL_0_SEEDS)))
        if label_0_data:
            label_0_samples.extend(
                [email for email, _ in random.sample(label_0_data, min(2, len(label_0_data)))]
            )

        focus = BATCH_FOCUS[(iteration - 1) % len(BATCH_FOCUS)]
        print(f"Iteration {iteration}: batch focus → {focus[:60]}…")

        batch = generate_batch(label_1_samples, label_0_samples, batch_focus=focus)

        for email, label in batch:
            if email not in seen:
                seen.add(email)
                if label == 1 and len(label_1_data) < target_per_label:
                    label_1_data.append((email, label))
                elif label == 0 and len(label_0_data) < target_per_label:
                    label_0_data.append((email, label))

        print(f"  Total: {len(label_1_data)} label 1, {len(label_0_data)} label 0")

    return label_1_data[:target_per_label], label_0_data[:target_per_label]


def save_dataset(label_1, label_0, train_ratio=0.8):
    """Save dataset to JSONL (v2 filenames)."""
    os.makedirs("data", exist_ok=True)

    all_data = label_1 + label_0
    random.shuffle(all_data)

    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    dev_data = all_data[split_idx:]

    train_path = os.path.join("data", "email_dataset_train_augmented_v2.jsonl")
    dev_path = os.path.join("data", "email_dataset_dev_augmented_v2.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for email, label in train_data:
            email_text = email.replace("\n", " ").strip()
            f.write(json.dumps({"text": email_text, "label": label}) + "\n")

    with open(dev_path, "w", encoding="utf-8") as f:
        for email, label in dev_data:
            email_text = email.replace("\n", " ").strip()
            f.write(json.dumps({"text": email_text, "label": label}) + "\n")

    print(f"\n[v2] Training data saved to {train_path} ({len(train_data)} examples)")
    print(f"[v2] Dev data saved to {dev_path} ({len(dev_data)} examples)")
    print(f"Total: {len(all_data)} examples")
    print(f"Label 1: {len(label_1)} | Label 0: {len(label_0)}")


if __name__ == "__main__":
    print("Generating v2 email dataset (probe-targeted prompts)...")
    label_1, label_0 = generate_dataset(target_per_label=500)
    save_dataset(label_1, label_0)
