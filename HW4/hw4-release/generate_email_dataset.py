"""
Generate a binary email dataset using few-shot prompting.
Label 1: Emails asking Bob to exercise OR do something in the morning.
Label 0: All other emails.
"""

import os
import json
import random
from openai import OpenAI

random.seed(42)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Seed templates for few-shot prompting
LABEL_1_SEEDS = [
    # Morning meetings/activities
    "meet tmrw?",
    "talk at 8am?",
    # Pure exercise 
    "run?",
    "Hi bob, do you want to go to gym?",
    # Morning exercise
    "run tmrw?",
]

LABEL_0_SEEDS = [
    # Work-related
    "Hi Bob,\n\nDo you have the quarterly report?\n\nThanks",
    "Bob,\n\nCan you send the budget spreadsheet?\n\nCheers",
    # Fun activities that are NOT morning and NOT exercise
    "Bob, lunch?",
    "Call at 2PM?",
    "Meeting at 7PM?",
]

PROMPT_TEMPLATE = """You are generating realistic email examples for binary classification.

TASK: Generate emails that ask Bob a question.
- LABEL 1: Emails asking Bob to exercise (gym, running, sports, etc.) OR do something in the morning (early meetings, early calls, etc.)
- LABEL 0: Everything else. Work emails (reports, deadlines, feedback) AND fun social emails that are NOT morning and NOT exercise (lunch, drinks, movies, dinner, etc.)

LABEL 1 EXAMPLES:
{label_1_examples}

LABEL 0 EXAMPLES:
{label_0_examples}

Generate 20 NEW emails for each label (40 total). Make them diverse and realistic.
Some emails should be very short and informal (3-10 words), and use casual language and abbreviations (e.g., 'tmrw' insteand of 'tomorrow').
Some emails should use specific times insteand of 'morning/afternoon/evening'

Return ONLY a valid JSON array, no other text:
[{{"email": "...", "label": 1}}, {{"email": "...", "label": 0}}, ...]
"""


def extract_json(response_text):
    """Extract JSON array from response."""
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the response
    start = response_text.find('[')
    end = response_text.rfind(']')
    if start != -1 and end != -1:
        try:
            return json.loads(response_text[start:end+1])
        except json.JSONDecodeError:
            pass

    return None


def generate_batch(label_1_samples, label_0_samples):
    """Call LLM to generate a batch of emails."""
    label_1_str = "\n".join([f"- {ex}" for ex in label_1_samples])
    label_0_str = "\n".join([f"- {ex}" for ex in label_0_samples])

    prompt = PROMPT_TEMPLATE.format(
        label_1_examples=label_1_str,
        label_0_examples=label_0_str
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano", # change to gpt-4.1-nano
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=4000
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
    max_iterations = 50

    while (len(label_1_data) < target_per_label or len(label_0_data) < target_per_label) and iteration < max_iterations:
        iteration += 1

        # Sample few-shot examples (mix of seeds + previously generated)
        label_1_samples = random.sample(LABEL_1_SEEDS, min(3, len(LABEL_1_SEEDS)))
        if label_1_data:
            label_1_samples.extend([email for email, _ in random.sample(label_1_data, min(2, len(label_1_data)))])

        label_0_samples = random.sample(LABEL_0_SEEDS, min(3, len(LABEL_0_SEEDS)))
        if label_0_data:
            label_0_samples.extend([email for email, _ in random.sample(label_0_data, min(2, len(label_0_data)))])

        print(f"Iteration {iteration}: Generating {batch_size} examples per label...")
        batch = generate_batch(label_1_samples, label_0_samples)

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
    """Save dataset to JSONL format with train/dev split."""
    os.makedirs("data", exist_ok=True)

    # Combine and shuffle
    all_data = label_1 + label_0
    random.shuffle(all_data)

    # Split into train and dev
    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    dev_data = all_data[split_idx:]

    # ========= update the save data path =========
    # Save training data
    train_path = os.path.join("data", "email_dataset_train_augmented.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for email, label in train_data:
            email_text = email.replace("\n", " ").strip()
            f.write(json.dumps({"text": email_text, "label": label}) + "\n")

    # Save dev data
    dev_path = os.path.join("data", "email_dataset_dev_augmented.jsonl")
    with open(dev_path, "w", encoding="utf-8") as f:
        for email, label in dev_data:
            email_text = email.replace("\n", " ").strip()
            f.write(json.dumps({"text": email_text, "label": label}) + "\n")

    print(f"\nTraining data saved to {train_path} ({len(train_data)} examples)")
    print(f"Dev data saved to {dev_path} ({len(dev_data)} examples)")
    print(f"Total: {len(all_data)} examples")
    print(f"Label 1: {len(label_1)} | Label 0: {len(label_0)}")


if __name__ == "__main__":
    print("Generating email dataset with few-shot prompting...")
    # generate 1000 dataset
    label_1, label_0 = generate_dataset(target_per_label=500)
    save_dataset(label_1, label_0)
