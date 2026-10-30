import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


BATCH_JUDGE_INSTRUCTIONS = """You are a deterministic binary rubric evaluator auditing 10 roleplay turns (`turn_01`..`turn_10`) and 10 in-character social media posts (`post_01`..`post_10`) from a TTRPG simulation.

==============================
PART 1: ROLEPLAY TURN GATES (A2, A3, A4)
==============================
GATE A2 — ACTION-RESULT & PARTNER BOUNDARY:
- Score 0 (FAIL) IF `public_content` does ANY of the following:
  1. GOD-MODES WORLD OUTCOME: Narrates the external physical success or environmental compliance of a contested action before a [SYSTEM] roll (e.g., stating a jammed lever snaps/moves, a sealed door slides open, or a hazard is neutralized). Declaring the physical attempt/effort ("I brace my boots and try to haul the grate") is 1 (PASS).
  2. PUPPETS TEAMMATE/NPC: Writes spoken dialogue, internal thoughts, or physical actions/reactions for another named character (e.g., "Dax nods and says...").
- Score 1 (PASS) otherwise.

GATE A3 — ANTI-ASSISTANT & PERSONA REGISTER:
- Score 0 (FAIL) IF `public_content` does ANY of the following:
  1. AI ASSISTANT / THERAPY-SPEAK COLLAPSE: Uses polite customer-service apologies, HR/therapy validation ("I understand why you are overwhelmed", "your frustration is valid", "let's take a deep breath"), or numbered advice lists.
  2. CONTRABAND VIOLATION: Violates the explicit negative constraints in the character's bio.
- Score 1 (PASS) otherwise.

GATE A4 — MECHANICAL & PRONOUN FIDELITY:
- Score 0 (FAIL) IF `public_content` does ANY of the following:
  1. RETCONS ROLL OUTCOME: Contradicts a [SYSTEM] roll result in the transcript (e.g., claiming the lower lever is pried loose when evt_107 logged failure).
  2. PHANTOM INVENTORY USE: Uses an item listed under `lost_or_transferred_items` or not in `current_inventory` (e.g., using the Plasma Torch after tossing it to Soren in evt_102).
  3. MISGENDERING: Uses the wrong pronouns for self or any teammate (check the character pronoun roster carefully, especially singular `they/them` for Soren).
- Score 1 (PASS) otherwise.

==============================
PART 2: SOCIAL MEDIA POST GATES (D2, D3, D4)
==============================
SYNTACTIC TIE-BREAKER RULES (MANDATORY):
1. COLLECTIVE "WE" RULE: If the author writes "we [achieved X]" (e.g., "we got the upper conduit open") and the author was [WITNESSED] in the room when a teammate succeeded at X, score D2 = 1 (PASS), provided the author does NOT claim personal singular agency ("I cut", "I helped cut") for an action they did not perform.
2. PASSIVE OBSERVATIONAL FRAMING: Stating a witnessed environmental state change in passive voice ("Upper conduit lock is finally cut through") without claiming personal agency is always D2 = 1 (PASS).
3. ATTEMPT VS. OUTCOME: Describing a failed attempt or its negative consequence ("Tried forcing the lower lever—jammed solid") is D2 = 1 (PASS). Only score D2 = 0 if the author claims the failed action succeeded (or vice versa).
4. SINGULAR "THEY": Referring to a `they/them` teammate (like Soren) as "they" is accurate third-person attribution (D2 = 1).
5. SUBJECTIVE COMMENTARY: In-character opinions, insults, or hyperbole about witnessed events ("Dax stood there whining like a useless suit") are D3 = 1 (PASS), NOT hallucinations.

GATE D2 — SELF VS. OTHER ATTRIBUTION:
- Score 0 (FAIL) IF ANY claim in the post commits:
  1. ACTOR_SWAP: Uses first-person singular agency ("I", "me", "my", or "I helped [verb]") to claim a physical/technical action performed solely by another character.
  2. OUTCOME_INVERSION: Claims a failed [SYSTEM] check succeeded (or a successful check failed).
- Score 1 (PASS) otherwise.

GATE D3 — EPISTEMIC & CANON GROUNDING:
- Score 0 (FAIL) IF ANY claim in the post commits:
  1. UNWITNESSED_LEAK: Mentions or describes content from any transcript line marked `[UNWITNESSED]`.
  2. HALLUCINATED: Fabricates a concrete physical event, creature, injury, or item discovery that never occurred in the transcript.
- Score 1 (PASS) otherwise.

GATE D4 — IN-CHARACTER SOCIAL REGISTER:
- Score 0 (FAIL) IF the post does ANY of the following:
  1. THIRD-PERSON RECAP: Refers to the focal author in the third person ("Vance tossed the torch...").
  2. RAW MECHANIC LEAK: Mentions raw TTRPG mechanics ("Slot 1", "DC 15", "rolled a 5", "Athletics").
  3. CORPORATE ASSISTANT TONE: Reads like a cheerful AI assistant or corporate HR announcement ("Hello everyone! Today we experienced a challenging event...").
- Score 1 (PASS) otherwise.

==============================
REQUIRED JSON OUTPUT FORMAT
==============================
Return ONLY a single valid JSON object with this exact structure (evaluate `reason` BEFORE assigning the 0/1 integers):
{
  "roleplay_turns": {
    "turn_01": {"reason": "brief quote/rule check", "A2": 1, "A3": 1, "A4": 1},
    ... through "turn_10"
  },
  "social_posts": {
    "post_01": {"reason": "brief quote/rule check", "D2": 1, "D3": 1, "D4": 1},
    ... through "post_10"
  }
}"""


def compute_kappa_and_ac1(y1: List[int], y2: List[int]) -> Tuple[float, float, float]:
    """Returns (raw_agreement, cohens_kappa, gwets_ac1) for two binary 0/1 lists."""
    n = len(y1)
    agree = sum(1 for a, b in zip(y1, y2) if a == b)
    p_o = agree / n

    p1_a = sum(y1) / n
    p1_b = sum(y2) / n
    p0_a = 1.0 - p1_a
    p0_b = 1.0 - p1_b

    p_e_kappa = (p1_a * p1_b) + (p0_a * p0_b)
    kappa = 1.0 if p_e_kappa == 1.0 else (p_o - p_e_kappa) / (1.0 - p_e_kappa)

    pi_1 = (p1_a + p1_b) / 2.0
    pi_0 = 1.0 - pi_1
    p_e_ac1 = 2.0 * pi_1 * pi_0
    ac1 = 1.0 if p_e_ac1 == 1.0 else (p_o - p_e_ac1) / (1.0 - p_e_ac1)

    return round(p_o, 4), round(kappa, 4), round(ac1, 4)


def export_prompt(packet_path: str, out_path: str) -> None:
    packet = json.loads(Path(packet_path).read_text(encoding="utf-8"))
    ctx = packet["scene_context"]

    blind_turns = [
        {"item_id": t["item_id"], "public_content": t["public_content"]}
        for t in packet["roleplay_turns"]
    ]
    blind_posts = [
        {"item_id": p["item_id"], "content": p["content"]}
        for p in packet["social_posts"]
    ]

    prompt_text = (
        f"{BATCH_JUDGE_INSTRUCTIONS}\n\n"
        f"==============================\n"
        f"SCENE CONTEXT & TRANSCRIPT\n"
        f"==============================\n"
        f"FOCAL ACTOR: {ctx['focal_actor']['actor_id']} ({ctx['focal_actor']['pronouns']})\n"
        f"BIO & CONSTRAINTS: {ctx['focal_actor']['bio']}\n"
        f"CURRENT INVENTORY: {ctx['focal_actor']['current_inventory']}\n"
        f"LOST / TRANSFERRED ITEMS: {ctx['focal_actor']['lost_or_transferred_items']}\n"
        f"TEAMMATE PRONOUN ROSTER: {json.dumps(ctx['teammates'])}\n\n"
        + "\n".join(ctx["transcript"])
        + "\n\n==============================\n"
        + "ITEMS TO EVALUATE\n"
        + "==============================\n"
        + json.dumps({"roleplay_turns": blind_turns, "social_posts": blind_posts}, indent=2)
    )

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(prompt_text, encoding="utf-8")
    print(f"[+] Blind calibration prompt exported to: {out_file}")
    print("    1. Paste its contents into Claude Sonnet and save JSON reply to: data/claude_scores.json")
    print("    2. Save Gemini Pro JSON reply to                               : data/gemini_scores.json")
    print("    3. Run: python scripts/calibrate_rubric.py --mode score")


def score_responses(packet_path: str, claude_path: str, gemini_path: str) -> None:
    packet = json.loads(Path(packet_path).read_text(encoding="utf-8"))
    claude = json.loads(Path(claude_path).read_text(encoding="utf-8"))
    gemini = json.loads(Path(gemini_path).read_text(encoding="utf-8"))

    gates = ["A2", "A3", "A4", "D2", "D3", "D4"]
    gold: Dict[str, List[int]] = {g: [] for g in gates}
    scores_a: Dict[str, List[int]] = {g: [] for g in gates}
    scores_b: Dict[str, List[int]] = {g: [] for g in gates}
    disagreements: List[str] = []

    for item in packet["roleplay_turns"]:
        iid = item["item_id"]
        for g in ["A2", "A3", "A4"]:
            exp = item["expected"][g]
            va = int(claude["roleplay_turns"][iid][g])
            vb = int(gemini["roleplay_turns"][iid][g])
            gold[g].append(exp)
            scores_a[g].append(va)
            scores_b[g].append(vb)
            if len({exp, va, vb}) > 1:
                disagreements.append(
                    f"  [MISMATCH] {iid} {g}: Gold={exp}, Claude={va}, Gemini={vb} | {item['note']}"
                )

    for item in packet["social_posts"]:
        iid = item["item_id"]
        for g in ["D2", "D3", "D4"]:
            exp = item["expected"][g]
            va = int(claude["social_posts"][iid][g])
            vb = int(gemini["social_posts"][iid][g])
            gold[g].append(exp)
            scores_a[g].append(va)
            scores_b[g].append(vb)
            if len({exp, va, vb}) > 1:
                disagreements.append(
                    f"  [MISMATCH] {iid} {g}: Gold={exp}, Claude={va}, Gemini={vb} | {item['note']}"
                )

    print("\n=== SUBSCRIPTION JUDGE CALIBRATION SCORECARD (Target Kappa >= 0.90) ===")
    print(f"{'Gate':<8} | {'Claude/Gem %':<13} | {'Claude/Gem κ':<13} | {'Gwet AC1':<10} | {'Claude/Gold':<11} | {'Gemini/Gold':<11}")
    print("-" * 78)

    all_a, all_b, all_gold = [], [], []
    for g in gates:
        p_o, k_ab, ac1 = compute_kappa_and_ac1(scores_a[g], scores_b[g])
        _, k_ag, _ = compute_kappa_and_ac1(scores_a[g], gold[g])
        _, k_bg, _ = compute_kappa_and_ac1(scores_b[g], gold[g])
        print(f"{g:<8} | {p_o*100:>11.1f}% | {k_ab:>13.3f} | {ac1:>10.3f} | {k_ag:>11.3f} | {k_bg:>11.3f}")
        all_a.extend(scores_a[g])
        all_b.extend(scores_b[g])
        all_gold.extend(gold[g])

    print("-" * 78)
    tot_po, tot_k, tot_ac1 = compute_kappa_and_ac1(all_a, all_b)
    _, tot_ag, _ = compute_kappa_and_ac1(all_a, all_gold)
    _, tot_bg, _ = compute_kappa_and_ac1(all_b, all_gold)
    print(f"{'OVERALL':<8} | {tot_po*100:>11.1f}% | {tot_k:>13.3f} | {tot_ac1:>10.3f} | {tot_ag:>11.3f} | {tot_bg:>11.3f}")
    print("=======================================================================\n")

    if disagreements:
        print("Flagged Disagreements:")
        for d in disagreements:
            print(d)
    else:
        print("[+] 100% Agreement across Claude Sonnet, Gemini Pro, and Gold Labels (κ = 1.000)!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subscription-based Rubric Calibration")
    parser.add_argument("--mode", choices=["export", "score"], default="export")
    parser.add_argument("--packet", default="benchmarks/rubric_calibration_v1.json")
    parser.add_argument("--out-prompt", default="data/judge_calibration_prompt.txt")
    parser.add_argument("--claude-json", default="data/claude_scores.json")
    parser.add_argument("--gemini-json", default="data/gemini_scores.json")
    args = parser.parse_args()

    if args.mode == "export":
        export_prompt(args.packet, args.out_prompt)
    else:
        score_responses(args.packet, args.claude_json, args.gemini_json)