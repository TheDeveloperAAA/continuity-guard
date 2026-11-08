"""Generate the public demo show (original story, written for this project) in Kuku's timed script format.

Two versions are written:
  demo_show/        - with 6 continuity problems planted on purpose (see demo_show/ANSWER_KEY.md)
  demo_show_clean/  - the same episodes with those problems removed (used to measure false alarms)
Both keep two DELIBERATE changes that must NOT be flagged: a hidden-identity speaker label and a haircut on screen.
"""
import os

EPISODES = [
    {
        "num": 1,
        "logline": "At her engagement in Mumbai, chai-seller's daughter Meera is dumped by Vikram for her cousin Tanya, until a silent delivery rider steps in.",
        "notes": "First episode. Establishes Meera, Vikram, cousin Tanya and the mysterious rider.",
        "heading": "INT. BANQUET HALL - NIGHT",
        "blocks": [
            ("CAMERA", "", "Wide shot of a glittering banquet hall decorated with marigold strings as guests turn toward the stage.", 3.0),
            ("ACTION", "", "Meera steps onto the stage in her late mother's red saree, her long black hair braided with jasmine.", 3.5),
            ("VIKRAM", "Cold", "\"Stop right there, Meera.\"", 1.8),
            ("MEERA", "Confused", "\"Vikram? Everyone is waiting for the rings.\"", 2.4),
            ("CAMERA", "", "Close-up on Vikram slipping the engagement ring onto Tanya's finger instead.", 2.6),
            ("TANYA", "Smug", "\"He chose family money, cousin. He chose me.\"", 2.6),
            ("MEERA", "Heartbroken", "\"Tanya... you are my cousin. How could you?\"", 2.8),
            ("VIKRAM", "Dismissive", "\"The Raichand Group will never hire a chai-seller's daughter's husband.\"", 3.4),
            ("ACTION", "", "Guests laugh as a delivery rider in a dusty helmet walks in carrying a stack of tiffin boxes.", 3.2),
            ("CAMERA", "", "Close-up as the rider lifts his visor, revealing calm hazel eyes and a thin scar on his left eyebrow.", 3.0),
            ("MYSTERIOUS RIDER", "Calm", "\"She is worth more than this entire hall.\"", 2.6),
            ("VIKRAM", "Mocking", "\"And who are you? The tiffin boy?\"", 2.2),
            ("MYSTERIOUS RIDER", "Quiet", "\"Someone who keeps his promises.\"", 2.2),
            ("SOUND FX", "", "A tense tabla roll rises to a sharp cliffhanger hit.", 2.0),
            ("CAMERA", "", "The rider offers Meera his hand as the screen cuts to black.", 2.0),
        ],
    },
    {
        "num": 2,
        "logline": "Meera cuts her hair to start over and learns the rider is named Arjun, while Vikram brags about a contract with the Raichand Group.",
        "notes": "Picks up after Episode 1's cliffhanger. The rider is now named Arjun.",
        "heading": "INT. MEERA'S CHAI STALL - DAY",
        "blocks": [
            ("CAMERA", "", "Morning light falls on a small chai stall as steam curls from a dented kettle.", 3.0),
            ("ACTION", "", "Meera looks into a cracked mirror and cuts her long braid short with steady hands.", 3.4),
            ("CAMERA", "", "Medium shot of Meera, her hair now cut short at the jaw, pouring tea with new resolve.", 3.0),
            ("ARJUN", "Warm", "\"Short hair suits a fresh start.\"", 2.0),
            ("MEERA", "Surprised", "\"You again. You never told me your name.\"", 2.4),
            ("ARJUN", "Smiling", "\"Arjun. Just Arjun, for now.\"", 2.0),
            ("CAMERA", "", "Close-up of Arjun's hazel eyes as he studies an embossed Raichand Group card in Meera's stall.", 3.0),
            ("ACTION", "", "Vikram pulls up in a white sedan, waving a contract printed on Raichand Group letterhead.", 3.2),
            ("VIKRAM", "Boastful", "\"The Raichand Group signed my company today, Meera. You lost a fortune.\"", 3.6),
            ("MEERA", "Firm", "\"Chai thandi ho gayi, Vikram. Go home.\"", 2.4),
            ("ARJUN", "Quiet", "\"Contracts can be cancelled.\"", 2.0),
            ("SOUND FX", "", "A low synth pulse swells into a cliffhanger sting.", 2.0),
        ],
    },
    {
        "num": 3,
        "logline": "At the Raichand Group gala, Tanya humiliates Meera, and Arjun walks in wearing the chairman's ring.",
        "notes": "Picks up from Episode 2. Meera's hair is short since Episode 2.",
        "heading": "INT. RAICHAND TOWER BALLROOM - NIGHT",
        "blocks": [
            ("CAMERA", "", "Wide shot of the Raichand Tower ballroom glittering under crystal chandeliers.", 3.0),
            ("ACTION", "", "Meera enters in a simple white saree, her short hair pinned back, carrying a tray of chai.", 3.4),
            ("TANYA", "Cruel", "\"Look, my cousin serves tea at the Raichand Group gala now.\"", 3.0),
            ("VIKRAM", "Laughing", "\"Some people are born to serve.\"", 2.2),
            ("CAMERA", "", "The doors open as Arjun walks in wearing a navy bandhgala and a gold signet ring.", 3.0),
            ("CAMERA", "", "Close-up of Arjun's hazel eyes locking on Vikram.", 2.4),
            ("ARJUN", "Commanding", "\"The Raichand Group cancelled your contract an hour ago, Vikram.\"", 3.4),
            ("VIKRAM", "Stunned", "\"Who gave you the authority?\"", 2.0),
            ("ACTION", "", "The board members bow to Arjun as the host announces the new chairman of the Raichand Group.", 3.4),
            ("MEERA", "Breathless", "\"Arjun... you are the chairman?\"", 2.2),
            ("SOUND FX", "", "Strings swell into a final cliffhanger chord.", 2.0),
        ],
    },
]

# Problems planted in the 'errors' version: (episode, find, replace, what it tests)
PLANTED = [
    (2, "Close-up of Arjun's hazel eyes as he studies an embossed Raichand Group card",
        "Close-up of Arjun's hazel eyes as he studies an embossed Raychand Group card", "place/company name variant (camera line)"),
    (3, "\"The Raichand Group cancelled your contract an hour ago, Vikram.\"",
        "\"The Raichandani Group cancelled your contract an hour ago, Vikram.\"", "company name variant (spoken line)"),
    (3, "Close-up of Arjun's hazel eyes locking on Vikram.", "Close-up of Arjun's dark brown eyes locking on Vikram.", "eye colour contradiction"),
    (3, "\"Who gave you the authority?\"", "\"Who gave you the authority, Vikrant?\"", "person name variant (spoken)"),
    (3, "**MEERA (Breathless):**", "**MS. MEERA (Breathless):**", "speaker label variant"),
    (3, "\"Look, my cousin serves tea", "\"Look, my sister serves tea", "relationship flip (OUT OF SCOPE: expected to be missed)"),
]
DELIBERATE = ["MYSTERIOUS RIDER -> ARJUN (identity hidden in Episode 1, revealed in Episode 2)",
              "Meera cuts her long hair short on screen in Episode 2"]


def ts(x):
    return f"{int(x // 60):02d}:{x % 60:05.2f}"


def render(ep):
    out = [f"EPISODE NUMBER: Episode {ep['num']}", f"EPISODE LOGLINE: {ep['logline']}", f"CONTINUITY NOTES: {ep['notes']}", "",
           "--- SCRIPT START ---", "", "SCENE 01", f"Scene heading: {ep['heading']}", "Scene start: 00:00.00"]
    t, body = 0.0, []
    for head, emo, text, dur in ep["blocks"]:
        h = head if head in ("CAMERA", "ACTION", "SOUND FX") else f"{head} ({emo})"
        body += [f"**{h}:**", f"Start: ({ts(t)}) | End: ({ts(t + dur)})", text, ""]
        t = round(t + dur + 0.1, 2)
    out += [f"Scene end: {ts(t)}", "", "Actions and dialogue:", ""] + body + ["--- SCRIPT END ---", ""]
    return "\n".join(out)


def main(base):
    for folder, with_errors in (("demo_show", True), ("demo_show_clean", False)):
        d = os.path.join(base, folder)
        os.makedirs(d, exist_ok=True)
        for ep in EPISODES:
            txt = render(ep)
            if with_errors:
                for n, a, b, _ in PLANTED:
                    if n == ep["num"]:
                        assert a in txt, (n, a)
                        txt = txt.replace(a, b, 1)
            with open(os.path.join(d, f"episode-{ep['num']:03d}.txt"), "w", encoding="utf-8") as fh:
                fh.write(txt)
    key = ["# Demo show answer key", "", "Planted problems (demo_show only):", ""]
    key += [f"- Ep {n}: {why} - `{a[:60]}` -> `{b[:60]}`" for n, a, b, why in PLANTED]
    key += ["", "Deliberate changes that must NOT be flagged as errors (both versions):", ""] + [f"- {x}" for x in DELIBERATE]
    with open(os.path.join(base, "demo_show", "ANSWER_KEY.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(key) + "\n")


if __name__ == "__main__":
    main(os.path.dirname(os.path.abspath(__file__)))
