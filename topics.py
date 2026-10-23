"""Feedback categories used to group what players talk about.

Each category is a list of regex patterns. A review "mentions" a category if any
pattern matches its lowercased text. Edit these lists to tune the analysis.
"""
import re

TOPICS = {
    "Gameplay": [
        r"gameplay", r"mechanic", r"combat", r"controls?\b", r"level design",
        r"missions?\b", r"quests?\b", r"grind", r"difficult", r"replay",
    ],
    "Graphics & Visuals": [
        r"graphic", r"visual", r"art ?style", r"textures?\b", r"animations?\b",
        r"beautiful", r"gorgeous", r"ugly",
    ],
    "Bugs & Performance": [
        r"\bbugs?\b", r"buggy", r"glitch", r"crash", r"\bfps\b", r"\blag",
        r"stutter", r"optimi[sz]", r"performance", r"freez", r"broken",
    ],
    "Price & Value": [
        r"\bprice", r"worth", r"money", r"refund", r"overpriced", r"\bsale\b",
        r"\bdlcs?\b", r"microtransaction", r"pay ?to ?win", r"\$\d",
    ],
    "Story & Characters": [
        r"\bstory", r"\bplot\b", r"characters?\b", r"writing", r"ending",
        r"narrative", r"\blore\b",
    ],
    "Multiplayer & Servers": [
        r"multiplayer", r"servers?\b", r"online", r"matchmaking", r"co-?op",
        r"cheat(?:er|ing|s)\b", r"hacker", r"toxic",
    ],
    "Sound & Music": [
        r"soundtrack", r"\bmusic", r"sound design", r"voice act", r"\baudio",
    ],
    "Updates & Developers": [
        r"\bupdates?\b", r"\bdevs?\b", r"developers?\b", r"early access",
        r"abandon", r"roadmap", r"\bpatch",
    ],
}

TOPIC_REGEX = {name: re.compile("|".join(pats)) for name, pats in TOPICS.items()}
