import re
import sys
import unicodedata

from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer

import emoji

CUSTOM_TABLE = str.maketrans(
    {
        "\xad": None,
        "\x7f": None,
        "\ufeff": None,
        "\u200b": None,
        "\u200e": None,
        "\u202a": None,
        "\u202c": None,
        "‘": "'",
        "’": "'",
        "`": "'",
        "“": '"',
        "”": '"',
        "«": '"',
        "»": '"',
        "ɢ": "G",
        "ɪ": "I",
        "ɴ": "N",
        "ʀ": "R",
        "ʏ": "Y",
        "ʙ": "B",
        "ʜ": "H",
        "ʟ": "L",
        "ғ": "F",
        "ᴀ": "A",
        "ᴄ": "C",
        "ᴅ": "D",
        "ᴇ": "E",
        "ᴊ": "J",
        "ᴋ": "K",
        "ᴍ": "M",
        "Μ": "M",
        "ᴏ": "O",
        "ᴘ": "P",
        "ᴛ": "T",
        "ᴜ": "U",
        "ᴡ": "W",
        "ᴠ": "V",
        "ĸ": "K",
        "в": "B",
        "м": "M",
        "н": "H",
        "т": "T",
        "ѕ": "S",
        "—": "-",
        "–": "-",
    }
)

WORDS_REPLACER = [
    ("sh*t", "shit"),
    ("s**t", "shit"),
    ("f*ck", "fuck"),
    ("fu*k", "fuck"),
    ("f**k", "fuck"),
    ("f*****g", "fucking"),
    ("f***ing", "fucking"),
    ("f**king", "fucking"),
    ("p*ssy", "pussy"),
    ("p***y", "pussy"),
    ("pu**y", "pussy"),
    ("p*ss", "piss"),
    ("b*tch", "bitch"),
    ("bit*h", "bitch"),
    ("h*ll", "hell"),
    ("h**l", "hell"),
    ("cr*p", "crap"),
    ("d*mn", "damn"),
    ("stu*pid", "stupid"),
    ("st*pid", "stupid"),
    ("n*gger", "nigger"),
    ("n***ga", "nigger"),
    ("f*ggot", "faggot"),
    ("scr*w", "screw"),
    ("pr*ck", "prick"),
    ("g*d", "god"),
    ("s*x", "sex"),
    ("a*s", "ass"),
    ("a**hole", "asshole"),
    ("a***ole", "asshole"),
    ("a**", "ass"),
]

REGEX_REPLACER = [
    (re.compile(pat.replace("*", "\*"), flags=re.IGNORECASE), repl)
    for pat, repl in WORDS_REPLACER
]

RE_SPACE = re.compile(r"\s")
RE_MULTI_SPACE = re.compile(r"\s+")

NMS_TABLE = dict.fromkeys(
    i for i in range(sys.maxunicode + 1) if unicodedata.category(chr(i)) == "Mn"
)

HEBREW_TABLE = {i: "א" for i in range(0x0590, 0x05FF)}
ARABIC_TABLE = {i: "ا" for i in range(0x0600, 0x06FF)}
CHINESE_TABLE = {i: "是" for i in range(0x4E00, 0x9FFF)}
KANJI_TABLE = {i: "ッ" for i in range(0x2E80, 0x2FD5)}
HIRAGANA_TABLE = {i: "ッ" for i in range(0x3041, 0x3096)}
KATAKANA_TABLE = {i: "ッ" for i in range(0x30A0, 0x30FF)}

TABLE = dict()
TABLE.update(CUSTOM_TABLE)
TABLE.update(NMS_TABLE)
# Non-english languages
TABLE.update(CHINESE_TABLE)
TABLE.update(HEBREW_TABLE)
TABLE.update(ARABIC_TABLE)
TABLE.update(HIRAGANA_TABLE)
TABLE.update(KATAKANA_TABLE)
TABLE.update(KANJI_TABLE)


EMOJI_REGEXP = emoji.get_emoji_regexp()

UNICODE_EMOJI_MY = {
    k: f" EMJ {v.strip(':').replace('_', ' ')} "
    for k, v in emoji.UNICODE_EMOJI_ALIAS.items()
}


def my_demojize(string: str) -> str:
    """
    Modified version of the demojize function from emoji
    """

    def replace(match):
        return UNICODE_EMOJI_MY.get(match.group(0), match.group(0))

    return re.sub("\ufe0f", "", EMOJI_REGEXP.sub(replace, string))


def fix_tokens(tokens):
    """
    Expects a list of lower-cased tokens from TweeterTokenizer
    """
    for token in tokens:
        if token == "gov't" or token == "govt":
            yield "government"
        elif token == "i'm":
            yield "i"
            yield "am"
        elif token.endswith("n't"):
            yield token[:-3]
            yield "not"
        elif token.endswith("'re"):
            yield token[:-3]
            yield "are"
        elif token.endswith("'ll"):
            yield token[:-3]
            yield "will"
        elif token.endswith("'ve"):
            yield token[:-3]
            yield "have"
        elif token.endswith("'s"):
            yield token[:-2]
            yield "'s"
        else:
            yield token


def normalize(text: str) -> str:
    text = my_demojize(text)
    text = RE_SPACE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = text.translate(TABLE)
    text = RE_MULTI_SPACE.sub(" ", text).strip()

    for pattern, repl in REGEX_REPLACER:
        text = pattern.sub(repl, text)

    return text
