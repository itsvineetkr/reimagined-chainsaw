"""Field normalisation shared by every method.

Design rules
  * Canonicalise *both* sides of a pair to the same token (``street``/``st``/``str`` -> ``st``);
    whether ``st`` meant Street or Saint is irrelevant as long as both sides agree.
  * Never throw identity information away: numbers, postal codes and landmarks are extracted
    into their own fields instead of being deleted.
  * Country is an open set: known aliases collapse (``USA`` -> ``us``) and anything unseen
    passes through lower-cased, so France (test-only) is handled like any other label.
  * The abbreviation tables are generic linguistic normalisation (no business lookup).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import cache

import pandas as pd

from er_common.progress import progress

# --------------------------------------------------------------------------------------------
# Lexicons
# --------------------------------------------------------------------------------------------

_SHARED_MAP: dict[str, str] = {
    "saint": "st",
    "sainte": "ste",
    "mount": "mt",
    "fort": "ft",
    "sri": "sri",
    "shri": "sri",
    "shree": "sri",
    "sree": "sri",
    "shrii": "sri",
    "shreee": "sri",
    "smt": "smt",
    "shrimati": "smt",
    "srimati": "smt",
    "mohd": "mohd",
    "mohammed": "mohd",
    "mohammad": "mohd",
    "muhammad": "mohd",
    "mohamed": "mohd",
    "muhammed": "mohd",
    "mohamad": "mohd",
    "mahmood": "mahmud",
    "mahmud": "mahmud",
    "and": "and",
}

_NAME_MAP: dict[str, str] = {
    **_SHARED_MAP,
    "private": "pvt",
    "pvt": "pvt",
    "pte": "pvt",
    "limited": "ltd",
    "ltd": "ltd",
    "ltda": "ltd",
    "corporation": "corp",
    "corpn": "corp",
    "corp": "corp",
    "incorporated": "inc",
    "inc": "inc",
    "company": "co",
    "companies": "co",
    "cos": "co",
    "compagnie": "co",
    "cie": "co",
    "international": "intl",
    "intl": "intl",
    "internatl": "intl",
    "national": "natl",
    "natl": "natl",
    "manufacturing": "mfg",
    "manufacturers": "mfg",
    "manufacturer": "mfg",
    "mfrs": "mfg",
    "mfg": "mfg",
    "services": "svc",
    "service": "svc",
    "svcs": "svc",
    "svc": "svc",
    "serv": "svc",
    "associates": "assoc",
    "association": "assoc",
    "assn": "assoc",
    "assoc": "assoc",
    "brothers": "bros",
    "bros": "bros",
    "freres": "bros",
    "enterprises": "ent",
    "enterprise": "ent",
    "entp": "ent",
    "ent": "ent",
    "hospital": "hosp",
    "hospitals": "hosp",
    "hosp": "hosp",
    "center": "ctr",
    "centre": "ctr",
    "cntr": "ctr",
    "ctr": "ctr",
    "department": "dept",
    "dept": "dept",
    "university": "univ",
    "univ": "univ",
    "institute": "inst",
    "inst": "inst",
    "technologies": "tech",
    "technology": "tech",
    "techs": "tech",
    "tech": "tech",
    "systems": "sys",
    "system": "sys",
    "sys": "sys",
    "management": "mgmt",
    "mgmt": "mgmt",
    "development": "dev",
    "dev": "dev",
    "industries": "ind",
    "industry": "ind",
    "inds": "ind",
    "ind": "ind",
    "market": "mkt",
    "markets": "mkt",
    "mkt": "mkt",
    "doctor": "dr",
    "dr": "dr",
    "pharmaceuticals": "pharma",
    "pharmaceutical": "pharma",
    "pharma": "pharma",
    "medical": "med",
    "med": "med",
    "group": "grp",
    "grp": "grp",
    "holdings": "hldg",
    "hldgs": "hldg",
    "hldg": "hldg",
    "financial": "fin",
    "finance": "fin",
    "fin": "fin",
    "solutions": "soln",
    "solns": "soln",
    "soln": "soln",
    "consultants": "consult",
    "consultancy": "consult",
    "consulting": "consult",
    "consult": "consult",
    "engineering": "eng",
    "engg": "eng",
    "eng": "eng",
    "electricals": "elec",
    "electrical": "elec",
    "electric": "elec",
    "elec": "elec",
    "travels": "travel",
    "travel": "travel",
    "traders": "trade",
    "trader": "trade",
    "trading": "trade",
    "automobiles": "auto",
    "automobile": "auto",
    "automotive": "auto",
    "auto": "auto",
    "motors": "motor",
    "motor": "motor",
    "laboratories": "lab",
    "laboratory": "lab",
    "labs": "lab",
    "lab": "lab",
    "clinics": "clinic",
    "clinic": "clinic",
    "agencies": "agency",
    "agency": "agency",
    "insurance": "ins",
    "ins": "ins",
    "communications": "comm",
    "communication": "comm",
    "comm": "comm",
    "products": "prod",
    "product": "prod",
    "prod": "prod",
    "restaurants": "restaurant",
    "hotels": "hotel",
    "stores": "store",
    "shoppe": "shop",
    "etablissements": "ets",
    "etablissement": "ets",
    "ets": "ets",
    "societe": "ste",
    "ste": "ste",
}

_ADDR_MAP: dict[str, str] = {
    **_SHARED_MAP,
    "road": "rd",
    "rd": "rd",
    "street": "st",
    "st": "st",
    "str": "st",
    "stt": "st",
    "avenue": "ave",
    "ave": "ave",
    "av": "ave",
    "avn": "ave",
    "boulevard": "blvd",
    "blvd": "blvd",
    "bd": "blvd",
    "boul": "blvd",
    "bld": "blvd",
    "lane": "ln",
    "ln": "ln",
    "drive": "dr",
    "dr": "dr",
    "drv": "dr",
    "court": "ct",
    "ct": "ct",
    "place": "pl",
    "pl": "pl",
    "square": "sq",
    "sq": "sq",
    "highway": "hwy",
    "hwy": "hwy",
    "hiway": "hwy",
    "parkway": "pkwy",
    "pkwy": "pkwy",
    "expressway": "expy",
    "expy": "expy",
    "terrace": "ter",
    "terr": "ter",
    "circle": "cir",
    "cir": "cir",
    "crescent": "cres",
    "cres": "cres",
    "plaza": "plz",
    "plz": "plz",
    "suite": "ste",
    "suit": "ste",
    "apartment": "apt",
    "apt": "apt",
    "floor": "fl",
    "flr": "fl",
    "fl": "fl",
    "building": "bldg",
    "bldg": "bldg",
    "bldng": "bldg",
    "room": "rm",
    "rm": "rm",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "northeast": "ne",
    "northwest": "nw",
    "southeast": "se",
    "southwest": "sw",
    "nagar": "nagar",
    "ngr": "nagar",
    "nagr": "nagar",
    "marg": "marg",
    "mrg": "marg",
    "colony": "colony",
    "clny": "colony",
    "col": "colony",
    "sector": "sec",
    "sect": "sec",
    "sec": "sec",
    "phase": "phase",
    "ph": "phase",
    "block": "blk",
    "blk": "blk",
    "cross": "cross",
    "crs": "cross",
    "layout": "layout",
    "lyt": "layout",
    "extension": "extn",
    "extn": "extn",
    "ext": "extn",
    "bazaar": "bazar",
    "bazar": "bazar",
    "bzr": "bazar",
    "chowk": "chowk",
    "chk": "chowk",
    "mohalla": "mohalla",
    "mohala": "mohalla",
    "enclave": "enclave",
    "encl": "enclave",
    "estate": "estate",
    "est": "estate",
    "industrial": "indl",
    "indl": "indl",
    "plot": "plot",
    "plt": "plot",
    "opposite": "opp",
    "opp": "opp",
    "near": "near",
    "nr": "near",
    "behind": "behind",
    "bhd": "behind",
    "junction": "jn",
    "jn": "jn",
    "jct": "jn",
    "junc": "jn",
    "station": "stn",
    "stn": "stn",
    "railway": "rly",
    "rly": "rly",
    "district": "dist",
    "distt": "dist",
    "dist": "dist",
    "village": "vill",
    "vill": "vill",
    "vil": "vill",
    "taluka": "taluk",
    "taluk": "taluk",
    "chemin": "chemin",
    "ch": "chemin",
    "impasse": "imp",
    "imp": "imp",
    "route": "rte",
    "rte": "rte",
    "faubourg": "fbg",
    "fbg": "fbg",
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
    # renamed cities: both spellings are live in real address data
    "bombay": "mumbai",
    "madras": "chennai",
    "calcutta": "kolkata",
    "bangalore": "bengaluru",
    "gurgaon": "gurugram",
    "poona": "pune",
    "trivandrum": "thiruvananthapuram",
    "baroda": "vadodara",
    "benares": "varanasi",
    "banaras": "varanasi",
    "benaras": "varanasi",
    "cochin": "kochi",
    "mysore": "mysuru",
    "mangalore": "mangaluru",
    "allahabad": "prayagraj",
    "pondicherry": "puducherry",
    "simla": "shimla",
    "calicut": "kozhikode",
    "belgaum": "belagavi",
    "vizag": "visakhapatnam",
    "nyc": "ny",
    # full state names -> postal codes (codes themselves are left untouched)
    "alabama": "al",
    "alaska": "ak",
    "arizona": "az",
    "arkansas": "ar",
    "california": "ca",
    "colorado": "co",
    "connecticut": "ct",
    "delaware": "de",
    "florida": "fl",
    "georgia": "ga",
    "hawaii": "hi",
    "idaho": "id",
    "illinois": "il",
    "indiana": "in",
    "iowa": "ia",
    "kansas": "ks",
    "kentucky": "ky",
    "louisiana": "la",
    "maine": "me",
    "maryland": "md",
    "massachusetts": "ma",
    "michigan": "mi",
    "minnesota": "mn",
    "mississippi": "ms",
    "missouri": "mo",
    "montana": "mt",
    "nebraska": "ne",
    "nevada": "nv",
    "ohio": "oh",
    "oklahoma": "ok",
    "oregon": "or",
    "pennsylvania": "pa",
    "tennessee": "tn",
    "texas": "tx",
    "utah": "ut",
    "vermont": "vt",
    "virginia": "va",
    "washington": "wa",
    "wisconsin": "wi",
    "wyoming": "wy",
    "maharashtra": "mh",
    "karnataka": "ka",
    "kerala": "kl",
    "gujarat": "gj",
    "rajasthan": "rj",
    "bihar": "br",
    "odisha": "od",
    "orissa": "od",
    "telangana": "ts",
    "punjab": "pb",
    "haryana": "hr",
    "jharkhand": "jh",
    "chhattisgarh": "cg",
    "uttarakhand": "uk",
    "assam": "as",
    "goa": "ga",
}

# Multi-word phrases rewritten before token mapping (regex, replacement).
_ADDR_PHRASES: tuple[tuple[str, str], ...] = (
    ("mahatma gandhi", "mg"),
    ("new york", "ny"),
    ("new jersey", "nj"),
    ("new mexico", "nm"),
    ("new hampshire", "nh"),
    ("north carolina", "nc"),
    ("south carolina", "sc"),
    ("north dakota", "nd"),
    ("south dakota", "sd"),
    ("west virginia", "wv"),
    ("rhode island", "ri"),
    ("district of columbia", "dc"),
    ("uttar pradesh", "up"),
    ("madhya pradesh", "mp"),
    ("andhra pradesh", "ap"),
    ("himachal pradesh", "hp"),
    ("arunachal pradesh", "ar"),
    ("tamil nadu", "tn"),
    ("west bengal", "wb"),
    ("jammu and kashmir", "jk"),
    ("post office", "po"),
    ("p o", "po"),
    ("united states of america", ""),
    ("united states", ""),
)
_ADDR_PHRASE_RES = tuple((re.compile(rf"\b{p}\b"), r) for p, r in _ADDR_PHRASES)

LEGAL_TOKENS: frozenset[str] = frozenset(
    {
        "pvt",
        "ltd",
        "llc",
        "llp",
        "inc",
        "corp",
        "co",
        "plc",
        "lp",
        "pllc",
        "pc",
        "gmbh",
        "sarl",
        "sas",
        "sasu",
        "sa",
        "eurl",
        "snc",
        "sci",
        "ag",
        "bv",
        "nv",
        "opc",
        "lllp",
    }
)
# country words that appear as name tails ("XYZ (India) Pvt Ltd")
_NAME_TAIL_EXTRA: frozenset[str] = frozenset({"india", "usa", "us", "america", "france", "and"})
_NAME_STOP: frozenset[str] = frozenset({"the", "and", "of", "le", "la", "les", "de", "du", "des", "et"})
_ADDR_DROP: frozenset[str] = frozenset({"india", "usa", "france", "cedex"})
_NUMBER_WORDS: frozenset[str] = frozenset({"no", "number", "num", "nos"})

_COUNTRY_ALIASES: dict[str, str] = {
    "us": "us",
    "usa": "us",
    "u s": "us",
    "u s a": "us",
    "united states": "us",
    "united states of america": "us",
    "america": "us",
    "in": "in",
    "ind": "in",
    "india": "in",
    "bharat": "in",
    "republic of india": "in",
    "fr": "fr",
    "fra": "fr",
    "france": "fr",
    "republique francaise": "fr",
    "french republic": "fr",
}

# keyword + at most two following words, never crossing a comma: without separators we cannot
# tell where "Near Clock Tower Mumbai" ends, and eating the city is worse than keeping a word.
_LANDMARK_RE = re.compile(
    r"\b(?:near|nr|opp|opposite|behind|beside|besides|next to|adjacent to|adj to|adj|in front of|"
    r"infront of|close to|facing|across from|across)\b\.?(?:[ \t]+[^\s,;]+){0,2}"
)
_DBA_RE = re.compile(r"\b(?:d b a|dba|doing business as|t a|trading as|aka|a k a)\b")

_APOS_RE = re.compile(r"[’‘'`´]")
_ZIP4_RE = re.compile(r"\b(\d{5})\s*-\s*\d{4}\b")
_PIN_SPLIT_RE = re.compile(r"\b(\d{3})\s(\d{3})\s*$")
_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
_PUNCT_RE = re.compile(r"[^\w\s]|_")
_ALNUM_SPLIT_RE = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")
_WS_RE = re.compile(r"\s+")
_POSTAL_RE = re.compile(r"^\d{5,6}$")
_DIGITS_RE = re.compile(r"\d+")
_VOWELS = frozenset("aeiou")


# --------------------------------------------------------------------------------------------
# Primitive steps
# --------------------------------------------------------------------------------------------


def fold_unicode(text: str) -> str:
    """NFKD + strip combining marks (é -> e, ç -> c) + lower-case."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower()


def _clean(text: str) -> str:
    text = fold_unicode(text).replace("&", " and ").replace("@", " at ")
    text = _APOS_RE.sub("", text)
    text = _ZIP4_RE.sub(r"\1", text)
    text = _ORDINAL_RE.sub(r"\1", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _ALNUM_SPLIT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _join_initials(tokens: list[str]) -> list[str]:
    """``s b i`` -> ``sbi`` (initialisms written with dots/spaces)."""
    out: list[str] = []
    run: list[str] = []
    for tok in tokens:
        if len(tok) == 1 and tok.isalpha():
            run.append(tok)
            continue
        if len(run) >= 2:
            out.append("".join(run))
        else:
            out.extend(run)
        run = []
        out.append(tok)
    if len(run) >= 2:
        out.append("".join(run))
    else:
        out.extend(run)
    return out


def _map_tokens(tokens: list[str], mapping: dict[str, str]) -> list[str]:
    return [m for m in (mapping.get(t, t) for t in tokens) if m]


def skeleton(token: str) -> str:
    """Transliteration-tolerant consonant skeleton: ``shree``/``sri`` -> ``sr``, ``baalaji`` -> ``blj``."""
    t = token
    for a, b in (
        ("ph", "f"),
        ("sh", "s"),
        ("kh", "k"),
        ("gh", "g"),
        ("th", "t"),
        ("dh", "d"),
        ("bh", "b"),
        ("jh", "j"),
        ("ch", "c"),
        ("ck", "k"),
        ("q", "k"),
        ("w", "v"),
        ("z", "j"),
        ("x", "ks"),
        ("y", "i"),
    ):
        t = t.replace(a, b)
    if not t:
        return t
    head, rest = t[0], "".join(ch for ch in t[1:] if ch not in _VOWELS)
    out = [head]
    for ch in rest:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


def normalize_country(value: str) -> str:
    key = _WS_RE.sub(" ", _PUNCT_RE.sub(" ", fold_unicode(value))).strip()
    return _COUNTRY_ALIASES.get(key, key)


# --------------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NormName:
    norm: str  # all tokens, canonicalised
    core: str  # legal suffixes / country tail / stop words removed
    alt: str  # trade name after "dba", "" if none
    legal: str  # sorted legal tokens found
    skel: str  # consonant skeleton of core tokens
    acronym: str  # initials of core tokens (>= 2 tokens), "" otherwise


def _strip_tail(tokens: list[str]) -> tuple[list[str], list[str]]:
    legal: list[str] = []
    end = len(tokens)
    while end > 1 and (tokens[end - 1] in LEGAL_TOKENS or tokens[end - 1] in _NAME_TAIL_EXTRA):
        if tokens[end - 1] in LEGAL_TOKENS:
            legal.append(tokens[end - 1])
        end -= 1
    return tokens[:end], legal


def _core_tokens(tokens: list[str]) -> tuple[list[str], list[str]]:
    body, legal = _strip_tail(tokens)
    legal += [t for t in body if t in LEGAL_TOKENS and t not in {"co", "sa", "ag", "pc"}]
    core = [t for t in body if t not in _NAME_STOP and not (t in LEGAL_TOKENS and t != body[0])]
    return (core or body), legal


@cache
def normalize_name(raw: str) -> NormName:
    cleaned = _clean(raw)
    parts = _DBA_RE.split(cleaned, maxsplit=1)
    main = _map_tokens(_join_initials(parts[0].split()), _NAME_MAP)
    alt_tokens = _map_tokens(_join_initials(parts[1].split()), _NAME_MAP) if len(parts) > 1 else []
    core, legal = _core_tokens(main) if main else ([], [])
    alt_core = _core_tokens(alt_tokens)[0] if alt_tokens else []
    acronym = "".join(t[0] for t in core if t) if len(core) >= 2 else ""
    return NormName(
        norm=" ".join(main + alt_tokens),
        core=" ".join(core),
        alt=" ".join(alt_core),
        legal=" ".join(sorted(set(legal))),
        skel=" ".join(skeleton(t) for t in core),
        acronym=acronym,
    )


# --------------------------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NormAddress:
    norm: str  # full canonical address
    core: str  # landmark phrase + postal code removed
    landmark: str  # canonical landmark phrase ("near sbi atm"), "" if none
    postal: str  # 5-6 digit postal/ZIP/PIN code, "" if none
    nums: str  # space-joined sorted unique digit runs (postal excluded)
    house: str  # first number of the core address, "" if none


def _canon_address_tokens(text: str) -> list[str]:
    text = _clean(text)
    for rx, rep in _ADDR_PHRASE_RES:
        text = rx.sub(rep, text)
    toks = _join_initials(text.split())
    toks = [
        t for i, t in enumerate(toks) if not (t in _NUMBER_WORDS and i + 1 < len(toks) and toks[i + 1][:1].isdigit())
    ]
    return [t for t in _map_tokens(toks, _ADDR_MAP) if t not in _ADDR_DROP]


def _find_postal(tokens: list[str]) -> str:
    """Last 5-6 digit token. A leading one is usually a (US) house number, so it only counts as
    postal when it is 6 digits or directly followed by another number (reordered components,
    e.g. ``33005 50 impasse voltaire``)."""
    for i in range(len(tokens) - 1, 0, -1):
        if _POSTAL_RE.match(tokens[i]):
            return tokens[i]
    if tokens and _POSTAL_RE.match(tokens[0]):
        if len(tokens[0]) == 6 or (len(tokens) > 1 and tokens[1].isdigit()):
            return tokens[0]
    return ""


@cache
def normalize_address(raw: str) -> NormAddress:
    folded = fold_unicode(raw)
    folded = _PIN_SPLIT_RE.sub(r"\1\2", folded.strip())
    landmark_raw = " ".join(m.group(0) for m in _LANDMARK_RE.finditer(folded))
    without_landmark = _LANDMARK_RE.sub(" ", folded)

    full = _canon_address_tokens(folded)
    core = _canon_address_tokens(without_landmark)
    landmark = _canon_address_tokens(landmark_raw) if landmark_raw else []

    postal = _find_postal(core)
    if postal:
        core = [t for t in core if t != postal]
    digits = [t for t in core if t.isdigit()]
    nums = sorted(set(digits), key=lambda x: (len(x), x))
    return NormAddress(
        norm=" ".join(full),
        core=" ".join(core),
        landmark=" ".join(landmark),
        postal=postal,
        nums=" ".join(nums),
        house=digits[0] if digits else "",
    )


# --------------------------------------------------------------------------------------------
# DataFrame API
# --------------------------------------------------------------------------------------------

NORM_COLUMNS = (
    "name_norm",
    "name_core",
    "name_alt",
    "name_legal",
    "name_skel",
    "name_acronym",
    "addr_norm",
    "addr_core",
    "addr_landmark",
    "postal",
    "nums",
    "house",
    "country_norm",
)


def normalize_records(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with the normalised columns in :data:`NORM_COLUMNS` appended."""
    n = len(df)
    names = [normalize_name(x) for x in progress(df["business_name"].tolist(), "normalise names", n, "rec")]
    addrs = [normalize_address(x) for x in progress(df["business_address"].tolist(), "normalise addresses", n, "rec")]
    out = df.copy()
    out["name_norm"] = [n.norm for n in names]
    out["name_core"] = [n.core for n in names]
    out["name_alt"] = [n.alt for n in names]
    out["name_legal"] = [n.legal for n in names]
    out["name_skel"] = [n.skel for n in names]
    out["name_acronym"] = [n.acronym for n in names]
    out["addr_norm"] = [a.norm for a in addrs]
    out["addr_core"] = [a.core for a in addrs]
    out["addr_landmark"] = [a.landmark for a in addrs]
    out["postal"] = [a.postal for a in addrs]
    out["nums"] = [a.nums for a in addrs]
    out["house"] = [a.house for a in addrs]
    out["country_norm"] = [normalize_country(x) for x in df["country"].tolist()]
    return out
