"""Synthetic dataset generator that mimics the challenge's documented noise patterns.

Used for smoke tests and for exercising the pipelines when the real data is not at hand.
Scores on this data say nothing about leaderboard performance; every threshold and weight in
the methods is re-tuned on whatever training data is supplied.

Hard cases produced on purpose:
  * chains       -- same brand at several addresses (S1 contains several of them)
  * siblings     -- same / near-identical name in another branch or city (distractor)
  * neighbours   -- different business at the same address (distractor)
  * near-names   -- one token changed ("Balaji Hospital" vs "Balaji Medical Store"), same city
  * singletons   -- S1 entities with no match at all
Test split additionally contains France, which never occurs in train.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------------------------

US_SURNAMES = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Wilson",
    "Anderson",
    "Taylor",
    "Thomas",
    "Moore",
    "Jackson",
    "Martin",
    "Lee",
    "Thompson",
    "White",
    "Harris",
    "Clark",
    "Lewis",
    "Walker",
    "Hall",
    "Young",
    "King",
    "Wright",
    "Lopez",
    "Hill",
    "Green",
    "Adams",
    "Baker",
    "Nelson",
    "Carter",
    "Mitchell",
    "Roberts",
    "Turner",
    "Phillips",
    "Campbell",
    "Parker",
    "Evans",
    "Edwards",
    "Collins",
    "Stewart",
]
US_ADJ = [
    "Golden",
    "Blue",
    "Silver",
    "Pacific",
    "Liberty",
    "Eagle",
    "Summit",
    "Pioneer",
    "Evergreen",
    "Sunrise",
    "Northern",
    "Premier",
    "Royal",
    "Heritage",
    "Capital",
    "Metro",
    "Coastal",
    "Valley",
    "Crown",
    "Diamond",
    "Atlas",
    "Keystone",
    "Lakeside",
    "Riverside",
    "Highland",
]
US_INDUSTRY = [
    "Pizza",
    "Dental",
    "Auto Repair",
    "Bakery",
    "Plumbing",
    "Law Offices",
    "Consulting",
    "Hardware",
    "Pharmacy",
    "Cafe",
    "Grill",
    "Salon",
    "Fitness",
    "Insurance Agency",
    "Realty",
    "Construction",
    "Landscaping",
    "Veterinary Clinic",
    "Medical Center",
    "Hospital",
    "Motors",
    "Electric",
    "Cleaners",
    "Florist",
    "Books",
    "Deli",
    "Tavern",
    "Services",
    "Technologies",
    "Manufacturing",
    "Associates",
    "Brothers",
    "International",
    "Laboratories",
    "Travel",
]
US_SAINTS = ["Mary", "Joseph", "Luke", "John", "Francis", "Vincent", "Anthony", "Jude"]
US_LEGAL = ["", "", "Inc", "Inc.", "LLC", "Corp", "Corporation", "Co", "Company", "Ltd"]
US_STREETS = [
    "Main",
    "Oak",
    "Maple",
    "Washington",
    "Lincoln",
    "Park",
    "Elm",
    "Cedar",
    "Lake",
    "Hill",
    "Sunset",
    "Broadway",
    "Market",
    "Church",
    "Mill",
    "Spring",
    "River",
    "Highland",
    "Madison",
    "Jefferson",
    "Franklin",
    "Pine",
    "Walnut",
    "Chestnut",
    "Center",
    "Union",
    "Grand",
    "Santa Monica",
]
US_STREET_TYPES = ["Street", "Avenue", "Road", "Boulevard", "Drive", "Lane", "Court", "Place", "Way", "Parkway"]
US_CITIES = [
    ("New York", "NY", "New York", "100"),
    ("Los Angeles", "CA", "California", "900"),
    ("Chicago", "IL", "Illinois", "606"),
    ("Houston", "TX", "Texas", "770"),
    ("Phoenix", "AZ", "Arizona", "850"),
    ("Philadelphia", "PA", "Pennsylvania", "191"),
    ("San Diego", "CA", "California", "921"),
    ("Dallas", "TX", "Texas", "752"),
    ("Austin", "TX", "Texas", "787"),
    ("Seattle", "WA", "Washington", "981"),
    ("Denver", "CO", "Colorado", "802"),
    ("Boston", "MA", "Massachusetts", "021"),
    ("Atlanta", "GA", "Georgia", "303"),
    ("Miami", "FL", "Florida", "331"),
    ("Portland", "OR", "Oregon", "972"),
    ("Columbus", "OH", "Ohio", "432"),
    ("Charlotte", "NC", "North Carolina", "282"),
    ("Nashville", "TN", "Tennessee", "372"),
]
US_CHAINS = ["Starbucks", "McDonald's", "Walgreens", "CVS Pharmacy", "Subway", "Chase Bank", "Home Depot", "Dunkin'"]

IN_PREFIX = ["Shree", "Sri", "Shri", "", "", "", "New", "Royal", "Maa", "Jai"]
IN_ROOTS = [
    "Balaji",
    "Ganesh",
    "Krishna",
    "Laxmi",
    "Durga",
    "Sai",
    "Hanuman",
    "Gupta",
    "Sharma",
    "Agarwal",
    "Patel",
    "Reddy",
    "Iyer",
    "Khan",
    "Singh",
    "Verma",
    "Mehta",
    "Jain",
    "Kumar",
    "Rao",
    "Nair",
    "Bansal",
    "Mittal",
    "Chopra",
    "Kapoor",
    "Joshi",
    "Desai",
    "Shah",
    "Mohammed",
    "Ansari",
    "Ambika",
    "Shiv",
    "Om",
    "Vinayak",
    "Gayatri",
    "Radhe",
    "Mahalaxmi",
    "Annapurna",
    "Bharat",
]
IN_INDUSTRY = [
    "Hospital",
    "Medical Store",
    "Traders",
    "Enterprises",
    "Textiles",
    "Jewellers",
    "Sweets",
    "Electricals",
    "Hardware",
    "Motors",
    "Garments",
    "Pharmacy",
    "Clinic",
    "Restaurant",
    "Hotel",
    "Steel Industries",
    "Agencies",
    "Constructions",
    "Tours and Travels",
    "Kirana Store",
    "Mobile Shop",
    "Diagnostic Centre",
    "Opticals",
    "Bakery",
    "Book Depot",
    "Furniture",
    "Auto Works",
    "Tailors",
    "Computers",
    "Brothers",
    "Associates",
    "Technologies",
]
IN_LEGAL = ["", "", "", "Pvt Ltd", "Pvt. Ltd.", "Private Limited", "LLP", "& Sons", "& Co", "Ltd"]
IN_AREAS = [
    "MG",
    "Gandhi",
    "Nehru",
    "Rajendra",
    "Shivaji",
    "Laxmi",
    "Ashok",
    "Indira",
    "Subhash",
    "Tilak",
    "Patel",
    "Lajpat",
    "Model",
    "Sadar",
    "Station",
    "Karol",
    "Anna",
    "Ram",
    "Shastri",
    "Azad",
    "Vivekananda",
    "Tagore",
    "Ambedkar",
    "Netaji",
    "Kasturba",
]
IN_AREA_TYPES = ["Nagar", "Colony", "Marg", "Road", "Bazar", "Chowk", "Market", "Vihar", "Layout", "Extension"]
IN_CITIES = [
    ("Mumbai", "Maharashtra", "MH", "400"),
    ("Delhi", "Delhi", "DL", "110"),
    ("Bengaluru", "Karnataka", "KA", "560"),
    ("Chennai", "Tamil Nadu", "TN", "600"),
    ("Kolkata", "West Bengal", "WB", "700"),
    ("Hyderabad", "Telangana", "TS", "500"),
    ("Pune", "Maharashtra", "MH", "411"),
    ("Ahmedabad", "Gujarat", "GJ", "380"),
    ("Jaipur", "Rajasthan", "RJ", "302"),
    ("Lucknow", "Uttar Pradesh", "UP", "226"),
    ("Kanpur", "Uttar Pradesh", "UP", "208"),
    ("Nagpur", "Maharashtra", "MH", "440"),
    ("Indore", "Madhya Pradesh", "MP", "452"),
    ("Patna", "Bihar", "BR", "800"),
    ("Gurugram", "Haryana", "HR", "122"),
    ("Kochi", "Kerala", "KL", "682"),
]
IN_LANDMARKS = [
    "Near SBI ATM",
    "Opp. City Mall",
    "Behind Bus Stand",
    "Near Railway Station",
    "Opposite Govt Hospital",
    "Next to HDFC Bank",
    "Near Hanuman Mandir",
    "Beside Petrol Pump",
    "Near Clock Tower",
    "Opp. Police Station",
    "Behind Post Office",
    "Near Water Tank",
]
IN_CHAINS = [
    "State Bank of India",
    "HDFC Bank",
    "Apollo Pharmacy",
    "Reliance Fresh",
    "Big Bazaar",
    "Domino's Pizza",
    "Cafe Coffee Day",
    "Tanishq",
]

FR_KIND = [
    "Boulangerie",
    "Pharmacie",
    "Café",
    "Garage",
    "Librairie",
    "Boucherie",
    "Fromagerie",
    "Cabinet",
    "Restaurant",
    "Hôtel",
    "Salon",
    "Pâtisserie",
    "Brasserie",
    "Épicerie",
    "Atelier",
    "Clinique",
]
FR_NAMES = [
    "Dupont",
    "Martin",
    "Bernard",
    "Dubois",
    "Durand",
    "Lefèvre",
    "Moreau",
    "Laurent",
    "Simon",
    "Michel",
    "Garcia",
    "David",
    "Bertrand",
    "Roux",
    "Vincent",
    "Fournier",
    "Girard",
    "Bonnet",
    "François",
    "Mercier",
    "du Centre",
    "de la Gare",
    "Saint-Michel",
    "des Arts",
    "du Port",
]
FR_LEGAL = ["", "", "SARL", "SAS", "SA", "EURL", "& Fils", "et Cie"]
FR_STREET_TYPES = ["Rue", "Avenue", "Boulevard", "Place", "Chemin", "Allée", "Impasse", "Quai"]
FR_STREETS = [
    "de la Paix",
    "de la République",
    "Victor Hugo",
    "Jean Jaurès",
    "Gambetta",
    "Pasteur",
    "de la Gare",
    "de la Liberté",
    "de l'Église",
    "du Moulin",
    "du Général de Gaulle",
    "Foch",
    "Saint-Honoré",
    "Voltaire",
    "des Écoles",
    "Nationale",
]
FR_CITIES = [
    ("Paris", "750"),
    ("Lyon", "690"),
    ("Marseille", "130"),
    ("Toulouse", "310"),
    ("Nice", "060"),
    ("Nantes", "440"),
    ("Strasbourg", "670"),
    ("Bordeaux", "330"),
    ("Lille", "590"),
    ("Rennes", "350"),
]
FR_CHAINS = ["Carrefour City", "Monoprix", "Crédit Agricole", "BNP Paribas", "Franprix"]

COUNTRY_LABEL = {"us": "US", "in": "India", "fr": "France"}
COUNTRY_ALIASES = {"us": ["USA", "United States", "U.S."], "in": ["IN", "INDIA", "Bharat"], "fr": ["FR", "FRANCE"]}

NAME_SWAPS = [
    ("Private", "Pvt"),
    ("Limited", "Ltd"),
    ("Corporation", "Corp"),
    ("Company", "Co"),
    ("Incorporated", "Inc"),
    ("&", "and"),
    ("Hospital", "Hosp."),
    ("Centre", "Center"),
    ("Center", "Ctr"),
    ("Medical", "Med."),
    ("Brothers", "Bros"),
    ("International", "Intl"),
    ("Services", "Svcs"),
    ("Saint", "St."),
    ("Enterprises", "Ent."),
    ("Technologies", "Tech"),
    ("Laboratories", "Labs"),
    ("Associates", "Assoc."),
    ("Manufacturing", "Mfg"),
]
TRANSLIT = [
    ("Shree", "Sri"),
    ("Shree", "Shri"),
    ("Sri", "Shri"),
    ("Mohammed", "Mohd"),
    ("Mohammed", "Muhammad"),
    ("Laxmi", "Lakshmi"),
    ("Ganesh", "Ganesha"),
    ("Krishna", "Krishnaa"),
    ("Mahalaxmi", "Mahalakshmi"),
    ("Vinayak", "Vinayaka"),
    ("Shiv", "Shiva"),
    ("Iyer", "Aiyar"),
    ("Agarwal", "Aggarwal"),
    ("Chopra", "Chopraa"),
    ("Jewellers", "Jewelers"),
    ("Textiles", "Textile"),
]
ADDR_SWAPS = [
    ("Road", "Rd"),
    ("Street", "St"),
    ("Avenue", "Ave"),
    ("Boulevard", "Blvd"),
    ("Drive", "Dr"),
    ("Lane", "Ln"),
    ("Court", "Ct"),
    ("Parkway", "Pkwy"),
    ("Nagar", "Ngr"),
    ("Sector", "Sec"),
    ("Suite", "Ste"),
    ("Floor", "Flr"),
    ("Colony", "Col"),
    ("Extension", "Extn"),
    ("MG", "Mahatma Gandhi"),
    ("Bazar", "Bazaar"),
    ("Market", "Mkt"),
    ("Place", "Pl"),
    ("Rue", "R."),
    ("Boulevard", "Bd"),
    ("Avenue", "Av."),
    ("Saint", "St"),
]
CITY_ALIASES = [
    ("Mumbai", "Bombay"),
    ("Bengaluru", "Bangalore"),
    ("Gurugram", "Gurgaon"),
    ("Kolkata", "Calcutta"),
    ("Chennai", "Madras"),
    ("Kochi", "Cochin"),
]


@dataclass(frozen=True)
class Entity:
    country: str
    name: str
    parts: tuple[str, ...]  # address components; last-but-one holds city/state, last holds postal
    postal: str


@dataclass(frozen=True)
class SynthConfig:
    n_s1: int = 3000
    countries: tuple[str, ...] = ("us", "in")
    country_weights: tuple[float, ...] = (0.5, 0.5)
    match_count_probs: tuple[float, ...] = (0.30, 0.36, 0.20, 0.10, 0.04)
    chain_share: float = 0.12
    distractor_ratio: float = 1.0
    distractor_mix: tuple[float, float, float, float] = (0.20, 0.30, 0.25, 0.25)  # fresh, sibling, neighbour, near-name
    s2_noise: float = 1.5
    s3_noise: float = 2.2
    partial_address_rate: float = 0.06
    country_alias_rate: float = 0.05
    country_missing_rate: float = 0.01
    seed: int = 7
    id_offset: int = 0
    extra: dict[str, str] = field(default_factory=dict)


class _Gen:
    def __init__(self, cfg: SynthConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    # ---------------------------------------------------------------- canonical entities
    def entity(self, country: str, chain: str | None = None, name: str | None = None) -> Entity:
        r = self.rng
        if country == "us":
            if name is None:
                if chain:
                    name = chain
                else:
                    pat = r.random()
                    if pat < 0.35:
                        name = f"{r.choice(US_SURNAMES)} {r.choice(US_INDUSTRY)}"
                    elif pat < 0.55:
                        name = f"{r.choice(US_SURNAMES)} & {r.choice(US_SURNAMES)} {r.choice(US_INDUSTRY)}"
                    elif pat < 0.85:
                        name = f"{r.choice(US_ADJ)} {r.choice(US_INDUSTRY)}"
                    else:
                        name = f"Saint {r.choice(US_SAINTS)}'s {r.choice(US_INDUSTRY)}"
                    legal = r.choice(US_LEGAL)
                    name = f"{name} {legal}".strip()
            city, st, _state, zp = r.choice(US_CITIES)
            num = str(r.choice([r.randint(1, 999), r.randint(100, 9999), r.randint(10000, 19999)]))
            street = f"{num} {r.choice(US_STREETS)} {r.choice(US_STREET_TYPES)}"
            parts = [street]
            if r.random() < 0.25:
                parts.append(f"Suite {r.randint(1, 9) * 100 + r.randint(0, 20)}")
            postal = zp + f"{r.randint(0, 40):02d}"
            parts += [city, st, postal]
            return Entity(country, name, tuple(parts), postal)
        if country == "in":
            if name is None:
                if chain:
                    name = chain
                else:
                    prefix = r.choice(IN_PREFIX)
                    name = f"{prefix} {r.choice(IN_ROOTS)} {r.choice(IN_INDUSTRY)} {r.choice(IN_LEGAL)}"
                    name = " ".join(name.split())
            city, state, _code, pin = r.choice(IN_CITIES)
            unit = r.choice(
                [
                    f"Shop No. {r.randint(1, 60)}",
                    f"Plot No {r.randint(1, 400)}",
                    f"{r.randint(1, 200)}/{r.randint(1, 20)}",
                    f"H.No. {r.randint(1, 99)}-{r.randint(1, 9)}",
                    f"{r.randint(1, 500)}",
                    f"{r.choice(['1st', '2nd', '3rd'])} Floor, {r.randint(1, 90)}",
                ]
            )
            area = f"{r.choice(IN_AREAS)} {r.choice(IN_AREA_TYPES)}"
            parts = [unit, area]
            if r.random() < 0.3:
                parts.append(f"Sector {r.randint(1, 60)}")
            if r.random() < 0.35:
                parts.append(r.choice(IN_LANDMARKS))
            postal = pin + f"{r.randint(1, 40):03d}"
            parts += [city, state, postal]
            return Entity(country, name, tuple(parts), postal)
        if country == "fr":
            if name is None:
                if chain:
                    name = chain
                else:
                    name = f"{r.choice(FR_KIND)} {r.choice(FR_NAMES)} {r.choice(FR_LEGAL)}".strip()
            city, cp = r.choice(FR_CITIES)
            num = str(r.randint(1, 180)) + (" bis" if r.random() < 0.08 else "")
            street = f"{num} {r.choice(FR_STREET_TYPES)} {r.choice(FR_STREETS)}"
            postal = cp + f"{r.randint(1, 20):02d}"
            return Entity(country, name, (street, postal, city), postal)
        raise ValueError(country)

    def sibling(self, e: Entity) -> Entity:
        """Same name, different branch: half the time another city, otherwise same postal area."""
        other = self.entity(e.country, name=e.name)
        if self.rng.random() < 0.5:
            return other
        parts = list(e.parts)
        parts[0] = self._renumber(other.parts[0] if self.rng.random() < 0.5 else parts[0], force=True)
        return Entity(e.country, e.name, tuple(parts), e.postal)

    def neighbour(self, e: Entity) -> Entity:
        """Different business at the same address."""
        other = self.entity(e.country)
        return Entity(e.country, other.name, e.parts, e.postal)

    def near_name(self, e: Entity) -> Entity:
        """One token of the name replaced; same city / postal area, different street number."""
        r = self.rng
        toks = e.name.split()
        pool = {"us": US_INDUSTRY + US_SURNAMES, "in": IN_INDUSTRY + IN_ROOTS, "fr": FR_KIND + FR_NAMES}[e.country]
        i = r.randrange(len(toks))
        toks[i] = r.choice(pool)
        parts = list(e.parts)
        parts[0] = self._renumber(parts[0], force=True)
        return Entity(e.country, " ".join(toks), tuple(parts), e.postal)

    # ---------------------------------------------------------------- noise
    def _typo(self, word: str) -> str:
        r = self.rng
        if len(word) < 4 or not word.isalpha():
            return word
        i = r.randrange(1, len(word) - 1)
        op = r.random()
        if op < 0.25:
            return word[:i] + word[i + 1 :]
        if op < 0.5:
            return word[:i] + r.choice("aeioulnrst") + word[i:]
        if op < 0.75:
            return word[:i] + r.choice("aeioulnrst") + word[i + 1 :]
        return word[: i - 1] + word[i] + word[i - 1] + word[i + 1 :]

    def _renumber(self, part: str, force: bool = False) -> str:
        """Change the first digit run: ``force`` -> a different number, else a formatting variant."""
        r = self.rng
        m = re.search(r"\d+", part)
        if not m:
            return part
        t = m.group(0)
        if force:
            new = str(int(t) + r.randint(1, 30))
        else:
            new = r.choice([f"{t}/{r.randint(1, 5)}", f"No. {t}", f"#{t}", f"{t}-A", t])
        return part[: m.start()] + new + part[m.end() :]

    def _swap(self, text: str, table: list[tuple[str, str]]) -> str:
        r = self.rng
        hits = [(a, b) for a, b in table if a in text.split() or b in text.split()]
        if not hits:
            return text
        a, b = r.choice(hits)
        toks = text.split()
        return " ".join(b if t == a else a if t == b else t for t in toks)

    def noisy_name(self, name: str, level: float) -> str:
        r = self.rng
        n_ops = min(4, int(level + r.random() * level))
        out = name
        for _ in range(n_ops):
            op = r.random()
            if op < 0.25:
                out = self._swap(out, NAME_SWAPS)
            elif op < 0.37:
                out = self._swap(out, TRANSLIT)
            elif op < 0.52:
                toks = out.split()
                j = r.randrange(len(toks))
                toks[j] = self._typo(toks[j])
                out = " ".join(toks)
            elif op < 0.62:
                legal = {
                    "Pvt",
                    "Ltd",
                    "Pvt.",
                    "Ltd.",
                    "Private",
                    "Limited",
                    "LLC",
                    "Inc",
                    "Inc.",
                    "Corp",
                    "Co",
                    "LLP",
                    "SARL",
                    "SAS",
                    "SA",
                    "EURL",
                    "Corporation",
                    "Company",
                }
                toks = [t for t in out.split() if t not in legal]
                if r.random() < 0.5:
                    toks.append(r.choice(["Pvt Ltd", "LLC", "Inc", "Ltd", "Co", "SARL"]))
                out = " ".join(toks) or out
            elif op < 0.70:
                toks = out.split()
                if len(toks) >= 3:
                    j = r.randrange(len(toks) - 1)
                    toks[j], toks[j + 1] = toks[j + 1], toks[j]
                out = " ".join(toks)
            elif op < 0.80:
                out = out.replace(".", "") if "." in out else out.replace(" ", " ", 1)
                out = r.choice([out.upper(), out.lower(), out.title(), out])
            elif op < 0.86:
                toks = out.split()
                if len(toks) >= 3:
                    del toks[r.randrange(1, len(toks))]
                out = " ".join(toks)
            elif op < 0.90:
                out = out.replace("'", "").replace("-", " ")
            elif op < 0.93 and " dba " not in out:
                trade = f"{self.rng.choice(US_ADJ)} {self.rng.choice(['Store', 'Shop', 'Mart', 'Outlet'])}"
                out = f"{out} dba {trade}"
            elif op < 0.97:
                toks = out.split()
                if len(toks) >= 3:
                    toks = toks[1:] if r.random() < 0.5 else toks[:-1]
                out = " ".join(toks)
            elif op < 0.985:
                toks = out.split()
                j = r.randrange(len(toks))
                if len(toks[j]) > 3 and toks[j].isalpha():
                    toks[j] = toks[j][: r.randint(3, max(3, len(toks[j]) - 2))] + "."
                out = " ".join(toks)
        return " ".join(out.split())

    def noisy_address(self, e: Entity, level: float) -> str:
        r = self.rng
        parts = list(e.parts)
        n_ops = min(5, int(level + r.random() * (level + 0.5)))
        for _ in range(n_ops):
            op = r.random()
            if op < 0.22:
                j = r.randrange(len(parts))
                parts[j] = self._swap(parts[j], ADDR_SWAPS)
            elif op < 0.32 and e.postal in parts:
                parts.remove(e.postal)
            elif op < 0.40 and len(parts) > 3:
                # drop the state / an intermediate component
                del parts[r.randrange(1, len(parts) - 1)]
            elif op < 0.50 and e.country == "in":
                lm = [p for p in parts if p in IN_LANDMARKS]
                if lm:
                    parts.remove(lm[0])
                    if r.random() < 0.5:
                        parts.insert(r.randrange(1, len(parts)), r.choice(IN_LANDMARKS))
                else:
                    parts.insert(r.randrange(1, len(parts)), r.choice(IN_LANDMARKS))
            elif op < 0.58 and len(parts) >= 3:
                i, j = sorted(r.sample(range(len(parts) - 1), 2))
                parts[i], parts[j] = parts[j], parts[i]
            elif op < 0.68:
                parts[0] = self._renumber(parts[0])
            elif op < 0.80:
                j = r.randrange(len(parts))
                toks = parts[j].split()
                if toks:
                    k = r.randrange(len(toks))
                    toks[k] = self._typo(toks[k])
                parts[j] = " ".join(toks)
            elif op < 0.86:
                parts = [self._swap(p, CITY_ALIASES) for p in parts]
            elif op < 0.92:
                if e.country == "in" and e.postal in parts and r.random() < 0.5:
                    parts[parts.index(e.postal)] = f"{e.postal[:3]} {e.postal[3:]}"
                elif e.country == "us" and e.postal in parts:
                    parts[parts.index(e.postal)] = f"{e.postal}-{r.randint(1000, 9999)}"
                else:
                    parts = [self._expand_state(p, e.country) for p in parts]
            else:
                parts = [p.upper() if r.random() < 0.5 else p.lower() for p in parts]
        if r.random() < self.cfg.partial_address_rate and len(parts) >= 3:
            keep = sorted(r.sample(range(len(parts)), 2))
            parts = [parts[k] for k in keep]
        sep = r.choice([", ", ", ", " ", ","])
        return sep.join(p for p in parts if p)

    def _expand_state(self, part: str, country: str) -> str:
        if country == "us":
            for _c, st, full, _z in US_CITIES:
                if part == st:
                    return full
                if part == full:
                    return st
        if country == "in":
            for _c, full, code, _p in IN_CITIES:
                if part == full:
                    return code
        return part

    def country_label(self, country: str, noisy: bool) -> str:
        r = self.rng
        if noisy and r.random() < self.cfg.country_missing_rate:
            return ""
        if noisy and r.random() < self.cfg.country_alias_rate:
            return r.choice(COUNTRY_ALIASES[country])
        return COUNTRY_LABEL[country]

    def clean_address(self, e: Entity) -> str:
        return ", ".join(e.parts)


def generate(cfg: SynthConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Return (s1, s2, s3, gt) for one split."""
    g = _Gen(cfg)
    r = g.rng
    countries = r.choices(cfg.countries, weights=cfg.country_weights, k=cfg.n_s1)

    entities: list[Entity] = []
    chain_pool = {"us": US_CHAINS, "in": IN_CHAINS, "fr": FR_CHAINS}
    for c in countries:
        chain = r.choice(chain_pool[c]) if r.random() < cfg.chain_share else None
        entities.append(g.entity(c, chain=chain))

    s1_rows: list[tuple[str, str, str]] = []
    s2_rows: list[tuple[str, str, str, int]] = []  # (name, address, country, owner index or -1)
    s3_rows: list[tuple[str, str, str, int]] = []
    for i, e in enumerate(entities):
        s1_name = g.noisy_name(e.name, 0.3) if r.random() < 0.3 else e.name
        s1_rows.append((s1_name, g.clean_address(e), g.country_label(e.country, noisy=False)))
        n_match = r.choices(range(len(cfg.match_count_probs)), weights=cfg.match_count_probs)[0]
        for _ in range(n_match):
            if r.random() < 0.5:
                s2_rows.append(
                    (
                        g.noisy_name(e.name, cfg.s2_noise),
                        g.noisy_address(e, cfg.s2_noise),
                        g.country_label(e.country, True),
                        i,
                    )
                )
            else:
                s3_rows.append(
                    (
                        g.noisy_name(e.name, cfg.s3_noise),
                        g.noisy_address(e, cfg.s3_noise),
                        g.country_label(e.country, True),
                        i,
                    )
                )

    n_distract = int(cfg.distractor_ratio * cfg.n_s1)
    for _ in range(n_distract):
        base = r.choice(entities)
        kind = r.choices(range(4), weights=cfg.distractor_mix)[0]
        if kind == 0:
            d = g.entity(r.choices(cfg.countries, weights=cfg.country_weights)[0])
        elif kind == 1:
            d = g.sibling(base)
        elif kind == 2:
            d = g.neighbour(base)
        else:
            d = g.near_name(base)
        level = cfg.s2_noise if r.random() < 0.5 else cfg.s3_noise
        row = (g.noisy_name(d.name, level), g.noisy_address(d, level), g.country_label(d.country, True), -1)
        (s2_rows if r.random() < 0.5 else s3_rows).append(row)

    r.shuffle(s2_rows)
    r.shuffle(s3_rows)
    order = list(range(len(s1_rows)))
    r.shuffle(order)
    s1_id = {i: f"S1-{cfg.id_offset + k + 1:05d}" for k, i in enumerate(order)}

    def frame(rows: list[tuple[str, str, str]], ids: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "entity_id": ids,
                "business_name": [x[0] for x in rows],
                "business_address": [x[1] for x in rows],
                "country": [x[2] for x in rows],
            }
        )

    s1_sorted = sorted(range(len(s1_rows)), key=lambda i: s1_id[i])
    s1 = frame([s1_rows[i] for i in s1_sorted], [s1_id[i] for i in s1_sorted])
    gt: dict[str, list[str]] = {s1_id[i]: [] for i in range(len(s1_rows))}
    for prefix, rows in (("S2", s2_rows), ("S3", s3_rows)):
        ids = [f"{prefix}-{cfg.id_offset + k + 1:05d}" for k in range(len(rows))]
        for cid, row in zip(ids, rows, strict=True):
            if row[3] >= 0:
                gt[s1_id[row[3]]].append(cid)
        df = frame([row[:3] for row in rows], ids)
        if prefix == "S2":
            s2 = df
        else:
            s3 = df
    return s1, s2, s3, gt


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _write_gt(gt: dict[str, list[str]], order: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for s in order:
            fh.write(f"{s}\t{','.join(gt[s])}\n")


def write_dataset(out_dir: str | Path, n_train: int = 3000, n_test: int = 1500, seed: int = 7) -> Path:
    """Write ``train/`` (with ground truth) and ``test/`` (hidden ground truth, France included)."""
    out = Path(out_dir)
    train_cfg = SynthConfig(n_s1=n_train, seed=seed)
    test_cfg = replace(
        train_cfg, n_s1=n_test, seed=seed + 1000, countries=("us", "in", "fr"), country_weights=(0.35, 0.35, 0.30)
    )
    for split, cfg in (("train", train_cfg), ("test", test_cfg)):
        s1, s2, s3, gt = generate(cfg)
        _write(s1, out / split / f"{split}_source1.tsv")
        _write(s2, out / split / f"{split}_source2.tsv")
        _write(s3, out / split / f"{split}_source3.tsv")
        gt_name = f"{split}_ground_truth.tsv" if split == "train" else "_hidden_test_ground_truth.tsv"
        _write_gt(gt, s1["entity_id"].tolist(), out / split / gt_name)
    return out
