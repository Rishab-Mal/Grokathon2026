"""NOAA Lake Erie HAB bulletin archive (NCEI): list, download, extract text, parse bloom area.

Each season is a separate NCEI accession. Only the extracted text is cached
(data/raw/bulletin_text/), PDFs are ~4-5 MB each and are discarded.
"""
import io
import re
import time
from datetime import date, datetime
from pathlib import Path

import requests
from pypdf import PdfReader

NCEI = "https://www.ncei.noaa.gov/data/oceans/archive"

# year -> bulletin directory inside its NCEI accession
SEASONS = {
    2018: f"{NCEI}/arc0130/0183933/1.1/data/0-data/bulletin/",
    2019: f"{NCEI}/arc0152/0209038/1.1/data/0-data/bulletin/",
    2020: f"{NCEI}/arc0183/0239146/1.1/data/0-data/bulletin/",
    2021: f"{NCEI}/arc0213/0279234/1.1/data/0-data/bulletins/",
    2022: f"{NCEI}/arc0213/0279446/1.1/data/0-data/bulletins/",
    2023: f"{NCEI}/arc0222/0288351/1.1/data/0-data/bulletins/",
    2024: f"{NCEI}/arc0235/0299571/1.1/data/0-data/bulletins/",
}

TEXT_CACHE = Path(__file__).resolve().parent.parent / "data" / "raw" / "bulletin_text"


def _date_from_name(name):
    """Filenames differ by year: HAB20180625_2018001_LE.pdf, 2020061617.pdf, 2023-Aug-01_bulletin.pdf."""
    m = re.match(r"HAB(\d{8})_", name)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    m = re.match(r"(\d{4}-[A-Za-z]{3}-\d{2})_bulletin", name)
    if m:
        return datetime.strptime(m.group(1), "%Y-%b-%d").date()
    m = re.match(r"(\d{8})\d{2}\.pdf$", name)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    return None


def list_bulletins(year):
    """Return {date: url} for every bulletin PDF in a season (first file wins per day)."""
    base = SEASONS[year]
    html = _get(base).text
    out = {}
    for name in sorted(set(re.findall(r'href="([^"?/][^"]*\.pdf)"', html))):
        d = _date_from_name(name)
        if d and d not in out:
            out[d] = base + name
    return dict(sorted(out.items()))


def _get(url, timeout=45, tries=4):
    """NCEI drops connections under load; retry with backoff."""
    for i in range(tries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def bulletin_text(d: date, url: str) -> str:
    """Extracted text, or '' if the PDF could not be fetched (not cached, so a rerun retries it)."""
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    path = TEXT_CACHE / f"{d.isoformat()}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    try:
        pdf = _get(url).content
    except requests.RequestException as e:
        print(f"  skip {d}: {e.__class__.__name__}", flush=True)
        return ""
    reader = PdfReader(io.BytesIO(pdf))
    text = "\n".join(p.extract_text() or "" for p in reader.pages)
    path.write_text(text, encoding="utf-8")
    return text


# ---- Label rule (see LABELS.md) -------------------------------------------
# A bulletin is a BLOOM day (1) if it reports a bloom area >= AREA_THRESHOLD
# sq mi, or (when no area is given) describes a Microcystis bloom as present /
# continuing in the western basin. It is NO BLOOM (0) if it reports an area
# below the threshold, or says cyanobacteria are undetectable, only traces,
# or low concentrations. Anything else is unknown (None) and is skipped.
AREA_THRESHOLD = 50.0

_AREA = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:square\s+miles|sq\.?\s*mi)", re.I)
_AREA_KM = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:square\s+kilometers|km2|km²|sq\.?\s*km)", re.I)
_NEGATIVE = re.compile(
    r"not\s+(?:currently\s+)?detectable|no\s+(?:cyanobacteria\s+|microcystis\s+)?bloom|"
    r"bloom\s+(?:is\s+)?not\s+(?:present|detected)|only\s+traces|traces\s+of\s+cyanobacteria|"
    r"at\s+low\s+concentrations?|relatively\s+low|small\s+patches|cyanobacteria\s+ha(?:ve|s)\s+been\s+detected\s+in\s+lake\s+erie|"
    r"early\s+season\s+bloom\s+projection|bloom\s+has\s+(?:ended|dissipated)|no\s+longer\s+(?:present|detect)",
    re.I,
)
_POSITIVE = re.compile(
    r"bloom\s+(?:continues|persists|is\s+present|remains|extends|has\s+(?:grown|expanded|increased))|"
    r"bloom\s+(?:is\s+)?(?:present|visible|observed)\s+in|bloom\s+in\s+(?:the\s+)?western\s+(?:basin|lake\s+erie)|"
    r"bloom\s+of\s+microcystis",
    re.I,
)


def _unspace(text):
    """2018-19 PDFs extract with NULs between letters, or as 'C y a n o b a c t e r i a   i s'."""
    text = text.replace("\x00", "")
    return re.sub(r"(?:\S ){3,}\S", lambda m: m.group(0).replace(" ", ""), text)


def summary(text, n=700):
    """The opening analysis paragraph, whitespace-normalised."""
    flat = " ".join(_unspace(text).split())
    i = flat.find("Analysis")
    return flat[i if i >= 0 else 0:][:n]


def parse_area_sq_mi(flat):
    m = _AREA.search(flat)
    if m:
        return float(m.group(1).replace(",", ""))
    m = _AREA_KM.search(flat)
    if m:
        return round(float(m.group(1).replace(",", "")) * 0.386102, 1)
    return None


def classify(text):
    """Return (bloom 0/1/None, area_sq_mi or None, reason)."""
    s = summary(text)
    area = parse_area_sq_mi(s)
    if area is not None:
        return int(area >= AREA_THRESHOLD), area, f"area {area:g} sq mi"
    neg = _NEGATIVE.search(s)
    if neg:
        return 0, None, f"text: '{neg.group(0)}'"
    pos = _POSITIVE.search(s)
    if pos:
        return 1, None, f"text: '{pos.group(0)}'"
    return None, None, "unclassified"
