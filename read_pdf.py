from dataclasses import dataclass
from pathlib import Path
import re

@dataclass(frozen=True)
class Chunk:
    id: str
    page: int
    text: str


def clean_text(text: str) -> str:
    # PyMuPDF preserves word boundaries in this designed PDF better than pypdf.
    text = text.replace('\x02', '').replace('\u00ad', '')
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    # Keep line boundaries: a person's name followed by their role must remain
    # distinguishable from the next person's name and role.
    lines = [re.sub(r'[^\S\n]+', ' ', line).strip() for line in text.splitlines()]
    return '\n'.join(line for line in lines if line).strip()


def split_text(text: str, size: int = 150, overlap: int = 35) -> list[str]:
    if not 0 <= overlap < size:
        raise ValueError('Require 0 <= overlap < chunk size.')
    words = list(re.finditer(r'\S+', text))
    parts = []
    for start in range(0, len(words), size - overlap):
        last = min(start + size, len(words)) - 1
        parts.append(text[words[start].start():words[last].end()])
        if start + size >= len(words):
            break
    return parts


def team_cards(page):
    """Reflow the handbook's visual staff cards without inventing staff data.

    Names use larger type. Their roles sit directly beneath the same horizontal
    centre. Separate cards avoid merging a role with another staff member.
    """
    spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b
             for line in b['lines'] for s in line['spans'] if s['text'].strip()]
    cards = []
    for name in spans:
        if not 14 <= name['size'] <= 22 or not 2 <= len(name['text'].split()) <= 4:
            continue
        box = name['bbox']
        centre = (box[0] + box[2]) / 2
        details = [s for s in spans if 0.5 * name['size'] < s['size'] < 0.85 * name['size']
                   and abs((s['bbox'][0] + s['bbox'][2]) / 2 - centre) < 15
                   and -2 <= s['bbox'][1] - box[3] <= 35]
        if details:
            details.sort(key=lambda s: s['bbox'][1])
            cards.append(name['text'].strip() + ' — ' + details[0]['text'].strip()
                         + ''.join('\n' + s['text'].strip() for s in details[1:]))
    return cards


def extract_chunks(pdf: Path) -> list[Chunk]:
    chunks = []
    import fitz
    with fitz.open(pdf) as document:
        for page_number, page in enumerate(document, start=1):
            # Native block order preserves the team member/role pairs on page 15.
            text = clean_text(page.get_text())
            if 'THE TEAM' in text:
                cards = team_cards(page)
                if len(cards) >= 3:
                    chunks.extend(Chunk(f'p{page_number}-c{i}', page_number, card)
                                  for i, card in enumerate(cards))
                    continue
            if len(text.split()) < 15:
                continue
            for number, part in enumerate(split_text(text)):
                chunks.append(Chunk(f'p{page_number}-c{number}', page_number, part))
    if not chunks:
        raise ValueError('No usable PDF text found. Scanned PDFs require OCR first.')
    return chunks

