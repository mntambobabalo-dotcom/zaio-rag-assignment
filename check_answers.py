from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
from dotenv import load_dotenv
from ai import LLM, ProviderError, REFUSAL
from database import KnowledgeBase
from main import Assistant


def run(live=False, output=Path('results/check_results.json')):
    load_dotenv()
    kb = KnowledgeBase.load(Path('data/index'))
    assistant = Assistant(kb, LLM())
    results = []
    for case in json.loads(Path('questions.json').read_text()):
        hits = kb.search(case['question'])
        pages = sorted({c.page for c, _ in hits})
        row = {**case, 'retrieved_sources': [f'Page {p}' for p in pages],
               'retrieval_hit': bool(set(pages) & set(case['expected_pages']))
                                if case['expected_pages'] else None,
               'retrieved_chunks': [{'page': c.page, 'chunk_id': c.id,
                                     'score': round(s, 6)} for c, s in hits],
               'generated_answer': None, 'returned_source': None,
               'status': 'NOT_RUN_LLM' if hits else 'REFUSED_WITHOUT_LLM'}
        if not hits:
            row['generated_answer'] = REFUSAL
        elif live:
            try:
                response = assistant.ask(case['question'])
                row.update(generated_answer=response['answer'],
                           returned_source=response['source'], status='LIVE_RESPONSE',
                           model_output=getattr(assistant.llm, 'last_output', None))
            except ProviderError as exc:
                row.update(status='PROVIDER_ERROR', error=str(exc))
        results.append(row)
        print(f"{len(results):02d}: {row['status']} — {case['question']}", flush=True)
    report = {'run_at_utc': datetime.now(timezone.utc).isoformat(),
              'live_requested': live, 'model': assistant.llm.model if live else None,
              'note': 'Reference answers are hand-reviewed expectations, not generated answers. '
                      'Live answers require manual factual review even when evidence validation passes.',
              'index': kb.manifest, 'cases': results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'{len(results)} cases written to {output}')
    if live and any(r['status'] == 'PROVIDER_ERROR' for r in results):
        raise SystemExit('Live evaluation incomplete: provider errors. See report.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/check_results.json'))
    args = parser.parse_args()
    run(args.live, args.output)
