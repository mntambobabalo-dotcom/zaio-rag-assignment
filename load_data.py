from pathlib import Path
import argparse
from database import KnowledgeBase

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build handbook embeddings and FAISS index.')
    parser.add_argument('--pdf', type=Path, default=Path('data/handbook.pdf'))
    parser.add_argument('--index', type=Path, default=Path('data/index'))
    parser.add_argument('--backend', choices=['semantic', 'tfidf'], default='semantic')
    args = parser.parse_args()
    knowledge = KnowledgeBase.build(args.pdf, args.index, backend=args.backend)
    print(knowledge.manifest)
