#!/usr/bin/env python3
"""
Extract Wikipedia articles from the bz2 XML dump into clean JSONL files.
No external dependencies — uses only stdlib.

Usage:
    python3 scripts/extract_wikipedia.py [source_bz2] [output_dir]

Defaults:
    source: /opt/data/wikipedia/enwiki-latest-pages-articles.xml.bz2
    output: /opt/data/wikipedia/text
"""
import bz2
import xml.etree.ElementTree as ET
import json
import os
import re
import sys
import time


def clean_wiki_markup(text: str) -> str:
    """Strip common wiki markup to produce cleaner text for RAG."""
    # Remove templates {{ ... }}
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    # Convert [[link|display]] to display, [[link]] to link
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    # Remove external links [http://... display] -> display
    text = re.sub(r'\[https?://[^\s\]]+ ([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://[^\]]+\]', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+/?>', '', text)
    # Remove ref tags and contents
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*/>', '', text)
    # Remove category/file links
    text = re.sub(r'\[\[(?:Category|File|Image):[^\]]*\]\]', '', text, flags=re.IGNORECASE)
    # Remove bold/italic markup
    text = re.sub(r"'{2,5}", '', text)
    # Remove section headers (== Heading ==)
    text = re.sub(r'={2,6}\s*(.+?)\s*={2,6}', r'\1.', text)
    # Remove multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove leading/trailing whitespace per line
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()


def extract_wikipedia(src: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Try common namespace URIs — the dump version changes over time
    ns_candidates = [
        '{http://www.mediawiki.org/xml/export-0.11/}',
        '{http://www.mediawiki.org/xml/export-0.10/}',
    ]

    count = 0
    batch = []
    file_idx = 0
    batch_size = 1000
    skipped_redirects = 0
    skipped_short = 0
    ns_prefix = None
    start_time = time.time()

    print(f'Source: {src}')
    print(f'Output: {out_dir}')
    file_size_gb = os.path.getsize(src) / (1024**3)
    print(f'File size: {file_size_gb:.1f} GB')
    print(f'Streaming Wikipedia XML (this takes 2-4 hours)...')
    print()

    with bz2.open(src, 'rb') as f:
        for event, elem in ET.iterparse(f, events=('end',)):
            # Auto-detect namespace from first element
            if ns_prefix is None:
                tag = elem.tag
                if '}' in tag:
                    ns_prefix = tag[:tag.index('}') + 1]
                    print(f'Detected XML namespace: {ns_prefix}')
                else:
                    # No namespace
                    ns_prefix = ''

            if elem.tag == ns_prefix + 'page':
                title_el = elem.find(ns_prefix + 'title')
                text_el = elem.find(f'.//{ns_prefix}text')
                ns_el = elem.find(ns_prefix + 'ns')

                # Only namespace 0 = articles
                if ns_el is not None and ns_el.text == '0' and text_el is not None and text_el.text:
                    raw_text = text_el.text

                    if raw_text.startswith('#REDIRECT') or raw_text.startswith('#redirect'):
                        skipped_redirects += 1
                    else:
                        cleaned = clean_wiki_markup(raw_text)
                        if len(cleaned) >= 100:
                            batch.append({
                                'id': str(count),
                                'title': title_el.text if title_el is not None else '',
                                'text': cleaned
                            })
                            count += 1

                            if len(batch) >= batch_size:
                                subdir = os.path.join(out_dir, f'{file_idx // 100:02X}')
                                os.makedirs(subdir, exist_ok=True)
                                fpath = os.path.join(subdir, f'wiki_{file_idx:05d}.jsonl')
                                with open(fpath, 'w', encoding='utf-8') as of:
                                    for doc in batch:
                                        of.write(json.dumps(doc, ensure_ascii=False) + '\n')
                                file_idx += 1
                                batch = []

                                if count % 50000 == 0:
                                    elapsed = time.time() - start_time
                                    rate = count / elapsed
                                    print(f'  {count:>10,} articles  |  '
                                          f'{skipped_redirects:,} redirects skipped  |  '
                                          f'{elapsed/60:.0f} min  |  '
                                          f'{rate:.0f} articles/sec',
                                          flush=True)
                        else:
                            skipped_short += 1

                # Free memory
                elem.clear()

    # Write remaining batch
    if batch:
        subdir = os.path.join(out_dir, f'{file_idx // 100:02X}')
        os.makedirs(subdir, exist_ok=True)
        fpath = os.path.join(subdir, f'wiki_{file_idx:05d}.jsonl')
        with open(fpath, 'w', encoding='utf-8') as of:
            for doc in batch:
                of.write(json.dumps(doc, ensure_ascii=False) + '\n')

    elapsed = time.time() - start_time
    print()
    print(f'Done!')
    print(f'  Articles extracted: {count:,}')
    print(f'  Redirects skipped: {skipped_redirects:,}')
    print(f'  Too-short skipped: {skipped_short:,}')
    print(f'  Output files: {file_idx + 1}')
    print(f'  Time: {elapsed/3600:.1f} hours')
    print(f'  Output dir: {out_dir}')


if __name__ == '__main__':
    source = sys.argv[1] if len(sys.argv) > 1 else '/opt/data/wikipedia/enwiki-latest-pages-articles.xml.bz2'
    output = sys.argv[2] if len(sys.argv) > 2 else '/opt/data/wikipedia/text'

    if not os.path.exists(source):
        print(f'ERROR: Source file not found: {source}')
        sys.exit(1)

    extract_wikipedia(source, output)
