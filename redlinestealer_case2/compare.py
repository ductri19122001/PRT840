#!/usr/bin/env python3
"""Prepare, execute and audit one matched local Ollama comparison. Standard library only."""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = 'llama3.1:8b'
FIXED = ['T1071.001', 'T1105', 'T1041', 'T1048.003', 'T1046']
ARMS = ['fixed_reference', 'retrieved_reference']
OPTIONS = {'seed': 42, 'temperature': 0, 'num_ctx': 8192, 'num_predict': 2304,
           'top_k': 40, 'top_p': 0.9, 'repeat_penalty': 1.0,
           'num_batch': 64, 'num_thread': 8, 'num_gpu': 0}
STOP = set('a an and are as at be been being by can could for from has have in into is it its may of on or that the their them these they this those to use used using was were will with would'.split())
INSTRUCTIONS = '''You assist a human network-forensics analyst. Use only the supplied evidence and official ATT&CK reference excerpts. Return one valid JSON object, no Markdown.
Copy observed_facts_to_copy exactly into observed_facts. Assess every supplied technique exactly once, in supplied order, using its exact ID and name. Do not add other techniques. Cite evidence IDs E01-E06 or C01-C02. Reference definitions are not evidence that the behaviour occurred. Separate supporting observations from the missing behavioural requirements. Keep explanations concise.
supported requires direct case evidence of the technique's defining attacker behaviour. possible is an explicitly unconfirmed hypothesis with both a specific case observation and missing behavioural evidence. not_supported means the supplied evidence does not establish the defining behaviour; it does not mean it was disproved.
HTTP service labels support a protocol observation only. They do not establish C2, commands, responses or exfiltration. Counts, timing, byte totals, one IP/port, and retrospective malware CTI do not by themselves establish attacker intent, scanning, beaconing, host compromise or any attack stage. Missing byte values are not zero. Use C01 only as historical context and respect the no-port-match limit C02. Dataset labels and detector outcomes are withheld.
Give at most two conditional next events, with the additional evidence needed and telemetry to monitor. Do not report a completed Cyber Kill Chain when the observed stages are unconfirmed. Give three practical response actions: immediate triage, short-term corroboration, and evidence preservation, with the trigger or prerequisite for each. These are proposed analyst actions, not executed containment. Do not recommend running suspicious files. State the major unanswered questions. Human validation is required.
'''

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(value): return hashlib.sha256(value).hexdigest()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def tokens(value): return [t for t in re.findall(r'[a-z][a-z0-9]+', value.lower()) if t not in STOP]

def retrieve(records, common):
    service = next(e['value'] for e in common['evidence'] if e['id'] == 'E05')
    # Protocol expansion follows the Case Study 1 BM25 service-facet method.
    # The query never uses family names, CTI or desired technique IDs.
    query = 'http https web protocols' if service.get('http', 0) else 'tcp network protocol'
    excluded = [r['id'] for r in records if set(r.get('platforms', [])) == {'PRE'}]
    eligible = [r for r in records if r['id'] not in excluded]
    counts = [collections.Counter(tokens(' '.join([r['name'], ' '.join(r['tactics']), r['description']]))) for r in eligible]
    lengths = [sum(c.values()) for c in counts]
    mean = sum(lengths) / len(lengths)
    df = collections.Counter(t for c in counts for t in c)
    ranked = []
    for record, count, length in zip(eligible, counts, lengths):
        term_scores = {}
        for term in sorted(set(tokens(query))):
            tf = count[term]
            if tf:
                idf = math.log(1 + (len(eligible) - df[term] + .5) / (df[term] + .5))
                term_scores[term] = idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * length / mean))
        if term_scores:
            ranked.append({'id': record['id'], 'name': record['name'], 'score': sum(term_scores.values()), 'term_scores': term_scores})
    ranked.sort(key=lambda r: (-r['score'], r['id']))
    trace = {'algorithm': 'BM25', 'k1': 1.5, 'b': .75, 'top_k': 5, 'query': query,
             'query_source': 'E05.http service label; predefined service-family expansion. HTTPS is a lookup synonym, not observed evidence.',
             'indexed_fields': ['name', 'tactics', 'description'], 'corpus_size': len(records),
             'eligible_size': len(eligible), 'excluded_PRE_only_ids': excluded, 'ranking': ranked[:20],
             'selected_ids': [r['id'] for r in ranked[:5]],
             'limitation': 'Lexical relevance does not validate the technique, and one service facet may favour overlapping protocol techniques.'}
    return trace

def obj(properties): return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}
def arr(item, **kw): return {'type': 'array', 'items': item, **kw}
def schema():
    string = {'type': 'string'}
    integer = {'type': 'integer'}
    evidence_ids = arr({'type': 'string', 'enum': ['E01','E02','E03','E04','E05','E06','C01','C02']}, minItems=1)
    facts = obj({'selected_flows': integer, 'connection_states': obj({k: integer for k in ['RSTO','S1','S0','OTH']}),
                 'service_labels': obj({'http': integer, '-': integer}), 'total_orig_bytes': integer, 'total_resp_bytes': integer,
                 'orig_bytes_missing_rows': integer, 'resp_bytes_missing_rows': integer})
    mapping = obj({'technique_id': string, 'technique_name': string,
                   'status': {'type': 'string', 'enum': ['supported','possible','not_supported']},
                   'confidence': {'type': 'string', 'enum': ['low','medium','high']},
                   'supporting_observations': string, 'missing_behavioural_evidence': string, 'evidence_ids': evidence_ids})
    event = obj({'event': string, 'preconditions': string, 'telemetry_to_monitor': string, 'evidence_ids': evidence_ids})
    action = obj({'phase': {'type': 'string', 'enum': ['immediate','short_term','evidence_preservation']},
                  'action': string, 'trigger_or_prerequisite': string, 'evidence_ids': evidence_ids})
    return obj({'observed_facts': facts, 'candidate_attack_mappings': arr(mapping, minItems=5, maxItems=5),
                'plausible_next_events': arr(event, maxItems=2), 'response_actions': arr(action, minItems=3, maxItems=3),
                'unanswered_questions': arr(string, minItems=1, maxItems=4), 'human_validation_required': {'type': 'boolean'}})

def excerpt(record):
    # Same excerpt policy in both conditions; all source text remains in references.
    text = record['description']
    if len(text) > 1600:
        head = text[:1600]; cut = head.rfind('. ')
        text = head[:cut + 1] if cut > 800 else head.rsplit(' ', 1)[0] + '…'
    return {'technique_id': record['id'], 'name': record['name'], 'tactics': record['tactics'],
            'description': text, 'description_truncated': text != record['description'], 'source_url': record['url']}

def verify_assets():
    corpus = ROOT / 'references/enterprise_techniques.json'
    manifest = read(ROOT / 'references/corpus_manifest.json')
    assert sha(corpus.read_bytes()) == manifest['normalised_sha256'], 'Corpus checksum mismatch'
    records = read(corpus)
    assert len(records) == manifest['active_techniques'] == 697
    sources = read(ROOT / 'source/source_manifest.json')
    for item in sources['files']:
        assert sha((ROOT / 'source/duc_baseline' / item['name']).read_bytes()) == item['sha256'], item['name']
    return records, manifest

def prepare(out):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Prepared comparison already exists; choose a new directory to preserve it.')
    records, corpus = verify_assets(); by_id = {r['id']: r for r in records}
    common = read(ROOT / 'inputs/common_evidence.json'); trace = retrieve(records, common)
    selections = {'fixed_reference': FIXED, 'retrieved_reference': trace['selected_ids']}
    write(out / 'common_evidence.json', common); write(out / 'output_schema.json', schema()); write(out / 'retrieval_trace.json', trace)
    manifests = {}
    for arm in ARMS:
        payload = {'case_evidence': common, 'attack_reference': [excerpt(by_id[i]) for i in selections[arm]]}
        prompt = INSTRUCTIONS + '\nINPUT\n' + json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        request = {'model': MODEL, 'prompt': prompt, 'format': schema(), 'stream': False, 'keep_alive': '30m', 'options': OPTIONS}
        d = out / arm; write(d / 'input.json', payload); write(d / 'request.json', request)
        (d / 'prompt.txt').write_text(prompt, encoding='utf-8')
        manifests[arm] = {'technique_ids': selections[arm], 'request_sha256': sha(canonical(request)),
                          'prompt_sha256': sha(prompt.encode()), 'reference_sha256': sha(canonical(payload['attack_reference']))}
    write(out / 'experiment_manifest.json', {'prepared_utc': now(), 'model': MODEL, 'options': OPTIONS,
          'seeds': [42], 'replicates_per_condition': 1, 'execution_order': ARMS, 'reference_top_k': 5,
          'common_evidence_sha256': sha(canonical(common)), 'instructions_sha256': sha(INSTRUCTIONS.encode()),
          'schema_sha256': sha(canonical(schema())), 'corpus': corpus, 'arms': manifests,
          'comparison_boundary': 'Only the selected reference records differ within this pair. Duc\'s original run used different evidence, prompt, settings and Ollama version and is a separate pilot. Single paired run; no accuracy, significance or repeatability estimate.'})
    print('Prepared:', out); print('Retrieved:', ', '.join(selections['retrieved_reference']))

def strict_json(raw):
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d: raise ValueError('Duplicate JSON key: ' + k)
            d[k] = v
        return d
    def bad(value): raise ValueError('Non-finite JSON number: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)

def schema_errors(value, spec, path='$'):
    errors = []; kind = spec['type']
    ok = {'object': isinstance(value, dict), 'array': isinstance(value, list), 'string': isinstance(value, str),
          'integer': isinstance(value, int) and not isinstance(value, bool), 'boolean': isinstance(value, bool)}[kind]
    if not ok: return [path + ': expected ' + kind]
    if 'enum' in spec and value not in spec['enum']: errors.append(path + ': invalid enum')
    if kind == 'object':
        props = spec['properties']
        for k in spec['required']:
            if k not in value: errors.append(path + ': missing ' + k)
        for k, v in value.items():
            if k not in props: errors.append(path + ': unknown ' + k)
            else: errors.extend(schema_errors(v, props[k], path + '.' + k))
    if kind == 'array':
        if len(value) < spec.get('minItems', 0) or len(value) > spec.get('maxItems', 10**9): errors.append(path + ': wrong item count')
        for i, v in enumerate(value): errors.extend(schema_errors(v, spec['items'], path + f'[{i}]'))
    return errors

def audit(raw, payload, envelope):
    report = {'valid_json': False, 'schema_valid': False, 'errors': [],
              'semantic_validation': 'Pending independent human review; automated checks are structural and factual-copy checks only.'}
    try: value = strict_json(raw)
    except (ValueError, TypeError) as e:
        report['errors'].append(str(e)); return report, None
    report['valid_json'] = True
    report['errors'] = schema_errors(value, schema()); report['schema_valid'] = not report['errors']
    if not report['schema_valid']: return report, value
    expected = payload['case_evidence']['observed_facts_to_copy']
    report['exact_observed_facts'] = value['observed_facts'] == expected
    if not report['exact_observed_facts']: report['errors'].append('Observed facts differ from supplied values')
    references = payload['attack_reference']; mappings = value['candidate_attack_mappings']
    report['exact_technique_coverage'] = [m['technique_id'] for m in mappings] == [r['technique_id'] for r in references]
    report['exact_technique_names'] = all(m['technique_name'] == r['name'] for m,r in zip(mappings,references))
    if not report['exact_technique_coverage']: report['errors'].append('Technique set/order differs from supplied reference')
    if not report['exact_technique_names']: report['errors'].append('Technique names differ from reference')
    report['three_action_phases'] = collections.Counter(a['phase'] for a in value['response_actions']) == collections.Counter(['immediate','short_term','evidence_preservation'])
    if not report['three_action_phases']: report['errors'].append('Response phases incomplete or duplicated')
    report['human_validation_required'] = value['human_validation_required'] is True
    if not report['human_validation_required']: report['errors'].append('Human review incorrectly omitted')
    report['done_reason'] = envelope.get('done_reason'); report['generation_complete'] = envelope.get('done') is True and envelope.get('done_reason') == 'stop'
    if not report['generation_complete']: report['errors'].append('Generation incomplete or stopped by length')
    report['prompt_eval_count'] = envelope.get('prompt_eval_count'); report['eval_count'] = envelope.get('eval_count')
    report['context_budget_ok'] = isinstance(report['prompt_eval_count'], int) and report['prompt_eval_count'] + OPTIONS['num_predict'] + 512 <= OPTIONS['num_ctx']
    if not report['context_budget_ok']: report['errors'].append('Insufficient context budget or missing prompt token count')
    report['automated_checks_passed'] = not report['errors']
    return report, value

def api(base, endpoint, value=None, timeout=30):
    req = urllib.request.Request(base.rstrip('/') + endpoint, data=None if value is None else canonical(value),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response: return response.read()

def verify_prepared(out):
    verify_assets(); manifest = read(out / 'experiment_manifest.json')
    requests = [read(out / a / 'request.json') for a in ARMS]
    inputs = [read(out / a / 'input.json') for a in ARMS]
    assert inputs[0]['case_evidence'] == inputs[1]['case_evidence']
    assert sha(canonical(inputs[0]['case_evidence'])) == manifest['common_evidence_sha256']
    for arm, request, payload in zip(ARMS, requests, inputs):
        assert sha(canonical(request)) == manifest['arms'][arm]['request_sha256']
        assert sha(request['prompt'].encode()) == manifest['arms'][arm]['prompt_sha256']
        assert request['prompt'] == INSTRUCTIONS + '\nINPUT\n' + json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    assert {k:v for k,v in requests[0].items() if k!='prompt'} == {k:v for k,v in requests[1].items() if k!='prompt'}
    return manifest

def run(out, base):
    manifest = verify_prepared(out)
    if (out / 'execution_started.json').exists(): raise FileExistsError('Run already started; outputs are preserved. No automatic retries.')
    tags = strict_json(api(base, '/api/tags'))
    identity = next(m for m in tags['models'] if m['name'] == MODEL)
    metadata = {'started_utc': now(), 'api_base': 'http://127.0.0.1:11434', 'model': identity,
                'ollama_version': strict_json(api(base, '/api/version')), 'execution_order': ARMS,
                'scope': 'Local text inference only. No response action executed.'}
    write(out / 'execution_started.json', metadata)
    model_info = strict_json(api(base, '/api/show', {'model': MODEL}))
    # Retain model provenance without exporting the installation-specific Modelfile path.
    model_info.pop('modelfile', None)
    model_info['captured_fields_note'] = 'The API Modelfile field is omitted because it contains the local installation path. Model identity and generation template are retained.'
    write(out / 'model_show.json', model_info)
    for arm in ARMS:
        d = out / arm; print('Generating', arm, flush=True)
        started = now()
        try:
            raw_bytes = api(base, '/api/generate', read(d / 'request.json'), timeout=3600)
            (d / 'ollama_response.json').write_bytes(raw_bytes)
            envelope = strict_json(raw_bytes)
            raw = envelope.get('response', '')
            (d / 'response_raw.txt').write_text(raw, encoding='utf-8')
            validation, parsed = audit(raw, read(d / 'input.json'), envelope)
            if parsed is not None: write(d / 'response_parsed.json', parsed)
            write(d / 'validation.json', validation)
            write(d / 'timing.json', {'started_utc': started, 'finished_utc': now(), **{k:envelope.get(k) for k in ['total_duration','load_duration','prompt_eval_duration','eval_duration','prompt_eval_count','eval_count','done_reason']}})
            print(arm, 'finished:', validation, flush=True)
        except Exception as e:
            write(d / 'execution_error.json', {'started_utc': started, 'failed_utc': now(), 'error_type': type(e).__name__, 'message': str(e)})
            raise
    after = strict_json(api(base, '/api/tags')); identity_after = next(m for m in after['models'] if m['name']==MODEL)
    assert identity_after['digest'] == identity['digest'], 'Model changed during comparison'
    write(out / 'execution_complete.json', {'finished_utc': now(), 'model_digest_unchanged': True, 'independent_human_review': 'pending'})

def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['prepare','verify','run']); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:11434'); a = p.parse_args()
    if a.command=='prepare': prepare(a.out)
    elif a.command=='verify': verify_prepared(a.out); print('Prepared pair and source checksums verified')
    else: run(a.out, a.base_url)
if __name__ == '__main__': main()
