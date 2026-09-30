import csv
import collections
import datetime as dt
from decimal import Decimal
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source/duc_baseline'

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

rows = list(csv.DictReader((SOURCE / 'redline_endpoint_evidence_only_flows.csv').open(encoding='utf-8-sig')))
assert len(rows) == len({r['uid'] for r in rows}) == 2755
counts = lambda field: dict(collections.Counter(r[field] for r in rows))
def total(field):
    values = [Decimal(r[field]) for r in rows if r[field] != '']
    assert all(v == v.to_integral_value() for v in values)
    return int(sum(values))
summary = {
    'selected_flows': len(rows),
    'source_ip': next(iter(counts('id.orig_h'))),
    'destination_ip': next(iter(counts('id.resp_h'))),
    'destination_port': 2567,
    'protocol': 'tcp',
    'first_seen_utc': min(r['timestamp_utc'] for r in rows),
    'last_seen_utc': max(r['timestamp_utc'] for r in rows),
    'connection_states': counts('conn_state'),
    'service_labels': counts('service'),
    'total_orig_bytes': total('orig_bytes'),
    'total_resp_bytes': total('resp_bytes'),
    'orig_bytes_missing_rows': sum(r['orig_bytes'] == '' for r in rows),
    'resp_bytes_missing_rows': sum(r['resp_bytes'] == '' for r in rows),
    'total_orig_packets': total('orig_pkts'),
    'total_resp_packets': total('resp_pkts'),
}
assert summary['total_orig_bytes'] == 885750 and summary['total_resp_bytes'] == 0
assert counts('id.orig_h') == {'192.168.1.106': 2755}
assert counts('id.resp_h') == {'4.234.116.12': 2755}
assert counts('id.resp_p') == {'2567': 2755}
write(ROOT / 'analysis/verified_flow_summary.json', summary)
hourly = collections.Counter(r['timestamp_utc'][:13] + ':00:00+00:00' for r in rows)
for filename, fields, values in [
    ('hourly_flow_starts.csv', ['hour_utc', 'flow_starts'], sorted(hourly.items())),
    ('connection_states.csv', ['connection_state', 'flow_count'], summary['connection_states'].items()),
    ('service_labels.csv', ['zeek_service', 'flow_count'], summary['service_labels'].items()),
]:
    p = ROOT / 'analysis' / filename
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(fields); w.writerows(values)

original = (SOURCE / 'case002_ollama_prompt_with_retrospective_cti.txt').read_text(encoding='utf-8')
payload = json.loads(original.split('\nINPUT\n', 1)[1])
facts = [
    {'id': 'E01', 'kind': 'network_observation', 'value': {k: summary[k] for k in ['selected_flows', 'source_ip', 'destination_ip', 'destination_port', 'protocol']}},
    {'id': 'E02', 'kind': 'network_observation', 'value': {k: summary[k] for k in ['first_seen_utc', 'last_seen_utc']}},
    {'id': 'E03', 'kind': 'network_observation', 'value': summary['connection_states']},
    {'id': 'E04', 'kind': 'network_observation', 'value': {k: summary[k] for k in ['total_orig_bytes', 'total_resp_bytes', 'orig_bytes_missing_rows', 'resp_bytes_missing_rows', 'total_orig_packets', 'total_resp_packets']}, 'boundary': 'Byte totals sum available values only; 134 rows in each byte column are missing and have not been imputed as zero.'},
    {'id': 'E05', 'kind': 'network_observation', 'value': summary['service_labels'], 'boundary': 'Zeek service labels only. HTTP request/response fields, URIs, hosts and payload are not supplied. An http label does not establish command-and-control or exfiltration.'},
    {'id': 'E06', 'kind': 'evidence_availability', 'value': 'Only these endpoint-selected conn.log-derived records and the team CTI register are supplied. No process, executable, credential-access, file-transfer, HTTP application content or wider host/service probing evidence is supplied. Absence from this input does not establish absence from the capture.'},
    {'id': 'C01', 'kind': 'retrospective_cti_context', 'value': payload['retrospective_endpoint_cti']['source'], 'retrieved_utc_date': '2026-09-22', 'boundary': 'Team-recorded historical file-to-IP association, not an independently reverified live CTI result. The 2023-02-17 date is the related-file analysis date in the source prompt, not a proven date of publication or observed connection. Family labels are external CTI context, not dataset ground truth.'},
    {'id': 'C02', 'kind': 'retrospective_cti_limit', 'value': payload['retrospective_endpoint_cti']['port_specific_cti'], 'boundary': 'Do not infer that the related file used port 2567, that these flows are RedLineStealer traffic, or that the host is compromised.'},
]
common = {
    'case_id': 'CTU-SME-11-VM2-case-002',
    'evidence': facts,
    'observed_facts_to_copy': {k: summary[k] for k in ['selected_flows', 'connection_states', 'service_labels', 'total_orig_bytes', 'total_resp_bytes', 'orig_bytes_missing_rows', 'resp_bytes_missing_rows']},
    'controls': {'dataset_labels': 'withheld', 'detector_outcomes': 'withheld', 'external_cti_family_names': 'context only'},
    'interpretation_rules': [
        'S0 indicates a connection attempt with no reply seen; RSTO means the originator reset an established connection. Neither proves malicious intent or successful C2.',
        'Zero resp_bytes is the sum of available responder payload-byte values. It does not mean no responder packets or that missing values are zero: use E04.',
        'Repeated traffic to one endpoint is not, by itself, evidence of scanning, beaconing, an attack stage, successful C2 or exfiltration.',
        'The reference describes what adversaries may do; it is not observed case evidence.',
    ],
}
write(ROOT / 'inputs/common_evidence.json', common)
write(ROOT / 'source/source_manifest.json', {
    'repository': 'https://github.com/ductri19122001/PRT840',
    'branch': 'CTI--Remaining-Results',
    'commit': 'f0fe547ba4d67e2ec5de6b900abc39ac9560f5c0',
    'directory': 'CTU-SME-11_Windows7full-2/outputs/case_studies/redline_cti',
    'reviewed_utc_date': '2026-09-30',
    'files': [{'name': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size} for p in sorted(SOURCE.iterdir())],
    'scope': 'Supplied endpoint-selected CSV files and saved run; original Zeek capture was not independently re-extracted.',
    'timeline_note': 'The supplied endpoint_timeline.csv duplicates the 2755 evidence rows. The hourly_flow_starts.csv in analysis is a new aggregate.',
})
cache = ROOT / 'references'
manifest = json.loads((cache / 'corpus_manifest.json').read_text())
assert hashlib.sha256((cache / 'enterprise_techniques.json').read_bytes()).hexdigest() == manifest['normalised_sha256']
print(json.dumps(summary, indent=2))
