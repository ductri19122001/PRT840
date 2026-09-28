import pandas as pd
from pathlib import Path
from datetime import datetime,timezone
p=Path(__file__).resolve().parent
root=p.parent
m=pd.read_csv(root/'cti_mapping_metadata(2).csv')
f=pd.read_csv(root/'clean_network_flows(2).csv')
assert len(m)==len(f) and m.flow_row_id.is_unique and f.flow_row_id.is_unique
assert (m.sort_values('flow_row_id')['flow_row_id'].to_numpy()==f.sort_values('flow_row_id')['flow_row_id'].to_numpy()).all()
assert (m.set_index('flow_row_id')['id.orig_p'].sort_index().to_numpy()==f.set_index('flow_row_id')['id.orig_p'].sort_index().to_numpy()).all()
assert (m.set_index('flow_row_id')['id.resp_p'].sort_index().to_numpy()==f.set_index('flow_row_id')['id.resp_p'].sort_index().to_numpy()).all()
assert (m.set_index('flow_row_id')['label'].sort_index().map({'Benign':0,'Malicious':1}).to_numpy()==f.set_index('flow_row_id')['label'].sort_index().to_numpy()).all()
joined=m.drop(columns=['label','detailedlabel','source_file','capture_date']).merge(f.drop(columns=['label']),on=['flow_row_id','id.orig_p','id.resp_p'],validate='one_to_one')
case=joined[(joined['id.resp_h']=='66.63.168.35')&(joined['id.resp_p']==5888)].copy().sort_values(['ts','flow_row_id'])
case.insert(0,'evidence_id',case.flow_row_id.map(lambda x:f'FLOW-{x}'))
case['timestamp_utc']=pd.to_datetime(case.ts,unit='s',utc=True).dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
case['cti_indicator']='66.63.168.35:5888'
case['cti_status']='MATCH_PUBLIC_MALWARE_CONFIG'
case['cti_source']='MalwareBazaar sample b121d9b4950ef1d71a99bbd8a0f93e71a150ec01f7986b1eebc1c557e9810819'
case['cti_source_url']='https://bazaar.abuse.ch/sample/b121d9b4950ef1d71a99bbd8a0f93e71a150ec01f7986b1eebc1c557e9810819/'
case['cti_caveat']='Public malware configuration contains endpoint; does not prove individual flow payload, exfiltration, or host compromise.'
case.to_csv(p/'remcos_endpoint_enriched_flows.csv',index=False)
cols=['evidence_id','timestamp_utc','uid','id.orig_h','id.orig_p','id.resp_h','id.resp_p','proto','service','conn_state','duration','orig_bytes','resp_bytes','orig_pkts','resp_pkts','history']
case[cols].to_csv(p/'remcos_evidence_only_flows.csv',index=False)
register=pd.DataFrame([{'indicator':'66.63.168.35:5888','indicator_type':'ip_port','status':'MATCH_PUBLIC_MALWARE_CONFIG','source':'MalwareBazaar','source_url':'https://bazaar.abuse.ch/sample/b121d9b4950ef1d71a99bbd8a0f93e71a150ec01f7986b1eebc1c557e9810819/','source_sample_sha256':'b121d9b4950ef1d71a99bbd8a0f93e71a150ec01f7986b1eebc1c557e9810819','finding':'Public sample report lists C2 Extraction: 66.63.168.35:5888','source_first_seen_utc':'2023-02-21 09:02:10 UTC','lookup_date_utc':datetime.now(timezone.utc).strftime('%Y-%m-%d'),'limitations':'Historical sample-specific config; public source may derive from other sandbox feeds; does not establish traffic contents or independently validate dataset labels.'},{'indicator':'66.63.168.35:5888','indicator_type':'ip_port','status':'MATCH_PUBLIC_MALWARE_CONFIG','source':'MalwareBazaar (second sample)','source_url':'https://bazaar.abuse.ch/sample/08c829e7056b8e022539076acbc962dea072e6506184d4036b785cb0e4592371/','source_sample_sha256':'08c829e7056b8e022539076acbc962dea072e6506184d4036b785cb0e4592371','finding':'Public executable sample report lists C2 Extraction: 66.63.168.35:5888','source_first_seen_utc':'2023-02-21 09:02:16 UTC','lookup_date_utc':datetime.now(timezone.utc).strftime('%Y-%m-%d'),'limitations':'Same sample chain as archive; not independent corroborating intelligence provider.'}]);register.to_csv(p/'cti_evidence_register.csv',index=False)
summary=case.groupby(case.timestamp_utc.str[:10]).agg(flows=('flow_row_id','size'),first_utc=('timestamp_utc','min'),last_utc=('timestamp_utc','max'),total_orig_bytes=('orig_bytes','sum'),total_resp_bytes=('resp_bytes','sum')).reset_index().rename(columns={'timestamp_utc':'date_utc'});summary.to_csv(p/'case_daily_summary.csv',index=False)
print('JOIN_VERIFIED',len(joined),'TARGET_ENDPOINT_FLOWS',len(case),'OUTPUT_FILES',*[x.name for x in p.iterdir() if x.is_file()],sep=' | ')
print(summary.to_string(index=False))
