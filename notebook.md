Numerical ML features: (Train DT + IF)
id.orig_p
id.resp_p 
duration 
orig_bytes 
resp_bytes
missed_bytes 
orig_pkts 
orig_ip_bytes 
resp_pkts 
resp_ip_bytes

Categorical ML features: (Encode then train DT +TF):
proto 
service
conn_state 
history

Target: (Benign / Malicious)
label (benign signed as 0 and 1 for malicious)

CTI/ metadata retained: (Sprint 2 / traceability)
ts
uid
id.orig_h
id.resp_h
source_file 
capture_date

Detailed ground truth: (Analysis only)
detailedlabel

Excluded: (Missing data)
local_orig
local_resp 
tunnel_parents