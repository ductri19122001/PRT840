*local_orig, local_resp, tunnel_parents (remove by 100% missing value)

*detailedlabel (contains data of RemcosRAT, Malware_data_exfiltration, PyExfil-DNSRemcosRAT, Malware_data_exfiltration, PyExfil-DNS causing leakage)

*id.orig_h, id.resp_h is IP If you feed it directly into the baseline model, the model might learn "which IPs are malicious" instead of learning network behavior. For our project, we need to retain the IPs for Sprint 2's CTI enrichment.

*uid: is just a Zeek connection ID, essentially a technical identifier, with no behavioral significance for training purposes.

*source_file: only indicates which file the record comes from, not network characteristics.

*capture_date: primarily for provenance/distribution verification by date/time split; should not be used as a baseline feature because there is a risk of the model learning "which day had an attack".

*label == "Unknown"

Therefore, we are going  14 candidate features + label + flow_row_id. However, flow_row_id is for tracking only, not a model feature.
