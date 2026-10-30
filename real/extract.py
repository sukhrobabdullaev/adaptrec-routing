# -*- coding: utf-8 -*-
"""Amazon Reviews 2023 (Video_Games) -> ixcham so'rov oqimi."""
import gzip, json, csv, time
T0=time.time()
def log(*a): print("[%6.1fs]"%(time.time()-T0), *a, flush=True)

n=bad=0
with gzip.open("Video_Games.jsonl.gz","rt",encoding="utf-8") as f, \
     open("vg_events_raw.csv","w",newline="",encoding="utf-8") as o:
    w=csv.writer(o); w.writerow(["user_id","item","rating","ts","verified","helpful"])
    for line in f:
        try:
            d=json.loads(line)
            w.writerow([d["user_id"],d["parent_asin"],d["rating"],d["timestamp"],
                        int(bool(d.get("verified_purchase"))),d.get("helpful_vote",0)])
            n+=1
        except Exception: bad+=1
log("xom hodisalar:",n,"buzuq:",bad)

m=bad2=0
with gzip.open("meta_Video_Games.jsonl.gz","rt",encoding="utf-8") as f, \
     open("vg_items.csv","w",newline="",encoding="utf-8") as o:
    w=csv.writer(o)
    w.writerow(["item","title","main_category","store","categories","price","avg_rating","rating_number"])
    for line in f:
        try:
            d=json.loads(line)
            w.writerow([d.get("parent_asin"),(d.get("title") or "").replace("\n"," ")[:300],
                        d.get("main_category") or "",(d.get("store") or "")[:120],
                        "|".join(str(c) for c in (d.get("categories") or []))[:300],
                        d.get("price") if d.get("price") is not None else "",
                        d.get("average_rating") or "", d.get("rating_number") or ""])
            m+=1
        except Exception: bad2+=1
log("itemlar:",m,"buzuq:",bad2)
