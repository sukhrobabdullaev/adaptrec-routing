# -*- coding: utf-8 -*-
"""5-core filtr va vaqt bo'yicha tartiblangan so'rov oqimini qurish."""
import csv, time, collections, datetime, json
T0=time.time()
def log(*a): print("[%5.1fs]"%(time.time()-T0), *a, flush=True)

rows=[]
with open("vg_events_raw.csv", encoding="utf-8") as f:
    r=csv.reader(f); next(r)
    for u,i,rt,ts,v,h in r:
        rows.append((u,i,float(rt),int(ts)))
log("yuklandi:", len(rows))

# iterativ 5-core
K=5
while True:
    cu=collections.Counter(x[0] for x in rows)
    ci=collections.Counter(x[1] for x in rows)
    keep=[x for x in rows if cu[x[0]]>=K and ci[x[1]]>=K]
    if len(keep)==len(rows): break
    rows=keep
    log("5-core iteratsiyasi ->", len(rows))

rows.sort(key=lambda x: x[3])
users={u for u,_,_,_ in rows}; items={i for _,i,_,_ in rows}
t0,t1=rows[0][3], rows[-1][3]
def d(ms): return datetime.datetime.utcfromtimestamp(ms/1000).strftime("%Y-%m-%d")

with open("vg_stream.csv","w",newline="",encoding="utf-8") as o:
    w=csv.writer(o); w.writerow(["user","item","rating","ts"])
    w.writerows(rows)

stat={"hodisalar":len(rows),"foydalanuvchilar":len(users),"itemlar":len(items),
      "zichlik":round(len(rows)/(len(users)*len(items))*100,6),
      "boshlanish":d(t0),"tugash":d(t1),
      "hodisa_per_user":round(len(rows)/len(users),2),
      "hodisa_per_item":round(len(rows)/len(items),2)}
json.dump(stat, open("stream_stats.json","w"), indent=1)
log("YAKUNIY OQIM:", json.dumps(stat, ensure_ascii=False))

# yillar bo'yicha taqsimot (drift uchun muhim)
yil=collections.Counter(datetime.datetime.utcfromtimestamp(t/1000).year for _,_,_,t in rows)
for y in sorted(yil): print("  %d: %8d" % (y, yil[y]))
