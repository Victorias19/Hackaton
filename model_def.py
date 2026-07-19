import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.model_selection import GroupKFold

DAILY = ["temp_rel","nightly_temperature","wrist_temp_diff","resting_hr",
         "hrv_rmssd","resp_rate","minutesasleep","efficiency","steps_total",
         "active_minutes","calories_total"]
HISTORY = ["prior_cycle_length_mean","prior_cycle_length_sd","n_prior_cycles"]

class CyclePredictor:
    def __init__(self, event="menses"):
        self.event=event
        self.end="next_onset_day" if event=="menses" else "ovulation_day"
        self.rsf=RandomSurvivalForest(n_estimators=300,min_samples_leaf=8,
                    max_depth=5,max_features="sqrt",random_state=42,n_jobs=-1)
    def _feat(self, cd, hist, t):
        s0=cd[cd["day"]<=t]; row={"day_now":float(t)}
        for c in DAILY:
            s=s0[c].dropna() if c in s0 else pd.Series(dtype=float)
            if len(s):
                row[c+"_mean"]=s.mean(); row[c+"_last"]=s.iloc[-1]
                row[c+"_slope"]=(s.iloc[-1]-s.iloc[0])/max(t,1)
            else: row[c+"_mean"]=row[c+"_last"]=row[c+"_slope"]=np.nan
        for h in HISTORY: row[h]=hist.get(h,np.nan)
        return row
    def fit(self, cs, pn, cf):
        H=cf.set_index(["id","cycle_index"]).to_dict("index")
        rows,tt,ev,gr=[],[],[],[]
        for _,c in cs.iterrows():
            o,e=c["onset_day"],c[self.end]
            if pd.isna(e) or e<=o: continue
            tot=int(e-o); h=H.get((c["id"],c.get("cycle_index")),{})
            d=pn[(pn["id"]==c["id"])&(pn["day_in_study"]>=o)&(pn["day_in_study"]<e)].copy()
            if not len(d): continue
            d["day"]=d["day_in_study"]-o
            for t in range(2,tot,2):
                rows.append(self._feat(d,h,t)); tt.append(tot-t); ev.append(True); gr.append(c["id"])
        X=pd.DataFrame(rows); self.features_=X.columns.tolist(); self.fill_=X.median(); X=X.fillna(self.fill_)
        y=Surv.from_arrays(event=np.array(ev),time=np.array(tt,float))
        self._X,self._y,self._g=X,y,np.array(gr)
        self.rsf.fit(X,y); self.train_c_=self.rsf.score(X,y); return self
    def cv(self,n=5):
        sc=[]
        for tr,te in GroupKFold(n).split(self._X,self._y,self._g):
            m=RandomSurvivalForest(n_estimators=300,min_samples_leaf=8,max_depth=5,
                max_features="sqrt",random_state=42,n_jobs=-1).fit(self._X.iloc[tr],self._y[tr])
            sc.append(m.score(self._X.iloc[te],self._y[te]))
        return np.array(sc)
    def predict_day(self, cd, hist, t):
        X=pd.DataFrame([self._feat(cd,hist,t)]).reindex(columns=self.features_).fillna(self.fill_)
        fn=self.rsf.predict_survival_function(X)[0]
        p=-np.diff(np.r_[1.0,fn.y])
        return pd.DataFrame({"day":t+fn.x,"p_event_day":np.round(p,4),"surv":np.round(fn.y,4)})
