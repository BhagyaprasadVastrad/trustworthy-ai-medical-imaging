import argparse,json
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,brier_score_loss,roc_auc_score
from .metrics import bootstrap_auroc_ci,bootstrap_error_rate_ci,predictive_entropy
from .project import RESEARCH_DATA_DIR

def load(seeds):
    data=[np.load(RESEARCH_DATA_DIR/f"stage2_seed_{s}.npz",allow_pickle=True) for s in seeds]
    uids=data[0]["study_uids"].astype(str); labels=data[0]["labels"].astype(int); probs=[]
    for s,d in zip(seeds,data):
        du=d["study_uids"].astype(str); dl=d["labels"].astype(int)
        if not np.array_equal(uids,du): raise ValueError(f"StudyInstanceUID order mismatch for seed {s}")
        if not np.array_equal(labels,dl): raise ValueError(f"Label mismatch for seed {s}")
        probs.append(d["probabilities"].astype(float))
    return uids,labels,np.vstack(probs)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44,45,46]); a=p.parse_args()
    uids,labels,members=load(a.seeds); ensemble=members.mean(0); pred=(ensemble>=.5).astype(int); error=(pred!=labels).astype(int)
    disagreement=members.std(0); maxprob_unc=1-np.maximum(ensemble,1-ensemble); entropy=predictive_entropy(ensemble)
    single=members[0]
    out={"single_seed":a.seeds[0],"single_accuracy":float(accuracy_score(labels,single>=.5)),"single_auroc":float(roc_auc_score(labels,single)),"single_brier":float(brier_score_loss(labels,single)),"ensemble_accuracy":float(accuracy_score(labels,pred)),"ensemble_auroc":float(roc_auc_score(labels,ensemble)),"ensemble_brier":float(brier_score_loss(labels,ensemble)),"disagreement_error_auroc":float(roc_auc_score(error,disagreement)),"max_probability_uncertainty_error_auroc":float(roc_auc_score(error,maxprob_unc)),"entropy_error_auroc":float(roc_auc_score(error,entropy)),"single_auroc_ci":bootstrap_auroc_ci(labels,single),"ensemble_auroc_ci":bootstrap_auroc_ci(labels,ensemble),"disagreement_error_auroc_ci":bootstrap_auroc_ci(error,disagreement),"max_probability_error_auroc_ci":bootstrap_auroc_ci(error,maxprob_unc),"entropy_error_auroc_ci":bootstrap_auroc_ci(error,entropy),"overall_error_rate_ci":bootstrap_error_rate_ci(error)}
    top=disagreement>=np.percentile(disagreement,90); out["top10_error_rate_ci"]=bootstrap_error_rate_ci(error[top]); out["top10_count"]=int(top.sum())
    pd.DataFrame({"StudyInstanceUID":uids,"label":labels,"ensemble_probability":ensemble,"disagreement_std":disagreement,"max_probability_uncertainty":maxprob_unc,"entropy":entropy,"error":error}).to_csv(RESEARCH_DATA_DIR/"stage5_aligned_results.csv",index=False)
    with open(RESEARCH_DATA_DIR/"stage5_aligned_results.json","w",encoding="utf-8") as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()