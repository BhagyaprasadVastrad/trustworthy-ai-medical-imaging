import argparse, json, random
from pathlib import Path
import numpy as np, pandas as pd, pydicom, torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .models import SmallCNN
from .metrics import binary_metrics, bootstrap_auroc_ci, bootstrap_error_rate_ci, confidence
from .project import MODEL_DIR, RSNA_ANNOTATION_PATH, RSNA_DICOM_ROOT, RESEARCH_DATA_DIR, ensure_output_dirs

SEEDS=[42,43,44,45,46]

class RSNADataset(Dataset):
    def __init__(self, df):
        self.df=df
        self.transform=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; ds=pydicom.dcmread(r.dicom_path)
        return self.transform(Image.fromarray(ds.pixel_array,mode="L")),int(r.label),r.StudyInstanceUID

def build_labels():
    with open(RSNA_ANNOTATION_PATH,"r",encoding="utf-8") as f: data=json.load(f)
    lm={x["id"]:x["name"] for x in data["labelGroups"][0]["labels"]}
    grouped={}
    for a in data["datasets"][0]["annotations"]: grouped.setdefault(a["StudyInstanceUID"],set()).add(lm[a["labelId"]])
    return pd.DataFrame([(u,1 if "Lung Opacity" in s else 0) for u,s in grouped.items() if "Lung Opacity" in s or "No Lung Opacity" in s],columns=["StudyInstanceUID","label"])

def build_index(labels):
    lookup=dict(zip(labels.StudyInstanceUID,labels.label)); rows=[]
    for path in sorted(RSNA_DICOM_ROOT.rglob("*.dcm")):
        ds=pydicom.dcmread(path,stop_before_pixels=True); uid=getattr(ds,"StudyInstanceUID",None)
        if uid in lookup: rows.append({"dicom_path":str(path),"StudyInstanceUID":uid,"label":int(lookup[uid])})
    return pd.DataFrame(rows).reset_index(drop=True)

def infer(seed,df):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=SmallCNN().to(device)
    ck=MODEL_DIR/f"smallcnn_pneumoniamnist_seed_{seed}.pt"
    model.load_state_dict(torch.load(ck,map_location=device)); model.eval()
    loader=DataLoader(RSNADataset(df),batch_size=128,shuffle=False,num_workers=0)
    rows=[]
    with torch.no_grad():
        for images,labels,uids in loader:
            logits=model(images.to(device)); probs=torch.sigmoid(logits).cpu().numpy(); raw=logits.cpu().numpy()
            for uid,label,z,p in zip(uids,labels.numpy(),raw,probs): rows.append((uid,int(label),float(z),float(p)))
    out=pd.DataFrame(rows,columns=["StudyInstanceUID","label","logit","probability"]).sort_values("StudyInstanceUID").reset_index(drop=True)
    np.savez(RESEARCH_DATA_DIR/f"stage2_seed_{seed}.npz",study_uids=out.StudyInstanceUID.to_numpy(),labels=out.label.to_numpy(),logits=out.logit.to_numpy(),probabilities=out.probability.to_numpy())
    return out

def stage3_4(seed,data):
    labels=data["labels"]; probs=data["probabilities"]; logits=data["logits"]
    idx=np.arange(len(labels)); cal,ev=train_test_split(idx,test_size=.5,stratify=labels,random_state=42)
    x=torch.tensor(logits[cal],dtype=torch.float32); y=torch.tensor(labels[cal],dtype=torch.float32)
    lt=torch.nn.Parameter(torch.zeros(1)); opt=torch.optim.LBFGS([lt],lr=.01,max_iter=100); loss_fn=torch.nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad(); loss=loss_fn(x/torch.exp(lt),y); loss.backward(); return loss
    opt.step(closure); T=float(torch.exp(lt).item())
    scaled=1/(1+np.exp(-logits/T)); before=binary_metrics(labels[ev],probs[ev]); after=binary_metrics(labels[ev],scaled[ev])
    pred=(probs>=.5).astype(int); error=(pred!=labels).astype(int); conf=confidence(probs)
    return {"seed":seed,"temperature":T,"stage2":binary_metrics(labels,probs),"stage3_before":before,"stage3_after":after,"stage4_accuracy":float(1-error.mean()),"stage4_error_rate_ci":bootstrap_error_rate_ci(error),"stage4_confidence_error_auroc_ci":bootstrap_auroc_ci(error,conf)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--seeds",nargs="+",type=int,default=SEEDS); p.add_argument("--skip-inference",action="store_true"); a=p.parse_args()
    ensure_output_dirs(); labels=build_labels(); df=build_index(labels); results=[]
    for seed in a.seeds:
        path=RESEARCH_DATA_DIR/f"stage2_seed_{seed}.npz"
        if not (a.skip_inference and path.exists()): infer(seed,df)
        d=np.load(path,allow_pickle=True)
        if not (len(d["study_uids"]) == len(d["labels"]) == len(d["logits"]) == len(d["probabilities"])): raise ValueError("Prediction arrays differ in length")
        uids=d["study_uids"].astype(str); order=np.argsort(uids); d2={"labels":d["labels"][order].astype(int),"logits":d["logits"][order].astype(float),"probabilities":d["probabilities"][order].astype(float)}
        results.append(stage3_4(seed,d2))
    with open(RESEARCH_DATA_DIR/"multiseed_stage2_to_4.json","w",encoding="utf-8") as f: json.dump(results,f,indent=2)
    rows=[]
    for r in results:
        row={"seed":r["seed"],"temperature":r["temperature"],"stage2_accuracy":r["stage2"]["accuracy"],"stage2_auroc":r["stage2"]["auroc"],"stage2_brier":r["stage2"]["brier"],"stage2_ece":r["stage2"]["ece"],"stage3_before_brier":r["stage3_before"]["brier"],"stage3_after_brier":r["stage3_after"]["brier"],"stage3_before_auroc":r["stage3_before"]["auroc"],"stage3_after_auroc":r["stage3_after"]["auroc"],"stage4_accuracy":r["stage4_accuracy"]}
        rows.append(row)
    table=pd.DataFrame(rows)
    summary={col:{"mean":float(table[col].mean()),"std":float(table[col].std(ddof=1))} for col in table.columns if col!="seed"}
    table.to_csv(RESEARCH_DATA_DIR/"multiseed_stage2_to_4_by_seed.csv",index=False)
    with open(RESEARCH_DATA_DIR/"multiseed_stage2_to_4_summary.json","w",encoding="utf-8") as f: json.dump(summary,f,indent=2)
    print(table.to_string(index=False))
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()