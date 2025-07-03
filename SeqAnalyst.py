import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import re
from itertools import combinations
from collections import Counter

from Bio import SeqIO
from Bio.SeqUtils import molecular_weight, gc_fraction
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Data import CodonTable
from Bio.Restriction import Analysis, AllEnzymes
from Bio.Align import PairwiseAligner
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

KD_SCALE = {
    'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,
    'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,
    'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,
    'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3
}

def GC(seq):
    return 100.0 * gc_fraction(seq, ambiguous="ignore")

def parse_sequences(files, text):
    seqs = []
    if files:
        for f in files:
            handle = StringIO(f.getvalue().decode("utf-8"))
            seqs.extend(SeqIO.parse(handle, "fasta"))
    elif text:
        handle = StringIO(text)
        seqs.extend(SeqIO.parse(handle, "fasta"))
    return seqs

def get_kmer_freqs(seqs, k=4):
    df = []
    for rec in seqs:
        s = str(rec.seq).upper()
        counts = Counter(s[i:i+k] for i in range(len(s)-k+1) if "N" not in s[i:i+k])
        total = sum(counts.values()) or 1
        df.append({kmer:cnt/total for kmer,cnt in counts.items()})
    return pd.DataFrame(df).fillna(0).to_numpy()

def get_aa_comp_vectors(seqs):
    df = []
    for rec in seqs:
        s = str(rec.seq).replace("*","")
        try:
            pa = ProteinAnalysis(s)
            df.append(pa.get_amino_acids_percent())
        except:
            df.append({aa:0.0 for aa in KD_SCALE})
    return pd.DataFrame(df).fillna(0).to_numpy()

def sliding_hydro(seq, window=9):
    vals = [KD_SCALE.get(aa,0) for aa in seq]
    half = window//2
    out = []
    for i in range(len(vals)):
        st = max(0,i-half); en = min(len(vals),i+half+1)
        out.append(sum(vals[st:en])/(en-st))
    return out

def analyze_single_gene(rec):
    seq = rec.seq; s = str(seq)
    stats = {
        "ID":rec.id,"Length (bp)":len(seq),
        "GC (%)":round(GC(seq),2),
        "MW (DNA)":round(molecular_weight(seq,seq_type="DNA"),2)
    }
    df = pd.DataFrame([stats])
    aligner = PairwiseAligner(); aligner.mode="local"
    score = aligner.score(s,s)
    rc = str(seq.reverse_complement())

    frames={}
    for i in range(3):
        f = seq[i:].translate(to_stop=True)
        r = seq.reverse_complement()[i:].translate(to_stop=True)
        frames[f"F{i+1}"]=len(f); frames[f"F-{i+1}"]=len(r)
    fig1,ax1=plt.subplots(); ax1.bar(frames.keys(),frames.values())
    ax1.set_title("6‑Frame Lengths"); plt.tight_layout()

    table=CodonTable.unambiguous_dna_by_name["Standard"]
    orfs=[]
    for strand,nuc in [(1,seq),(-1,seq.reverse_complement())]:
        for frame in range(3):
            for prot in nuc[frame:].translate(table).split("*"):
                if "M" in prot and len(prot)>50:
                    start=frame+prot.find("M")*3
                    end=start+len(prot)*3
                    orfs.append({"Strand":strand,"Frame":frame+1,
                                 "Start":start,"End":end,"AA length":len(prot)})
    orf_df=pd.DataFrame(orfs)

    rmap=Analysis(AllEnzymes, seq, linear=True).full()
    rmap_df=pd.DataFrame([{"Enzyme":e,"Sites":",".join(map(str,sites))} for e,sites in rmap.items()])

    return df,{"RC":rc,"Score":score},orf_df,rmap_df,{"frames":fig1}

def analyze_single_protein(rec):
    s = str(rec.seq).replace("*","")
    pa=ProteinAnalysis(s)
    stats={
        "ID":rec.id,"Length":len(s),
        "MW (Da)":round(pa.molecular_weight(),2),
        "pI":round(pa.isoelectric_point(),2),
        "Aromaticity":round(pa.aromaticity(),2),
        "Instability":round(pa.instability_index(),2),
        "Gravy":round(pa.gravy(),2)
    }
    df=pd.DataFrame([stats])
    comp=pa.get_amino_acids_percent()
    fig2,ax2=plt.subplots(); ax2.bar(comp.keys(),comp.values())
    ax2.set_title("AA Composition"); plt.tight_layout()
    hyd=sliding_hydro(s)
    fig3,ax3=plt.subplots(); ax3.plot(hyd); ax3.set_title("Hydropathy"); plt.tight_layout()
    aligner=PairwiseAligner(); aligner.mode="local"
    score=aligner.score(s,s)
    patterns={
        "N‑gly":"N[^P][ST][^P]","PKC":"[ST][^P]K[RK]",
        "CK2":"[ST][^P]{2}[DE]","Ploop":"[AG].{4}GK[ST]"
    }
    doms=[]
    for name,pat in patterns.items():
        for m in re.finditer(pat,s):
            doms.append({"Domain":name,"Start":m.start(),"End":m.end(),"Motif":m.group()})
    dom_df=pd.DataFrame(doms)
    return df,{"Score":score},pd.DataFrame([comp]),dom_df,{"aac":fig2,"hyd":fig3}

def analyze_multiple(seqs,typ):
    ids = [r.id for r in seqs]
    n = len(seqs)
    
    base_font = 10
    min_font = 4
    font_size = max(min_font, base_font - n//10)
    heatmap_size = max(6, min(12, n*0.4))
    dendro_height = max(4, min(10, n*0.3))
    bar_width = max(6, min(12, n*0.5))

    dist=np.zeros((n,n))
    aligner=PairwiseAligner(); aligner.mode="global"
    for i,j in combinations(range(n),2):
        a,b=str(seqs[i].seq),str(seqs[j].seq)
        sc=aligner.score(a,b); dist[i,j]=dist[j,i]=sc
    M=dist.max() or 1
    dm=pd.DataFrame(M-dist,index=ids,columns=ids)
    plots={}
    
    if n>2:
        lm=linkage((M-dist)[np.triu_indices(n,1)],"average")
        fig4,ax4=plt.subplots(figsize=(6,dendro_height))
        dendrogram(lm,labels=ids,orientation="right",ax=ax4,leaf_font_size=font_size)
        plt.tight_layout(); plots["dendrogram"]=fig4
    
    fig5,ax5=plt.subplots(figsize=(heatmap_size,heatmap_size))
    cax=ax5.matshow(dm,cmap="viridis_r"); fig5.colorbar(cax)
    ax5.set_xticks(range(n)); ax5.set_xticklabels(ids,rotation=90,fontsize=font_size)
    ax5.set_yticks(range(n)); ax5.set_yticklabels(ids,fontsize=font_size)
    ax5.set_title("Distance",fontsize=font_size+2); plt.tight_layout(); plots["heatmap"]=fig5

    if typ.startswith("Gene"):
        feats=get_kmer_freqs(seqs)
        stats={"ID":ids,"Length(bp)":[len(r.seq) for r in seqs],"GC%":[round(GC(r.seq),2) for r in seqs]}
        fig6,ax6=plt.subplots(figsize=(bar_width,6))
        ax6.bar(ids,stats["GC%"])
        ax6.set_title("GC%",fontsize=font_size+2)
        ax6.tick_params(axis='x',rotation=90,labelsize=font_size)
        ax6.tick_params(axis='y',labelsize=font_size)
        plt.tight_layout(); plots["gc"]=fig6
    else:
        feats=get_aa_comp_vectors(seqs)
        stats={"ID":ids,"Length":[len(r.seq) for r in seqs]}
    stats_df=pd.DataFrame(stats)

    sf=StandardScaler().fit_transform(feats)
    if sf.shape[1]>=2:
        pr=PCA(2).fit_transform(sf)
        fig7,ax7=plt.subplots(figsize=(8,6))
        ax7.scatter(pr[:,0],pr[:,1])
        for i,t in enumerate(ids): ax7.annotate(t,(pr[i,0],pr[i,1]),fontsize=font_size,alpha=0.7 if n>20 else 1.0)
        ax7.set_title("PCA",fontsize=font_size+2)
        ax7.tick_params(labelsize=font_size)
        plt.tight_layout(); plots["pca"]=fig7
    if n>2 and sf.shape[0]>n-1:
        emb=umap.UMAP(n_neighbors=min(n-1,5),min_dist=0.1,random_state=42).fit_transform(sf)
        fig8,ax8=plt.subplots(figsize=(8,6))
        ax8.scatter(emb[:,0],emb[:,1])
        for i,t in enumerate(ids): ax8.annotate(t,(emb[i,0],emb[i,1]),fontsize=font_size,alpha=0.7 if n>20 else 1.0)
        ax8.set_title("UMAP",fontsize=font_size+2)
        ax8.tick_params(labelsize=font_size)
        plt.tight_layout(); plots["umap"]=fig8

    max_len = max(len(r.seq) for r in seqs)
    aligned = []
    for r in seqs:
        s = str(r.seq)
        aligned.append(list(s + "-"*(max_len - len(s))))
    cons = "".join(Counter(col).most_common(1)[0][0] for col in zip(*aligned))
    return stats_df, dm, cons, plots

def main():
    st.set_page_config(page_title="Bioinformatics Suite",layout="wide")
    plt.style.use("seaborn-v0_8-paper")

    st.title("Interactive Bioinformatics Analysis")
    method = st.sidebar.radio("Input Method",["Upload FASTA","Paste FASTA"])
    seqs=[]
    if method=="Upload FASTA":
        files=st.sidebar.file_uploader("FASTA files",type=["fa","fasta","fna","faa"],accept_multiple_files=True)
        if files: seqs=parse_sequences(files,None)
    else:
        txt=st.sidebar.text_area("Paste FASTA",height=200)
        if txt: seqs=parse_sequences(None,txt)

    if not seqs:
        st.info("Provide FASTA sequences."); return

    seq_type=st.sidebar.radio("Sequence Type",["Gene (DNA/RNA)","Protein"])
    if len(seqs)==1:
        rec=seqs[0]
        if seq_type.startswith("Gene"):
            df,align,orf_df,rmap_df,pls=analyze_single_gene(rec)
            t1,t2,t3,t4=st.tabs(["Summary","Plots","Details","Download"])
            with t1: st.dataframe(df); st.subheader("ORFs"); st.dataframe(orf_df)
            with t2: st.pyplot(pls["frames"])
            with t3:
                st.write("RC:",align["RC"]); st.write("Score:",align["Score"])
                st.subheader("Restriction map"); st.dataframe(rmap_df)
            with t4: st.download_button("Download CSV",df.to_csv(index=False).encode(),"stats.csv")
        else:
            df,align,aa_df,dom_df,pls=analyze_single_protein(rec)
            t1,t2,t3,t4=st.tabs(["Summary","Plots","Details","Download"])
            with t1: st.dataframe(df); st.subheader("AA%"); st.dataframe(aa_df); st.subheader("Domains"); st.dataframe(dom_df)
            with t2: st.pyplot(pls["aac"]); st.pyplot(pls["hyd"])
            with t3: st.write("Score:",align["Score"])
            with t4: st.download_button("Download CSV",df.to_csv(index=False).encode(),"stats.csv")
    else:
        stats_df,dm,cons,pls=analyze_multiple(seqs,seq_type)
        t1,t2,t3=st.tabs(["Stats","Plots","Consensus"])
        with t1: st.subheader("Stats"); st.dataframe(stats_df); st.subheader("Distance"); st.dataframe(dm)
        with t2:
            for fig in pls.values(): st.pyplot(fig)
        with t3: st.code(cons)

if __name__=="__main__":
    main()
