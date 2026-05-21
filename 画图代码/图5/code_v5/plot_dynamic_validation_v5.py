#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data_v5'; FIG=ROOT/'figures_v5'; FIG.mkdir(exist_ok=True)
MATERIAL_ORDER=['Solar-Cells','MLI','OSRs']
SPEEDS=['0.5','0.7','0.9']
GRID_MOTIONS=[['Pitch sweep','Roll sweep','Pitch sine'],['Roll sine','Two-axis sine phase 0','Two-axis sine phase 90'],['Pitch-roll coupled','Diagonal same','Diagonal opposite']]
plt.rcParams.update({
    'font.family':'Times New Roman',
    'mathtext.fontset':'stix',
    'font.size':7.5,
    'axes.labelsize':8,
    'axes.titlesize':8,
    'legend.fontsize':6.6,
    'xtick.labelsize':7,
    'ytick.labelsize':7,
    'axes.linewidth':0.8,
    'xtick.direction':'in',
    'ytick.direction':'in',
    'xtick.top':True,
    'ytick.right':True,
    'pdf.fonttype':42,
    'ps.fonttype':42,
    'svg.fonttype':'none',
    'figure.dpi':300,
    'savefig.dpi':600,
})
COL={'d1':'#0072B2','d2':'#E69F00','d3':'#009E73','d4':'#D55E00','pitch':'#CC79A7','roll':'#56B4E9'}
def safe(s): return s.lower().replace(' ','_').replace('-','_').replace('/','_')
def load(p):
    with p.open('r',encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f); rows=list(r); cols=r.fieldnames
    return {c:np.array([float(row[c]) for row in rows]) for c in cols}
def subcaption(ax, text, y=-0.35):
    ax.text(0.5, y, text, transform=ax.transAxes, ha='center', va='top', fontsize=7)
# Fig. 6: distance, 3x3, captions below.
fig,axes=plt.subplots(3,3,figsize=(7.16,5.75),sharex=False,sharey=True)
letters=list('abcdefghi')
for i,mat in enumerate(MATERIAL_ORDER):
    for j,sp in enumerate(SPEEDS):
        motion=GRID_MOTIONS[i][j]
        p=DATA/(safe(mat)+'__'+safe(motion)+'__v'+sp.replace('.','p')+'.csv')
        a=load(p); t=a['time_s']-a['time_s'][0]; ax=axes[i,j]
        for d in ['d1','d2','d3','d4']:
            ax.plot(t,a[f'{d}_true'],color=COL[d],lw=1.0,label=f'{d} true')
            ax.plot(t,a[f'{d}_pred'],color=COL[d],lw=0.82,ls='--',alpha=0.95,label=f'{d} pred.')
        ax.grid(True,ls=':',lw=0.45,color='0.82')
        if j==0: ax.set_ylabel('Distance (mm)')
        if i==2: ax.set_xlabel('Time (s)')
        subcaption(ax, f'({letters[i*3+j]}) {mat}; {sp} mm/s; {motion}', y=-0.22 if i<2 else -0.30)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles[:8],labels[:8],loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(0.5,0.025),columnspacing=1.0,handlelength=1.8)
fig.tight_layout(rect=[0,0.075,1,0.985],pad=0.45,w_pad=0.30,h_pad=0.95)
fig.savefig(FIG/'fig6_material_speed_9motions_four_distance.png',dpi=600,bbox_inches='tight')
fig.savefig(FIG/'fig6_material_speed_9motions_four_distance.pdf',bbox_inches='tight')
fig.savefig(FIG/'fig6_material_speed_9motions_four_distance.svg',bbox_inches='tight')
fig.savefig(FIG/'fig6_material_speed_9motions_four_distance.tif',dpi=600,bbox_inches='tight')
# Fig. 7: attitude, also 3x3, same cases, captions below.
fig,axes=plt.subplots(3,3,figsize=(7.16,5.75),sharex=False,sharey=True)
for i,mat in enumerate(MATERIAL_ORDER):
    for j,sp in enumerate(SPEEDS):
        motion=GRID_MOTIONS[i][j]
        p=DATA/(safe(mat)+'__'+safe(motion)+'__v'+sp.replace('.','p')+'.csv')
        a=load(p); x=a['d_mean_true']; idx=np.argsort(x)[::-1]; ax=axes[i,j]
        ax.plot(x[idx],a['pitch_true'][idx],color=COL['pitch'],lw=1.0,label='Pitch true')
        ax.plot(x[idx],a['pitch_pred'][idx],color=COL['pitch'],lw=0.82,ls='--',label='Pitch pred.')
        ax.plot(x[idx],a['roll_true'][idx],color=COL['roll'],lw=1.0,label='Roll true')
        ax.plot(x[idx],a['roll_pred'][idx],color=COL['roll'],lw=0.82,ls='--',label='Roll pred.')
        ax.invert_xaxis(); ax.grid(True,ls=':',lw=0.45,color='0.82')
        if j==0: ax.set_ylabel('Angle (deg)')
        if i==2: ax.set_xlabel(r'$d_{mean}$ (mm)')
        subcaption(ax, f'({letters[i*3+j]}) {mat}; {sp} mm/s; {motion}', y=-0.22 if i<2 else -0.30)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(0.5,0.025),columnspacing=1.0,handlelength=1.8)
fig.tight_layout(rect=[0,0.075,1,0.985],pad=0.45,w_pad=0.30,h_pad=0.95)
fig.savefig(FIG/'fig7_material_speed_9motions_attitude_vs_distance.png',dpi=600,bbox_inches='tight')
fig.savefig(FIG/'fig7_material_speed_9motions_attitude_vs_distance.pdf',bbox_inches='tight')
fig.savefig(FIG/'fig7_material_speed_9motions_attitude_vs_distance.svg',bbox_inches='tight')
fig.savefig(FIG/'fig7_material_speed_9motions_attitude_vs_distance.tif',dpi=600,bbox_inches='tight')
print('saved figures to',FIG)
