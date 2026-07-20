#!/usr/bin/env python
# coding: utf-8

# In[4]:


import uproot
import numpy as np
file = uproot.open("/eos/user/a/aasiri/TracksAndTrackstersLinking/histoSinglePi.root")


alltracksters = load_branch_with_highest_cycle(file, 'ticlDumper/ticlTrackstersCLUE3DHigh')
allassociations = load_branch_with_highest_cycle(file, 'ticlDumper/associations')

tsKeys = ['time', 'timeError', 'regressed_energy', 'raw_energy', 'raw_em_energy', 'raw_pt', 'raw_em_pt', 'barycenter_x', 'barycenter_y', 'barycenter_z', 'barycenter_eta', 'barycenter_phi', 'id_probabilities', 'vertices_x', 'vertices_y', 'vertices_z', 'vertices_energy']


assKeys = ['ticlTracksterLinks_simToReco_CP', 'ticlTracksterLinks_simToReco_CP_score', 'ticlTracksterLinks_simToReco_CP_sharedE', 'ticlTracksterLinks_simToReco_SC', 'ticlTracksterLinks_simToReco_SC_score', 'ticlTracksterLinks_simToReco_SC_sharedE', 'ticlCandidate_simToReco_SC', 'ticlCandidate_simToReco_SC_score', 'ticlCandidate_simToReco_SC_sharedE', 'ticlCandidate_simToReco_CP', 'ticlCandidate_simToReco_CP_score', 'ticlCandidate_simToReco_CP_sharedE']


all_tracksters = alltracksters.arrays(tsKeys)
associations = allassociations.arrays(assKeys)


# In[5]:


len(all_tracksters)==len(associations)


# In[3]:


def load_branch_with_highest_cycle(file, branch_name):
    all_keys = file.keys()
    matching_keys = [key for key in all_keys if key.startswith(branch_name)]
    if not matching_keys:
        raise ValueError(f"No branch with name '{branch_name}' found in the file.")
    highest_cycle_key = max(matching_keys, key=lambda key: int(key.split(";")[1]))
    return file[highest_cycle_key]


# In[23]:


# label_utils.py

import awkward as ak

def assign_trackster_labels(associations, all_tracksters):
    labels = []
    scores = []
    shared_energies = []

    for ev_idx in range(len(associations)):
        if(ev_idx==10):
            return {
                #my comment: ak array used here because we have variable number of ts in each event
                "y": ak.Array(labels),
                "score": ak.Array(scores),
                "shared_e": ak.Array(shared_energies)
            }
        print(ev_idx)
        assEv = associations[ev_idx]

        #my comment:
        # simToReco: indexed by sim_ts -> list of matching resco_ts (0/1/many) + their scores/sharedE
        simToRecoEv = assEv.ticlTracksterLinks_simToReco_CP
        simToRecoScores = assEv.ticlTracksterLinks_simToReco_CP_score
        simToRecoSharedE = assEv.ticlTracksterLinks_simToReco_CP_sharedE

        #mycomment: all the energies of the reco_ts
        trackster_energies = all_tracksters.raw_energy[ev_idx]

        #mycomment: these will be filled like this: one entry per each reco_ts
        n_ts = len(trackster_energies)
        reco_labels  = [[] for _ in range(n_ts)]   # list-of-lists in case we have more than one reco corresponding to the same sim
        reco_scores  = [[] for _ in range(n_ts)]
        reco_sharedE = [[] for _ in range(n_ts)]

        for sim_idx in range(len(simToRecoEv)):


            # my comment: list of the reco_ts that matches this exact sim_idx
            simRecoMatches = simToRecoEv[sim_idx]
            simRecoScores = simToRecoScores[sim_idx]
            simRecoSharedE = simToRecoSharedE[sim_idx]



            print('sim_idx=',sim_idx,'is in',end=" ")
            for i, matched_reco in enumerate(simRecoMatches):


                trackster_energy = trackster_energies[matched_reco]

                #checking energy purity. 
                if simRecoScores[i] < 0.99 and (simRecoSharedE[i] / trackster_energy) > 0.4:
                    print(matched_reco, end=" ")
                    reco_labels[matched_reco].append(sim_idx)
                    reco_scores[matched_reco].append(simRecoScores[i])
                    reco_sharedE[matched_reco].append(simRecoSharedE[i])





        labels.append(reco_labels)
        scores.append(reco_scores)
        shared_energies.append(reco_sharedE)

    return {
        #my comment: ak array used here because we have variable number of ts in each event
        "y": ak.Array(labels),
        "score": ak.Array(scores),
        "shared_e": ak.Array(shared_energies)
    }


# ## output structure :
# {
#     "y":        ak.Array,   # shape: events → reco_tracksters → list of matched simtracksters (or [] if none)
#     "score":    ak.Array,   # same shape, parallel to "y" — each score aligned to its sim_idx
#     "shared_e": ak.Array,   # same shape, parallel to "y" — each shared-energy value aligned to its sim_idx
# }
# 
# Three levels of nesting for each field:  **result[" y "][event][reco_idx]** → a list of matched simtracksters, e.g. [2, 6, 7] or [].
# 

# In[24]:


res = assign_trackster_labels(associations, all_tracksters)


# In[28]:


for i,recos in enumerate(res['y']):
    print('ev:',i)
    for reco in recos:
        print(reco)



# In[ ]:




