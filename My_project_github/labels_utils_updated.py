

import numpy as np



import awkward as ak
from tqdm import tqdm

def assign_trackster_labels(associations, all_tracksters):
    labels = []
    scores = []
    shared_energies = []

    for ev_idx in tqdm(range(len(associations))):
        
        
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



            for i, matched_reco in enumerate(simRecoMatches):


                trackster_energy = trackster_energies[matched_reco]

                #checking energy purity. 
                if simRecoScores[i] < 0.99 and (simRecoSharedE[i] / trackster_energy) > 0.4:
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





