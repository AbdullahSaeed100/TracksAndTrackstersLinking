
from tqdm import tqdm
from collections import defaultdict
import numpy as np



class EtaPhiTile:
    def __init__(self, eta_bins, phi_bins):
        #when an object is created these two vars decide how to devide the buckets
        self.eta_bins = eta_bins
        self.phi_bins = phi_bins
        self.tile = defaultdict(list)

    def _get_bin(self, eta, phi):
        #it decides the bucket this eta and phi fills into
        eta_bin = np.digitize([eta], self.eta_bins)[0]
        phi_bin = np.digitize([phi], self.phi_bins)[0]
        return (eta_bin, phi_bin)

    def fill(self, eta, phi, index):
        b = self._get_bin(eta, phi)
        self.tile[b].append(index)
        #this tile will be like this 
        # self.tile = {
        #   (9, 73): [0, 5],    # (eta_bucket, phi_bucket) -> list of tracks or tracksters in this bucket
        #    (4, 120): [2]
        #}


class Node:
    def __init__(self, index: int, is_trackster: bool):
        self.index = index
        self.is_trackster = is_trackster
        self.neighbours = []  



#my comment: trks and all_track — every track's and trkst raw properties, all events
#just uses these for the selection cuts and spatial tiling 
#all_valid_track_indices is used to allow only these tracks that have true particle match(checked in download funct) become Node objects at all
#tsCP just to iterate over events
def construct_nodes(trks, all_tracksters, all_valid_track_indices, tsCP):
    all_events_nodes = []
    n=0
    print("starting node construction")
    for ev in tqdm(range(len(tsCP))):
        
        event_nodes = []
        delta = 0.2#tunable 
        min_eta_pos, max_eta_pos = 1.5, 3.2
        min_eta_neg, max_eta_neg = -3.2, -1.5

        # Set up eta-phi tiling
        eta_edges_pos = np.linspace(min_eta_pos, max_eta_pos, 24)
        eta_edges_neg = np.linspace(min_eta_neg, max_eta_neg, 24)
        phi_edges = np.linspace(-np.pi, np.pi, 126)


        tracksterTilePos = EtaPhiTile(eta_edges_pos, phi_edges)
        tracksterTileNeg = EtaPhiTile(eta_edges_neg, phi_edges)


        for idx, (eta, phi) in enumerate(zip(all_tracksters[ev]["barycenter_eta"], all_tracksters[ev]["barycenter_phi"])):
            if eta > 0:
                tracksterTilePos.fill(eta, phi, idx)
            else:
                tracksterTileNeg.fill(eta, phi, idx)
        #tracksterTilePos.tile = {
            #  (eta_bin,phi_bin):[trkstIdx1,trkkstIdx2] 
            #  (9, 63): [0, 1],   # idx 0 and 1 share a bucket
            #  (18, 3):  [3],      # idx 3 is alone in its bucket
        #}
        # Extract track and trackster features for this event
        track_eta   = trks[ev]["track_hgcal_eta"]
        track_phi   = trks[ev]["track_hgcal_phi"]
        track_z     = trks[ev]["track_hgcal_z"]
        track_pt    = trks[ev]["track_pt"]# there are another one called track_hgcal_pt needed to be verified?
        track_hits  = trks[ev]["track_missing_outer_hits"]
        track_qual  = trks[ev]["track_quality"]
        track_id    = trks[ev]["track_id"]
        mtd_x       = trks[ev]["track_pos_mtd/track_pos_mtd.theVector.theX"]
        mtd_y       = trks[ev]["track_pos_mtd/track_pos_mtd.theVector.theY"]
        mtd_z       = trks[ev]["track_pos_mtd/track_pos_mtd.theVector.theZ"]
        trkst_eta   = all_tracksters[ev]['barycenter_eta']
        trkst_phi   = all_tracksters[ev]['barycenter_phi']
        trackster_z = all_tracksters[ev]["barycenter_z"]
        for track_idx, (eta, phi) in enumerate(zip(track_eta, track_phi)):
            # Selection
            #print(" track_idx ", track_idx, " all_valid_track_indices[ev] ", all_valid_track_indices[ev])
            if track_idx not in all_valid_track_indices[ev]:
                continue
            # An exact origin position is an unset MTD placeholder, not a
            # physical measurement. Exclude the track until the upstream
            # MTD-position issue is resolved.
            if mtd_x[track_idx] == 0 and mtd_y[track_idx] == 0 and mtd_z[track_idx] == 0:
                continue
            if track_hits[track_idx] > 4:
                continue
            if track_pt[track_idx] <= 1.0:#
                continue
            if track_qual[track_idx] < 1:
                continue
            #print("track_z[track_idx] ", track_z[track_idx])

            node = Node(index=track_idx, is_trackster=False)

            if eta > 0:
                #clipped (max/min) so it doesn't go past the physical endcap boundary (min_eta_pos=1.5, max_eta_pos=3.2
                eta_min = max(eta - delta, min_eta_pos)
                eta_max = min(eta + delta, max_eta_pos)
                tile = tracksterTilePos
            else:
                eta_min = max(eta - delta, min_eta_neg)#
                eta_max = min(eta + delta, max_eta_neg)
                tile = tracksterTileNeg

            phi_min = phi - delta
            phi_max = phi + delta

            eta_bin_min = np.digitize([eta_min], tile.eta_bins)[0]
            eta_bin_max = np.digitize([eta_max], tile.eta_bins)[0]
            phi_bin_min = np.digitize([phi_min], tile.phi_bins)[0]
            phi_bin_max = np.digitize([phi_max], tile.phi_bins)[0]

            if phi_bin_min > phi_bin_max:
                phi_bin_max += len(tile.phi_bins)

            for eta_i in range(eta_bin_min, eta_bin_max + 1):
                for phi_i in range(phi_bin_min, phi_bin_max + 1):
                    wrapped_phi_i = phi_i % len(tile.phi_bins)
                    bin_key = (eta_i, wrapped_phi_i)
                    for ts_idx in tile.tile.get(bin_key, []):
                        if np.sign(trackster_z[ts_idx]) == np.sign(track_z[track_idx]) and distWrap2(eta, phi, trkst_eta[ts_idx], trkst_phi[ts_idx])<delta**2:
                            node.neighbours.append(ts_idx)
                            #print("trackster_z[ts_idx] ", trackster_z[ts_idx])

            event_nodes.append(node)


        #---------------------------trkst starts from here
        

        #you can uncomment this and change it if you want diffrent values for trkst window
        #delta = 0.1
        #min_eta_pos, max_eta_pos = 1.5, 3.2
        #min_eta_neg, max_eta_neg = -3.2, -1.5


        for trkst_idx, (eta, phi) in enumerate(zip(trkst_eta, trkst_phi)):

            node = Node(index=trkst_idx, is_trackster=True)

            if eta > 0:
                #clipped (max/min) so it doesn't go past the physical endcap boundary (min_eta_pos=1.5, max_eta_pos=3.2
                eta_min = max(eta - delta, min_eta_pos)
                eta_max = min(eta + delta, max_eta_pos)
                tile = tracksterTilePos
            else:
                eta_min = max(eta - delta, min_eta_neg)
                eta_max = min(eta + delta, max_eta_neg)
                tile = tracksterTileNeg

            phi_min = phi - delta
            phi_max = phi + delta

            eta_bin_min = np.digitize([eta_min], tile.eta_bins)[0]
            eta_bin_max = np.digitize([eta_max], tile.eta_bins)[0]
            phi_bin_min = np.digitize([phi_min], tile.phi_bins)[0]
            phi_bin_max = np.digitize([phi_max], tile.phi_bins)[0]

            if phi_bin_min > phi_bin_max:
                phi_bin_max += len(tile.phi_bins)
            #here
            for eta_i in range(eta_bin_min, eta_bin_max + 1):

                for phi_i in range(phi_bin_min, phi_bin_max + 1):

                    wrapped_phi_i = phi_i % len(tile.phi_bins)
                    bin_key = (eta_i, wrapped_phi_i)

                    for other_idx in tile.tile.get(bin_key, []):


                        # self-exclusion + ordering rule, combined in one check
                        if abs(trackster_z[other_idx]) <= abs(trackster_z[trkst_idx]):
                            continue

                        # dR test
                        # deta = eta - trkst_eta[other_idx]
                        # dphi = phi - trkst_phi[other_idx]
                        # dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
                        # if deta**2 + dphi**2 < delta**2:
                        #     node.neighbours.append(other_idx)
                        #alternative  
                        if distWrap2(eta, phi, trkst_eta[other_idx], trkst_phi[other_idx])<delta**2:
                            node.neighbours.append(other_idx)


            event_nodes.append(node)
        #---------------------------


            #print("event_nodes ", event_nodes)
        #this commented code with if down there is for checking first 10 events output
        #n=n+1
        #print(n)

        all_events_nodes.append(event_nodes)
        #if n==10:
        #    return all_events_nodes
    return all_events_nodes

def distWrap2(refEta, refPhi, otherTsEta, otherTsPhi):
    deltaPhi = otherTsPhi - refPhi
    deltaPhi = (deltaPhi + np.pi) % (2 * np.pi) - np.pi
    return ((otherTsEta - refEta) ** 2 + deltaPhi ** 2) 
#my comment: output for event_nodes :
# event_nodes = [
#    Node(index=1, is_trackster=False, neighbours=[5, 12]),
#   Node(index=2, is_trackster=False, neighbours=[]),
#    Node(index=4, is_trackster=False, neighbours=[5]),
#]
# output for all_events_nodes:
# all_events_nodes = [
# event_nodes_for_event_0,   
#  event_nodes_for_event_1,   
#   event_nodes_for_event_2,
#]
#or
#all_events_nodes = [
#    [ Node(1, False, [5, 12]), Node(2, False, []), Node(4, False, [5]) ],   # event 0
#    [ Node(0, False, [3]), Node(7, False, [3, 9, 14]) ],                     # event 1
#]


