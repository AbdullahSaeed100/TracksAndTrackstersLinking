import numpy as np
import awkward as ak
import torch
from torch_geometric.data import Data

# Speed of light in cm/ns (same constant as prepare_data.py / d1)
C_CM_PER_NS = 29.9792458

# MTD position branch names as loaded into data_trk in ClusterDataset download()
# (loaded there via `self.feature_trk + mtd_pos_keys`, NOT part of feature_trk itself,
#  so they never become node-table columns — they are TOF ingredients only)
MTD_X_KEY = "track_pos_mtd/track_pos_mtd.theVector.theX"
MTD_Y_KEY = "track_pos_mtd/track_pos_mtd.theVector.theY"
MTD_Z_KEY = "track_pos_mtd/track_pos_mtd.theVector.theZ"


# ---------------------------------------------------------------------------
# Unified edge feature schema (Decision C4: one shared table + edge_type col).
#
# Every column has the SAME physical meaning for both edge types, so no
# zero-padding of type-specific blocks is needed. The 7+7 "individual
# trackster properties" from d1's compute_ts_ts_edge_features are NOT
# duplicated here: the model (EdgeConvBlock) already concatenates
# x[src] and x[dst] with edge_attr in every message, so each endpoint's
# individual node features are already delivered per-edge. edge_attr
# therefore holds only genuine pair-level quantities.
#
#   col 0: deta        (src eta - tgt eta)
#   col 1: dphi        (wrapped to [-pi, pi])
#   col 2: dR          sqrt(deta^2 + dphi^2)
#   col 3: dist3D      3D distance between the two positions
#   col 4: distXY      transverse (xy) distance
#   col 5: dE          energy difference   (tgt energy - src energy/momentum)
#   col 6: E_ratio     energy ratio        (tgt / (src + eps))
#   col 7: deltaTime   TOF-corrected timing difference (0 where timing invalid)
#   col 8: dtime_sig   raw timing difference / combined timing error (0 where invalid)
#   col 9: edge_type   0.0 = trk-ts edge, 1.0 = ts-ts edge
# ---------------------------------------------------------------------------
NUM_EDGE_FEATURES = 10
EDGE_TYPE_TRK_TS = 0.0
EDGE_TYPE_TS_TS = 1.0


def construct_graphs(event, data_trk, data_trkst, event_nodes,
                     node_feature_trkst, node_feature_trk):
    """Build one PyG Data graph for one event, containing BOTH
    track->trackster edges and (mirrored) trackster<->trackster edges.

    event_nodes: list of Node objects from nodes_construction.construct_nodes.
        Track nodes:     is_trackster=False, neighbours = nearby trackster indices.
        Trackster nodes: is_trackster=True,  neighbours = nearby OTHER trackster
                         indices (one-directional discovery: only tracksters
                         farther from the origin are listed).
    """
    try:
        # -------------------------------------------------------------------
        # Column index lookups — computed from the feature lists, NOT
        # hardcoded, so this file survives feature-list edits (e.g. the
        # barycenter_etaError/phiError removal) without silent misalignment.
        # -------------------------------------------------------------------
        trkster_offset = len(node_feature_trk)

        def trk_col(name):
            return node_feature_trk.index(name)

        def ts_col(name):
            return trkster_offset + node_feature_trkst.index(name)

        T_ETA = trk_col("track_hgcal_eta")
        T_PHI = trk_col("track_hgcal_phi")
        T_X = trk_col("track_hgcal_x")
        T_Y = trk_col("track_hgcal_y")
        T_Z = trk_col("track_hgcal_z")
        T_P = trk_col("track_p")
        T_TIME = trk_col("track_time_mtd")
        T_TIMEERR = trk_col("track_time_mtd_err")

        S_ETA = ts_col("barycenter_eta")
        S_PHI = ts_col("barycenter_phi")
        S_X = ts_col("barycenter_x")
        S_Y = ts_col("barycenter_y")
        S_Z = ts_col("barycenter_z")
        S_E = ts_col("raw_energy")
        S_TIME = ts_col("time")
        S_TIMEERR = ts_col("timeError")

        # Trackster truth labels for this event.
        # NOTE: with the rewritten label_utils.py, y is LIST-VALUED per
        # trackster: [] = unmatched, [5] or [5, 9] = matched sim indices.
        ts_labels_full = data_trkst[event]["y"]

        # -------------------------------------------------------------------
        # 1) SEEDING — which nodes enter the graph
        # -------------------------------------------------------------------
        # Track seeds: track nodes with at least one trackster neighbour.
        valid_tracks = [(node.index, node)
                        for node in event_nodes
                        if (not node.is_trackster) and len(node.neighbours) > 0]
        N_trk = len(valid_tracks)

        # Trackster seeds: trackster nodes with at least one ts-ts neighbour.
        valid_ts_nodes = [(node.index, node)
                          for node in event_nodes
                          if node.is_trackster and len(node.neighbours) > 0]

        # Collect every trackster that participates in the graph, deduplicated,
        # from BOTH sources: (a) tracks' neighbour lists (trk-ts) and
        # (b) trackster seeds themselves + their ts-ts neighbours.
        # Doing (b) here is what finally lets an isolated trackster (no nearby
        # track) enter the graph, per Option B.
        used_ts_indices = []
        seen = set()

        def _add_ts(ts_idx):
            if ts_idx not in seen:
                used_ts_indices.append(ts_idx)
                seen.add(ts_idx)
        # adding ts related to valid tracks
        for _, node in valid_tracks:
            for ts_idx in node.neighbours:
                _add_ts(ts_idx)
        # adding the origin ts in ts-ts linking then the the other one
        for ts_idx, node in valid_ts_nodes:
            _add_ts(ts_idx)                 # the origing 
            for other_idx in node.neighbours:
                _add_ts(other_idx)          # its partners

        # original_idx -> graph_local_position
        ts_index_map = {ts_idx: i for i, ts_idx in enumerate(used_ts_indices)}
        
        # graph_local_position -> original_idx
        ts_index_inv_map = {v: k for k, v in ts_index_map.items()}
        N_ts = len(ts_index_map)
        N_nodes = N_trk + N_ts

        if N_nodes == 0:
            return None

        # to go back from the trk local-graph index to original index  (needed to read
        # per-track fields that are NOT node columns, e.g. MTD position).
        trk_index_inv = [trk_idx for trk_idx, _ in valid_tracks]

        # -------------------------------------------------------------------
        # 2) EDGES
        # -------------------------------------------------------------------
        edges = [[], []]
        edge_types = []

        # (a) trk-ts edges: track -> trackster, one direction (unchanged).
        for local_trk_idx, (original_trk_idx, node) in enumerate(valid_tracks):
            for original_ts_idx in node.neighbours:
                local_ts_idx = N_trk + ts_index_map[original_ts_idx]
                edges[0].append(local_trk_idx)
                edges[1].append(local_ts_idx)
                edge_types.append(EDGE_TYPE_TRK_TS)

        # (b) ts-ts edges: MIRRORED into both directions.
        # nodes_construction discovers each true pair exactly once
        # (inner trackster -> outer trackster); here we deliberately add
        # both i->j and j->i so message passing is symmetric.
        #an important decision here needs to be revised.<------------------------------------
        for ts_idx, node in valid_ts_nodes:
            src_new = N_trk + ts_index_map[ts_idx]
            for other_idx in node.neighbours:
                tgt_new = N_trk + ts_index_map[other_idx]
                edges[0].append(src_new)
                edges[1].append(tgt_new)
                edge_types.append(EDGE_TYPE_TS_TS)
                edges[0].append(tgt_new)
                edges[1].append(src_new)
                edge_types.append(EDGE_TYPE_TS_TS)

        num_edges = len(edges[0])
        if num_edges == 0:
            return None

        # -------------------------------------------------------------------
        # 3) NODE FEATURE TABLE (zero-padded by node type, as before)
        # -------------------------------------------------------------------
        feature_dim = len(node_feature_trk) + len(node_feature_trkst)
        nodes = np.zeros((N_nodes, feature_dim), dtype=np.float64)

        for i, key in enumerate(node_feature_trk):
            arr = ak.to_numpy(data_trk[event][key])
            for new_idx, (trk_idx, _) in enumerate(valid_tracks):
                nodes[new_idx, i] = arr[trk_idx]

        for j, key in enumerate(node_feature_trkst):
            arr = ak.to_numpy(data_trkst[event][key])
            for ts_orig_idx, ts_new_idx in ts_index_map.items():
                nodes[N_trk + ts_new_idx, trkster_offset + j] = arr[ts_orig_idx]
                # it would be something like this but with more features of course
                            # track features   # trckst features
        # columns:            [eta, phi, p,     eta, phi, energy]
        # row 0 (TRACK):      [2.0, 0.3, 15.0,  0.0, 0.0, 0.0]   <- original track 0
        # row 1 (TRACK):      [2.3, 0.5, 18.0,  0.0, 0.0, 0.0]   <- original track 2
        # row 2 (TRACKSTER):  [0.0, 0.0, 0.0,   2.9, 0.4, 12.0]  <- original trackster 2
        # row 3 (TRACKSTER):  [0.0, 0.0, 0.0,   3.0, 0.6, 30.0]  <- original trackster 4
        # row 4 (TRACKSTER):  [0.0, 0.0, 0.0,   3.1, 0.9, 20.0]  <- original trackster 0
        # -------------------------------------------------------------------
        # 4) EDGE FEATURES — unified schema, computed per edge type
        # -------------------------------------------------------------------
        src = np.array(edges[0])
        tgt = np.array(edges[1])
        etype = np.array(edge_types)
        edge_features = np.zeros((num_edges, NUM_EDGE_FEATURES))

        is_tt = etype == EDGE_TYPE_TS_TS      # ts-ts edge mask(list of true and false values per edge answering if this edge is ts-ts link)
        is_kt = ~is_tt                        # trk-ts edge mask(same but the opposit. is the edge trk-ts link)

        # --- Per-endpoint quantities, chosen by edge type -----------------
        # For trk-ts: src is a track row  -> track columns.
        # For ts-ts:  src is a trackster  -> trackster columns.
        src_eta = np.where(is_tt, nodes[src, S_ETA], nodes[src, T_ETA])
        src_phi = np.where(is_tt, nodes[src, S_PHI], nodes[src, T_PHI])
        src_x = np.where(is_tt, nodes[src, S_X], nodes[src, T_X])
        src_y = np.where(is_tt, nodes[src, S_Y], nodes[src, T_Y])
        src_z = np.where(is_tt, nodes[src, S_Z], nodes[src, T_Z])
        # "Energy-like" quantity: trackster raw_energy vs track momentum
        # (same convention d1 uses: track p compared against trackster E).
        src_E = np.where(is_tt, nodes[src, S_E], nodes[src, T_P])
        src_time = np.where(is_tt, nodes[src, S_TIME], nodes[src, T_TIME])
        src_terr = np.where(is_tt, nodes[src, S_TIMEERR], nodes[src, T_TIMEERR])

        # tgt is a trackster for BOTH edge types.
        tgt_eta = nodes[tgt, S_ETA]
        tgt_phi = nodes[tgt, S_PHI]
        tgt_x = nodes[tgt, S_X]
        tgt_y = nodes[tgt, S_Y]
        tgt_z = nodes[tgt, S_Z]
        tgt_E = nodes[tgt, S_E]
        tgt_time = nodes[tgt, S_TIME]
        tgt_terr = nodes[tgt, S_TIMEERR]

        # --- cols 0-2: angular separation ---------------------------------
        deta = src_eta - tgt_eta
        dphi = src_phi - tgt_phi
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        edge_features[:, 0] = deta
        edge_features[:, 1] = dphi
        edge_features[:, 2] = np.sqrt(deta**2 + dphi**2)

        # --- cols 3-4: spatial separation ----------------------------------
        # For ts-ts this is barycenter-to-barycenter (also reused for TOF).
        # For trk-ts this is track-HGCAL-position to barycenter (the TOF for
        # trk-ts is computed separately from the MTD position below, matching
        # d1, which measures flight FROM the timing detector, not from HGCAL).
        dx = src_x - tgt_x
        dy = src_y - tgt_y
        dz = src_z - tgt_z
        dist3d = np.sqrt(dx**2 + dy**2 + dz**2)
        edge_features[:, 3] = dist3d
        edge_features[:, 4] = np.sqrt(dx**2 + dy**2)

        # --- cols 5-6: energy comparison -----------------------------------
        edge_features[:, 5] = tgt_E - src_E
        edge_features[:, 6] = tgt_E / (src_E + 1e-8)

        # --- cols 7-8: timing (deltaTime + significance) --------------------
        valid_time = (src_terr > 0) & (tgt_terr > 0)

        # TOF distance: ts-ts -> barycenter-to-barycenter distance (dist3d);
        # trk-ts -> MTD position to trackster barycenter, read per-edge from
        # data_trk (Option C: MTD fields live in data_trk, not the node table)
        tof_dist = dist3d.copy()
        if np.any(is_kt):
            mtd_x = ak.to_numpy(data_trk[event][MTD_X_KEY])
            mtd_y = ak.to_numpy(data_trk[event][MTD_Y_KEY])
            mtd_z = ak.to_numpy(data_trk[event][MTD_Z_KEY])
            kt_pos = np.where(is_kt)[0]
            for e in kt_pos:
                orig_trk = trk_index_inv[src[e]]
                ddx = mtd_x[orig_trk] - tgt_x[e]
                ddy = mtd_y[orig_trk] - tgt_y[e]
                ddz = mtd_z[orig_trk] - tgt_z[e]
                tof_dist[e] = np.sqrt(ddx**2 + ddy**2 + ddz**2)

        tof = tof_dist / C_CM_PER_NS
        # d1 conventions: trk-ts uses signed (ts_time - trk_time - tof);
        # ts-ts uses |t_i - t_j| - tof (symmetric, so identical for both
        # mirrored directions of the same pair).
        dt_kt = (tgt_time - src_time) - tof
        dt_tt = np.abs(src_time - tgt_time) - tof
        deltaTime = np.where(is_tt, dt_tt, dt_kt)
        edge_features[:, 7] = np.where(valid_time, deltaTime, 0.0)

        terr_comb = np.sqrt(src_terr**2 + tgt_terr**2 + 1e-8)
        edge_features[:, 8] = np.where(valid_time,
                                       (tgt_time - src_time) / terr_comb, 0.0)

        # --- col 9: edge type ------------------------------------------------
        edge_features[:, 9] = etype

        # -------------------------------------------------------------------
        # 5) LABELS — genuine same-particle check (overlap of sim-id lists)
        # -------------------------------------------------------------------
        # Trackster side: ts_labels_full[orig_idx] is a list of sim indices.
        # Track side: needs data_trk["sim_match"] (list of sim indices per
        # track), built in download() alongside all_valid_track_indices.
        # If absent, falls back to the old loose "trackster is real" label,
        # with a warning, so the pipeline still runs.
        have_trk_sim = "sim_match" in ak.fields(data_trk)
        if have_trk_sim:
            trk_sim_full = data_trk[event]["sim_match"]
        else:
            print(f"[graph_construction] WARNING (event {event}): "
                  f"data_trk has no 'sim_match' field - trk-ts labels fall "
                  f"back to trackster-validity only (known-loose).")

        # Pre-convert trackster label lists to python sets once per used ts.
        ts_simsets = {}
        for ts_orig_idx in used_ts_indices:
            ts_simsets[ts_orig_idx] = set(ak.to_list(ts_labels_full[ts_orig_idx]))

        y = np.zeros(num_edges)
        for e in range(num_edges):
            tgt_orig = ts_index_inv_map[tgt[e] - N_trk]
            tgt_sims = ts_simsets[tgt_orig]
            if etype[e] == EDGE_TYPE_TS_TS:# if the src is ts this will work
                src_orig = ts_index_inv_map[src[e] - N_trk]
                src_sims = ts_simsets[src_orig]
                # this edge will get 1 if the two tracksters share at least one sim
                y[e] = 1.0 if (src_sims & tgt_sims) else 0.0
            else:# if the src is trk this will work
                if have_trk_sim:
                    orig_trk = trk_index_inv[src[e]]
                    trk_sims = set(ak.to_list(trk_sim_full[orig_trk]))
                # this edge will get 1 if the track and the tracksters share at least one sim                    
                    y[e] = 1.0 if (trk_sims & tgt_sims) else 0.0
                else:
                    y[e] = 1.0 if len(tgt_sims) > 0 else 0.0

        # -------------------------------------------------------------------
        # 6) PACKAGE
        # -------------------------------------------------------------------
        graph = Data(
            x=torch.tensor(nodes, dtype=torch.float32),
            edge_index=torch.tensor(np.array(edges), dtype=torch.long),
            edge_attr=torch.tensor(edge_features, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            num_nodes=N_nodes,
        )
        # Bookkeeping used downstream (e.g. adding_addFeatures needs to map
        # trackster node rows back to original trackster indices).
        graph.n_trk = N_trk
        graph.n_ts = N_ts
        graph.trk_index_inv = torch.tensor(trk_index_inv, dtype=torch.long)
        graph.ts_index_inv = torch.tensor(
            [ts_index_inv_map[i] for i in range(N_ts)], dtype=torch.long)
        graph.edge_type = torch.tensor(etype, dtype=torch.float)
        graph.event = event
        return graph

    except Exception as e:
        # Deliberately NOT fully silent anymore: print the event and the
        # actual error so failures are visible during development.
        import traceback
        print(f"[graph_construction] event {event} failed: {e}")
        traceback.print_exc()
        return None
