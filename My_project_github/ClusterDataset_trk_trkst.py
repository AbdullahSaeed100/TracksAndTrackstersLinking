import os
import os.path as osp
from glob import glob

import tqdm as tqdm
from tqdm import tqdm

import uproot as uproot
import awkward as ak
import numpy as np

from graph_construction_updated import construct_graphs
import torch
from torch_geometric.data import Dataset
from labels_utils_updated import assign_trackster_labels

from node_construction_updated import construct_nodes

#from graph_construction import construct_graphs
#from min_max_dist import adding_addFeatures
         
class ClusterDataset(Dataset):
    feature_trk = [
    "track_hgcal_eta",
    "track_hgcal_phi",
    #"track_hgcal_etaErr",
    #"track_hgcal_phiErr",
    "track_hgcal_x",
    "track_hgcal_y",
    "track_hgcal_z",
    "track_p",
    "track_hgcal_pt",        # replaces track_pt
    "track_nhits",
    "track_time_mtd",        # new — genuine node feature (confirmed: passed through as its own output column in d1)
    "track_time_mtd_err",    # new — same reason
    ]

    feature_trkst = [
    "barycenter_eta",
    "barycenter_phi",
    #"barycenter_etaError",
    #"barycenter_phiError",
    "barycenter_x",
    "barycenter_y",
    "barycenter_z",
    "raw_energy",
    "time",
    "timeError",
    "raw_em_energy",
    "raw_em_pt",
    "raw_pt",
    ]    
    model_feature_keys = np.arange(len(feature_trk) + len(feature_trkst))
    def __init__(self, root, histo_path, transform=None, test=False, pre_transform=None, pre_filter=None):
        self.test = test
        self.histo_path = histo_path
        '''
        this is the work of superclass that its init called here
        1.Stores root (so self.raw_dir/self.processed_dir become available, derived from it).
        2.Checks self.raw_file_names — are the expected raw files present?
        3.If not, calls self.download().
        4.Checks self.processed_file_names — are the expected processed files present?
        5.If not, calls self.process().
        '''
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return glob(f"{self.raw_dir}/*")

    @property
    def processed_file_names(self):
        return glob(f"{self.processed_dir}/data_*.pt")

    # use this to load the tree if some of file.keys() are duplicates ending with different numbers
    def load_branch_with_highest_cycle(self, file, branch_name):

        # Get all keys in the file
        all_keys = file.keys()

        # Filter keys that match the specified branch name
        matching_keys = [
            key for key in all_keys if key.startswith(branch_name)]

        if not matching_keys:
            raise ValueError(
                f"No branch with name '{branch_name}' found in the file.")

        # Find the key with the highest cycle
        highest_cycle_key = max(
            matching_keys, key=lambda key: int(key.split(";")[1]))

        # Load the branch with the highest cycle
        branch = file[highest_cycle_key]

        return branch
    # my comment: prepare raw data. the if else test has no meaning here, both if and else doing the same work.
    def download(self):
        if (self.test):
            files = glob(f"{self.histo_path}/*.root")
            
        else:
            files = glob(f"{self.histo_path}/*.root")
            print("working")
        for id in range(len(files)):
            file = uproot.open(files[id])
            alltracksters = self.load_branch_with_highest_cycle(file,'ticlDumper/ticlTracksterLinks')
            # allclusters = self.load_branch_with_highest_cycle(file,'ticlDumper/clusters')
            allsimtrackstersCP = self.load_branch_with_highest_cycle(file, 'ticlDumper/simtrackstersCP')
            allsimtrackstersSC = self.load_branch_with_highest_cycle(file, 'ticlDumper/simtrackstersSC')
            allassociations = self.load_branch_with_highest_cycle(file, 'ticlDumper/associations')
            alltracks = self.load_branch_with_highest_cycle(file, 'ticlDumper/tracks')

            # node_feature_keys_before_trkst = ["barycenter_eta","barycenter_phi","barycenter_etaError",
            #                                   "barycenter_phiError","barycenter_x","barycenter_y",
            #                                   "barycenter_z","raw_energy","time","timeError",
            #                                   "raw_em_energy","raw_em_pt","raw_pt"]
            # node_feature_keys_before_trk = ["track_hgcal_eta","track_hgcal_phi","track_hgcal_etaErr",
            #                                 "track_hgcal_phiErr","track_hgcal_x","track_hgcal_y",
            #                                 "track_hgcal_z","track_p","track_pt","track_nhits"]
            #my comment: pulls just the listed fields out of a tree into an awkward array. 
            mtd_pos_keys = [
                "track_pos_mtd/track_pos_mtd.theVector.theX",
                "track_pos_mtd/track_pos_mtd.theVector.theY",
                "track_pos_mtd/track_pos_mtd.theVector.theZ",
            ]
            data_trkst = alltracksters.arrays(self.feature_trkst+["vertices_x", "vertices_y", "vertices_z"])
            data_trk = alltracks.arrays(self.feature_trk+mtd_pos_keys+['track_pt' ,'track_quality','track_missing_outer_hits',"track_id"]) # my comment:data_trk = every track's 10 features, all events. the track node features.
            # my comment: "pdgID" seems the only one used here.
            #simTracksters = allsimtrackstersSC.arrays(['raw_em_energy','raw_energy', 'regressed_energy',
            #                                           'pdgID', 'NTracksters','NClusters'])
            tsCP=allsimtrackstersCP.arrays(["trackIdx","regressed_energy","regressed_pt"])
            # my comment for below all_trackstrs and all_tracks: used for truth-matching (assign_trackster_labels) and min/max distance (min_max_dist.py), which need hit-level detail
            #my comment: we can merge both this all_tracksters and data_trkst because they are from the same tree but we need to consider other things.
            # all_tracksters = alltracksters.arrays(["barycenter_phi","barycenter_eta","raw_energy",
            #                                        "time", "timeError","barycenter_etaError",
            #                                        "barycenter_phiError", 'barycenter_x', 'barycenter_y',
            #                                        'barycenter_z','vertices_x','vertices_y','vertices_z',
            #                                        'vertices_indexes','raw_em_pt', 'raw_pt'])
            associations = allassociations.arrays(['ticlTracksterLinks_recoToSim_CP_sharedE',
                                                   "ticlTracksterLinks_recoToSim_CP",
                                                   "ticlTracksterLinks_recoToSim_CP_score",
                                                   "ticlTracksterLinks_simToReco_CP_score",
                                                   "ticlTracksterLinks_simToReco_CP",
                                                   "ticlTracksterLinks_simToReco_CP_sharedE"])
           
            # trks=alltracks.arrays(["track_hgcal_eta","track_hgcal_phi","track_pt","track_id",
            #                        'track_hgcal_etaErr','track_hgcal_phiErr','track_hgcal_etaphiCov',
            #                        'track_p','track_beta', 'track_quality',
            #                        'track_missing_outer_hits','track_nhits', 'track_time_quality',
            #                        'track_time','track_missing_inner_hits', 'track_hgcal_pt',
            #                        'track_hgcal_xyCov', 'track_hgcal_yErr','track_hgcal_xErr',
            #                        'track_hgcal_z', 'track_hgcal_y', 'track_hgcal_x'])
        
            all_valid_track_indices = []
            all_track_sim_match = []
            print("starting the node construction")
            for ev in tqdm(range(len(tsCP))):
                
                    
                valid_indices = set()
                track_ids = ak.to_numpy(data_trk[ev]["track_id"])
                n_trk = len(track_ids)
                track_sim_match = [-1] * n_trk   # one entry per track, aligned by position; -1 = unmatched

                for sim_idx in range(len(tsCP[ev]["trackIdx"])):
                    for track_id_value in tsCP[ev]["trackIdx"][sim_idx]:   # loop ALL, not just sims[0]
                        matches = np.where(track_ids == track_id_value)[0]
                        if len(matches) == 0:
                            continue
                        track_position = matches[0]
                        valid_indices.add(track_position)
                        track_sim_match[track_position] = sim_idx   # store BY POSITION, not by ID

                all_track_sim_match.append(track_sim_match)
                all_valid_track_indices.append(list(valid_indices))

            #labels_dict = assign_trackster_labels(associations, all_tracksters)
            print("start labe")
            labels_dict = assign_trackster_labels(associations, data_trkst)
            print("finish Label")
            data_trkst["y"]        = labels_dict["y"]
            data_trkst["score"]    = labels_dict["score"]
            data_trkst["shared_e"] = labels_dict["shared_e"]
            data_trkst["vertices"] = ak.concatenate([data_trkst["vertices_x"][:, :, :, np.newaxis], data_trkst["vertices_y"][:, :, :, np.newaxis], data_trkst["vertices_z"][:, :, :, np.newaxis]], axis=-1)
            #data_trkst["vertices"] = np.stack([all_tracksters["vertices_x"], all_tracksters["vertices_y"], all_tracksters["vertices_z"]],axis=-1)
            all_nodes = construct_nodes(data_trk, data_trkst, all_valid_track_indices, tsCP)
            data_trkst["barycenter_eta"] = np.abs(data_trkst["barycenter_eta"])
            data_trkst["barycenter_z"] = np.abs(data_trkst["barycenter_z"])            
            data_trk["track_hgcal_eta"] = np.abs(data_trk["track_hgcal_eta"])
            data_trk["track_hgcal_z"] = np.abs(data_trk["track_hgcal_z"])
            data_trk["sim_match"] = ak.Array(all_track_sim_match)#this is used in graph_construction 
            torch.save(all_nodes, osp.join(self.raw_dir, f'all_nodes_id_{id}.pt'))
            torch.save(data_trkst, osp.join(self.raw_dir, f'data_trkst_id_{id}.pt'))
            torch.save(data_trk, osp.join(self.raw_dir, f'data_trk_id_{id}.pt'))

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            if "all_nodes" not in raw_path :
                continue 
            print(f"Loading: {raw_path}")
            # Extract ID from filename (e.g. 'data_trkst_id_5.pt')
            file_id = int(os.path.basename(raw_path).split('_')[-1].split('.')[0])

            # Load associated trk and trkst files
            trk_path = osp.join(self.raw_dir, f'data_trk_id_{file_id}.pt')
            trkst_path = osp.join(self.raw_dir, f'data_trkst_id_{file_id}.pt')
            nodes_path = osp.join(self.raw_dir, f'all_nodes_id_{file_id}.pt')
            if not osp.exists(trk_path) or not osp.exists(trkst_path) or not osp.exists(nodes_path):
                print(f"Missing files for ID {file_id}, skipping.")
                continue

            data_trk = torch.load(trk_path, weights_only=False)
            data_trkst = torch.load(trkst_path, weights_only=False)
            all_nodes = torch.load(nodes_path, weights_only=False)
            graph_list = []
            #print(" len(all_nodes) ", len(all_nodes))
            #print(" len(data_trkst[y] ", len(data_trkst["y"]))
            print("starting the graph construction")
            for event in tqdm(range(len(all_nodes))):
                
                # Reuse your own graph construction logic here
                #print(all_nodes[event])
                #intialgraph = construct_graphs(event, data_trk, data_trkst, all_nodes[event],self.feature_trkst, self.feature_trk)
                #graph = adding_addFeatures(event, intialgraph, data_trkst, all_nodes[event],self.feature_trk ,self.feature_trkst)
                graph = construct_graphs(event, data_trk, data_trkst, all_nodes[event],self.feature_trkst, self.feature_trk)

                if graph is None or graph.x.size(0) == 0 or graph.edge_index.size(1) == 0:
                    continue
                if self.pre_filter is not None and not self.pre_filter(graph):
                    continue
                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)
                    #print(" graph ", graph)
                torch.save(graph, osp.join(self.processed_dir, f'data_{idx}.pt'))# this is better
                idx += 1
    
       
    def len(self):
        return len(self.processed_file_names)
    '''
    def get(self, idx):
        #data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        #data = torch.load(osp.join(self.processed_dir, f'data_*_{idx}.pt'), weights_only=False)
        pattern = osp.join(self.processed_dir, f"data_*_{idx}.pt")
        matches = glob(pattern)
        print(" matches ", matches)
        if len(matches) == 0:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        
        data_list = [torch.load(f, weights_only=False) for f in matches]        
        if len(data_list) > 1:
            from torch_geometric.data import Batch
            data = Batch.from_data_list(data_list)
        else:
            data = data_list[0]
        print(data)
        return data
    '''
    # def get(self, idx):
    #     pattern = osp.join(self.processed_dir, f"data_*_{idx}.pt")
    #     matches = glob(pattern)
    #     print("matches:", matches)
    
    #     if len(matches) == 0:
    #         print(f"[Warning] Skipping missing file for pattern: {pattern}")
    #         return None  # Skip missing samples
        
    #     data_list = [torch.load(f, weights_only=False) for f in matches]

    #     if len(data_list) > 1:
    #         from torch_geometric.data import Batch
    #         data = Batch.from_data_list(data_list)
    #     else:
    #         data = data_list[0]

    #     return data
    def get(self, idx):
        path = osp.join(self.processed_dir, f"data_{idx}.pt")
        if not osp.exists(path):
            print(f"[Warning] Skipping missing file: {path}")
            return None
        # Runtime transforms are applied by PyG Dataset.__getitem__().
        # Applying self.transform here as well would preprocess normal
        # dataset[index]/DataLoader samples twice.
        return torch.load(path, weights_only=False)
