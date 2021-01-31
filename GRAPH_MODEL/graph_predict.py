import os
import torch
import numpy as np
import dgl
import cv2
import pandas as pd
from GRAPH_MODEL.dataset import MyDataset
from GRAPH_MODEL.gated_gcn import GatedGCNNet
from unidecode import unidecode
from itertools import combinations


def _text_encode(text):
        text_encode = []
        for t in text.upper():   #   unidecode(           
            if t not in alphabet:
                text_encode.append(alphabet.index(" "))
            else:
                text_encode.append(alphabet.index(t))
        return np.array(text_encode)
def _load_annotation(annotation_file):
    texts = []
    text_lengths = []
    boxes = []
    labels = []
    original = []
    lines = annotation_file
    for line in lines:
        splits = line.split('\t')
        if len(splits) < 10:
            continue
        text_encode = _text_encode(splits[8])
        original.append(splits[8])
        text_lengths.append(text_encode.shape[0])
        texts.append(text_encode)
        box_info = [int(x) for x in splits[:8]]
    
        box_info.append(np.max(box_info[0::2]) - np.min(box_info[0::2]))
        box_info.append(np.max(box_info[1::2]) - np.min(box_info[1::2]))
        boxes.append([int(x) for x in box_info])
        labels.append(node_labels.index(splits[9]))
    return np.array(texts), np.array(text_lengths), np.array(boxes), np.array(labels),original

def _prepapre_pipeline(boxes, edge_data, text, text_length):
    box_min = boxes.min(0)
    box_max = boxes.max(0)

    boxes = (boxes - box_min) / (box_max - box_min)
    boxes = (boxes - 0.5) / 0.5

    edge_min = edge_data.min(0)
    edge_max = edge_data.max(0)

    edge_data = (edge_data - edge_min) / (edge_max - edge_min)
    edge_data = (edge_data - 0.5) / 0.5

    return boxes, edge_data, text, text_length

def load_data(annotation_file):
    texts, text_lengths, boxes, labels,original = _load_annotation(annotation_file)

    origin_boxes = boxes
    node_nums = text_lengths.shape[0]
    src = []
    dst = []
    edge_data = []
    for i in range(node_nums):
        for j in range(node_nums):
            if i == j:
                continue
                
            edata = []
            #y distance
            y_distance = np.mean(boxes[i][:8][1::2]) - np.mean(boxes[j][:8][1::2])
            x_distance = np.mean(boxes[i][:8][0::2]) - np.mean(boxes[j][:8][0::2])
            w = boxes[i, 8]
            h = boxes[i, 9]

            if np.abs(y_distance) >  3 * h:
                continue
            
            edata.append(y_distance)
            edata.append(x_distance)

            edge_data.append(edata)
            src.append(i)
            dst.append(j)

    edge_data = np.array(edge_data)
    g = dgl.DGLGraph()
    g = g.to('cuda:0')
    g.add_nodes(node_nums)
    g.add_edges(src, dst)
    

    boxes, edge_data, text, text_length = _prepapre_pipeline(boxes, edge_data, texts, text_lengths)

    boxes = torch.from_numpy(boxes).float()
    edge_data = torch.from_numpy(edge_data).float()

    tab_sizes_n = g.number_of_nodes()
    tab_snorm_n = torch.FloatTensor(tab_sizes_n, 1).fill_(1./float(tab_sizes_n))
    snorm_n = tab_snorm_n.sqrt()  

    tab_sizes_e = g.number_of_edges()
    tab_snorm_e = torch.FloatTensor(tab_sizes_e, 1).fill_(1./float(tab_sizes_e))
    snorm_e = tab_snorm_e.sqrt()

    max_length = text_lengths.max()
    new_text = [np.expand_dims(np.pad(t, (0, max_length - t.shape[0]), 'constant'), axis=0) for t in text]
    texts = np.concatenate(new_text)

    labels = torch.from_numpy(np.array(labels))
    texts = torch.from_numpy(np.array(texts))
    text_length = torch.from_numpy(np.array(text_length))

    graph_node_size = [g.number_of_nodes()]
    graph_edge_size = [g.number_of_edges()]

    return g, labels, boxes, edge_data, snorm_n, snorm_e, texts, text_length, origin_boxes, annotation_file, graph_node_size, graph_edge_size, original


def load_gate_gcn_net(device, checkpoint_path):
    net_params = {}
    net_params['in_dim_text'] = len(alphabet)
    net_params['in_dim_node'] = 10
    net_params['in_dim_edge'] = 2
    net_params['hidden_dim'] = 512
    net_params['out_dim'] = 512
    net_params['n_classes'] = 5
    net_params['in_feat_dropout'] = 0.
    net_params['dropout'] = 0.0
    net_params['L'] = 8
    net_params['readout'] = True
    net_params['graph_norm'] = True
    net_params['batch_norm'] = True
    net_params['residual'] = True
    net_params['device'] = 'cuda'
    net_params['OHEM'] = 3

    model = GatedGCNNet(net_params)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model

node_labels = ['other', 'company', 'address', 'date', 'total']
alphabet = ' "$(),-./0123456789:;ABCDEFGHIJKLMNOPQRSTUVWXYZ_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ'
checkpoint_path = '/media/thorpham/PROJECT/OCR-challenge/FULL_FOLLOW/GRAPH_MODEL/weights/graph_weight.pkl'

class GRAPH_MODEL:
    def __init__(self,node_labels,alphabet,weight,device="cuda"):
        self.node_labels = node_labels
        self.alphabet = alphabet
        self.model = load_gate_gcn_net(device, weight)
        self.device = device
        
    def predict(self,input):
        batch_graphs, batch_labels, batch_x, batch_e, batch_snorm_n, batch_snorm_e, text, text_length, boxes, ann_file, graph_node_size, graph_edge_size,original = load_data(input)
        # preprocessing data
        batch_x = batch_x.to(self.device)  # num x feat
        batch_e = batch_e.to(self.device)

        text = text.to(self.device)
        text_length =  text_length.to(self.device)        
        batch_snorm_e = batch_snorm_e.to(self.device)
        batch_snorm_n = batch_snorm_n.to(self.device) 
        batch_scores = self.model.forward(batch_graphs, batch_x, batch_e, text, text_length, batch_snorm_n, batch_snorm_e,graph_node_size, graph_edge_size)
        batch_scores = batch_scores.cpu().softmax(1)
        values, pred = batch_scores.max(1)
        length = pred.shape[0]
        results = []
        for i in range(length):
            if pred[i] == batch_labels[i]:
                if pred[i] == 0:
                    continue

                msg = "{}".format(node_labels[pred[i]])
            else:
                msg = "{}".format(node_labels[pred[i]])
            s = original[i]
            results.append(s + "  |||   " + msg)
        return results
