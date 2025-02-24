import torch.nn as nn
from torch.nn import functional as F
import torch
from .gatconv import GATConv

    
class DUPLEX(nn.Module):
    def __init__(self, args):
        super(DUPLEX, self).__init__()
        self.args = args
        self.activation = F.relu
        self.am_layers = nn.ModuleList()
        self.ph_layers = nn.ModuleList()
        self.dropout = nn.Dropout(args.dr_rate)
        self.n_layers = args.n_layers
        self.fusion_layer = args.fusion_layer
        # ----- am layer -----
        self.am_layers.append(GATConv(args.input_dim, args.hidden_dim//args.head, num_heads=args.head))
        self.ph_layers.append(GATConv(args.input_dim, args.hidden_dim//args.head, num_heads=args.head))
        if args.fusion == 'add':
            for i in range(1, args.n_layers-1):
                self.am_layers.append(GATConv(args.hidden_dim, args.hidden_dim//args.head, num_heads=args.head))
                self.ph_layers.append(GATConv(args.hidden_dim, args.hidden_dim//args.head, num_heads=args.head))
            if self.n_layers==1:
                self.am_agg_layer = GATConv(args.input_dim, args.hidden_dim//args.head, num_heads=args.head)
                self.ph_agg_layer = GATConv(args.input_dim, args.hidden_dim//args.head, num_heads=args.head)
            else:
                self.am_agg_layer = GATConv(args.hidden_dim, args.hidden_dim//args.head, num_heads=args.head)
                self.ph_agg_layer = GATConv(args.hidden_dim, args.hidden_dim//args.head, num_heads=args.head)
            self.am_layers.append(GATConv(args.hidden_dim, args.output_dim, num_heads=args.head))
            self.ph_layers.append(GATConv(args.hidden_dim, args.output_dim, num_heads=args.head))

        else:
            for i in range(0, args.n_layers-2):
                self.am_layers.append(GATConv(args.hidden_dim, args.hidden_dim//args.head, num_heads=args.head))
                self.ph_layers.append(GATConv(args.hidden_dim, args.hidden_dim//args.head, num_heads=args.head))
            self.ph_layers.append(GATConv(args.hidden_dim, args.output_dim, num_heads=args.head))
            self.am_layers.append(GATConv(args.hidden_dim, args.output_dim, num_heads=args.head))
        self.projector = nn.Linear(args.output_dim*2, args.output_dim)

    def forward(self, g, input_am, input_ph):
        h_am = input_am
        h_ph = input_ph
        for i in range(self.args.n_layers):
            am_layer = self.am_layers[i]
            ph_layer = self.ph_layers[i]

            if self.args.fusion == 'add':
                if i == self.fusion_layer:
                    h_am_agg = self.am_agg_layer(g, h_ph) # agg new
                    h_ph_agg = self.ph_agg_layer(g, h_am) # agg new

                    h_am = am_layer(g, h_am) # am_new
                    h_ph = ph_layer(g, h_ph) # ph_new
                
                    h_am = h_am + h_am_agg
                    h_ph = h_ph + h_ph_agg
                else:
                    h_am = am_layer(g, h_am) # am_new
                    h_ph = ph_layer(g, h_ph) # ph_new
            else:
                h_am = am_layer(g, h_am) # am_new
                h_ph = ph_layer(g, h_ph)
                
            if i < self.n_layers-1:
                h_am = h_am.flatten(1)
                h_am = self.activation(h_am)
                h_am = self.dropout(h_am)
                h_ph = h_ph.flatten(1)
                h_ph = self.activation(h_ph)
                h_ph = self.dropout(h_ph)
            else:
                h_am = h_am.mean(1)
                h_ph = h_ph.mean(1)
            if i < self.n_layers-1:
                h_am = self.activation(h_am)
                h_am = self.dropout(h_am)
                h_ph = self.activation(h_ph)
                h_ph = self.dropout(h_ph)
            else:
                output = self.projector(torch.cat((h_am, h_ph), dim=-1))
                # pred_score = self.classifier(torch.cat((h_am, h_ph), dim=-1))
                # continue
        return output
