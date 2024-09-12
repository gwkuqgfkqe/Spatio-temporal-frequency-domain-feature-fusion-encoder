
import torch
from torch import nn
#from sklearn import
import torch.nn.functional as F
def transpose_to_4d_input(x):
    while len(x.shape) < 4:
        x = x.unsqueeze(-1)
    return x.permute(0, 3, 1, 2)
def get_edges(dataset):
    edges = []
    if dataset == 'bci2a':
        edges = [
            (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
            (2, 3), (2, 7), (2, 8),
            (3, 4), (3, 9),
            (4, 5), (4, 10),
            (5, 6), (5, 11),
             (6, 12) ,(6, 13),
            (7, 8), (7, 14),
            (8, 9), (8, 14),
            (9, 10), (9, 15),
            (10, 11), (10, 16),
            (11, 12), (11, 17),
            (12, 13), (12, 18),
            (13, 18),
            (14, 15), (14, 19),
            (15, 16), (15, 19),
            (16, 17), (16, 20),
            (17, 18), (17, 21),
            (18, 21),
            (19, 20), (19, 22),
            (20, 21), (20, 22),
            (21, 22)
        ]
    return edges

def get_adjacency_matrix(n_electrodes, graph_strategy):
    adjacency = torch.zeros(n_electrodes, n_electrodes)
    if graph_strategy == 'CG':
        edges = get_edges('bci2a')
        for i, j in edges:
             adjacency[i - 1][j - 1] = 1
             adjacency[j - 1][i - 1] = 1
        for i in range(n_electrodes):
             adjacency[i][i] = 1
    

        #adjacency = torch.tensor(result)
    return adjacency




class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):

        self.max_norm = max_norm

        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):

        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )

        return super(Conv2dWithConstraint, self).forward(x)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class SpectralAttentionModule(nn.Module):
    def __init__(self, num_channels):
        super(SpectralAttentionModule, self).__init__()
        self.num_channels = num_channels
        self.query_w = nn.Parameter(torch.randn(13, 13)).unsqueeze(0).repeat(40, 1, 1).to(device)
        self.key_w = nn.Parameter(torch.randn(13, 13)).unsqueeze(0).repeat(40, 1, 1).to(device)
        self.value_w = nn.Parameter(torch.randn(13, 13)).unsqueeze(0).repeat(40, 1, 1).to(device)
        self.key = None
        self.query = None
        self.value = None
        nn.init.normal_(self.query_w)
        nn.init.normal_(self.key_w)
        nn.init.normal_(self.value_w)
        # batch_size, _, _ = x_fft.size()
        # self.query_w = self.query_w.unsqueeze(0).repeat(1, 1, 1)
        # self.key_w = self.key_w.unsqueeze(0).repeat(1, 1, 1)
        # self.value_w = self.value_w.unsqueeze(0).repeat(1, 1, 1)

    def forward(self, x):


        x_fft = torch.fft.rfft(x, dim=-1)


        num_freqs = x_fft.shape[-1]



        self.key_w = self.key_w.type(x_fft.dtype)
        self.query_w = self.query_w.type(x_fft.dtype)
        self.value_w = self.value_w.type(x_fft.dtype)
        # print(x_fft.transpose(-1, -2).shape,self.key_w.shape)
        self.key = torch.matmul(x_fft.transpose(-1, -2), self.key_w)
        self.query = torch.matmul(x_fft.transpose(-1, -2), self.query_w)
        # value = nn.Parameter(torch.randn(num_freqs, self.num_channels)).to(device)
        self.value = torch.matmul(x_fft.transpose(-1, -2), self.value_w)



        # query = query.to(value.dtype)
        # attn_logits = torch.matmul(query, key.transpose(-1, -2)).to(device)
        attn_weights = torch.matmul(self.query, self.key.transpose(-1, -2))
        attn_weights = torch.abs(attn_weights)
        weights = attn_weights
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_weights = attn_weights
        x_fft = x_fft

        # print(attn_weights.dtype, attn_weights.device)
        # print(value.dtype, value.device)
        attn_weights = attn_weights.type(x_fft.dtype)
        # print(attn_weights.dtype, attn_weights.device)
        # print(value.dtype, value.device)
        attn_output = torch.matmul(attn_weights, self.value)  # * x_fft.to(device)
        attn_output = attn_output.transpose(-1, -2)

        x_ifft = torch.fft.irfft(attn_output, dim=-1)
        # print(x_ifft.shape)

        return x_ifft


class GraphTemporalConvolution(nn.Module):
    def __init__(self, adjacency, in_channels, out_channels, kernel_length):

        super(GraphTemporalConvolution, self).__init__()

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.register_buffer('adjacency', adjacency)
        self.importance = nn.Parameter(torch.randn(in_channels, self.adjacency.size()[0], self.adjacency.size()[0]))
        self.temporal_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=(1, kernel_length), stride=1, bias=False, padding='same')

    def forward(self, x):

        x = torch.matmul(torch.mul(self.adjacency, self.importance), x)

        x = self.temporal_conv(x)
        return x


class GCNEEGNet(nn.Module):

    def __init__(self, n_channels, n_classes, input_window_size,
                 F1=8, D=2, F2=16, kernel_length=64, drop_p=0.5):
        super(GCNEEGNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.drop_p = drop_p
        self.input_windows_size = input_window_size
        adjacency = get_adjacency_matrix(self.n_channels, 'CG')
        self.frenquencyatten=SpectralAttentionModule(self.n_channels)
        self.block_temporal_conv = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_length),
                      stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            GraphTemporalConvolution(adjacency, self.F1, self.F1, 8),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            #nn.Dropout(p=self.drop_p),
            GraphTemporalConvolution(adjacency, self.F1, self.F1, 8),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            #nn.Dropout(p=self.drop_p),
            GraphTemporalConvolution(adjacency, self.F1, self.F1, 8),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            #nn.Dropout(p=self.drop_p)
        )
        self.block_spacial_conv = nn.Sequential(
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.n_channels, 1),
                                 max_norm=1, stride=1, bias=False, groups=self.F1, padding=(0, 0)),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=self.drop_p)
        )
        self.block_separable_conv = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16),
                      stride=1, bias=False, groups=self.F1 * self.D, padding=(0, 16 // 2)),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                      stride=1, bias=False, padding=(0, 0)),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=self.drop_p),
            nn.Flatten()
        )
        block_conv = nn.Sequential(
            self.block_temporal_conv,
            self.block_spacial_conv,
            self.block_separable_conv
        )
        out = block_conv(torch.ones((1, 1, self.n_channels, self.input_windows_size), dtype=torch.float32))
        self.block_classifier = nn.Sequential(
            nn.Linear(out.cpu().data.numpy().shape[1], self.n_classes),
            #nn.Softmax(dim=1)
        )

    def forward(self, x):#(64,22,1125)
        #x=self.frenquencyatten(x)
        x = transpose_to_4d_input(x)#(64,1,22,1125)
        x = self.block_temporal_conv(x)#(64,8,22,1125)
        x = self.block_spacial_conv(x)#(64,16,1,281)
        x = self.block_separable_conv(x)#(64,560)
        x = self.block_classifier(x)#（64，4）
        return x




from braindecode.models import EEGConformer,ATCNet,EEGNetv4

class STFEnc(nn.Module):
    def __init__(self, n_channels, n_classes, input_window_size,kernel_length=64):
        super(STFEnc, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.kernel_length=kernel_length
        #self.batch_size=batch_size
        self.input_windoe_size = input_window_size
        self.frenquencyatten=SpectralAttentionModule(self.n_channels)
        self.lstm = nn.LSTM(124, 124, 3, batch_first=True)
        self.gcnEEGNet =  GCNEEGNet(n_channels=self.n_channels, n_classes=self.n_classes,
                                     input_window_size=self.input_windoe_size)
        self.change_layer = nn.Linear(in_features=n_classes, out_features=self.input_windoe_size)
        self.convfinal = nn.Conv2d(1, self.n_channels, (1, self.kernel_length),
                                   stride=1, bias=False, padding='same')
        self.convfinal1 = nn.Conv1d( self.n_channels,1, self.kernel_length,
                                   stride=1, bias=False, padding='same')
        self.L=nn.Linear(self.input_windoe_size, self.n_classes)
        self.eegtransformer = ATCNet(n_channels=self.n_channels, n_classes=self.n_classes,n_times=self.input_windoe_size,
                                         add_log_softmax=False,input_window_seconds=4.5,sfreq=250,conv_block_pool_size_1=8,conv_block_pool_size_2=7)


    def forward(self, x):

        
        gcn_output = self.gcnEEGNet(x)#(64,4)


        
        # change_output = self.change_layer(gcn_output)#(64,11

        
        eegtransformer_output = self.eegtransformer(x)

        
        #fused_features = torch.cat((gcn_output, eegtransformer_output), dim=1)


        
        fused_features=(eegtransformer_output+gcn_output)



        return fused_features
# 示例用法

