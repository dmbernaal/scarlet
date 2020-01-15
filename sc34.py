#export
def sconv_layer(ni, nf, padding, ks=3, stride=2):
    """
    Selu activation, no batch norm.
    """ 
    bias = False # SeLU Paper
    act_fn = Selu()
    layers = [nn.Conv2d(ni, nf, ks, padding=padding, stride=stride, bias=bias), nn.BatchNorm2d(nf, eps=1e-5, momentum=0.1), act_fn]
    return nn.Sequential(*layers)

class SCNN(nn.Sequential):
    def __init__(self, c_in, c_out, layers):
        super().__init__()

        # STEM
        sizes = [c_in, 32, 64, 64]
        stem = [sconv_layer(sizes[i], sizes[i+1], stride=2 if i==0 else 1, padding=1) for i in range(len(sizes)-1)]
                
        # BODY: 34 layer preset
        block_szs = [64, 128, 256, 512]
        
        if layers==34: 
            mult = 2 # expansion
            block_depths = [3, 4, 6, 3] # depth per block
            
        # blocks: mirrors layer size    
        block1 = [ nn.ModuleList([sconv_layer(block_szs[0], block_szs[0], stride=1, padding=1) for i in range(block_depths[0])]) for n in range(mult) ]
        block2 = [ nn.ModuleList([sconv_layer(block_szs[1]//2 if i==0 and n==0 else block_szs[1], block_szs[1], stride=2 if i==0 and n==0 else 1, padding=1) for i in range(block_depths[1])]) for n in range(mult) ]
        block3 = [ nn.ModuleList([sconv_layer(block_szs[2]//2 if i==0 and n==0 else block_szs[2], block_szs[2], stride=2 if i==0 and n==0 else 1, padding=1) for i in range(block_depths[2])]) for n in range(mult) ]
        block4 = [ nn.ModuleList([sconv_layer(block_szs[3]//2 if i==0 and n==0 else block_szs[3], block_szs[3], stride=2 if i==0 and n==0 else 1, padding=1) for i in range(block_depths[3])]) for n in range(mult) ]
        
        # body
        body = block1+block2+block3+block4
        
        self.stem = nn.Sequential(*stem)
        self.body = nn.Sequential(*body) 
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(block_szs[-1], c_out))
        
    def forward(self, x):
        x = self.stem(x)
        
        for seq in self.body:
            for l in seq:
                x = l(x)
        
        x = self.head(x)
        return x
        
        
def selu_normal_(tensor, mode1='fan_in', mode2='fan_out'):
    fan_in = nn.init._calculate_correct_fan(tensor, mode1)
    fan_out = nn.init._calculate_correct_fan(tensor, mode2)
    with torch.no_grad():
        return torch.randn(fan_in, fan_out) / math.sqrt(1./fan_in)

nn.init.selu_normal_ = selu_normal_ # adding modified init

def init_scnn(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)): nn.init.selu_normal_(m.weight)
    for l in m.children(): init_scnn(l)
        
def create_scnn_model(data, layers=34):
    c_in = data.train_ds[0][0].shape[0]
    c_out = data.c
    model = SCNN(c_in, c_out, layers)
    init_scnn(model)
    return model