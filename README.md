# aps360

## Architecture (ResNet18 + CBAM + Soft Voting)

%%{init: { "flowchart": { "htmlLabels": false } }}%%
flowchart TB
  I[Input 3xHxW] --> C1[Conv7x7 s2, 64] --> BN1[BN] --> R1[ReLU] --> MP[MaxPool 3x3 s2]

  %% 定义一个“隐形占位”样式
  classDef ghost fill:#0000,stroke:#0000,stroke-width:0;

  subgraph L1 ["Layer1 x2 blocks (64)"]
    L1pad[" "]:::ghost
    B1a["Conv3x3 s1,64 | BN | ReLU | Conv3x3 64 | BN | CBAM(64)"] --> ADD1a["+ skip"] --> R1a[ReLU]
    B1b["Conv3x3 s1,64 | BN | ReLU | Conv3x3 64 | BN | CBAM(64)"] --> ADD1b["+ skip"] --> R1b[ReLU]
  end

  subgraph L2 ["Layer2 x2 blocks (128, first s2)"]
    L2pad[" "]:::ghost           %% 关键：顶一个空白节点，避免标题被遮挡
    B2a["Conv3x3 s2,128 | BN | ReLU | Conv3x3 128 | BN | CBAM(128)\n+ Downsample skip"] --> ADD2a["+"]
    B2b["Conv3x3 s1,128 | BN | ReLU | Conv3x3 128 | BN | CBAM(128)"] --> ADD2b["+"]
  end

  subgraph L3 ["Layer3 x2 blocks (256, first s2)"]
    L3pad[" "]:::ghost
    B3a["Conv3x3 s2,256 | BN | ReLU | Conv3x3 256 | BN | CBAM(256)"] --> ADD3a["+"]
    B3b["Conv3x3 s1,256 | BN | ReLU | Conv3x3 256 | BN | CBAM(256)"] --> ADD3b["+"]
  end

  subgraph L4 ["Layer4 x2 blocks (512, first s2)"]
    L4pad[" "]:::ghost
    B4a["Conv3x3 s2,512 | BN | ReLU | Conv3x3 512 | BN | CBAM(512)"] --> ADD4a["+"]
    B4b["Conv3x3 s1,512 | BN | ReLU | Conv3x3 512 | BN | CBAM(512)"] --> ADD4b["+"]
  end

  MP --> L1 --> L2 --> L3 --> L4 --> GAP[AdaptiveAvgPool 1x1] --> FC[Linear 512->num_classes] --> SM[Softmax]
