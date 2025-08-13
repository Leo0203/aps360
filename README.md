# aps360

## Architecture (ResNet18 + CBAM + Soft Voting)

```mermaid
flowchart TB
  I[Input 3xHxW] --> C1[Conv7x7 s2, 64] --> BN1[BN] --> R1[ReLU] --> MP[MaxPool 3x3 s2]

  subgraph L1 ["Layer1 x2 blocks (64)"]
    B1a["Conv3x3 s1,64 | BN | ReLU | Conv3x3 64 | BN | CBAM(64)"] --> R1a[ReLU]
    B1b["Conv3x3 s1,64 | BN | ReLU | Conv3x3 64 | BN | CBAM(64)"] --> R1b[ReLU]
  end

  subgraph L2 ["Layer2 x2 blocks (128, first s2)"]
    B2a["Conv3x3 s2,128 | BN | ReLU | Conv3x3 128 | BN | CBAM(128)<br/>+ Downsample skip"]
    B2b["Conv3x3 s1,128 | BN | ReLU | Conv3x3 128 | BN | CBAM(128)"]
  end

  subgraph L3 ["Layer3 x2 blocks (256, first s2)"]
    B3a["Conv3x3 s2,256 | BN | ReLU | Conv3x3 256 | BN | CBAM(256)"]
    B3b["Conv3x3 s1,256 | BN | ReLU | Conv3x3 256 | BN | CBAM(256)"] 
  end

  subgraph L4 ["Layer4 x2 blocks (512, first s2)"]
    B4a["Conv3x3 s2,512 | BN | ReLU | Conv3x3 512 | BN | CBAM(512)"]
    B4b["Conv3x3 s1,512 | BN | ReLU | Conv3x3 512 | BN | CBAM(512)"] 
  end

  MP --> L1 --> L2 --> L3 --> L4 --> GAP[AdaptiveAvgPool 1x1] --> FC[Linear 512->num_classes] --> SM[Softmax]
